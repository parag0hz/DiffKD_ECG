"""
Generic single-teacher KD trainer.

Trains a target model on one view using a single frozen teacher on another view.
Loss: L = BCE + lambda_kd * MKD(teacher_logits → target_logits)

Usage:
    python dafd_mvkt/train_kd_view.py \\
        --config       dafd_mvkt/configs/hierarchical_chains_6.yaml \\
        --data_dir     comper_repo/ptb_xl \\
        --lead         II \\
        --teacher_view 12L100 \\
        --target_view  1L500 \\
        --teacher_ckpt dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt \\
        --output_name  H02_stage2_1L500_from_12L100_seed0 \\
        --seed         0
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset, SUPERCLASSES
from losses.mkd import multi_label_kd_loss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init


# ── view metadata ─────────────────────────────────────────────────────────────

VIEW_DEFS: dict[str, dict] = {
    "12L500": {"in_channels": 12, "input_key": "teacher_12l_500"},
    "12L100": {"in_channels": 12, "input_key": "teacher_12l_100"},
    "12L50":  {"in_channels": 12, "input_key": "teacher_12l_50"},
    "1L500":  {"in_channels":  1, "input_key": "teacher_1l_500"},
    "1L100":  {"in_channels":  1, "input_key": "teacher_1l_100"},
    "1L50":   {"in_channels":  1, "input_key": "teacher_1l_50"},
}


def _build_model(view: str, num_classes: int = 5) -> ResNet1d:
    in_ch = VIEW_DEFS[view]["in_channels"]
    return ResNet1d(
        in_channels=in_ch, num_classes=num_classes,
        layers=[3, 4, 6, 3], base_channels=64, proj_dim=128, dropout=0.0,
    )


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_eval(
    model: ResNet1d,
    loader: DataLoader,
    input_key: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch[input_key].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels, all_ids


# ── core training function ────────────────────────────────────────────────────

def train_kd_view(
    *,
    chain_cfg:     dict,
    data_dir:      str,
    lead:          str,
    teacher_view:  str,
    target_view:   str,
    teacher_ckpt:  str,
    output_name:   str,
    output_dir:    Path,
    target_init_ckpt: str | None = None,
    lambda_kd:     float = 1.0,
    temperature:   float = 2.0,
    seed:          int   = 0,
    epochs:        int | None = None,
    batch_size:    int | None = None,
    lr:            float | None = None,
    debug:         bool  = False,
) -> dict:
    """
    Train target_view model from frozen teacher_view teacher.

    Returns metrics dict (test set with tuned thresholds).
    Saves checkpoint and JSON metrics under output_dir.
    """
    tc = chain_cfg["training"]
    ep    = epochs    if epochs    is not None else tc["epochs"]
    bs    = batch_size if batch_size is not None else tc["batch_size"]
    lr_   = lr         if lr         is not None else tc["lr"]
    nw    = tc["num_workers"]
    pw    = tc.get("persistent_workers", False)
    wd    = tc["weight_decay"]
    gc    = tc["grad_clip"]
    lkd   = lambda_kd
    temp  = temperature

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"KD: {teacher_view} → {target_view}")
    print(f"output: {output_name}  device: {device}  seed: {seed}")
    print(f"{'='*60}")

    # ── load frozen teacher ───────────────────────────────────────────────────
    t_view_def = VIEW_DEFS[teacher_view]
    teacher = _build_model(teacher_view)
    load_checkpoint(teacher_ckpt, teacher, device)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── build target model ────────────────────────────────────────────────────
    tgt_view_def = VIEW_DEFS[target_view]
    target = _build_model(target_view).to(device)

    if target_init_ckpt and Path(target_init_ckpt).exists():
        load_encoder_init(target, target_init_ckpt, strict=False, device=device)
        print(f"Target initialized from SimCLR: {target_init_ckpt}")
    else:
        if target_init_ckpt:
            print(f"[WARN] SimCLR init not found, using random init: {target_init_ckpt}")
        else:
            print("Target: random init")

    t_key   = t_view_def["input_key"]
    tgt_key = tgt_view_def["input_key"]

    n_params = sum(p.numel() for p in target.parameters() if p.requires_grad)
    print(f"Target params: {n_params/1e6:.2f}M")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = PTBXLMultiTeacherDataset(data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=4 if nw > 0 else None)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=4 if nw > 0 else None)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=4 if nw > 0 else None)

    # ── debug ─────────────────────────────────────────────────────────────────
    if debug:
        first = next(iter(train_loader))
        print(f"\n[DEBUG] teacher_view={teacher_view}  key={t_key}")
        print(f"  teacher input shape: {first[t_key].shape}")
        print(f"[DEBUG] target_view={target_view}  key={tgt_key}")
        print(f"  target  input shape: {first[tgt_key].shape}")
        with torch.no_grad():
            t_out = teacher(first[t_key].to(device), return_features=False)
            tgt_out = target(first[tgt_key].to(device), return_features=False)
        print(f"  teacher logits shape: {t_out['logits'].shape}")
        print(f"  target  logits shape: {tgt_out['logits'].shape}")
        b_bce = nn.BCEWithLogitsLoss()(tgt_out["logits"], first["y"].to(device))
        b_kd  = multi_label_kd_loss(tgt_out["logits"], t_out["logits"], temp)
        print(f"  first batch BCE={b_bce:.4f}  KD={b_kd:.4f}")

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    bce_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(target.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    # ── paths ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    ckpt_path = output_dir / f"{output_name}_best.pt"
    log_path  = log_dir / f"{output_name}.csv"

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    log_rows  = []

    for epoch in range(1, ep + 1):
        target.train()
        ep_bce = ep_kd = ep_total = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}", dynamic_ncols=True,
                          leave=False):
            y       = batch["y"].to(device)
            tgt_in  = batch[tgt_key].to(device)
            tgt_out = target(tgt_in, return_features=False)
            s_logits = tgt_out["logits"]

            with torch.no_grad():
                t_in  = batch[t_key].to(device)
                t_out = teacher(t_in, return_features=False)
                t_logits = t_out["logits"]

            loss_bce = bce_fn(s_logits, y)
            loss_kd  = multi_label_kd_loss(s_logits, t_logits, temp)
            loss     = loss_bce + lkd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(target.parameters(), gc)
            optimizer.step()

            ep_bce   += loss_bce.item()
            ep_kd    += loss_kd.item()
            ep_total += loss.item()
            n_batches += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(target, val_loader, tgt_key, device)
        val_m   = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_m["macro_auc"]

        avg_bce = ep_bce / n_batches
        avg_kd  = ep_kd  / n_batches
        avg_tot = ep_total / n_batches
        is_best = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, target, epoch,
                            {"val_auc": val_auc},
                            {"teacher_view": teacher_view, "target_view": target_view},
                            extra={"thresholds": None})

        print(f"Ep {epoch:3d} | bce={avg_bce:.4f} kd={avg_kd:.4f} tot={avg_tot:.4f} "
              f"| val_auc={val_auc:.4f} {'★' if is_best else ''}")

        log_rows.append({"epoch": epoch, "bce": round(avg_bce, 5),
                         "kd": round(avg_kd, 5), "total": round(avg_tot, 5),
                         "val_auc": round(val_auc, 5),
                         "best_val_auc": round(best_auc, 5)})

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── evaluation ────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, target, device)

    val_probs, val_labels, _ = _run_eval(target, val_loader, tgt_key, device)
    grid = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(target, test_loader, tgt_key, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    metrics = {
        "output_name":       output_name,
        "teacher_view":      teacher_view,
        "target_view":       target_view,
        "macro_auc":         round(m05["macro_auc"], 5),
        "macro_f1_0_5":      round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned":    round(m_tune.get("macro_f1", 0), 5),
        "val_auc_best":      round(best_auc, 5),
        "per_class_thresholds": {
            SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
        },
        "_meta": {"seed": seed, "lambda_kd": lkd, "temperature": temp,
                  "teacher_ckpt": str(teacher_ckpt), "target_init": str(target_init_ckpt)},
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    # save val metrics
    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    val_metrics = {
        "output_name": output_name,
        "macro_auc":      round(val_m05["macro_auc"], 5),
        "macro_f1_0_5":   round(val_m05.get("macro_f1", 0), 5),
        "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5),
    }
    with open(output_dir / f"metrics_val_{output_name}.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    with open(output_dir / f"metrics_test_{output_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # predictions CSV
    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows = []
    for i, rid in enumerate(test_ids):
        row = {"record_id": rid}
        for ci, c in enumerate(SUPERCLASSES):
            row[f"y_{c}"]    = int(test_labels[i, ci])
            row[f"prob_{c}"] = round(float(test_probs[i, ci]), 5)
            row[f"pred_{c}"] = int(preds[i, ci])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / f"predictions_test_{output_name}.csv", index=False)

    print(f"\nAUC={metrics['macro_auc']:.4f}  F1_tuned={metrics['macro_f1_tuned']:.4f}")
    print(f"Saved: {output_dir / f'metrics_test_{output_name}.json'}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",           default="dafd_mvkt/configs/hierarchical_chains_6.yaml")
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--lead",             default="II")
    p.add_argument("--teacher_view",     required=True, choices=list(VIEW_DEFS))
    p.add_argument("--target_view",      required=True, choices=list(VIEW_DEFS))
    p.add_argument("--teacher_ckpt",     required=True)
    p.add_argument("--target_init_ckpt", default=None)
    p.add_argument("--output_name",      required=True)
    p.add_argument("--output_dir",       default="dafd_mvkt/outputs/hierchain")
    p.add_argument("--lambda_kd",        type=float, default=1.0)
    p.add_argument("--temperature",      type=float, default=2.0)
    p.add_argument("--seed",             type=int, default=0)
    p.add_argument("--epochs",           type=int, default=None)
    p.add_argument("--batch_size",       type=int, default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--debug",            action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        chain_cfg = yaml.safe_load(f)

    train_kd_view(
        chain_cfg=chain_cfg,
        data_dir=args.data_dir,
        lead=args.lead,
        teacher_view=args.teacher_view,
        target_view=args.target_view,
        teacher_ckpt=args.teacher_ckpt,
        output_name=args.output_name,
        output_dir=Path(args.output_dir),
        target_init_ckpt=args.target_init_ckpt,
        lambda_kd=args.lambda_kd,
        temperature=args.temperature,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        debug=args.debug,
    )
