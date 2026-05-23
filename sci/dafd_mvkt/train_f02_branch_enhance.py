"""
F02 Branch CRF Enhancement Trainer (Phase 5).

Retrains F02 branch teachers from 12L100 with CRF regularization.

Loss:
    L_branch = BCE
             + KD(12L100 → branch)
             + beta_branch_crf * CRF(12L100, branch)

CRF only — no FeatureKD from 12L100 to branch in this phase.

After branch training, runs final student using best weights from prior phases.

Usage:
    python dafd_mvkt/train_f02_branch_enhance.py \\
        --config         dafd_mvkt/configs/adaptive_50hz.yaml \\
        --data_dir       comper_repo/ptb_xl \\
        --t12l100_ckpt   dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt \\
        --enhance_1l500  --enhance_1l100 \\
        --beta_branch_crf 0.05 \\
        --variant_id     F02BR03 \\
        --output_dir     dafd_mvkt/outputs/f02_opt/branch
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
from losses.contrastive import cross_rate_contrastive_loss
from losses.mkd import multi_label_kd_loss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed

F02_AUC = 0.8447; F02_F1 = 0.6275
MVKT_AUC = 0.843; MVKT_F1 = 0.626

T12L100_KEY = "teacher_12l_100"   # [B, 12, 1000]
T1L500_KEY  = "teacher_1l_500"    # [B,  1, 5000]
T1L100_KEY  = "teacher_1l_100"    # [B,  1, 1000]
STUDENT_KEY = "student_x"         # [B,  1,  500]
CRF_T = 0.07


def _build_model(in_channels: int) -> ResNet1d:
    return ResNet1d(in_channels=in_channels, num_classes=5,
                    layers=[3, 4, 6, 3], base_channels=64,
                    proj_dim=128, dropout=0.0)


@torch.no_grad()
def _run_eval(model, loader, key, device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        out = model(batch[key].to(device), return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return 1/(1+np.exp(-logits)), labels, all_ids


def _save_metrics(model, val_loader, test_loader, key, device,
                  out_name, out_dir, meta):
    val_probs, val_labels, _ = _run_eval(model, val_loader, key, device)
    grid = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)
    test_probs, test_labels, test_ids = _run_eval(model, test_loader, key, device)
    m05 = compute_metrics(test_labels, test_probs, threshold=0.5)
    mt  = compute_metrics(test_labels, test_probs, threshold=thresholds)
    metrics = {"output_name": out_name,
               "macro_auc": round(m05["macro_auc"], 5),
               "macro_f1_0_5": round(m05.get("macro_f1", 0), 5),
               "macro_f1_tuned": round(mt.get("macro_f1", 0), 5),
               "per_class_thresholds": {SUPERCLASSES[i]: round(float(thresholds[i]), 4)
                                        for i in range(5)},
               "_meta": meta}
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(mt.get(f"f1_{c}", float("nan")), 5)
    val_m = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_mt = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc": round(val_m["macro_auc"], 5),
                   "macro_f1_tuned": round(val_mt.get("macro_f1", 0), 5)}, f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows = [{"record_id": rid,
              **{f"y_{c}": int(test_labels[i, ci]) for ci, c in enumerate(SUPERCLASSES)},
              **{f"prob_{c}": round(float(test_probs[i, ci]), 5) for ci, c in enumerate(SUPERCLASSES)},
              **{f"pred_{c}": int(preds[i, ci]) for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_test_{out_name}.csv", index=False)
    return metrics


def _train_branch(
    *,
    branch_view: str,      # "1L500" or "1L100"
    branch_in_ch: int,
    branch_key: str,
    t12l100: nn.Module,
    t12l100_key: str,
    cfg: dict,
    data_dir: str,
    lead: str,
    beta_crf: float,
    lambda_kd: float,
    temperature: float,
    seed: int,
    variant_id: str,
    out_dir: Path,
    device: torch.device,
    init_ckpt: str | None = None,
) -> tuple[nn.Module, str]:
    """Train one branch with CRF-regularized KD from 12L100. Returns (branch, ckpt_path)."""
    tc  = cfg["training"]
    ep  = tc.get("epochs", 100)
    bs  = tc.get("batch_size", 512)
    lr_ = tc.get("lr", 1e-3)
    nw  = tc.get("num_workers", 4)
    pw  = tc.get("persistent_workers", True)
    wd  = tc.get("weight_decay", 1e-4)
    gc  = tc.get("grad_clip", 1.0)

    set_seed(seed)
    branch_name = f"f02_branch_{branch_view.lower()}_{variant_id}_seed{seed}"
    ckpt_path   = out_dir / f"{branch_name}_best.pt"
    log_path    = out_dir.parent / "logs" / f"{branch_name}.csv"

    print(f"\n{'─'*50}")
    print(f"Branch {branch_view}: CRF-KD from 12L100  beta_crf={beta_crf}")
    print(f"output: {ckpt_path}")

    branch = _build_model(branch_in_ch).to(device)
    if init_ckpt and Path(init_ckpt).exists():
        load_checkpoint(init_ckpt, branch, device)
        print(f"  Init from: {init_ckpt}")
    else:
        print("  Init: random")

    train_ds = PTBXLMultiTeacherDataset(data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(data_dir, split="val",   lead=lead)
    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)

    bce_fn    = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    best_auc = 0.0
    log_rows  = []

    for epoch in range(1, ep + 1):
        branch.train()
        ep_bce = ep_kd = ep_crf = ep_tot = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Branch-{branch_view} Ep {epoch}/{ep}",
                          dynamic_ncols=True, leave=False):
            y = batch["y"].to(device)
            b_out  = branch(batch[branch_key].to(device), return_features=True)
            b_logits = b_out["logits"]

            with torch.no_grad():
                t_out = t12l100(batch[t12l100_key].to(device), return_features=True)
            t_logits = t_out["logits"]

            loss_bce = bce_fn(b_logits, y)
            loss_kd  = multi_label_kd_loss(b_logits, t_logits, temperature)
            loss_crf = cross_rate_contrastive_loss(
                b_out["pooled"], t_out["pooled"],
                b_out["proj"],   t_out["proj"],
                temperature=CRF_T,
            )
            loss = loss_bce + lambda_kd * loss_kd + beta_crf * loss_crf

            optimizer.zero_grad(); loss.backward()
            if gc > 0: nn.utils.clip_grad_norm_(branch.parameters(), gc)
            optimizer.step()

            ep_bce += loss_bce.item(); ep_kd += loss_kd.item()
            ep_crf += loss_crf.item(); ep_tot += loss.item(); nb += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(branch, val_loader, branch_key, device)
        val_auc = compute_metrics(val_labels, val_probs, threshold=0.5)["macro_auc"]
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, branch, epoch, {"val_auc": val_auc},
                            {"branch": branch_view, "beta_crf": beta_crf})

        print(f"  Ep {epoch:3d} | bce={ep_bce/nb:.4f} kd={ep_kd/nb:.4f} "
              f"crf={ep_crf/nb:.4f} tot={ep_tot/nb:.4f} | val_auc={val_auc:.4f}"
              f" {'★' if is_best else ''}")
        log_rows.append({"epoch": epoch, "bce": round(ep_bce/nb, 5),
                         "kd": round(ep_kd/nb, 5), "crf": round(ep_crf/nb, 5),
                         "total": round(ep_tot/nb, 5),
                         "val_auc": round(val_auc, 5), "best_val_auc": round(best_auc, 5)})

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    print(f"  Branch {branch_view} done. Best val_auc={best_auc:.4f}")
    load_checkpoint(ckpt_path, branch, device)
    return branch, str(ckpt_path)


def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_kd   = args.lambda_kd
    temperature = args.temperature
    beta_crf    = args.beta_branch_crf
    variant_id  = args.variant_id

    w_parts = [float(x) for x in args.student_weights.split(",")]
    w500, w100 = w_parts[0] / sum(w_parts), w_parts[1] / sum(w_parts)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "logs").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"F02 Branch Enhancement: {variant_id}")
    print(f"enhance_1l500={args.enhance_1l500}  enhance_1l100={args.enhance_1l100}")
    print(f"beta_branch_crf={beta_crf}  device={device}  seed={seed}")
    print(f"{'='*60}")

    # ── frozen 12L100 global teacher ──────────────────────────────────────────
    t12l100 = _build_model(in_channels=12).to(device)
    load_checkpoint(args.t12l100_ckpt, t12l100, device)
    t12l100.eval()
    for p in t12l100.parameters(): p.requires_grad_(False)
    print(f"12L100: {args.t12l100_ckpt}")

    lead = args.lead or cfg["data"].get("lead", "II")
    common = dict(cfg=cfg, data_dir=args.data_dir, lead=lead,
                  beta_crf=beta_crf, lambda_kd=lambda_kd, temperature=temperature,
                  seed=seed, variant_id=variant_id, out_dir=out_dir, device=device,
                  t12l100=t12l100, t12l100_key=T12L100_KEY)

    # ── branch 1L500 ──────────────────────────────────────────────────────────
    if args.enhance_1l500:
        b500, b500_ckpt = _train_branch(
            branch_view="1L500", branch_in_ch=1, branch_key=T1L500_KEY,
            init_ckpt=args.init_1l500_ckpt, **common)
    else:
        b500 = _build_model(in_channels=1).to(device)
        load_checkpoint(args.init_1l500_ckpt, b500, device)
        b500.eval(); [p.requires_grad_(False) for p in b500.parameters()]
        b500_ckpt = args.init_1l500_ckpt
        print(f"1L500: reusing {b500_ckpt}")

    # ── branch 1L100 ──────────────────────────────────────────────────────────
    if args.enhance_1l100:
        b100, b100_ckpt = _train_branch(
            branch_view="1L100", branch_in_ch=1, branch_key=T1L100_KEY,
            init_ckpt=args.init_1l100_ckpt, **common)
    else:
        b100 = _build_model(in_channels=1).to(device)
        load_checkpoint(args.init_1l100_ckpt, b100, device)
        b100.eval(); [p.requires_grad_(False) for p in b100.parameters()]
        b100_ckpt = args.init_1l100_ckpt
        print(f"1L100: reusing {b100_ckpt}")

    # ── train final student using enhanced branches ───────────────────────────
    print(f"\n{'─'*50}")
    print(f"Final student with enhanced branches")
    ep  = tc.get("epochs", 100)
    bs  = tc.get("batch_size", 512)
    lr_ = tc.get("lr", 1e-3)
    nw  = tc.get("num_workers", 4)
    pw  = tc.get("persistent_workers", True)
    wd  = tc.get("weight_decay", 1e-4)
    gc  = tc.get("grad_clip", 1.0)

    from utils.simclr_checkpoint import load_encoder_init
    s_name = f"student_II_50hz_f02_{variant_id}_seed{seed}"
    s_ckpt = out_dir.parent / f"{s_name}_best.pt"
    s_log  = out_dir.parent / "logs" / f"{s_name}.csv"

    b500.eval(); [p.requires_grad_(False) for p in b500.parameters()]
    b100.eval(); [p.requires_grad_(False) for p in b100.parameters()]

    student = _build_model(in_channels=1).to(device)
    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        load_encoder_init(student, args.student_init_ckpt, strict=False, device=device)
        print(f"Student init: {args.student_init_ckpt}")
    else:
        print("Student: random init")

    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)

    bce_fn    = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    best_auc = 0.0; log_rows = []
    for epoch in range(1, ep + 1):
        student.train()
        ep_bce = ep_kd = ep_tot = 0.0; nb = 0
        for batch in tqdm(train_loader, desc=f"Student Ep {epoch}/{ep}",
                          dynamic_ncols=True, leave=False):
            y = batch["y"].to(device)
            s_out = student(batch[STUDENT_KEY].to(device), return_features=False)
            s_logits = s_out["logits"]
            with torch.no_grad():
                l500 = b500(batch[T1L500_KEY].to(device), return_features=False)["logits"]
                l100 = b100(batch[T1L100_KEY].to(device), return_features=False)["logits"]
            loss_bce = bce_fn(s_logits, y)
            loss_kd  = (w500 * multi_label_kd_loss(s_logits, l500, temperature) +
                        w100 * multi_label_kd_loss(s_logits, l100, temperature))
            loss = loss_bce + lambda_kd * loss_kd
            optimizer.zero_grad(); loss.backward()
            if gc > 0: nn.utils.clip_grad_norm_(student.parameters(), gc)
            optimizer.step()
            ep_bce += loss_bce.item(); ep_kd += loss_kd.item()
            ep_tot += loss.item(); nb += 1
        scheduler.step()
        val_probs, val_labels, _ = _run_eval(student, val_loader, STUDENT_KEY, device)
        val_auc = compute_metrics(val_labels, val_probs, threshold=0.5)["macro_auc"]
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(s_ckpt, student, epoch, {"val_auc": val_auc},
                            {"variant": variant_id, "w500": w500, "w100": w100})
        print(f"Ep {epoch:3d} | bce={ep_bce/nb:.4f} kd={ep_kd/nb:.4f} "
              f"tot={ep_tot/nb:.4f} | val_auc={val_auc:.4f} {'★' if is_best else ''}")
        log_rows.append({"epoch": epoch, "bce": round(ep_bce/nb, 5),
                         "kd": round(ep_kd/nb, 5), "total": round(ep_tot/nb, 5),
                         "val_auc": round(val_auc, 5), "best_val_auc": round(best_auc, 5)})

    with open(s_log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    print(f"\nLoading best student checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(s_ckpt, student, device)
    meta = {"seed": seed, "variant_id": variant_id,
            "enhance_1l500": args.enhance_1l500, "enhance_1l100": args.enhance_1l100,
            "beta_branch_crf": beta_crf, "b500_ckpt": b500_ckpt, "b100_ckpt": b100_ckpt,
            "w500": w500, "w100": w100}
    sm = _save_metrics(student, val_loader, test_loader, STUDENT_KEY, device,
                       s_name, out_dir.parent, meta)

    auc = sm["macro_auc"]; f1t = sm["macro_f1_tuned"]
    print(f"\nAUC={auc:.4f}  F1_tuned={f1t:.4f}")
    print(f"vs F02:  {auc-F02_AUC:+.4f} AUC  {f1t-F02_F1:+.4f} F1")
    print(f"Beats F02: {'YES ✓' if auc>F02_AUC else 'no'}  MVKT: {'YES ✓' if auc>MVKT_AUC else 'no'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",            default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",          required=True)
    p.add_argument("--lead",              default="II")
    p.add_argument("--t12l100_ckpt",      required=True,
                   help="Frozen 12L100 global teacher checkpoint")
    p.add_argument("--init_1l500_ckpt",   required=True,
                   help="Existing 1L500 branch checkpoint (base or to retrain from)")
    p.add_argument("--init_1l100_ckpt",   required=True,
                   help="Existing 1L100 branch checkpoint (base or to retrain from)")
    p.add_argument("--student_init_ckpt", default=None)
    p.add_argument("--enhance_1l500",     action="store_true",
                   help="Retrain 1L500 branch with CRF")
    p.add_argument("--enhance_1l100",     action="store_true",
                   help="Retrain 1L100 branch with CRF")
    p.add_argument("--beta_branch_crf",   type=float, default=0.05)
    p.add_argument("--student_weights",   default="0.4,0.6",
                   help="w500,w100 for final student (comma-separated)")
    p.add_argument("--lambda_kd",         type=float, default=1.0)
    p.add_argument("--temperature",       type=float, default=2.0)
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--variant_id",        default="F02BR")
    p.add_argument("--output_dir",        default="dafd_mvkt/outputs/f02_opt/branch")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
