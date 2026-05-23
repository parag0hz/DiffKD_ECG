"""
Mutual learning between 1L100 (T) and 1L50 (S) with frozen upstream teacher.

Loss:
    L_T = BCE(T) + lambda_kd * KD(upstream → T)
          + lambda_mutual * KD(S.detach() → T)
    L_S = BCE(S) + lambda_kd * KD(T.detach() → S)

Upstream (12L100 or 12L500) is always frozen.
T (1L100) and S (1L50) are jointly trained via separate optimizers.
Mutual feedback from S to T uses small lambda (0.05 or 0.1).

Usage:
    python dafd_mvkt/train_mutual_1l100_1l50.py \\
        --config            dafd_mvkt/configs/adaptive_50hz.yaml \\
        --data_dir          comper_repo/ptb_xl \\
        --upstream_view     12L100 \\
        --upstream_ckpt     dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt \\
        --teacher_init_ckpt dafd_mvkt/outputs/ta_II_100hz_from_ta500_seed0_best.pt \\
        --student_init_ckpt dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \\
        --lambda_mutual     0.05 \\
        --seed              0
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

BCE_AUC  = 0.8061; BCE_F1  = 0.5740
PROG_AUC = 0.8396; PROG_F1 = 0.6176
C14_AUC  = 0.8425; C14_F1  = 0.6243
MVKT_AUC = 0.843;  MVKT_F1 = 0.626

VIEW_DEFS: dict[str, dict] = {
    "12L500": {"in_channels": 12, "input_key": "teacher_12l_500"},
    "12L100": {"in_channels": 12, "input_key": "teacher_12l_100"},
    "12L50":  {"in_channels": 12, "input_key": "teacher_12l_50"},
    "1L500":  {"in_channels":  1, "input_key": "teacher_1l_500"},
    "1L100":  {"in_channels":  1, "input_key": "teacher_1l_100"},
    "1L50":   {"in_channels":  1, "input_key": "teacher_1l_50"},
}

TEACHER_IN_CH = 1
TEACHER_KEY   = "teacher_1l_100"  # 1L100 (T)
STUDENT_IN_CH = 1
STUDENT_KEY   = "student_x"       # 1L50  (S) — II lead 50Hz


def _build_model(in_channels: int, num_classes: int = 5) -> ResNet1d:
    return ResNet1d(in_channels=in_channels, num_classes=num_classes,
                    layers=[3, 4, 6, 3], base_channels=64,
                    proj_dim=128, dropout=0.0)


@torch.no_grad()
def _run_eval(model: nn.Module, loader: DataLoader,
              input_key: str, device: torch.device):
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


def _eval_and_save(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_key: str,
    device: torch.device,
    out_name: str,
    out_dir: Path,
    extra_meta: dict,
) -> dict:
    val_probs, val_labels, _          = _run_eval(model, val_loader,  input_key, device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(model, test_loader, input_key, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    metrics = {
        "output_name":    out_name,
        "macro_auc":      round(m05["macro_auc"], 5),
        "macro_f1_0_5":   round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned": round(m_tune.get("macro_f1", 0), 5),
        "per_class_thresholds": {SUPERCLASSES[i]: round(float(thresholds[i]), 4)
                                 for i in range(5)},
        "_meta": extra_meta,
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc":      round(val_m05["macro_auc"], 5),
                   "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5)}, f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = [{"record_id": rid,
               **{f"y_{c}":    int(test_labels[i, ci])           for ci, c in enumerate(SUPERCLASSES)},
               **{f"prob_{c}": round(float(test_probs[i, ci]), 5) for ci, c in enumerate(SUPERCLASSES)},
               **{f"pred_{c}": int(preds[i, ci])                  for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_test_{out_name}.csv", index=False)
    return metrics


def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed if args.seed is not None else tc.get("seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep    = args.epochs     or tc.get("epochs", 100)
    bs    = args.batch_size or tc.get("batch_size", 512)
    lr_   = tc.get("lr", 1e-3)
    nw    = tc.get("num_workers", 4)
    pw    = tc.get("persistent_workers", True)
    wd    = tc.get("weight_decay", 1e-4)
    gc    = tc.get("grad_clip", 1.0)

    lambda_kd     = args.lambda_kd     if args.lambda_kd     is not None else tc.get("lambda_kd", 1.0)
    lambda_mutual = args.lambda_mutual if args.lambda_mutual is not None else tc.get("lambda_mutual", 0.05)
    temperature   = args.temperature   if args.temperature   is not None else tc.get("temperature", 2.0)

    upstream_view = args.upstream_view
    up_def        = VIEW_DEFS[upstream_view]
    up_key        = up_def["input_key"]

    lam_str  = str(lambda_mutual).replace(".", "")
    up_short = upstream_view.lower().replace("l", "l")  # e.g. "12l100"
    student_out_name = args.student_output_name or (
        f"student_II_50hz_mut_{up_short}_lam{lam_str}_seed{seed}")
    teacher_out_name = args.teacher_output_name or (
        f"teacher_1l100_mut_{up_short}_lam{lam_str}_seed{seed}")

    out_dir = Path(args.output_dir or cfg.get("outputs", {}).get("dir", "dafd_mvkt/outputs/adaptive"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    s_ckpt_path = out_dir / f"{student_out_name}_best.pt"
    t_ckpt_path = out_dir / f"{teacher_out_name}_best.pt"
    s_log_path  = out_dir / "logs" / f"{student_out_name}.csv"
    t_log_path  = out_dir / "logs" / f"{teacher_out_name}.csv"

    print(f"\n{'='*60}")
    print(f"Mutual Learning: upstream={upstream_view}  1L100(T) ⟷ 1L50(S)")
    print(f"lambda_kd={lambda_kd}  lambda_mutual={lambda_mutual}  T={temperature}")
    print(f"device={device}  seed={seed}  epochs={ep}  bs={bs}")
    print(f"Student out: {s_ckpt_path}")
    print(f"Teacher out: {t_ckpt_path}")
    print(f"{'='*60}")

    # ── frozen upstream teacher ───────────────────────────────────────────────
    upstream = _build_model(up_def["in_channels"]).to(device)
    load_checkpoint(args.upstream_ckpt, upstream, device)
    upstream.eval()
    for p in upstream.parameters():
        p.requires_grad_(False)
    print(f"Upstream ({upstream_view}): {args.upstream_ckpt}")

    # ── 1L100 teacher (T) — jointly trained ───────────────────────────────────
    t_model = _build_model(TEACHER_IN_CH).to(device)
    if args.teacher_init_ckpt and Path(args.teacher_init_ckpt).exists():
        load_checkpoint(args.teacher_init_ckpt, t_model, device)
        print(f"T (1L100) init: {args.teacher_init_ckpt}")
    else:
        print("T (1L100): random init")

    # ── 1L50 student (S) — jointly trained ───────────────────────────────────
    s_model = _build_model(STUDENT_IN_CH).to(device)
    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        load_encoder_init(s_model, args.student_init_ckpt, strict=False, device=device)
        print(f"S (1L50)  init: {args.student_init_ckpt}")
    else:
        print("S (1L50): random init")

    n_t = sum(p.numel() for p in t_model.parameters() if p.requires_grad)
    n_s = sum(p.numel() for p in s_model.parameters() if p.requires_grad)
    print(f"T params: {n_t/1e6:.2f}M   S params: {n_s/1e6:.2f}M")

    # ── data ──────────────────────────────────────────────────────────────────
    lead = args.lead or cfg["data"].get("lead", "II")
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)

    # ── optimizers & schedulers ───────────────────────────────────────────────
    bce_fn  = nn.BCEWithLogitsLoss()
    opt_t   = torch.optim.AdamW(t_model.parameters(), lr=lr_, weight_decay=wd)
    opt_s   = torch.optim.AdamW(s_model.parameters(), lr=lr_, weight_decay=wd)
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=ep)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=ep)

    # ── training loop ─────────────────────────────────────────────────────────
    best_s_auc = 0.0
    best_t_auc = 0.0
    s_log_rows = []
    t_log_rows = []

    for epoch in range(1, ep + 1):
        t_model.train()
        s_model.train()
        ep_t_bce = ep_t_kd_up = ep_t_kd_mut = ep_t_tot = 0.0
        ep_s_bce = ep_s_kd = ep_s_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}", dynamic_ncols=True, leave=False):
            y    = batch["y"].to(device)
            t_in = batch[TEACHER_KEY].to(device)
            s_in = batch[STUDENT_KEY].to(device)

            t_out    = t_model(t_in, return_features=False)
            t_logits = t_out["logits"]

            s_out    = s_model(s_in, return_features=False)
            s_logits = s_out["logits"]

            with torch.no_grad():
                up_out    = upstream(batch[up_key].to(device), return_features=False)
                up_logits = up_out["logits"]

            # L_T = BCE(T) + lambda_kd * KD(upstream→T) + lambda_mutual * KD(S.detach()→T)
            loss_t_bce    = bce_fn(t_logits, y)
            loss_t_kd_up  = multi_label_kd_loss(t_logits, up_logits,          temperature)
            loss_t_kd_mut = multi_label_kd_loss(t_logits, s_logits.detach(),  temperature)
            loss_t        = (loss_t_bce
                             + lambda_kd     * loss_t_kd_up
                             + lambda_mutual * loss_t_kd_mut)

            opt_t.zero_grad()
            loss_t.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(t_model.parameters(), gc)
            opt_t.step()

            # L_S = BCE(S) + lambda_kd * KD(T.detach()→S)
            loss_s_bce = bce_fn(s_logits, y)
            loss_s_kd  = multi_label_kd_loss(s_logits, t_logits.detach(), temperature)
            loss_s     = loss_s_bce + lambda_kd * loss_s_kd

            opt_s.zero_grad()
            loss_s.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(s_model.parameters(), gc)
            opt_s.step()

            ep_t_bce    += loss_t_bce.item()
            ep_t_kd_up  += loss_t_kd_up.item()
            ep_t_kd_mut += loss_t_kd_mut.item()
            ep_t_tot    += loss_t.item()
            ep_s_bce    += loss_s_bce.item()
            ep_s_kd     += loss_s_kd.item()
            ep_s_tot    += loss_s.item()
            n_batches   += 1

        sched_t.step()
        sched_s.step()

        s_probs, s_labels, _ = _run_eval(s_model, val_loader, STUDENT_KEY, device)
        t_probs, t_labels, _ = _run_eval(t_model, val_loader, TEACHER_KEY, device)
        s_val_auc = compute_metrics(s_labels, s_probs, threshold=0.5)["macro_auc"]
        t_val_auc = compute_metrics(t_labels, t_probs, threshold=0.5)["macro_auc"]

        if s_val_auc > best_s_auc:
            best_s_auc = s_val_auc
            save_checkpoint(s_ckpt_path, s_model, epoch, {"val_auc": s_val_auc},
                            {"upstream": upstream_view, "lambda_mutual": lambda_mutual})
        if t_val_auc > best_t_auc:
            best_t_auc = t_val_auc
            save_checkpoint(t_ckpt_path, t_model, epoch, {"val_auc": t_val_auc},
                            {"upstream": upstream_view, "lambda_mutual": lambda_mutual})

        n = n_batches
        print(f"Ep {epoch:3d} | "
              f"T: bce={ep_t_bce/n:.4f} kd_up={ep_t_kd_up/n:.4f} kd_mut={ep_t_kd_mut/n:.4f} "
              f"tot={ep_t_tot/n:.4f} auc={t_val_auc:.4f} {'★' if t_val_auc==best_t_auc else ''} | "
              f"S: bce={ep_s_bce/n:.4f} kd={ep_s_kd/n:.4f} tot={ep_s_tot/n:.4f} "
              f"auc={s_val_auc:.4f} {'★' if s_val_auc==best_s_auc else ''}")

        s_log_rows.append({"epoch": epoch,
                           "bce": round(ep_s_bce/n, 5), "kd": round(ep_s_kd/n, 5),
                           "total": round(ep_s_tot/n, 5),
                           "val_auc": round(s_val_auc, 5), "best_val_auc": round(best_s_auc, 5)})
        t_log_rows.append({"epoch": epoch,
                           "bce": round(ep_t_bce/n, 5),
                           "kd_upstream": round(ep_t_kd_up/n, 5),
                           "kd_mutual": round(ep_t_kd_mut/n, 5),
                           "total": round(ep_t_tot/n, 5),
                           "val_auc": round(t_val_auc, 5), "best_val_auc": round(best_t_auc, 5)})

    for rows, path in [(s_log_rows, s_log_path), (t_log_rows, t_log_path)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    # ── final evaluation — student ────────────────────────────────────────────
    print(f"\nLoading best student checkpoint (val_auc={best_s_auc:.4f}) …")
    load_checkpoint(s_ckpt_path, s_model, device)
    meta = {"seed": seed, "upstream": upstream_view,
            "lambda_kd": lambda_kd, "lambda_mutual": lambda_mutual,
            "temperature": temperature, "method": "mutual"}
    s_metrics = _eval_and_save(s_model, val_loader, test_loader,
                               STUDENT_KEY, device, student_out_name, out_dir, meta)

    # ── final evaluation — teacher ────────────────────────────────────────────
    print(f"\nLoading best teacher checkpoint (val_auc={best_t_auc:.4f}) …")
    load_checkpoint(t_ckpt_path, t_model, device)
    t_metrics = _eval_and_save(t_model, val_loader, test_loader,
                               TEACHER_KEY, device, teacher_out_name, out_dir, meta)

    for label, m in [("Student (1L50)", s_metrics), ("Teacher (1L100)", t_metrics)]:
        auc = m["macro_auc"]; f1t = m["macro_f1_tuned"]
        print(f"\n{label}:")
        print(f"  AUC={auc:.4f}  F1_tuned={f1t:.4f}")
        print(f"  vs Prog : {auc-PROG_AUC:+.4f}  vs C14: {auc-C14_AUC:+.4f}  "
              f"vs MVKT: {auc-MVKT_AUC:+.4f}")
        print(f"  Beats MVKT AUC: {'YES ✓' if auc>MVKT_AUC else 'no'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",              default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",            required=True)
    p.add_argument("--lead",                default="II")
    p.add_argument("--upstream_view",       default="12L100",
                   choices=["12L500", "12L100"])
    p.add_argument("--upstream_ckpt",       required=True)
    p.add_argument("--teacher_init_ckpt",   default=None,
                   help="Pre-trained 1L100 checkpoint to init T, e.g. ta_II_100hz_from_ta500_seed0_best.pt")
    p.add_argument("--student_init_ckpt",   default=None,
                   help="SimCLR encoder to init S")
    p.add_argument("--lambda_kd",           type=float, default=None)
    p.add_argument("--lambda_mutual",       type=float, default=None)
    p.add_argument("--temperature",         type=float, default=None)
    p.add_argument("--seed",                type=int,   default=0)
    p.add_argument("--epochs",              type=int,   default=None)
    p.add_argument("--batch_size",          type=int,   default=None)
    p.add_argument("--student_output_name", default=None)
    p.add_argument("--teacher_output_name", default=None)
    p.add_argument("--output_dir",          default=None)
    p.add_argument("--debug",               action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
