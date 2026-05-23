"""
Dual-teacher student trainer for fork-join KD.

Student S=1L50 trained from two frozen branch teachers t2 and t3.
Loss: BCE + lambda_kd * (w2*MKD(t2→S) + w3*MKD(t3→S))

Usage:
    python dafd_mvkt/train_student_dual_teacher.py \\
        --config          dafd_mvkt/configs/fork_join_dual_6.yaml \\
        --data_dir        comper_repo/ptb_xl \\
        --teacher2_view   1L500 \\
        --teacher3_view   1L100 \\
        --teacher2_ckpt   dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt \\
        --teacher3_ckpt   dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt \\
        --output_name     student_II_50hz_forkjoin_F02_12L100_to_1L500_1L100_simclr_seed0 \\
        --seed            0
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
from train_kd_view import VIEW_DEFS, _build_model, _run_eval
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init


def train_student_dual_teacher(
    *,
    fork_cfg:         dict,
    data_dir:         str,
    lead:             str,
    teacher2_view:    str,
    teacher3_view:    str,
    teacher2_ckpt:    str,
    teacher3_ckpt:    str,
    teacher_weights:  str  = "0.5,0.5",
    output_name:      str,
    output_dir:       Path,
    student_init_ckpt: str | None = None,
    lambda_kd:        float = 1.0,
    temperature:      float = 2.0,
    seed:             int   = 0,
    epochs:           int | None = None,
    batch_size:       int | None = None,
    lr:               float | None = None,
    debug:            bool  = False,
) -> dict:
    tc  = fork_cfg["training"]
    ep  = epochs     if epochs     is not None else tc["epochs"]
    bs  = batch_size if batch_size is not None else tc["batch_size"]
    lr_ = lr         if lr         is not None else tc["lr"]
    nw  = tc["num_workers"]
    pw  = tc.get("persistent_workers", False)
    wd  = tc["weight_decay"]
    gc  = tc["grad_clip"]
    temp = temperature

    w2, w3 = [float(w) for w in teacher_weights.split(",")]
    assert abs(w2 + w3 - 1.0) < 1e-6, f"teacher weights must sum to 1: {w2}+{w3}"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Dual-Teacher KD: {teacher2_view}(w={w2}) + {teacher3_view}(w={w3}) → 1L50")
    print(f"output: {output_name}  device: {device}  seed: {seed}")
    print(f"{'='*60}")

    # ── frozen teachers ───────────────────────────────────────────────────────
    t2 = _build_model(teacher2_view)
    load_checkpoint(teacher2_ckpt, t2, device)
    t2.to(device).eval()
    for p in t2.parameters(): p.requires_grad_(False)

    t3 = _build_model(teacher3_view)
    load_checkpoint(teacher3_ckpt, t3, device)
    t3.to(device).eval()
    for p in t3.parameters(): p.requires_grad_(False)

    # ── student ───────────────────────────────────────────────────────────────
    student = _build_model("1L50").to(device)
    if student_init_ckpt and Path(student_init_ckpt).exists():
        load_encoder_init(student, student_init_ckpt, strict=False, device=device)
        print(f"Student initialized from SimCLR: {student_init_ckpt}")
    else:
        if student_init_ckpt:
            print(f"[WARN] SimCLR init not found: {student_init_ckpt}  → random init")
        else:
            print("Student: random init")

    t2_key  = VIEW_DEFS[teacher2_view]["input_key"]
    t3_key  = VIEW_DEFS[teacher3_view]["input_key"]
    s_key   = "student_x"
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_params/1e6:.2f}M")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = PTBXLMultiTeacherDataset(data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs,  shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)

    # ── debug ─────────────────────────────────────────────────────────────────
    if debug:
        first = next(iter(train_loader))
        print(f"\n[DEBUG] teacher2_view={teacher2_view} key={t2_key} "
              f"shape={first[t2_key].shape}")
        print(f"[DEBUG] teacher3_view={teacher3_view} key={t3_key} "
              f"shape={first[t3_key].shape}")
        print(f"[DEBUG] student key={s_key} shape={first[s_key].shape}")
        with torch.no_grad():
            o2 = t2(first[t2_key].to(device), return_features=False)
            o3 = t3(first[t3_key].to(device), return_features=False)
            os = student(first[s_key].to(device), return_features=False)
        print(f"[DEBUG] t2 logits: {o2['logits'].shape}  "
              f"t3 logits: {o3['logits'].shape}  "
              f"student logits: {os['logits'].shape}")
        y = first["y"].to(device)
        bce  = nn.BCEWithLogitsLoss()(os["logits"], y)
        kd2  = multi_label_kd_loss(os["logits"], o2["logits"], temp)
        kd3  = multi_label_kd_loss(os["logits"], o3["logits"], temp)
        tot  = bce + lambda_kd * (w2 * kd2 + w3 * kd3)
        print(f"[DEBUG] BCE={bce:.4f} KD2={kd2:.4f} KD3={kd3:.4f} Total={tot:.4f}")

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    bce_fn    = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    # ── paths ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    ckpt_path = output_dir / f"{output_name}_best.pt"
    log_path  = log_dir / f"{output_name}.csv"

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc  = 0.0
    log_rows  = []

    for epoch in range(1, ep + 1):
        student.train()
        ep_bce = ep_kd2 = ep_kd3 = ep_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}",
                          dynamic_ncols=True, leave=False):
            y   = batch["y"].to(device)
            s_x = batch[s_key].to(device)
            s_out = student(s_x, return_features=False)
            s_logits = s_out["logits"]

            with torch.no_grad():
                t2_logits = t2(batch[t2_key].to(device),
                                return_features=False)["logits"]
                t3_logits = t3(batch[t3_key].to(device),
                                return_features=False)["logits"]

            loss_bce = bce_fn(s_logits, y)
            loss_kd2 = multi_label_kd_loss(s_logits, t2_logits, temp)
            loss_kd3 = multi_label_kd_loss(s_logits, t3_logits, temp)
            loss     = loss_bce + lambda_kd * (w2 * loss_kd2 + w3 * loss_kd3)

            optimizer.zero_grad()
            loss.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(student.parameters(), gc)
            optimizer.step()

            ep_bce  += loss_bce.item()
            ep_kd2  += loss_kd2.item()
            ep_kd3  += loss_kd3.item()
            ep_tot  += loss.item()
            n_batches += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(student, val_loader, s_key, device)
        val_m   = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_m["macro_auc"]

        avg_bce  = ep_bce  / n_batches
        avg_kd2  = ep_kd2  / n_batches
        avg_kd3  = ep_kd3  / n_batches
        avg_tot  = ep_tot  / n_batches
        is_best  = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch,
                            {"val_auc": val_auc},
                            {"teacher2_view": teacher2_view,
                             "teacher3_view": teacher3_view})

        print(f"Ep {epoch:3d} | bce={avg_bce:.4f} "
              f"kd2={avg_kd2:.4f} kd3={avg_kd3:.4f} tot={avg_tot:.4f} "
              f"| val_auc={val_auc:.4f} {'★' if is_best else ''}")

        log_rows.append({
            "epoch": epoch,
            "bce": round(avg_bce, 5), "kd2": round(avg_kd2, 5),
            "kd3": round(avg_kd3, 5), "total": round(avg_tot, 5),
            "val_auc": round(val_auc, 5), "best_val_auc": round(best_auc, 5),
        })

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── evaluation ────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, student, device)

    val_probs, val_labels, _ = _run_eval(student, val_loader, s_key, device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(student, test_loader, s_key, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    metrics = {
        "output_name":        output_name,
        "teacher2_view":      teacher2_view,
        "teacher3_view":      teacher3_view,
        "teacher_weights":    teacher_weights,
        "macro_auc":          round(m05["macro_auc"], 5),
        "macro_f1_0_5":       round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned":     round(m_tune.get("macro_f1", 0), 5),
        "val_auc_best":       round(best_auc, 5),
        "per_class_thresholds": {
            SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
        },
        "_meta": {"seed": seed, "lambda_kd": lambda_kd, "temperature": temp},
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    val_metrics = {
        "output_name":    output_name,
        "macro_auc":      round(val_m05["macro_auc"], 5),
        "macro_f1_0_5":   round(val_m05.get("macro_f1", 0), 5),
        "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5),
    }
    with open(output_dir / f"metrics_val_{output_name}.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(output_dir / f"metrics_test_{output_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

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
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",            default="dafd_mvkt/configs/fork_join_dual_6.yaml")
    p.add_argument("--data_dir",          required=True)
    p.add_argument("--lead",              default="II")
    p.add_argument("--teacher2_view",     required=True, choices=list(VIEW_DEFS))
    p.add_argument("--teacher3_view",     required=True, choices=list(VIEW_DEFS))
    p.add_argument("--teacher2_ckpt",     required=True)
    p.add_argument("--teacher3_ckpt",     required=True)
    p.add_argument("--teacher_weights",   default="0.5,0.5")
    p.add_argument("--student_init_ckpt", default=None)
    p.add_argument("--output_name",       required=True)
    p.add_argument("--output_dir",        default="dafd_mvkt/outputs/forkjoin")
    p.add_argument("--lambda_kd",         type=float, default=1.0)
    p.add_argument("--temperature",       type=float, default=2.0)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--epochs",            type=int, default=None)
    p.add_argument("--batch_size",        type=int, default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--debug",             action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        fork_cfg = yaml.safe_load(f)

    train_student_dual_teacher(
        fork_cfg=fork_cfg,
        data_dir=args.data_dir,
        lead=args.lead,
        teacher2_view=args.teacher2_view,
        teacher3_view=args.teacher3_view,
        teacher2_ckpt=args.teacher2_ckpt,
        teacher3_ckpt=args.teacher3_ckpt,
        teacher_weights=args.teacher_weights,
        output_name=args.output_name,
        output_dir=Path(args.output_dir),
        student_init_ckpt=args.student_init_ckpt,
        lambda_kd=args.lambda_kd,
        temperature=args.temperature,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        debug=args.debug,
    )
