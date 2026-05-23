"""
Train single-lead low-Hz student with optional distillation losses.

Total loss:
    L = L_BCE + α·L_MKD + β·L_CRF + γ·L_DAF

DAF loss module has learnable parameters (gate_logits) that are included
in the student optimiser.

Usage:
    python train_student.py \\
        --config configs/student_100hz.yaml \\
        --data_dir /path/to/ptbxl \\
        --teacher_ckpt outputs/teacher_best.pt \\
        --losses bce,mkd,crf,daf
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.ptbxl_dataset import PTBXLDataset
from losses.contrastive import cross_rate_contrastive_loss
from losses.frequency_distillation import DiagnosisAwareFrequencyDistillation
from losses.mkd import multi_label_kd_loss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics
from utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> ResNet1d:
    mc = cfg["model"]
    return ResNet1d(
        in_channels  = mc["in_channels"],
        num_classes  = mc["num_classes"],
        layers       = mc.get("layers", [3, 4, 6, 3]),
        base_channels= mc.get("base_channels", 64),
        proj_dim     = mc.get("proj_dim", 128),
        dropout      = mc.get("dropout", 0.0),
    )


def build_loaders(cfg: dict, data_dir: str) -> tuple[DataLoader, DataLoader]:
    dc = cfg["data"]
    tc = cfg["training"]

    def _ds(split: str) -> PTBXLDataset:
        return PTBXLDataset(
            ptbxl_root   = data_dir,
            split        = split,
            mode         = "student",
            sampling_rate= dc["sampling_rate"],
            student_lead = dc.get("student_lead", "II"),
            normalize    = dc.get("normalize", "zscore"),
        )

    train_loader = DataLoader(
        _ds("train"), batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc.get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        _ds("val"), batch_size=128, shuffle=False,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x   = batch["student_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics(labels, probs)


# ─────────────────────────────────────────────────────────────────────────────

def _check_nan(name: str, val: torch.Tensor) -> None:
    if torch.isnan(val):
        print(f"NaN detected in {name} — aborting.", file=sys.stderr)
        sys.exit(1)


def train(args: argparse.Namespace) -> None:
    cfg    = load_cfg(args.config)
    tc     = cfg["training"]
    dc     = cfg["distillation"]
    oc     = cfg["outputs"]

    active_losses = set(args.losses.split(","))
    print(f"Active losses: {sorted(active_losses)}")

    set_seed(tc.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir   = Path(args.output_dir or oc.get("dir", "outputs/student"))
    ckpt_path = args.ckpt_out or oc.get("checkpoint", str(out_dir / "best_student.pt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(cfg, args.data_dir)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    # ── teacher (frozen) ─────────────────────────────────────────────────────
    teacher_cfg_path = args.teacher_config or "configs/teacher_500hz.yaml"
    t_cfg = load_cfg(teacher_cfg_path)
    teacher = build_model(t_cfg).to(device)
    load_checkpoint(args.teacher_ckpt, teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher from {args.teacher_ckpt}")

    # ── student ───────────────────────────────────────────────────────────────
    student = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_params/1e6:.2f}M")

    # ── DAF loss (learnable gate) ─────────────────────────────────────────────
    daf_loss_fn: DiagnosisAwareFrequencyDistillation | None = None
    if "daf" in active_losses:
        mc = cfg["model"]
        daf_loss_fn = DiagnosisAwareFrequencyDistillation(
            num_classes = mc["num_classes"],
            teacher_ch  = teacher.feat_dim,
            student_ch  = student.feat_dim,
            common_dim  = dc.get("daf_common_dim", 256),
            pool_size   = dc.get("daf_pool_size", 128),
        ).to(device)

    # ── optimiser (student + DAF gate params) ─────────────────────────────────
    opt_params = list(student.parameters())
    if daf_loss_fn is not None:
        opt_params += list(daf_loss_fn.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    sched_name = tc.get("scheduler", "cosine")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5)

    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler(enabled=tc.get("amp", False))
    grad_clip = tc.get("grad_clip", 1.0)

    alpha = dc.get("alpha", 1.0)
    beta  = dc.get("beta", 0.1)
    gamma = dc.get("gamma", 0.1)
    mkd_T = dc.get("mkd_temperature", 2.0)
    crf_T = dc.get("crf_temperature", 0.07)

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    log = []

    for epoch in range(1, tc["epochs"] + 1):
        student.train()
        if daf_loss_fn is not None:
            daf_loss_fn.train()

        totals = {"bce": 0.0, "mkd": 0.0, "crf": 0.0, "daf": 0.0, "total": 0.0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{tc['epochs']}",
                          leave=False, dynamic_ncols=True):
            s_x = batch["student_x"].to(device)   # [B, 1, L]
            t_x = batch["teacher_x"].to(device)   # [B, 12, 5000]
            y   = batch["y"].to(device)            # [B, 5]

            optimizer.zero_grad()

            with autocast(enabled=tc.get("amp", False)):
                # student forward
                s_out = student(s_x, return_features=True)
                s_logits = s_out["logits"]     # [B, C]

                # teacher forward (no_grad)
                with torch.no_grad():
                    t_out = teacher(t_x, return_features=True)
                t_logits     = t_out["logits"]       # [B, C]
                t_pooled     = t_out["pooled"]       # [B, D]
                t_proj       = t_out["proj"]         # [B, 128]
                t_feature_map= t_out["feature_map"]  # [B, D, T']

                # ── BCE ──────────────────────────────────────────────────────
                loss_bce = criterion(s_logits, y)
                _check_nan("bce", loss_bce)

                total_loss = loss_bce
                totals["bce"] += loss_bce.item()

                # ── MKD ──────────────────────────────────────────────────────
                if "mkd" in active_losses:
                    loss_mkd = multi_label_kd_loss(s_logits, t_logits, mkd_T)
                    _check_nan("mkd", loss_mkd)
                    total_loss = total_loss + alpha * loss_mkd
                    totals["mkd"] += loss_mkd.item()

                # ── CRF ──────────────────────────────────────────────────────
                if "crf" in active_losses:
                    loss_crf = cross_rate_contrastive_loss(
                        s_out["pooled"], t_pooled,
                        s_out["proj"],   t_proj,
                        temperature=crf_T,
                    )
                    _check_nan("crf", loss_crf)
                    total_loss = total_loss + beta * loss_crf
                    totals["crf"] += loss_crf.item()

                # ── DAF ──────────────────────────────────────────────────────
                if "daf" in active_losses and daf_loss_fn is not None:
                    loss_daf = daf_loss_fn(
                        s_out["feature_map"], t_feature_map, t_logits
                    )
                    _check_nan("daf", loss_daf)
                    total_loss = total_loss + gamma * loss_daf
                    totals["daf"] += loss_daf.item()

                totals["total"] += total_loss.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(opt_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n_batches += 1

        if sched_name == "cosine":
            scheduler.step()

        # normalise totals
        for k in totals:
            totals[k] /= n_batches

        # ── validation ────────────────────────────────────────────────────────
        metrics = evaluate(student, val_loader, device)

        if sched_name == "plateau":
            scheduler.step(metrics["macro_auc"])

        print(f"[{epoch:4d}] "
              f"total={totals['total']:.4f} "
              f"bce={totals['bce']:.4f} "
              f"mkd={totals['mkd']:.4f} "
              f"crf={totals['crf']:.4f} "
              f"daf={totals['daf']:.4f}  "
              f"AUC={metrics['macro_auc']:.4f}  F1={metrics['macro_f1']:.4f}")

        row = {"epoch": epoch, **{k: round(v, 5) for k, v in totals.items()},
               **{k: round(v, 5) for k, v in metrics.items()}}
        log.append(row)

        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            extra = {}
            if daf_loss_fn is not None:
                extra["daf_state"] = daf_loss_fn.state_dict()
            save_checkpoint(ckpt_path, student, epoch, metrics, cfg, extra)
            print(f"  ✓ best AUC={best_auc:.4f} saved → {ckpt_path}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--teacher_ckpt",   required=True)
    p.add_argument("--teacher_config", default=None,
                   help="Path to teacher config YAML (default: configs/teacher_500hz.yaml)")
    p.add_argument("--losses",         default="bce",
                   help="Comma-separated: bce,mkd,crf,daf")
    p.add_argument("--output_dir",     default=None)
    p.add_argument("--ckpt_out",       default=None)
    p.add_argument("--seed",           type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
