"""
Train 12-lead 500 Hz teacher for PTB-XL 5-superclass multi-label classification.

Usage:
    python train_teacher.py --config configs/teacher_500hz.yaml --data_dir /path/to/ptbxl
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
from models.resnet1d import ResNet1d
from models.resnet1d_wang import ResNet1dWang
from utils.checkpoint import save_checkpoint
from utils.metrics import CLASSES, compute_metrics
from utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    mc = cfg["model"]
    arch = mc.get("arch", "resnet1d34")
    if arch == "resnet1d_wang":
        return ResNet1dWang(
            in_channels = mc["in_channels"],
            num_classes = mc["num_classes"],
            proj_dim    = mc.get("proj_dim", 128),
        )
    # default: ResNet1d-34 (He et al.)
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
            mode         = "teacher",
            sampling_rate= dc.get("sampling_rate", 500),
            normalize    = dc.get("normalize", "zscore"),
        )

    train_loader = DataLoader(
        _ds("train"), batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc.get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        _ds("val"), batch_size=256, shuffle=False,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x = batch["teacher_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


# ─────────────────────────────────────────────────────────────────────────────
# training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    tc  = cfg["training"]
    oc  = cfg["outputs"]

    set_seed(tc.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir or oc.get("dir", "outputs/teacher"))
    ckpt_path = args.ckpt_out or oc.get("checkpoint", str(out_dir / "best_teacher.pt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(cfg, args.data_dir)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M")

    # ── optimiser ─────────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    sched_name = tc.get("scheduler", "cosine")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5)

    scaler = GradScaler(enabled=tc.get("amp", False))
    grad_clip = tc.get("grad_clip", 1.0)

    # ── training ─────────────────────────────────────────────────────────────
    best_auc = 0.0
    log = []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{tc['epochs']}", leave=False,
                          dynamic_ncols=True):
            x = batch["teacher_x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=tc.get("amp", False)):
                out  = model(x, return_features=False)
                loss = criterion(out["logits"], y)

            if torch.isnan(loss):
                print("NaN in BCE loss — aborting.", file=sys.stderr)
                sys.exit(1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        if sched_name == "cosine":
            scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        metrics, _, _ = evaluate(model, val_loader, device)

        if sched_name == "plateau":
            scheduler.step(metrics["macro_auc"])

        print(f"[{epoch:4d}]  loss={epoch_loss:.4f}  "
              f"AUC={metrics['macro_auc']:.4f}  F1={metrics['macro_f1']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        row = {"epoch": epoch, "train_loss": round(epoch_loss, 5), **{
            k: round(v, 5) for k, v in metrics.items()
        }}
        log.append(row)

        # ── checkpoint ────────────────────────────────────────────────────────
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            save_checkpoint(ckpt_path, model, epoch, metrics, cfg)
            print(f"  ✓ best AUC={best_auc:.4f} saved → {ckpt_path}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ckpt_out",   default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        cfg = load_cfg(args.config)
        cfg["training"]["seed"] = args.seed
    train(args)
