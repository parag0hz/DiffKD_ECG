"""
Train 1-lead 500 Hz Teacher Assistant (TA) with supervision from frozen
12-lead 500 Hz teacher.

TA loss:
    L = L_BCE + alpha_ta * L_MKD(teacher→TA) + beta_ta * L_CRF(teacher↔TA)

Usage:
    python train_ta.py \\
        --config configs/ta_ii_500hz.yaml \\
        --data_dir /path/to/ptbxl \\
        --teacher_ckpt outputs/teacher_best.pt \\
        --lead II
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.ptbxl_dataset import PTBXLDataset
from losses.contrastive import cross_rate_contrastive_loss
from losses.mkd import multi_label_kd_loss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics
from utils.seed import set_seed


_HERE = Path(__file__).parent


def _load_encoder_ckpt(model: ResNet1d, ckpt_path: str, device: torch.device) -> None:
    """Load CLECG pretrained encoder weights (non-strict)."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in raw:
        sd = raw["state_dict"]
    elif "query_enc" in raw:
        sd = raw["query_enc"]
    else:
        sd = raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [init_encoder] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [init_encoder] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"  Encoder initialized from: {ckpt_path}")


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


def build_loaders(
    cfg: dict, data_dir: str, lead: str
) -> tuple[DataLoader, DataLoader]:
    dc = cfg["data"]
    tc = cfg["training"]
    kw = dict(num_workers=tc.get("num_workers", 4), pin_memory=True)
    train_ds = PTBXLDataset(
        ptbxl_root=data_dir, split="train", mode="ta",
        lead=lead, normalize=dc.get("normalize", "zscore"),
    )
    val_ds = PTBXLDataset(
        ptbxl_root=data_dir, split="val", mode="ta",
        lead=lead, normalize=dc.get("normalize", "zscore"),
    )
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        drop_last=True, **kw,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x   = batch["ta_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics(labels, probs)


def _check_nan(name: str, val: torch.Tensor) -> None:
    if torch.isnan(val):
        print(f"NaN detected in {name} — aborting.", file=sys.stderr)
        sys.exit(1)


def train(args: argparse.Namespace) -> None:
    cfg  = load_cfg(args.config)
    tc   = cfg["training"]
    oc   = cfg["output"]
    lead = args.lead or cfg["data"].get("lead", "II")

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    active_losses = set(args.losses.split(","))
    print(f"Active losses: {sorted(active_losses)}")

    seed = args.seed if args.seed is not None else tc.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {seed}")

    run_name  = args.run_name or oc.get("name", "ta")
    out_dir   = Path(args.output_dir) if args.output_dir else Path(oc.get("dir", "outputs"))
    ckpt_path = out_dir / f"{run_name}_best.pt"
    log_dir   = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(cfg, args.data_dir, lead)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    # ── frozen 12-lead teacher ────────────────────────────────────────────────
    t_cfg_path = args.teacher_config or str(_HERE / "configs/teacher_500hz.yaml")
    t_cfg      = load_cfg(t_cfg_path)
    teacher    = build_model(t_cfg).to(device)
    load_checkpoint(args.teacher_ckpt, teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher from {args.teacher_ckpt}")

    # ── 1-lead 500 Hz TA ──────────────────────────────────────────────────────
    ta = build_model(cfg).to(device)
    if args.init_encoder_ckpt:
        _load_encoder_ckpt(ta, args.init_encoder_ckpt, device)
    n_params = sum(p.numel() for p in ta.parameters() if p.requires_grad)
    print(f"TA params: {n_params/1e6:.2f}M")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        ta.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"]
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler("cuda", enabled=tc.get("amp", False))
    grad_clip = tc.get("grad_clip", 1.0)

    alpha = tc.get("alpha_ta", 1.0)
    beta  = tc.get("beta_ta",  0.1)
    mkd_T = tc.get("temperature", 2.0)
    crf_T = tc.get("contrastive_temperature", 0.07)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path   = log_dir / f"{run_name}.csv"
    log_fields = ["epoch", "bce", "mkd", "crf", "total", "macro_auc", "macro_f1"]
    csv_fh     = open(log_path, "w", newline="")
    csv_wr     = csv.DictWriter(csv_fh, fieldnames=log_fields, extrasaction="ignore")
    csv_wr.writeheader()

    best_auc = 0.0

    for epoch in range(1, tc["epochs"] + 1):
        ta.train()
        totals    = {"bce": 0.0, "mkd": 0.0, "crf": 0.0, "total": 0.0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{tc['epochs']}",
                          leave=False, dynamic_ncols=True):
            t_x  = batch["teacher_x"].to(device)   # [B, 12, 5000]
            ta_x = batch["ta_x"].to(device)          # [B, 1,  5000]
            y    = batch["y"].to(device)

            if args.debug and epoch == 1 and n_batches == 0:
                print(f"  [debug] teacher_x:{t_x.shape}  ta_x:{ta_x.shape}  y:{y.shape}")

            optimizer.zero_grad()

            with autocast("cuda", enabled=tc.get("amp", False)):
                ta_out = ta(ta_x, return_features=True)

                with torch.no_grad():
                    t_out = teacher(t_x, return_features=True)

                loss_bce = criterion(ta_out["logits"], y)
                _check_nan("bce", loss_bce)
                total_loss = loss_bce
                totals["bce"] += loss_bce.item()

                if "mkd" in active_losses:
                    lv = multi_label_kd_loss(ta_out["logits"], t_out["logits"], mkd_T)
                    _check_nan("mkd", lv)
                    total_loss = total_loss + alpha * lv
                    totals["mkd"] += lv.item()

                if "crf" in active_losses:
                    lv = cross_rate_contrastive_loss(
                        ta_out["pooled"], t_out["pooled"],
                        ta_out["proj"],   t_out["proj"],
                        temperature=crf_T,
                    )
                    _check_nan("crf", lv)
                    total_loss = total_loss + beta * lv
                    totals["crf"] += lv.item()

                totals["total"] += total_loss.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(ta.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n_batches += 1

        scheduler.step()

        for k in totals:
            totals[k] /= n_batches

        metrics = evaluate(ta, val_loader, device)

        print(f"[{epoch:4d}]  total={totals['total']:.4f}  "
              f"bce={totals['bce']:.4f}  mkd={totals['mkd']:.4f}  "
              f"crf={totals['crf']:.4f}  "
              f"AUC={metrics['macro_auc']:.4f}  F1={metrics['macro_f1']:.4f}")

        csv_wr.writerow({
            "epoch": epoch,
            **{k: round(v, 5) for k, v in totals.items()},
            "macro_auc": round(metrics["macro_auc"], 5),
            "macro_f1":  round(metrics["macro_f1"], 5),
        })
        csv_fh.flush()

        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            save_checkpoint(str(ckpt_path), ta, epoch, metrics, cfg)
            print(f"  ✓ best AUC={best_auc:.4f} → {ckpt_path}")

    csv_fh.close()
    print(f"\nDone. Best val AUC: {best_auc:.4f}")
    print(f"Log saved to {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--teacher_ckpt",   required=True)
    p.add_argument("--teacher_config", default=None,
                   help="Teacher config YAML (default: configs/teacher_500hz.yaml)")
    p.add_argument("--lead",           default=None,
                   help="Lead name override (e.g. II)")
    p.add_argument("--losses",         default="bce,mkd,crf",
                   help="Comma-separated: bce,mkd,crf")
    p.add_argument("--batch_size",     type=int, default=None,
                   help="Override batch size from config")
    p.add_argument("--seed",           type=int, default=None,
                   help="Random seed (overrides config)")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--output_dir",     default=None,
                   help="Override output directory from config")
    p.add_argument("--init_encoder_ckpt", default=None,
                   help="CLECG pretrained encoder checkpoint for initialization")
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
