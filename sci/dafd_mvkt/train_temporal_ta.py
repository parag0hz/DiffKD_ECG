"""
Train a low-Hz Teacher Assistant (TA) from a frozen high-Hz TA.

Progressive temporal distillation step:
  1-lead 500Hz TA (frozen)  →  1-lead 100Hz TA (trainable)

Loss:
  L = L_BCE
      + alpha_high_ta      * MKD(TA_high → TA_low)
      + beta_high_ta_crf   * CRF(TA_high ↔ TA_low)
      + gamma_high_ta_feat * FeatureKD(TA_high → TA_low)

Usage (from ~/sci):
    python dafd_mvkt/train_temporal_ta.py \\
        --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --high_ta_ckpt outputs/ta_ii_500hz_best.pt \\
        --lead II \\
        --seed 0 \\
        --batch_size 256 \\
        --output_dir dafd_mvkt/outputs
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
from losses.feature_kd import FeatureKDLoss
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


def build_loaders(cfg: dict, data_dir: str, lead: str) -> tuple[DataLoader, DataLoader]:
    dc = cfg["data"]
    tc = cfg["training"]
    hz = dc.get("sampling_hz", dc.get("sampling_rate", 100))
    kw = dict(num_workers=tc.get("num_workers", 4), pin_memory=True)

    def _ds(split: str) -> PTBXLDataset:
        return PTBXLDataset(
            ptbxl_root=data_dir, split=split, mode="progressive",
            sampling_rate=hz, lead=lead,
            normalize=dc.get("normalize", "zscore"),
        )

    train_loader = DataLoader(
        _ds("train"), batch_size=tc["batch_size"], shuffle=True,
        drop_last=True, **kw,
    )
    val_loader = DataLoader(_ds("val"), batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x   = batch["ta_low_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics(labels, probs)


def _check_nan(name: str, val: torch.Tensor) -> None:
    if torch.isnan(val):
        print(f"NaN in loss [{name}] — aborting.", file=sys.stderr)
        sys.exit(1)


def train(args: argparse.Namespace) -> None:
    cfg  = load_cfg(args.config)
    tc   = cfg["training"]
    oc   = cfg["output"]
    dc   = cfg["data"]
    lead = args.lead or dc.get("lead", "II")

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    seed = args.seed if args.seed is not None else tc.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hz     = dc.get("sampling_hz", dc.get("sampling_rate", 100))
    high_hz = tc.get("high_ta_hz", 500)

    run_name = args.run_name or f"ta_{lead}_{hz}hz_from_ta{high_hz}_seed{seed}"
    out_dir  = Path(args.output_dir) if args.output_dir else Path(oc.get("dir", "outputs"))
    ckpt_path = out_dir / f"{run_name}_best.pt"
    log_dir   = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  seed: {seed}  hz: {hz}  high_hz: {high_hz}")
    print(f"Run: {run_name}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(cfg, args.data_dir, lead)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    # ── frozen high-Hz TA ────────────────────────────────────────────────────
    high_cfg_path = args.high_ta_config or str(_HERE / "configs/ta_ii_500hz.yaml")
    high_cfg      = load_cfg(high_cfg_path)
    high_ta       = build_model(high_cfg).to(device)
    load_checkpoint(args.high_ta_ckpt, high_ta, device)
    high_ta.eval()
    for p in high_ta.parameters():
        p.requires_grad_(False)
    print(f"Loaded high-Hz TA from {args.high_ta_ckpt}")

    # ── trainable low-Hz TA ──────────────────────────────────────────────────
    low_ta   = build_model(cfg).to(device)
    if args.init_encoder_ckpt:
        _load_encoder_ckpt(low_ta, args.init_encoder_ckpt, device)
    n_params = sum(p.numel() for p in low_ta.parameters() if p.requires_grad)
    print(f"Low-Hz TA params: {n_params/1e6:.2f}M")

    # ── feature KD ───────────────────────────────────────────────────────────
    feat_kd_fn = FeatureKDLoss(
        student_channels = low_ta.feat_dim,
        ta_channels      = high_ta.feat_dim,
        pool_size        = tc.get("feature_pool_size", 128),
        normalize        = True,
    ).to(device)

    # ── optimiser ────────────────────────────────────────────────────────────
    opt_params = list(low_ta.parameters()) + list(feat_kd_fn.parameters())
    optimizer  = torch.optim.AdamW(opt_params, lr=tc["lr"],
                                   weight_decay=tc["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"]
    )
    criterion  = nn.BCEWithLogitsLoss()
    scaler     = GradScaler("cuda", enabled=tc.get("amp", False))
    grad_clip  = tc.get("grad_clip", 1.0)

    alpha = tc.get("alpha_high_ta",      1.0)
    beta  = tc.get("beta_high_ta_crf",   0.1)
    gamma = tc.get("gamma_high_ta_feat", 0.2)
    mkd_T = tc.get("temperature",             2.0)
    crf_T = tc.get("contrastive_temperature", 0.07)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path   = log_dir / f"{run_name}.csv"
    log_fields = ["epoch", "bce", "mkd", "crf", "feat", "total",
                  "macro_auc", "macro_f1"]
    csv_fh = open(log_path, "w", newline="")
    csv_wr = csv.DictWriter(csv_fh, fieldnames=log_fields, extrasaction="ignore")
    csv_wr.writeheader()

    best_auc = 0.0

    for epoch in range(1, tc["epochs"] + 1):
        low_ta.train()
        feat_kd_fn.train()

        totals = {k: 0.0 for k in ["bce", "mkd", "crf", "feat", "total"]}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{tc['epochs']}",
                          leave=False, dynamic_ncols=True):
            high_x = batch["ta_high_x"].to(device)   # [B, 1, 5000]
            low_x  = batch["ta_low_x"].to(device)    # [B, 1, L]
            y      = batch["y"].to(device)

            if args.debug and epoch == 1 and n_batches == 0:
                print(f"  [debug] ta_high_x:{high_x.shape}  "
                      f"ta_low_x:{low_x.shape}  y:{y.shape}")

            optimizer.zero_grad()

            with autocast("cuda", enabled=tc.get("amp", False)):
                low_out  = low_ta(low_x, return_features=True)
                with torch.no_grad():
                    high_out = high_ta(high_x, return_features=True)

                low_logits = low_out["logits"]

                loss_bce = criterion(low_logits, y)
                _check_nan("bce", loss_bce)
                total = loss_bce
                totals["bce"] += loss_bce.item()

                lv = multi_label_kd_loss(low_logits, high_out["logits"], mkd_T)
                _check_nan("mkd", lv)
                total = total + alpha * lv
                totals["mkd"] += lv.item()

                lv = cross_rate_contrastive_loss(
                    low_out["pooled"], high_out["pooled"],
                    low_out["proj"],   high_out["proj"],
                    temperature=crf_T,
                )
                _check_nan("crf", lv)
                total = total + beta * lv
                totals["crf"] += lv.item()

                lv = feat_kd_fn(low_out["feature_map"], high_out["feature_map"])
                _check_nan("feat", lv)
                total = total + gamma * lv
                totals["feat"] += lv.item()

                totals["total"] += total.item()

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(opt_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n_batches += 1

        scheduler.step()
        for k in totals:
            totals[k] /= n_batches

        metrics = evaluate(low_ta, val_loader, device)
        print(f"[{epoch:4d}]  total={totals['total']:.4f}  "
              f"bce={totals['bce']:.4f}  mkd={totals['mkd']:.4f}  "
              f"crf={totals['crf']:.4f}  feat={totals['feat']:.4f}  "
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
            extra = {"feat_kd_state": feat_kd_fn.state_dict()}
            save_checkpoint(str(ckpt_path), low_ta, epoch, metrics, cfg, extra)
            print(f"  ✓ best AUC={best_auc:.4f} → {ckpt_path}")

    csv_fh.close()
    print(f"\nDone. Best val AUC: {best_auc:.4f}")
    print(f"Log saved to {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--high_ta_ckpt",   required=True,
                   help="Path to frozen high-Hz TA checkpoint")
    p.add_argument("--high_ta_config", default=None,
                   help="Config for high-Hz TA (default: configs/ta_ii_500hz.yaml)")
    p.add_argument("--lead",           default=None)
    p.add_argument("--seed",           type=int, default=None)
    p.add_argument("--batch_size",     type=int, default=None)
    p.add_argument("--output_dir",     default=None)
    p.add_argument("--run_name",       default=None)
    p.add_argument("--init_encoder_ckpt", default=None,
                   help="CLECG pretrained encoder checkpoint for initialization")
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
