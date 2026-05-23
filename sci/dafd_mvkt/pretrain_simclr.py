"""
SimCLR contrastive pretraining for ECG encoders.

Architecture:
  - Encoder: ResNet1d (same backbone used for downstream HST-KD)
  - Projection head: pooled → Linear(feat_dim→proj_dim) → BN → ReLU → Linear(proj_dim→proj_dim)
  - Loss: NT-Xent (in-batch negatives, symmetric)

Unlike CLECG (MoCo), SimCLR:
  - Has NO momentum encoder and NO queue.
  - Both views are processed with gradient.
  - Requires larger batch size for sufficient negatives (≥256 recommended).

Output:
  {output_dir}/{run_name}_best.pt    — best val-loss checkpoint
  {output_dir}/{run_name}_last.pt    — final epoch checkpoint
  {output_dir}/{run_name}_encoder.pt — encoder weights only (for --init_encoder_ckpt)
  {output_dir}/{run_name}_log.csv    — epoch log

Usage:
    python dafd_mvkt/pretrain_simclr.py \\
        --config dafd_mvkt/configs/simclr_leadII_100hz.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --lead II \\
        --hz 100 \\
        --seed 0 \\
        --output_dir dafd_mvkt/outputs/simclr
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.simclr_dataset import SimCLRDataset
from losses.simclr_loss import NTXentLoss
from models.resnet1d import ResNet1d
from utils.seed import set_seed


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder(cfg: dict) -> ResNet1d:
    mc = cfg["model"]
    return ResNet1d(
        in_channels   = mc["in_channels"],
        num_classes   = mc["num_classes"],
        layers        = mc.get("layers", [3, 4, 6, 3]),
        base_channels = mc.get("base_channels", 64),
        proj_dim      = mc.get("proj_dim", 128),
        dropout       = mc.get("dropout", 0.0),
    )


class SimCLRProjectionHead(nn.Module):
    """2-layer MLP projection head with BN (SimCLR v2 style)."""

    def __init__(self, feat_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def run_epoch(
    encoder:    ResNet1d,
    proj_head:  SimCLRProjectionHead,
    criterion:  NTXentLoss,
    optimizer:  torch.optim.Optimizer | None,
    loader:     DataLoader,
    device:     torch.device,
    grad_clip:  float = 1.0,
    training:   bool = True,
) -> tuple[float, float]:
    """Run one epoch. Returns (avg_loss, avg_acc)."""
    encoder.train(training)
    proj_head.train(training)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="  train" if training else "  val",
                          leave=False, dynamic_ncols=True):
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)

            out1 = encoder(v1, return_features=True)
            out2 = encoder(v2, return_features=True)

            z1 = proj_head(out1["pooled"])
            z2 = proj_head(out2["pooled"])

            loss, info = criterion(z1, z2)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(proj_head.parameters()),
                    grad_clip,
                )
                optimizer.step()

            total_loss += loss.item()
            total_acc  += info["acc"]
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def _save(
    encoder:   ResNet1d,
    proj_head: SimCLRProjectionHead,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    path:      Path,
) -> None:
    torch.save({
        "epoch":               epoch,
        "encoder_state_dict":  encoder.state_dict(),
        "proj_head_state_dict": proj_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "temperature":         criterion.T,
    }, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--lead",       default=None)
    p.add_argument("--hz",         type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name",   default=None)
    args = p.parse_args()

    cfg   = load_cfg(args.config)
    tc    = cfg.get("training", {})
    dc    = cfg.get("data", {})
    prc   = cfg.get("pretraining", {})

    # CLI overrides
    lead       = args.lead       or dc.get("lead", "II")
    hz         = args.hz         or dc.get("sampling_hz", 100)
    seed       = args.seed       if args.seed is not None else tc.get("seed", 0)
    epochs     = args.epochs     or tc.get("epochs", 100)
    batch_size = args.batch_size or tc.get("batch_size", 256)
    out_dir    = Path(args.output_dir or cfg.get("output", {}).get("dir", "dafd_mvkt/outputs/simclr"))
    run_name   = args.run_name   or cfg.get("output", {}).get("name", f"simclr_{lead}_{hz}hz_seed{seed}")

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SimCLR pretraining: lead={lead} hz={hz} seed={seed} epochs={epochs}")
    print(f"Device: {device}  Output: {out_dir}/{run_name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    aug_cfg = prc.get("augment", {})
    train_ds = SimCLRDataset(args.data_dir, hz, lead, "train", aug_cfg, seed)
    val_ds   = SimCLRDataset(args.data_dir, hz, lead, "val",   aug_cfg, seed)

    num_workers = tc.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder   = build_encoder(cfg).to(device)
    feat_dim  = encoder.feat_dim
    proj_dim  = cfg["model"].get("proj_dim", 128)
    proj_head = SimCLRProjectionHead(feat_dim, proj_dim).to(device)

    temperature = prc.get("temperature", 0.1)
    criterion   = NTXentLoss(temperature=temperature)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    lr           = tc.get("lr", 3e-4)
    weight_decay = tc.get("weight_decay", 1e-5)
    grad_clip    = tc.get("grad_clip", 1.0)

    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_path  = out_dir / f"{run_name}_log.csv"
    best_val  = float("inf")
    best_ep   = 0

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(
                encoder, proj_head, criterion, optimizer, train_loader,
                device, grad_clip, training=True,
            )
            val_loss, val_acc = run_epoch(
                encoder, proj_head, criterion, None, val_loader,
                device, training=False,
            )
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            print(
                f"Ep {ep:3d}/{epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"lr={cur_lr:.2e}"
            )
            writer.writerow([ep, tr_loss, tr_acc, val_loss, val_acc, cur_lr])

            _save(encoder, proj_head, criterion, optimizer, ep,
                  out_dir / f"{run_name}_last.pt")

            if val_loss < best_val:
                best_val = val_loss
                best_ep  = ep
                _save(encoder, proj_head, criterion, optimizer, ep,
                      out_dir / f"{run_name}_best.pt")
                print(f"  → new best val_loss={best_val:.4f} at epoch {best_ep}")

    print(f"\nBest epoch: {best_ep}  val_loss={best_val:.4f}")

    # ── Save encoder for downstream init ──────────────────────────────────────
    # Load best checkpoint and save encoder-only in CLECG-compatible format
    best_ckpt = torch.load(out_dir / f"{run_name}_best.pt",
                           map_location="cpu", weights_only=False)
    enc_path  = out_dir / f"{run_name}_encoder.pt"
    torch.save(
        {"state_dict": best_ckpt["encoder_state_dict"], "hz": hz, "lead": lead,
         "type": "simclr", "best_epoch": best_ep},
        enc_path,
    )
    print(f"Encoder saved: {enc_path}")


if __name__ == "__main__":
    main()
