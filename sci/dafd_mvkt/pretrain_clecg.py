"""
CLECG-style MoCo contrastive pretraining for ECG encoders.

Architecture:
  - Query encoder  : ResNet1d (trainable)
  - Key encoder    : ResNet1d (EMA copy of query encoder, m=0.999)
  - Queue          : K=2048 L2-normalized key embeddings
  - Loss           : InfoNCE (NT-Xent via CLECGLoss)

Output:
  {output_dir}/{run_name}_best.pt   — best validation-loss checkpoint
  {output_dir}/{run_name}_last.pt   — final epoch checkpoint
  {output_dir}/{run_name}_encoder.pt — encoder weights only (for downstream init)
  {output_dir}/{run_name}_log.csv   — epoch log

Usage:
    python dafd_mvkt/pretrain_clecg.py \\
        --config dafd_mvkt/configs/clecg_leadII_100hz.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --lead II \\
        --hz 100 \\
        --seed 0 \\
        --output_dir dafd_mvkt/outputs
"""
from __future__ import annotations
import argparse
import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.clecg_dataset import CLECGDataset
from losses.clecg_loss import CLECGLoss
from models.resnet1d import ResNet1d
from utils.seed import set_seed


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder(cfg: dict) -> ResNet1d:
    mc = cfg["model"]
    return ResNet1d(
        in_channels  = mc["in_channels"],
        num_classes  = mc["num_classes"],
        layers       = mc.get("layers", [3, 4, 6, 3]),
        base_channels= mc.get("base_channels", 64),
        proj_dim     = mc.get("proj_dim", 128),
        dropout      = mc.get("dropout", 0.0),
    )


@torch.no_grad()
def momentum_update(query_enc: nn.Module, key_enc: nn.Module, m: float) -> None:
    """EMA update: key_enc ← m * key_enc + (1-m) * query_enc."""
    for q_param, k_param in zip(query_enc.parameters(), key_enc.parameters()):
        k_param.data.mul_(m).add_(q_param.data * (1.0 - m))


def run_epoch(
    query_enc: nn.Module,
    key_enc: nn.Module,
    criterion: CLECGLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    momentum: float,
    training: bool,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> dict:
    query_enc.train(training)
    key_enc.eval()

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for batch in tqdm(loader, desc="train" if training else "val",
                          dynamic_ncols=True, leave=False):
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)

            with autocast("cuda", enabled=use_amp):
                q_out = query_enc(v1, return_features=True)
                q = q_out["proj"]

                with torch.no_grad():
                    k_out = key_enc(v2, return_features=True)
                    k = k_out["proj"]

                # During validation, do NOT update queue (would corrupt training negatives)
                loss, info = criterion(q, k, update_queue=training)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(query_enc.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                momentum_update(query_enc, key_enc, momentum)

            total_loss += info["loss"]
            total_acc  += info["accuracy"]
            n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
    }


def pretrain(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    dc  = cfg["data"]
    tc  = cfg["training"]

    set_seed(args.seed if args.seed is not None else tc.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)

    hz   = args.hz   or int(dc.get("sampling_hz", dc.get("sampling_rate", 100)))
    lead = args.lead or dc.get("lead", "II")

    aug_cfg = cfg.get("augment", {})

    out_dir = Path(args.output_dir) if args.output_dir else Path(tc.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"clecg_{lead}_{hz}hz_seed{args.seed or 0}"
    print(f"Run    : {run_name}  lead={lead}  hz={hz}", flush=True)

    # ── datasets ─────────────────────────────────────────────────────────────
    num_workers = tc.get("num_workers", 4)
    batch_size  = args.batch_size or tc.get("batch_size", 256)

    train_ds = CLECGDataset(
        ptbxl_root=args.data_dir, sampling_rate=hz, lead=lead,
        split="train", augment_cfg=aug_cfg,
        seed=args.seed,
    )
    val_ds = CLECGDataset(
        ptbxl_root=args.data_dir, sampling_rate=hz, lead=lead,
        split="val", augment_cfg=aug_cfg,
        seed=(args.seed or 0) + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train  : {len(train_ds)}  Val: {len(val_ds)}", flush=True)

    # ── models ────────────────────────────────────────────────────────────────
    query_enc = build_encoder(cfg).to(device)
    key_enc   = copy.deepcopy(query_enc).to(device)

    # Key encoder never updated via gradients
    for p in key_enc.parameters():
        p.requires_grad_(False)

    proj_dim   = cfg["model"].get("proj_dim", 128)
    queue_size = tc.get("queue_size", 2048)
    temperature= tc.get("temperature", 0.07)
    momentum   = tc.get("momentum", 0.999)

    criterion = CLECGLoss(proj_dim, queue_size, temperature).to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    lr       = float(tc.get("lr", 3e-4))
    wd       = float(tc.get("weight_decay", 1e-4))
    epochs   = args.epochs or int(tc.get("epochs", 100))

    optimizer = torch.optim.AdamW(query_enc.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if tc.get("cosine_lr", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(tc.get("lr_min", 1e-6))
        )

    use_amp   = tc.get("amp", False)
    grad_clip = float(tc.get("grad_clip", 1.0))
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── log ───────────────────────────────────────────────────────────────────
    log_path = out_dir / f"{run_name}_log.csv"
    log_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        train_stats = run_epoch(
            query_enc, key_enc, criterion, train_loader,
            optimizer, scaler, device, momentum, training=True,
            use_amp=use_amp, grad_clip=grad_clip,
        )
        val_stats = run_epoch(
            query_enc, key_enc, criterion, val_loader,
            optimizer, scaler, device, momentum, training=False,
            use_amp=use_amp, grad_clip=grad_clip,
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch":     ep,
            "train_loss": round(train_stats["loss"], 5),
            "train_acc":  round(train_stats["accuracy"], 5),
            "val_loss":   round(val_stats["loss"], 5),
            "val_acc":    round(val_stats["accuracy"], 5),
            "lr":         f"{cur_lr:.2e}",
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        print(
            f"Ep {ep:3d}/{epochs}  "
            f"train_loss={train_stats['loss']:.4f}  acc={train_stats['accuracy']:.3f}  "
            f"val_loss={val_stats['loss']:.4f}  acc={val_stats['accuracy']:.3f}  "
            f"lr={cur_lr:.2e}",
            flush=True,
        )

        # Save last
        _save(query_enc, key_enc, criterion, optimizer, ep, out_dir / f"{run_name}_last.pt")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            _save(query_enc, key_enc, criterion, optimizer, ep, out_dir / f"{run_name}_best.pt")
            print(f"  → best val_loss={best_val_loss:.4f}", flush=True)

    # Save encoder-only weights for downstream initialization
    enc_path = out_dir / f"{run_name}_encoder.pt"
    torch.save({"state_dict": query_enc.state_dict(), "hz": hz, "lead": lead}, enc_path)
    print(f"\nEncoder saved: {enc_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


def _save(
    query_enc: nn.Module, key_enc: nn.Module, criterion: CLECGLoss,
    optimizer: torch.optim.Optimizer, epoch: int, path: Path,
) -> None:
    torch.save({
        "epoch":           epoch,
        "query_enc":       query_enc.state_dict(),
        "key_enc":         key_enc.state_dict(),
        "queue":           criterion.queue.queue,
        "queue_ptr":       criterion.queue.queue_ptr,
        "optimizer":       optimizer.state_dict(),
    }, path)


def parse_args() -> argparse.Namespace:
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
    return p.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
