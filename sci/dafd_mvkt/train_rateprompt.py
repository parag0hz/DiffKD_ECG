"""
RatePrompt-KD: Sampling-Rate Prompted Knowledge Distillation.

Student learns from multi-rate Lead-II signals conditioned on sampling rate.
Teachers (1L500, 1L100) and KD loss are identical to F02T01.
Test is always Lead-II 50 Hz.

Backward compat:
  --rate_conditioning none → identical architecture and loss to F02T01.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset, SUPERCLASSES
from losses.mkd import multi_label_kd_loss
from models.rateprompt_resnet1d import build_student
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import extract_encoder_state_dict

# ── local baseline ────────────────────────────────────────────────────────────
LOCAL_F02T01_AUC = 0.84644
LOCAL_F02T01_F1  = 0.62544


def _load_encoder_init_safe(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load SimCLR encoder weights, skipping keys with shape mismatches (e.g. stem for time_pe)."""
    sd = extract_encoder_state_dict(ckpt_path)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in sd if k not in filtered]
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    print(f"  [load_encoder_init] loaded {loaded}/{len(sd)} keys "
          f"(skipped {len(skipped)} shape-mismatch) from {ckpt_path}")
    if skipped:
        print(f"    shape-skipped: {skipped[:5]}" + ("..." if len(skipped) > 5 else ""))
    if missing:
        print(f"    still missing ({len(missing)}): {missing[:3]}" + ("..." if len(missing) > 3 else ""))


T1L500_KEY = "teacher_1l_500"   # [B, 1, 5000]
T1L100_KEY = "teacher_1l_100"   # [B, 1, 1000]
STUDENT_KEY = "student_x"       # [B, 1,  500]  50 Hz


# ── multi-rate input selection ────────────────────────────────────────────────

def _get_student_input(
    batch: dict[str, torch.Tensor],
    fs: int,
    device: torch.device,
) -> torch.Tensor:
    """Return Lead-II signal at requested sampling rate from a batch."""
    if fs == 50:
        return batch[STUDENT_KEY].to(device)                          # [B,1,500]
    elif fs == 100:
        return batch[T1L100_KEY].to(device)                           # [B,1,1000]
    elif fs == 250:
        x500 = batch[T1L500_KEY].to(device)                          # [B,1,5000]
        return F.interpolate(x500, size=2500,
                             mode="linear", align_corners=False)      # [B,1,2500]
    elif fs == 500:
        return batch[T1L500_KEY].to(device)                          # [B,1,5000]
    else:
        raise ValueError(f"Unsupported fs={fs}. Must be one of 50,100,250,500.")


def _student_forward(
    student: nn.Module,
    x: torch.Tensor,
    fs: int,
    rate_conditioning: str,
    return_features: bool = False,
) -> dict[str, torch.Tensor]:
    """Unified student forward: handles conditioning=none and otherwise."""
    if rate_conditioning == "none":
        return student(x, return_features=return_features)
    else:
        fs_t = torch.full((x.shape[0],), float(fs),
                          dtype=torch.float32, device=x.device)
        return student(x, fs=fs_t, return_features=return_features)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_eval(
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rate_conditioning: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    student.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch[STUDENT_KEY].to(device)        # always 50 Hz
        out = _student_forward(student, x, 50, rate_conditioning)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, torch.cat(all_labels).numpy(), all_ids


# ── summary CSV ───────────────────────────────────────────────────────────────

def _update_summary(out_dir: Path, row: dict) -> None:
    csv_path = out_dir / "summary.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("variant_id") != row["variant_id"]]
    rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep  = args.epochs     or tc.get("epochs", 100)
    bs  = args.batch_size or tc.get("batch_size", 512)
    lr_ = tc.get("lr", 1e-3)
    nw  = tc.get("num_workers", 4)
    pw  = tc.get("persistent_workers", True)
    wd  = tc.get("weight_decay", 1e-4)
    gc  = tc.get("grad_clip", 1.0)

    lambda_kd   = args.lambda_kd
    temperature = args.temperature
    w_parts     = [float(x) for x in args.weights.split(",")]
    w500        = w_parts[0] / sum(w_parts)
    w100        = w_parts[1] / sum(w_parts)

    rate_conditioning = args.rate_conditioning
    film_layers       = [s.strip() for s in args.film_layers.split(",")]
    time_pe_dim       = args.time_pe_dim
    fs_embed_dim      = args.fs_embed_dim

    multi_rate = args.multi_rate_train
    rate_set   = [int(x) for x in args.rate_set.split(",")]

    variant_id = args.variant_id or "RPopt"
    out_name   = f"student_II_50hz_rp_{variant_id}_seed{seed}"
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    ckpt_path = out_dir / f"{out_name}_best.pt"
    log_path  = out_dir / "logs" / f"{out_name}.csv"

    print(f"\n{'='*65}")
    print(f"RatePrompt-KD: {variant_id}")
    print(f"rate_conditioning={rate_conditioning}  multi_rate={multi_rate}  "
          f"rate_set={rate_set}")
    print(f"film_layers={film_layers}  time_pe_dim={time_pe_dim}  "
          f"fs_embed_dim={fs_embed_dim}")
    print(f"w500={w500:.3f}  w100={w100:.3f}  T={temperature}  λ_kd={lambda_kd}")
    print(f"device={device}  seed={seed}  ep={ep}  bs={bs}")
    print(f"output: {ckpt_path}")
    print(f"{'='*65}")

    # ── save config snapshot ─────────────────────────────────────────────────
    cfg_snap = {k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()}
    cfg_snap["__resolved"] = {
        "w500": w500, "w100": w100, "ep": ep, "bs": bs, "lr": lr_,
        "rate_set": rate_set, "film_layers": film_layers,
    }
    with open(out_dir / f"config_snapshot_{out_name}.json", "w") as f:
        json.dump(cfg_snap, f, indent=2)

    # ── frozen teachers ──────────────────────────────────────────────────────
    def _load_teacher(ckpt: str, in_ch: int) -> ResNet1d:
        m = ResNet1d(in_channels=in_ch, num_classes=5,
                     layers=[3, 4, 6, 3], base_channels=64,
                     proj_dim=128, dropout=0.0).to(device)
        load_checkpoint(ckpt, m, device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    t500 = _load_teacher(args.teacher_1l500_ckpt, in_ch=1)
    t100 = _load_teacher(args.teacher_1l100_ckpt, in_ch=1)
    print(f"Teacher 1L500: {args.teacher_1l500_ckpt}")
    print(f"Teacher 1L100: {args.teacher_1l100_ckpt}")

    # ── student ──────────────────────────────────────────────────────────────
    student = build_student(
        rate_conditioning=rate_conditioning,
        time_pe_dim=time_pe_dim,
        fs_embed_dim=fs_embed_dim,
        film_layers=film_layers if rate_conditioning in ("fs_film","time_pe_film") else None,
    ).to(device)

    # SimCLR init (shape-safe: skips stem when time_pe expands in_channels)
    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        target = student.backbone if hasattr(student, "backbone") else student
        _load_encoder_init_safe(target, args.student_init_ckpt, device)
        print(f"Student init: {args.student_init_ckpt}")
    else:
        print("Student: random init")

    n_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_p/1e6:.3f}M")

    # ── data ─────────────────────────────────────────────────────────────────
    lead = args.lead or cfg["data"].get("lead", "II")
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,  batch_size=512, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False, **kw)

    # ── sanity check ─────────────────────────────────────────────────────────
    print("\n[sanity] first batch check …")
    _first = next(iter(train_loader))
    for fs_chk in rate_set[:2]:
        _x = _get_student_input(_first, fs_chk, device)
        _out = _student_forward(student, _x, fs_chk, rate_conditioning)
        _log = _out["logits"]
        assert not _log.isnan().any() and not _log.isinf().any(), "NaN/Inf in logits!"
        if args.debug:
            print(f"  fs={fs_chk}: x={tuple(_x.shape)}  "
                  f"logits mean={_log.mean():.4f} std={_log.std():.4f}")
        else:
            print(f"  fs={fs_chk}: x={tuple(_x.shape)}  logits OK ({_log.mean():.3f}±{_log.std():.3f})")

    if hasattr(student, "time_pe_module") and student.time_pe_module is not None:
        _pe = student.time_pe_module(2, 500, 50.0, device)
        print(f"  time PE: {tuple(_pe.shape)} (dim={_pe.shape[1]})")
    if hasattr(student, "fs_embed_module") and student.fs_embed_module is not None:
        _fs_t = torch.tensor([50.0, 100.0], device=device)
        _emb  = student.fs_embed_module(_fs_t)
        print(f"  fs embed: {tuple(_emb.shape)} mean={_emb.mean():.4f} std={_emb.std():.4f}")
    print("[sanity] OK\n")

    # ── optimizer ────────────────────────────────────────────────────────────
    bce_fn    = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    # ── training loop ────────────────────────────────────────────────────────
    best_auc = 0.0
    log_rows: list[dict] = []
    fs_counts: dict[int, int] = {fs: 0 for fs in rate_set}

    for epoch in range(1, ep + 1):
        student.train()
        ep_bce = ep_kd = ep_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}",
                          dynamic_ncols=True, leave=False):
            y = batch["y"].to(device)

            # select fs for this step
            fs = random.choice(rate_set) if multi_rate else rate_set[0]
            fs_counts[fs] = fs_counts.get(fs, 0) + 1
            s_in = _get_student_input(batch, fs, device)

            s_out    = _student_forward(student, s_in, fs, rate_conditioning)
            s_logits = s_out["logits"]

            # teacher forward (always same inputs regardless of student fs)
            with torch.no_grad():
                x500 = batch[T1L500_KEY].to(device)
                x100 = batch[T1L100_KEY].to(device)
                l500 = t500(x500)["logits"]
                l100 = t100(x100)["logits"]

            kd500    = multi_label_kd_loss(s_logits, l500, temperature)
            kd100    = multi_label_kd_loss(s_logits, l100, temperature)
            loss_kd  = w500 * kd500 + w100 * kd100
            loss_bce = bce_fn(s_logits, y)
            loss     = loss_bce + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(student.parameters(), gc)
            optimizer.step()

            ep_bce  += loss_bce.item()
            ep_kd   += loss_kd.item()
            ep_tot  += loss.item()
            n_batches += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(student, val_loader, device, rate_conditioning)
        val_metrics = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_metrics["macro_auc"]
        val_f1  = val_metrics.get("macro_f1", 0.0)

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch,
                            {"val_auc": val_auc},
                            {"variant": variant_id,
                             "rate_conditioning": rate_conditioning,
                             "w500": w500, "w100": w100,
                             "temperature": temperature})

        n = n_batches
        print(f"Ep {epoch:3d} | bce={ep_bce/n:.4f} kd={ep_kd/n:.4f} "
              f"tot={ep_tot/n:.4f} | val_auc={val_auc:.4f} val_f1={val_f1:.4f} "
              f"{'★' if is_best else ''}")

        log_rows.append({
            "epoch":         epoch,
            "fs_sampled":    str(fs_counts),
            "bce":           round(ep_bce / n, 5),
            "kd":            round(ep_kd  / n, 5),
            "total":         round(ep_tot / n, 5),
            "val_auc":       round(val_auc, 5),
            "val_f1":        round(val_f1,  5),
            "best_val_auc":  round(best_auc, 5),
        })

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    # ── final evaluation ─────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, student, device)

    val_probs, val_labels, _          = _run_eval(student, val_loader,  device, rate_conditioning)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(student, test_loader, device, rate_conditioning)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    # ── metrics JSON ─────────────────────────────────────────────────────────
    metrics: dict = {
        "output_name":        out_name,
        "variant_id":         variant_id,
        "rate_conditioning":  rate_conditioning,
        "multi_rate_train":   multi_rate,
        "rate_set":           rate_set,
        "film_layers":        film_layers,
        "time_pe_dim":        time_pe_dim,
        "fs_embed_dim":       fs_embed_dim,
        "w500": w500, "w100": w100, "temperature": temperature,
        "macro_auc":          round(m05["macro_auc"],          5),
        "macro_f1_0_5":       round(m05.get("macro_f1", 0),   5),
        "macro_f1_tuned":     round(m_tune.get("macro_f1", 0), 5),
        "val_auc_best":       round(best_auc, 5),
        "delta_auc_vs_f02t01": round(m05["macro_auc"] - LOCAL_F02T01_AUC, 5),
        "per_class_thresholds": {
            SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
        },
        "_meta": {"seed": seed, "lambda_kd": lambda_kd, "n_params": n_p,
                  "local_f02t01_baseline": LOCAL_F02T01_AUC},
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    # val metrics
    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc":      round(val_m05["macro_auc"],          5),
                   "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5)},
                  f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # thresholds JSON
    with open(out_dir / f"thresholds_{out_name}.json", "w") as f:
        json.dump(metrics["per_class_thresholds"], f, indent=2)

    # predictions CSV
    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = [{"record_id": rid,
               **{f"y_{c}":    int(test_labels[i, ci]) for ci, c in enumerate(SUPERCLASSES)},
               **{f"prob_{c}": round(float(test_probs[i, ci]), 5) for ci, c in enumerate(SUPERCLASSES)},
               **{f"pred_{c}": int(preds[i, ci]) for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(
        out_dir / f"predictions_test_{out_name}.csv", index=False)

    # classwise CSV
    cw = [{"class": c,
            "auc":       round(m05.get(f"auc_{c}", float("nan")), 5),
            "f1_at_0.5": round(m05.get(f"f1_{c}",  float("nan")), 5),
            "f1_tuned":  round(m_tune.get(f"f1_{c}", float("nan")), 5),
            "threshold": round(float(thresholds[ci]), 4)}
           for ci, c in enumerate(SUPERCLASSES)]
    pd.DataFrame(cw).to_csv(out_dir / f"classwise_{out_name}.csv", index=False)

    # summary CSV
    auc = metrics["macro_auc"]
    summary_row = {
        "variant_id":              variant_id,
        "rate_conditioning":       rate_conditioning,
        "multi_rate_train":        multi_rate,
        "rate_set":                str(rate_set),
        "AUC_macro":               auc,
        "delta_AUC_vs_local_baseline": round(auc - LOCAL_F02T01_AUC, 5),
        "F1_tuned":                metrics["macro_f1_tuned"],
        "delta_F1":                round(metrics["macro_f1_tuned"] - LOCAL_F02T01_F1, 5),
        **{f"{c}_AUC": metrics[f"auc_{c}"]      for c in SUPERCLASSES},
        **{f"{c}_F1":  metrics[f"f1_tuned_{c}"] for c in SUPERCLASSES},
        "best_epoch":              round(best_auc, 5),
        "seed":                    seed,
        "notes":                   "",
    }
    _update_summary(out_dir, summary_row)

    da  = auc - LOCAL_F02T01_AUC
    f1t = metrics["macro_f1_tuned"]
    print(f"\nAUC={auc:.4f}  F1_tuned={f1t:.4f}")
    print(f"vs local F02T01 (0.84644): {da:+.4f} AUC")
    print(f"Beats F02T01: {'YES ★' if da > 0.001 else ('tie' if abs(da) <= 0.0005 else 'no')}")
    print(f"Saved: {out_dir}/metrics_test_{out_name}.json")


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RatePrompt-KD trainer")

    # core
    p.add_argument("--config",             default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",           required=True)
    p.add_argument("--lead",               default="II")
    p.add_argument("--teacher_1l500_ckpt", required=True)
    p.add_argument("--teacher_1l100_ckpt", required=True)
    p.add_argument("--student_init_ckpt",  default=None)
    p.add_argument("--weights",            default="0.60,0.40")
    p.add_argument("--lambda_kd",          type=float, default=1.0)
    p.add_argument("--temperature",        type=float, default=1.5)

    # rate conditioning
    p.add_argument("--rate_conditioning",  default="none",
                   choices=["none", "time_pe", "fs_film", "time_pe_film"])
    p.add_argument("--time_pe_dim",        type=int, default=15)
    p.add_argument("--fs_embed_dim",       type=int, default=64)
    p.add_argument("--film_layers",        default="layer1,layer2,layer3,layer4")

    # multi-rate training
    p.add_argument("--multi_rate_train",   action="store_true")
    p.add_argument("--rate_set",           default="50",
                   help="comma-separated fs values, e.g. '50,100,250,500'")

    # misc
    p.add_argument("--seed",               type=int,  default=0)
    p.add_argument("--epochs",             type=int,  default=None)
    p.add_argument("--batch_size",         type=int,  default=None)
    p.add_argument("--variant_id",         default=None)
    p.add_argument("--output_dir",         default="dafd_mvkt/outputs/rateprompt_f02t01")
    p.add_argument("--debug",              action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
