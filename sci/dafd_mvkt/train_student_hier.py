"""
Train single-lead low-Hz student with hierarchical Teacher Assistant supervision.

Three models are used:
  teacher  — frozen 12-lead 500 Hz ResNet (spatial gap supervisor)
  ta       — frozen 1-lead 500 Hz ResNet (temporal-resolution gap supervisor)
  student  — trainable 1-lead low-Hz ResNet

Loss components (select via --losses):
  bce          BCE against ground truth labels
  teacher_mkd  MKD from 12-lead teacher to student
  ta_mkd       MKD from 1-lead TA to student
  ta_crf       Symmetric InfoNCE between TA and student projections
  feature      Feature-map MSE (TA feature map → student feature map)

Default total loss (--losses bce,ta_mkd,ta_crf,feature):
  L = L_BCE
      + alpha_ta      * L_MKD(TA→student)
      + beta_ta_crf   * L_CRF(TA↔student)
      + gamma_feature * L_FeatureKD(TA→student)

Usage:
    python train_student_hier.py \\
        --config configs/student_100hz_ta.yaml \\
        --data_dir /path/to/ptbxl \\
        --teacher_ckpt outputs/teacher_best.pt \\
        --ta_ckpt outputs/ta_ii_500hz_best.pt \\
        --lead II \\
        --losses bce,ta_mkd,ta_crf,feature
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
from losses.class_adaptive_kd import ClassAdaptiveWrapper, load_class_weights
from losses.contrastive import cross_rate_contrastive_loss
from losses.feature_kd import FeatureKDLoss
from losses.gated_hierarchical_kd import (
    ConfidenceGatedHierarchicalKD,
    binary_multilabel_kl_per_class,
)
from losses.mkd import multi_label_kd_loss
from losses.segment_feature_kd import TemporalSegmentFeatureKDLoss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics
from utils.seed import set_seed


_HERE = Path(__file__).parent


def _load_encoder_ckpt(model: ResNet1d, ckpt_path: str, device: torch.device) -> None:
    """Load CLECG pretrained encoder weights (non-strict: missing keys OK)."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    # pretrain_clecg saves {"state_dict": ..., "hz": ..., "lead": ...} in _encoder.pt
    # or {"query_enc": ..., ...} in _best.pt
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


VALID_LOSS_TERMS = frozenset({
    "bce", "teacher_mkd", "ta_mkd", "ta_crf", "feature",
    "gated_kd", "segment_feature",
})


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
    cfg: dict, data_dir: str, lead: str, ta_hz: int = 500,
) -> tuple[DataLoader, DataLoader]:
    dc = cfg["data"]
    tc = cfg["training"]
    hz = dc.get("sampling_hz", dc.get("sampling_rate", 100))
    # ta_hz from config can be overridden by CLI --ta_hz
    cfg_ta_hz = dc.get("ta_hz", 500)
    effective_ta_hz = ta_hz if ta_hz != 500 else cfg_ta_hz
    kw = dict(num_workers=tc.get("num_workers", 4), pin_memory=True)
    train_ds = PTBXLDataset(
        ptbxl_root=data_dir, split="train", mode="hier",
        sampling_rate=hz, lead=lead, normalize=dc.get("normalize", "zscore"),
        ta_hz=effective_ta_hz,
    )
    val_ds = PTBXLDataset(
        ptbxl_root=data_dir, split="val", mode="hier",
        sampling_rate=hz, lead=lead, normalize=dc.get("normalize", "zscore"),
        ta_hz=effective_ta_hz,
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
        x   = batch["student_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics(labels, probs)


def _check_nan(
    name: str,
    val: torch.Tensor,
    s_logits: torch.Tensor | None = None,
) -> None:
    if torch.isnan(val):
        msg = f"NaN in loss component [{name}]"
        if s_logits is not None:
            msg += (f" | student logits min={s_logits.min().item():.4f}"
                    f" max={s_logits.max().item():.4f}")
        print(msg + " — aborting.", file=sys.stderr)
        sys.exit(1)


def train(args: argparse.Namespace) -> None:
    cfg  = load_cfg(args.config)
    tc   = cfg["training"]
    oc   = cfg["output"]
    dc   = cfg["data"]
    lead = args.lead or dc.get("lead", "II")

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.alpha_ta is not None:
        cfg["training"]["alpha_ta"] = args.alpha_ta
    if args.beta_ta_crf is not None:
        cfg["training"]["beta_ta_crf"] = args.beta_ta_crf
    if args.gamma_feature is not None:
        cfg["training"]["gamma_feature"] = args.gamma_feature
    if args.mkd_temperature is not None:
        cfg["training"]["temperature"] = args.mkd_temperature

    # ── loss validation ───────────────────────────────────────────────────────
    active_losses = set(args.losses.split(","))
    unknown = active_losses - VALID_LOSS_TERMS
    if unknown:
        raise ValueError(
            f"Unknown loss terms: {sorted(unknown)}. "
            f"Valid: {sorted(VALID_LOSS_TERMS)}"
        )
    if "bce" not in active_losses:
        raise ValueError("'bce' must always be included in --losses")

    # gated_kd is mutually exclusive with ta_mkd / teacher_mkd
    if "gated_kd" in active_losses and (
        "ta_mkd" in active_losses or "teacher_mkd" in active_losses
    ):
        raise ValueError(
            "gated_kd is mutually exclusive with ta_mkd and teacher_mkd. "
            "Use: bce,gated_kd  or  bce,ta_mkd,... — not both."
        )
    print(f"Active losses: {sorted(active_losses)}")

    # ── seed (CLI overrides config) ───────────────────────────────────────────
    seed = args.seed if args.seed is not None else tc.get("seed", 42)
    cfg["training"]["seed"] = seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {seed}")

    # ── auto checkpoint naming ────────────────────────────────────────────────
    hz          = dc.get("sampling_hz", dc.get("sampling_rate", 100))
    losses_fname = args.losses.replace(",", "_")
    run_name    = args.run_name or f"student_{lead}_{hz}hz_{losses_fname}_seed{seed}"
    out_dir     = Path(args.output_dir) if args.output_dir else Path(oc.get("dir", "outputs"))
    ckpt_path   = out_dir / f"{run_name}_best.pt"
    log_dir   = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    ta_hz = args.ta_hz if args.ta_hz != 500 else 500
    if ta_hz != 500:
        print(f"Progressive TA mode: ta_hz={ta_hz}")
    print("Building data loaders …")
    train_loader, val_loader = build_loaders(cfg, args.data_dir, lead, ta_hz=ta_hz)
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

    # ── frozen 1-lead TA ──────────────────────────────────────────────────────
    ta_cfg_path = args.ta_config or str(_HERE / "configs/ta_ii_500hz.yaml")
    ta_cfg      = load_cfg(ta_cfg_path)
    ta          = build_model(ta_cfg).to(device)
    load_checkpoint(args.ta_ckpt, ta, device)
    ta.eval()
    for p in ta.parameters():
        p.requires_grad_(False)
    print(f"Loaded TA from {args.ta_ckpt}")

    # ── trainable student ─────────────────────────────────────────────────────
    student  = build_model(cfg).to(device)
    if args.init_encoder_ckpt:
        _load_encoder_ckpt(student, args.init_encoder_ckpt, device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_params/1e6:.2f}M")

    # ── feature KD module (may have trainable channel projection) ─────────────
    feat_kd_fn: FeatureKDLoss | None = None
    if "feature" in active_losses:
        feat_kd_fn = FeatureKDLoss(
            student_channels = student.feat_dim,
            ta_channels      = ta.feat_dim,
            pool_size        = tc.get("feature_pool_size", 128),
            normalize        = True,
        ).to(device)

    # ── segment feature KD ────────────────────────────────────────────────────
    seg_feat_kd_fn: TemporalSegmentFeatureKDLoss | None = None
    if "segment_feature" in active_losses:
        seg_feat_kd_fn = TemporalSegmentFeatureKDLoss(
            student_channels  = student.feat_dim,
            ta_channels       = ta.feat_dim,
            num_segments      = tc.get("num_segments", 8),
            pool_per_segment  = tc.get("pool_per_segment", 16),
            normalize         = True,
            attention_weighted= args.attention_weighted_segment
                                or tc.get("attention_weighted_segment", False),
        ).to(device)

    # ── gated hierarchical KD ─────────────────────────────────────────────────
    gated_kd_fn: ConfidenceGatedHierarchicalKD | None = None
    if "gated_kd" in active_losses:
        gate_mode = args.gate_mode or tc.get("gate_mode", "confidence_agreement")
        gated_kd_fn = ConfidenceGatedHierarchicalKD(
            lambda_teacher = tc.get("lambda_teacher", 0.3),
            lambda_ta      = tc.get("lambda_ta", 1.0),
            temperature    = tc.get("temperature", 2.0),
            gate_mode      = gate_mode,
        ).to(device)
        print(f"GatedKD: gate_mode={gate_mode}")

    # ── class-adaptive weights ────────────────────────────────────────────────
    class_adaptive_wrapper: ClassAdaptiveWrapper | None = None
    if args.class_weights_json:
        weights = load_class_weights(args.class_weights_json)
        class_adaptive_wrapper = ClassAdaptiveWrapper(weights).to(device)
        print(f"Class-adaptive weights from {args.class_weights_json}: {weights}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    opt_params = list(student.parameters())
    if feat_kd_fn is not None:
        opt_params += list(feat_kd_fn.parameters())
    if seg_feat_kd_fn is not None:
        opt_params += list(seg_feat_kd_fn.parameters())

    optimizer = torch.optim.AdamW(
        opt_params, lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"]
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler("cuda", enabled=tc.get("amp", False))
    grad_clip = tc.get("grad_clip", 1.0)

    alpha_teacher     = tc.get("alpha_teacher",      0.3)
    alpha_ta          = tc.get("alpha_ta",            1.0)
    beta_ta_crf       = tc.get("beta_ta_crf",         0.1)
    gamma_feature     = tc.get("gamma_feature",       0.2)
    gated_kd_weight   = tc.get("gated_kd_weight",     1.0)
    seg_feat_weight   = tc.get("segment_feature_weight", 0.2)
    mkd_T             = tc.get("temperature",             2.0)
    crf_T             = tc.get("contrastive_temperature", 0.07)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path   = log_dir / f"{run_name}.csv"
    log_fields = ["epoch", "bce", "teacher_mkd", "ta_mkd", "ta_crf",
                  "feature", "gated_kd", "segment_feature",
                  "mean_w_teacher", "mean_w_ta", "mean_agreement",
                  "total", "macro_auc", "macro_f1"]
    csv_fh = open(log_path, "w", newline="")
    csv_wr = csv.DictWriter(csv_fh, fieldnames=log_fields, extrasaction="ignore")
    csv_wr.writeheader()

    best_auc = 0.0

    for epoch in range(1, tc["epochs"] + 1):
        student.train()
        if feat_kd_fn is not None:
            feat_kd_fn.train()
        if seg_feat_kd_fn is not None:
            seg_feat_kd_fn.train()

        totals = {k: 0.0 for k in
                  ["bce", "teacher_mkd", "ta_mkd", "ta_crf", "feature",
                   "gated_kd", "segment_feature",
                   "mean_w_teacher", "mean_w_ta", "mean_agreement", "total"]}
        n_batches = 0

        # Determine which frozen models are actually needed this epoch
        need_teacher = "teacher_mkd" in active_losses or "gated_kd" in active_losses
        need_ta      = bool(active_losses & {"ta_mkd", "ta_crf", "feature",
                                             "gated_kd", "segment_feature"})

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{tc['epochs']}",
                          leave=False, dynamic_ncols=True):
            t_x  = batch["teacher_x"].to(device)   # [B, 12, 5000]
            ta_x = batch["ta_x"].to(device)          # [B, 1,  5000]
            s_x  = batch["student_x"].to(device)    # [B, 1,  L]
            y    = batch["y"].to(device)

            if args.debug and epoch == 1 and n_batches == 0:
                print(f"  [debug] teacher_x:{t_x.shape}  ta_x:{ta_x.shape}  "
                      f"student_x:{s_x.shape}  y:{y.shape}")

            optimizer.zero_grad()

            with autocast("cuda", enabled=tc.get("amp", False)):
                s_out = student(s_x, return_features=True)

                with torch.no_grad():
                    t_out  = teacher(t_x,  return_features=True) if need_teacher else None
                    ta_out = ta(ta_x, return_features=True)      if need_ta      else None

                # BCE
                s_logits = s_out["logits"]

                loss_bce = criterion(s_logits, y)
                _check_nan("bce", loss_bce, s_logits)
                total_loss = loss_bce
                totals["bce"] += loss_bce.item()

                # Teacher → Student MKD
                if "teacher_mkd" in active_losses and t_out is not None:
                    lv = multi_label_kd_loss(
                        s_logits, t_out["logits"], mkd_T
                    )
                    _check_nan("teacher_mkd", lv, s_logits)
                    total_loss = total_loss + alpha_teacher * lv
                    totals["teacher_mkd"] += lv.item()

                # TA → Student MKD
                if "ta_mkd" in active_losses and ta_out is not None:
                    lv = multi_label_kd_loss(
                        s_logits, ta_out["logits"], mkd_T
                    )
                    _check_nan("ta_mkd", lv, s_logits)
                    total_loss = total_loss + alpha_ta * lv
                    totals["ta_mkd"] += lv.item()

                # TA ↔ Student CRF
                if "ta_crf" in active_losses and ta_out is not None:
                    lv = cross_rate_contrastive_loss(
                        s_out["pooled"], ta_out["pooled"],
                        s_out["proj"],   ta_out["proj"],
                        temperature=crf_T,
                    )
                    _check_nan("ta_crf", lv, s_logits)
                    total_loss = total_loss + beta_ta_crf * lv
                    totals["ta_crf"] += lv.item()

                # TA → Student Feature KD
                if "feature" in active_losses and feat_kd_fn is not None \
                        and ta_out is not None:
                    lv = feat_kd_fn(s_out["feature_map"], ta_out["feature_map"])
                    _check_nan("feature", lv, s_logits)
                    total_loss = total_loss + gamma_feature * lv
                    totals["feature"] += lv.item()

                # Gated Hierarchical KD (replaces ta_mkd + teacher_mkd)
                if "gated_kd" in active_losses and gated_kd_fn is not None \
                        and t_out is not None and ta_out is not None:
                    lv, gdiag = gated_kd_fn(
                        s_logits, t_out["logits"], ta_out["logits"]
                    )
                    _check_nan("gated_kd", lv, s_logits)
                    total_loss = total_loss + gated_kd_weight * lv
                    totals["gated_kd"]       += lv.item()
                    totals["mean_w_teacher"] += gdiag["mean_w_teacher"]
                    totals["mean_w_ta"]      += gdiag["mean_w_ta"]
                    totals["mean_agreement"] += gdiag["mean_agreement"]

                # Temporal Segment Feature KD
                if "segment_feature" in active_losses and seg_feat_kd_fn is not None \
                        and ta_out is not None:
                    lv = seg_feat_kd_fn(s_out["feature_map"], ta_out["feature_map"])
                    _check_nan("segment_feature", lv, s_logits)
                    total_loss = total_loss + seg_feat_weight * lv
                    totals["segment_feature"] += lv.item()

                totals["total"] += total_loss.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(opt_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Debug: verify frozen models have no gradients
            if args.debug and epoch == 1 and n_batches == 0:
                for mname, frozen_m in [("teacher", teacher), ("ta", ta)]:
                    has_grad = any(p.grad is not None
                                   for p in frozen_m.parameters())
                    status = "WARNING: has gradients!" if has_grad else "OK (frozen)"
                    print(f"  [debug] {mname} grad check: {status}")

            n_batches += 1

        scheduler.step()

        for k in totals:
            totals[k] /= n_batches

        metrics = evaluate(student, val_loader, device)

        log_line = (
            f"[{epoch:4d}]  total={totals['total']:.4f}  "
            f"bce={totals['bce']:.4f}  "
            f"t_mkd={totals['teacher_mkd']:.4f}  "
            f"ta_mkd={totals['ta_mkd']:.4f}  "
            f"ta_crf={totals['ta_crf']:.4f}  "
            f"feat={totals['feature']:.4f}  "
        )
        if "gated_kd" in active_losses:
            log_line += (
                f"gkd={totals['gated_kd']:.4f}  "
                f"w_t={totals['mean_w_teacher']:.3f}  "
                f"w_ta={totals['mean_w_ta']:.3f}  "
                f"agr={totals['mean_agreement']:.3f}  "
            )
        if "segment_feature" in active_losses:
            log_line += f"seg_feat={totals['segment_feature']:.4f}  "
        log_line += f"AUC={metrics['macro_auc']:.4f}  F1={metrics['macro_f1']:.4f}"
        print(log_line)

        csv_wr.writerow({
            "epoch": epoch,
            **{k: round(v, 5) for k, v in totals.items()},
            "macro_auc": round(metrics["macro_auc"], 5),
            "macro_f1":  round(metrics["macro_f1"], 5),
        })
        csv_fh.flush()

        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            extra = {}
            if feat_kd_fn is not None:
                extra["feat_kd_state"] = feat_kd_fn.state_dict()
            if seg_feat_kd_fn is not None:
                extra["seg_feat_kd_state"] = seg_feat_kd_fn.state_dict()
            save_checkpoint(str(ckpt_path), student, epoch, metrics, cfg, extra)
            print(f"  ✓ best AUC={best_auc:.4f} → {ckpt_path}")

    csv_fh.close()
    print(f"\nDone. Best val AUC: {best_auc:.4f}")
    print(f"Log saved to {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--teacher_ckpt",   required=True)
    p.add_argument("--ta_ckpt",        required=True)
    p.add_argument("--teacher_config", default=None,
                   help="Teacher config YAML (default: configs/teacher_500hz.yaml)")
    p.add_argument("--ta_config",      default=None,
                   help="TA config YAML (default: configs/ta_ii_500hz.yaml)")
    p.add_argument("--lead",           default=None,
                   help="Lead name override (e.g. II)")
    p.add_argument("--losses",         default="bce,ta_mkd,ta_crf,feature",
                   help="Comma-separated: bce,teacher_mkd,ta_mkd,ta_crf,feature")
    p.add_argument("--batch_size",     type=int, default=None,
                   help="Override batch size from config")
    p.add_argument("--seed",           type=int, default=None,
                   help="Random seed (overrides config)")
    p.add_argument("--output_dir",     default=None,
                   help="Override output directory from config (e.g. dafd_mvkt/outputs)")
    p.add_argument("--run_name",       default=None,
                   help="Override auto-generated run name")
    p.add_argument("--gate_mode",      default=None,
                   choices=["ta_only", "confidence", "confidence_agreement", "oracle_fixed"],
                   help="Gated KD mode (overrides config gate_mode)")
    p.add_argument("--ta_hz",          type=int, default=500,
                   help="TA sampling Hz: 500 (normal) or 100 (progressive)")
    p.add_argument("--class_weights_json", default=None,
                   help="Path to class weights JSON for class-adaptive KD")
    p.add_argument("--attention_weighted_segment", action="store_true",
                   help="Use attention-weighted segment feature KD")
    p.add_argument("--init_encoder_ckpt", default=None,
                   help="CLECG pretrained encoder checkpoint for initialization")
    p.add_argument("--lr",             type=float, default=None,
                   help="Learning rate override")
    p.add_argument("--alpha_ta",       type=float, default=None,
                   help="TA MKD weight override")
    p.add_argument("--beta_ta_crf",    type=float, default=None,
                   help="TA CRF weight override")
    p.add_argument("--gamma_feature",  type=float, default=None,
                   help="Feature KD weight override")
    p.add_argument("--mkd_temperature", type=float, default=None,
                   help="MKD temperature override")
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
