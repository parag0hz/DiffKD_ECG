"""
Evaluate a trained model on PTB-XL test (or val) set.

Supports teacher (12-lead 500Hz), TA (1-lead 500Hz), and student (1-lead low-Hz).
Always reports metrics at both threshold=0.5 and at per-class tuned thresholds.

Threshold sources (priority order):
  1. --thresholds_json  — load pre-computed thresholds from JSON
  2. --tune_thresholds  — tune on validation set, save thresholds JSON
  3. (default)          — threshold = 0.5 for all classes

Usage:
    # Teacher
    python evaluate.py --config configs/teacher_500hz.yaml \\
        --data_dir /path --ckpt outputs/teacher_best.pt \\
        --split test --model_name teacher

    # TA
    python evaluate.py --config configs/ta_ii_500hz.yaml \\
        --data_dir /path --ckpt outputs/ta_ii_500hz_best.pt \\
        --split test --model_name ta_ii_500hz --tune_thresholds

    # Hierarchical student
    python evaluate.py --config configs/student_100hz_ta.yaml \\
        --data_dir /path --ckpt outputs/student_II_100hz_bce_ta_mkd_seed0_best.pt \\
        --split test --model_name student_II_100hz_bce_ta_mkd_seed0 \\
        --tune_thresholds
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.ptbxl_dataset import PTBXLDataset, SUPERCLASSES
from models.resnet1d import ResNet1d
from models.resnet1d_wang import ResNet1dWang
from utils.checkpoint import load_checkpoint
from utils.metrics import CLASSES, compute_metrics, find_best_thresholds
from utils.seed import set_seed


def load_cfg(path: str) -> dict:
    with open(path) as f:
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
    return ResNet1d(
        in_channels  = mc["in_channels"],
        num_classes  = mc["num_classes"],
        layers       = mc.get("layers", [3, 4, 6, 3]),
        base_channels= mc.get("base_channels", 64),
        proj_dim     = mc.get("proj_dim", 128),
        dropout      = mc.get("dropout", 0.0),
    )


def _resolve_eval_params(
    cfg: dict, args: argparse.Namespace
) -> tuple[str, str, int, str]:
    """Returns (ds_mode, x_key, sampling_hz, lead)."""
    dc = cfg["data"]

    # Explicit eval_input_key in config takes highest priority
    explicit_key = dc.get("eval_input_key")
    if explicit_key:
        key_map = {"teacher_x": ("teacher", "teacher_x"),
                   "ta_x":      ("ta",      "ta_x"),
                   "student_x": ("student", "student_x")}
        if explicit_key in key_map:
            ds_mode, x_key = key_map[explicit_key]
            if args.hz:
                hz = args.hz
            elif ds_mode == "teacher":
                hz = 500
            elif ds_mode == "ta":
                hz = 500
            else:
                hz = int(dc.get("sampling_hz", dc.get("sampling_rate", 100)))
            lead = args.lead or dc.get("lead", dc.get("student_lead", "II"))
            return ds_mode, x_key, hz, lead

    cfg_mode = dc.get("mode", "student")

    # "progressive" config trains a low-Hz TA — evaluate it like a student
    if cfg_mode == "progressive" and not args.model_type:
        model_type = "student"
    elif args.model_type:
        model_type = args.model_type
    elif cfg_mode == "teacher":
        model_type = "teacher"
    elif cfg_mode == "ta":
        model_type = "ta"
    else:
        model_type = "student"

    if model_type == "teacher":
        ds_mode, x_key = "teacher", "teacher_x"
    elif model_type == "ta":
        # Sanity check: --model_type ta should only be used for 500Hz TAs.
        # A progressive (low-Hz) TA must be evaluated as student.
        cfg_hz = int(dc.get("sampling_hz", dc.get("sampling_rate", 500)))
        if cfg_mode == "progressive" or (args.hz and args.hz != 500) or cfg_hz != 500:
            import sys
            print(
                "WARNING: --model_type ta with a non-500Hz or progressive config. "
                "Switching to student mode (using student_x input at configured Hz). "
                "Use --model_type student explicitly to suppress this warning.",
                file=sys.stderr,
            )
            ds_mode, x_key = "student", "student_x"
        else:
            ds_mode, x_key = "ta", "ta_x"
    else:
        ds_mode, x_key = "student", "student_x"

    if args.hz:
        hz = args.hz
    elif ds_mode in ("teacher", "ta"):
        hz = 500
    else:
        hz = int(dc.get("sampling_hz", dc.get("sampling_rate", 100)))

    lead = args.lead or dc.get("lead", dc.get("student_lead", "II"))
    return ds_mode, x_key, hz, lead


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    x_key: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in tqdm(loader, desc="Inference", dynamic_ncols=True):
        x   = batch[x_key].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels, all_ids


def _build_metrics_dict(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    thresholds_src: str,
) -> dict:
    """
    Always compute metrics at both threshold=0.5 and at tuned thresholds.
    Returns merged dict with *_0_5 and *_tuned variants.
    """
    m05    = compute_metrics(labels, probs, threshold=0.5)
    m_tune = compute_metrics(labels, probs, threshold=thresholds)

    out: dict = {}

    # AUC is threshold-independent
    out["macro_auc"] = m05["macro_auc"]
    for c in CLASSES:
        out[f"auc_{c}"] = m05[f"auc_{c}"]

    # Macro scalars — both variants
    for key in ("macro_f1", "macro_precision", "macro_recall", "macro_specificity",
                "macro_sensitivity"):
        out[f"{key}_0_5"]   = m05.get(key, float("nan"))
        out[f"{key}_tuned"] = m_tune.get(key, float("nan"))

    # Backward-compat aliases (tuned values)
    out["macro_f1"]           = m_tune["macro_f1"]
    out["macro_precision"]    = m_tune["macro_precision"]
    out["macro_recall"]       = m_tune["macro_recall"]
    out["macro_specificity"]  = m_tune["macro_specificity"]

    # Per-class F1/prec/rec/spec — both variants
    for c in CLASSES:
        for metric in ("f1", "prec", "rec", "spec"):
            out[f"{metric}_0_5_{c}"]   = m05.get(f"{metric}_{c}", float("nan"))
            out[f"{metric}_tuned_{c}"] = m_tune.get(f"{metric}_{c}", float("nan"))
            # Backward-compat
            out[f"{metric}_{c}"]       = m_tune.get(f"{metric}_{c}", float("nan"))

    # Threshold record
    out["per_class_thresholds"] = {
        CLASSES[i]: round(float(thresholds[i]), 4) for i in range(len(CLASSES))
    }
    out["thresholds_src"] = thresholds_src

    return out


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    dc  = cfg["data"]
    tc  = cfg["training"]

    set_seed(tc.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_mode, x_key, hz, lead = _resolve_eval_params(cfg, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _make_ds(split: str) -> PTBXLDataset:
        return PTBXLDataset(
            ptbxl_root   = args.data_dir,
            split        = split,
            mode         = ds_mode,
            sampling_rate= hz,
            lead         = lead,
            normalize    = dc.get("normalize", "zscore"),
        )

    def _make_loader(split: str) -> DataLoader:
        return DataLoader(
            _make_ds(split), batch_size=256, shuffle=False,
            num_workers=tc.get("num_workers", 4), pin_memory=True,
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = load_checkpoint(args.ckpt, model, device)
    print(f"Loaded  : {args.ckpt}  (epoch={ckpt.get('epoch','?')})")
    print(f"Mode    : {ds_mode}  key: {x_key}  Hz: {hz}  lead: {lead}")

    # ── determine thresholds ──────────────────────────────────────────────────
    thresholds    = np.full(len(CLASSES), 0.5)
    thresholds_src = "fixed_0.5"

    if args.thresholds_json:
        with open(args.thresholds_json) as f:
            thr_data = json.load(f)
        thresholds = np.array([thr_data.get(c, 0.5) for c in CLASSES])
        thresholds_src = f"loaded:{args.thresholds_json}"
        print(f"Thresholds loaded from {args.thresholds_json}")

    elif args.tune_thresholds:
        print("Tuning thresholds on validation set …")
        grid = np.arange(
            args.threshold_grid_start,
            args.threshold_grid_end + 1e-9,
            args.threshold_grid_step,
        )
        val_probs, val_labels, _ = run_inference(
            model, _make_loader("val"), x_key, device
        )
        thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)
        thresholds_src = "tuned_on_val"

        thr_path = out_dir / f"thresholds_{args.model_name}.json"
        thr_dict = {CLASSES[i]: round(float(thresholds[i]), 4)
                    for i in range(len(CLASSES))}
        with open(thr_path, "w") as f:
            json.dump(thr_dict, f, indent=2)
        print(f"  Thresholds → {thr_path}")
        print("  " + "  ".join(f"{c}={thresholds[i]:.2f}"
                                for i, c in enumerate(CLASSES)))

    # ── inference ─────────────────────────────────────────────────────────────
    probs, labels, record_ids = run_inference(
        model, _make_loader(args.split), x_key, device
    )
    metrics = _build_metrics_dict(labels, probs, thresholds, thresholds_src)

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Split: {args.split}   Model: {args.model_name}")
    print(f"{'='*62}")
    print(f"Macro AUC        : {metrics['macro_auc']:.4f}")
    print(f"Macro F1  @0.5   : {metrics['macro_f1_0_5']:.4f}")
    print(f"Macro F1  tuned  : {metrics['macro_f1_tuned']:.4f}")
    print(f"Macro Prec@0.5   : {metrics['macro_precision_0_5']:.4f}")
    print(f"Macro Rec @0.5   : {metrics['macro_recall_0_5']:.4f}")
    print(f"Macro Spec@0.5   : {metrics['macro_specificity_0_5']:.4f}")
    print()
    hdr = f"{'Class':<6}  {'AUC':>7}  {'F1@0.5':>7}  {'F1_tun':>7}  {'Prec':>7}  {'Rec':>7}"
    print(hdr)
    for c in CLASSES:
        print(f"{c:<6}  {metrics[f'auc_{c}']:>7.4f}  "
              f"{metrics[f'f1_0_5_{c}']:>7.4f}  "
              f"{metrics[f'f1_tuned_{c}']:>7.4f}  "
              f"{metrics[f'prec_0_5_{c}']:>7.4f}  "
              f"{metrics[f'rec_0_5_{c}']:>7.4f}")

    # ── save predictions CSV ──────────────────────────────────────────────────
    tag   = f"{args.split}_{args.model_name}"
    preds = (probs >= thresholds[None]).astype(int)
    rows  = []
    for i, rid in enumerate(record_ids):
        row = {"record_id": rid}
        for ci, c in enumerate(CLASSES):
            row[f"y_{c}"]    = int(labels[i, ci])
            row[f"prob_{c}"] = round(float(probs[i, ci]), 5)
            row[f"pred_{c}"] = int(preds[i, ci])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_{tag}.csv", index=False)

    # ── save metrics JSON (with metadata) ─────────────────────────────────────
    metrics["_meta"] = {
        "model_name": args.model_name,
        "split": args.split,
        "lead": lead,
        "hz": hz,
        "ckpt": str(args.ckpt),
    }
    out_path = out_dir / f"metrics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({k: (round(v, 5) if isinstance(v, float) else v)
                   for k, v in metrics.items()}, f, indent=2)

    print(f"\nSaved: {out_dir}/predictions_{tag}.csv")
    print(f"       {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--model_name",  default="model")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--model_type",  default=None,
                   choices=["teacher", "ta", "student"],
                   help="Override model type (overrides config data.mode)")
    p.add_argument("--lead",        default=None, help="Override lead name")
    p.add_argument("--hz",          type=int, default=None,
                   help="Override sampling Hz")
    # Threshold options
    p.add_argument("--tune_thresholds", "--tune_threshold",
                   dest="tune_thresholds", action="store_true",
                   help="Tune per-class F1 thresholds on val set, then apply to split")
    p.add_argument("--thresholds_json", default=None,
                   help="Load pre-computed thresholds from JSON (skips tuning)")
    p.add_argument("--threshold_grid_start", type=float, default=0.05)
    p.add_argument("--threshold_grid_end",   type=float, default=0.95)
    p.add_argument("--threshold_grid_step",  type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
