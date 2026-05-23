"""
Compute per-class KD weights from validation statistics.

Modes:
  inverse_support      1 / sqrt(num_positive_c + 1)
  inverse_baseline_auc 1 - baseline_val_auc_c
  ta_student_gap       max(ta_val_auc_c - baseline_val_auc_c, 0)
  manual               read from --manual_weights

Weights are normalized so mean == 1.

Usage:
    python dafd_mvkt/experiments/compute_class_weights.py \\
        --mode inverse_baseline_auc \\
        --baseline_metrics dafd_mvkt/outputs/metrics_val_student_II_100hz_bce_seed0.json \\
        --out dafd_mvkt/outputs/class_weights/II_100hz_inverse_baseline_auc.json \\
        --lead II --hz 100

    python dafd_mvkt/experiments/compute_class_weights.py \\
        --mode inverse_support \\
        --data_dir comper_repo/ptb_xl \\
        --out dafd_mvkt/outputs/class_weights/II_100hz_inverse_support.json

    python dafd_mvkt/experiments/compute_class_weights.py \\
        --mode ta_student_gap \\
        --baseline_metrics dafd_mvkt/outputs/metrics_val_student_II_100hz_bce_seed0.json \\
        --ta_metrics dafd_mvkt/outputs/metrics_val_ta_ii_500hz.json \\
        --out dafd_mvkt/outputs/class_weights/II_100hz_ta_gap.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _normalize(weights: list[float]) -> list[float]:
    mean = sum(weights) / len(weights)
    return [w / (mean + 1e-8) for w in weights]


def _from_inverse_support(data_dir: str, split: str = "val") -> list[float]:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import os
    orig = os.getcwd()

    from dafd_mvkt.data.ptbxl_dataset import PTBXLDataset
    ds = PTBXLDataset(ptbxl_root=data_dir, split=split, mode="teacher")

    import torch
    labels = []
    for i in range(len(ds)):
        labels.append(ds[i]["y"].numpy())
    labels = np.stack(labels)   # [N, C]
    counts = labels.sum(axis=0)   # [C]
    weights = [1.0 / math.sqrt(c + 1.0) for c in counts]
    return weights


def _from_inverse_baseline_auc(baseline_metrics_path: str) -> list[float]:
    with open(baseline_metrics_path) as f:
        d = json.load(f)
    weights = []
    for c in CLASSES:
        auc = d.get(f"auc_{c}", float("nan"))
        if math.isnan(auc):
            weights.append(1.0)
        else:
            weights.append(max(1.0 - auc, 0.0))
    return weights


def _from_ta_student_gap(baseline_metrics_path: str, ta_metrics_path: str) -> list[float]:
    with open(baseline_metrics_path) as f:
        base = json.load(f)
    with open(ta_metrics_path) as f:
        ta = json.load(f)

    weights = []
    for c in CLASSES:
        base_auc = base.get(f"auc_{c}", float("nan"))
        ta_auc   = ta.get(f"auc_{c}", float("nan"))
        if math.isnan(base_auc) or math.isnan(ta_auc):
            weights.append(1.0)
        else:
            weights.append(max(ta_auc - base_auc, 0.0))
    return weights


def _from_manual(manual_str: str) -> list[float]:
    parts = manual_str.split(",")
    d = {}
    for part in parts:
        k, v = part.strip().split(":")
        d[k.strip()] = float(v.strip())
    return [d.get(c, 1.0) for c in CLASSES]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["inverse_support", "inverse_baseline_auc",
                            "ta_student_gap", "manual"])
    p.add_argument("--baseline_metrics", default=None,
                   help="Path to metrics_val_*.json for baseline model")
    p.add_argument("--ta_metrics", default=None,
                   help="Path to metrics_val_*.json for TA model (ta_student_gap)")
    p.add_argument("--data_dir", default=None,
                   help="PTB-XL root (inverse_support mode)")
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="Dataset split for inverse_support (default: val)")
    p.add_argument("--manual_weights", default=None,
                   help="e.g. 'NORM:1.0,MI:2.0,STTC:1.0,CD:1.5,HYP:2.0'")
    p.add_argument("--out", required=True,
                   help="Output JSON path")
    p.add_argument("--lead", default="II")
    p.add_argument("--hz", type=int, default=100)
    args = p.parse_args()

    if args.mode == "inverse_support":
        if not args.data_dir:
            raise ValueError("--data_dir required for inverse_support mode")
        raw = _from_inverse_support(args.data_dir, args.split)

    elif args.mode == "inverse_baseline_auc":
        if not args.baseline_metrics:
            raise ValueError("--baseline_metrics required for inverse_baseline_auc mode")
        raw = _from_inverse_baseline_auc(args.baseline_metrics)

    elif args.mode == "ta_student_gap":
        if not args.baseline_metrics or not args.ta_metrics:
            raise ValueError("--baseline_metrics and --ta_metrics required for ta_student_gap")
        raw = _from_ta_student_gap(args.baseline_metrics, args.ta_metrics)

    else:  # manual
        if not args.manual_weights:
            raise ValueError("--manual_weights required for manual mode")
        raw = _from_manual(args.manual_weights)

    normalized = _normalize(raw)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {c: round(w, 4) for c, w in zip(CLASSES, normalized)}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Class weights ({args.mode}, lead={args.lead}, hz={args.hz}):")
    for c, w_raw, w_norm in zip(CLASSES, raw, normalized):
        print(f"  {c}: raw={w_raw:.4f}  normalized={w_norm:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
