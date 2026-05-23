"""
Per-class diagnostic analysis for PTB-XL classification experiments.

Reads prediction CSV files produced by evaluate.py and computes:
  per-class AUC, F1@0.5, F1_tuned, precision, recall, specificity,
  support (positive count), positive_rate.

Outputs a CSV for side-by-side method comparison and prints a summary table.

Usage:
    python dafd_mvkt/experiments/classwise_analysis.py \\
        --predictions_dir dafd_mvkt/outputs \\
        --split test \\
        --hz 100 \\
        --out_dir dafd_mvkt/outputs/analysis
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix,
)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_LOSS_TERMS = ["teacher_mkd", "gated_kd", "segment_feature", "ta_mkd", "ta_crf", "feature", "bce"]


def _parse_losses_fname(s: str) -> str | None:
    terms, remaining = [], s
    while remaining:
        matched = False
        for t in _LOSS_TERMS:
            if remaining == t:
                terms.append(t); remaining = ""; matched = True; break
            if remaining.startswith(t + "_"):
                terms.append(t); remaining = remaining[len(t) + 1:]; matched = True; break
        if not matched:
            return None
    return ",".join(terms)


def _parse_model_name(name: str) -> dict | None:
    m = re.match(r"^student_(\w+)_(\d+)hz_(.+)_seed(\d+)$", name)
    if m:
        lead, hz, losses_part, seed = m.groups()
        variant_suffixes = [
            "_prog_clecgta_clecg", "_prog_clecg", "_clecgta_clecg", "_clecg", "_prog"
        ]
        variant_tag = ""
        for vs in variant_suffixes:
            if losses_part.endswith(vs):
                variant_tag = vs
                losses_part = losses_part[: -len(vs)]
                break
        losses = _parse_losses_fname(losses_part) or losses_part
        return {"lead": lead, "hz": int(hz), "losses": losses, "seed": int(seed),
                "method_name": f"{lead}_{hz}hz_{losses}{variant_tag}"}
    return None


def _compute_class_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict]:
    N = labels.shape[0]
    rows = []
    for ci, c in enumerate(CLASSES):
        y = labels[:, ci]
        p = probs[:, ci]
        thr = thresholds[ci]
        pred = (p >= thr).astype(int)

        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")

        f1_05   = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
        f1_tun  = float(f1_score(y, pred, zero_division=0))
        prec    = float(precision_score(y, pred, zero_division=0))
        rec     = float(recall_score(y, pred, zero_division=0))

        try:
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            spec = float(tn) / (tn + fp + 1e-8)
        except Exception:
            spec = float("nan")

        support = int(y.sum())
        pos_rate = support / (N + 1e-8)

        rows.append({
            "class": c,
            "auc": round(auc, 4),
            "f1_0_5": round(f1_05, 4),
            "f1_tuned": round(f1_tun, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "support": support,
            "positive_rate": round(pos_rate, 4),
        })
    return rows


def _load_predictions(csv_path: Path, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    df = pd.read_csv(csv_path)
    try:
        labels = df[[f"y_{c}" for c in CLASSES]].values.astype(float)
        probs  = df[[f"prob_{c}" for c in CLASSES]].values.astype(float)
        return labels, probs
    except KeyError:
        return None


def _load_thresholds(thr_path: Path) -> np.ndarray:
    if thr_path.exists():
        with open(thr_path) as f:
            d = json.load(f)
        return np.array([d.get(c, 0.5) for c in CLASSES])
    return np.full(len(CLASSES), 0.5)


def analyse(args: argparse.Namespace) -> None:
    pred_dir = Path(args.predictions_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"predictions_{args.split}_student_*_{args.hz}hz_*.csv"
    csv_files = sorted(pred_dir.glob(pattern))

    if not csv_files:
        print(f"No prediction files matching '{pattern}' in {pred_dir}")
        return

    all_rows = []
    for csv_path in csv_files:
        stem = csv_path.stem
        model_name = stem.replace(f"predictions_{args.split}_", "", 1)
        meta = _parse_model_name(model_name)
        if meta is None:
            print(f"  [skip] cannot parse: {csv_path.name}")
            continue

        result = _load_predictions(csv_path, args.split)
        if result is None:
            print(f"  [skip] missing columns: {csv_path.name}")
            continue
        labels, probs = result

        thr_path = pred_dir / f"thresholds_{model_name}.json"
        thresholds = _load_thresholds(thr_path)

        rows = _compute_class_metrics(labels, probs, thresholds)
        for row in rows:
            all_rows.append({
                "method": meta["method_name"],
                "lead": meta["lead"],
                "hz": meta["hz"],
                "losses": meta["losses"],
                "seed": meta["seed"],
                **row,
            })

    if not all_rows:
        print("No results loaded.")
        return

    df = pd.DataFrame(all_rows)
    out_path = out_dir / f"classwise_comparison_{args.hz}hz.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary table
    methods = df["method"].unique()
    print(f"\n{'='*80}")
    print(f"CLASS-WISE ANALYSIS  {args.hz}Hz  split={args.split}")
    print(f"{'='*80}")

    for c in CLASSES:
        cdf = df[df["class"] == c]
        support_str  = str(cdf['support'].iloc[0]) if len(cdf) else '?'
        pos_rate_str = f"{cdf['positive_rate'].iloc[0]:.3f}" if len(cdf) else '?'
        print(f"\n  Class: {c}  (support={support_str}, pos_rate={pos_rate_str})")
        print(f"  {'Method':<45} {'AUC':>7} {'F1@0.5':>7} {'F1tun':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7}")
        print(f"  {'-'*93}")
        for _, row in cdf.iterrows():
            print(f"  {row['method']:<45} {row['auc']:>7.4f} {row['f1_0_5']:>7.4f} "
                  f"{row['f1_tuned']:>7.4f} {row['precision']:>7.4f} "
                  f"{row['recall']:>7.4f} {row['specificity']:>7.4f}")

    print(f"\nSaved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_dir", default="dafd_mvkt/outputs")
    p.add_argument("--split",           default="test", choices=["val", "test"])
    p.add_argument("--hz",              type=int, default=100)
    p.add_argument("--out_dir",         default="dafd_mvkt/outputs/analysis")
    return p.parse_args()


if __name__ == "__main__":
    analyse(parse_args())
