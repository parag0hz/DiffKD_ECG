"""
Method comparison delta tables.

Loads metrics JSON files and produces delta tables comparing each method
against three baselines: BCE-only, TA-MKD, and the best known config.

Usage:
    python dafd_mvkt/experiments/compare_methods.py \\
        --results_dir dafd_mvkt/outputs \\
        --split test \\
        --hz 100 \\
        --out_dir dafd_mvkt/outputs/analysis
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
COMPARE_METRICS = (
    ["macro_auc", "macro_f1_0_5", "macro_f1_tuned"]
    + [f"auc_{c}" for c in CLASSES]
    + [f"f1_tuned_{c}" for c in CLASSES]
)

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
        losses = _parse_losses_fname(losses_part) or losses_part
        return {"lead": lead, "hz": int(hz), "losses": losses, "seed": int(seed),
                "method": f"{lead}_{hz}hz_{losses}"}
    return None


def _load_metrics(json_path: Path, split: str) -> dict | None:
    model_name = json_path.stem.replace(f"metrics_{split}_", "", 1)
    meta = _parse_model_name(model_name)
    if meta is None:
        return None
    with open(json_path) as f:
        d = json.load(f)
    rec = {**meta}
    for m in COMPARE_METRICS:
        rec[m] = d.get(m, float("nan"))
    return rec


def _make_delta_table(
    records: list[dict],
    baseline: dict | None,
    baseline_label: str,
) -> pd.DataFrame:
    if baseline is None:
        return pd.DataFrame()

    rows = []
    for rec in records:
        for metric in COMPARE_METRICS:
            val = rec.get(metric, float("nan"))
            base_val = baseline.get(metric, float("nan"))
            delta = val - base_val if not (np.isnan(val) or np.isnan(base_val)) else float("nan")
            rows.append({
                "method": rec["method"],
                "lead": rec["lead"],
                "hz": rec["hz"],
                "losses": rec["losses"],
                "metric": metric,
                "value": round(val, 4),
                "baseline_value": round(base_val, 4),
                "delta": round(delta, 4),
                "baseline": baseline_label,
            })
    return pd.DataFrame(rows)


def _print_macro_delta(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    macro_metrics = ["macro_auc", "macro_f1_0_5", "macro_f1_tuned"]
    sub = df[df["metric"].isin(macro_metrics)].copy()
    sub = sub.pivot(index="method", columns="metric", values="delta")
    print(f"\n  {title}")
    print(f"  {'Method':<50} {'ΔAUC':>8} {'ΔF1@0.5':>8} {'ΔF1_tun':>8}")
    print(f"  {'-'*76}")
    for method, row in sub.iterrows():
        vals = [row.get(m, float("nan")) for m in macro_metrics]
        fmt = lambda v: f"{v:+.4f}" if not np.isnan(v) else "   N/A"
        print(f"  {method:<50} {fmt(vals[0]):>8} {fmt(vals[1]):>8} {fmt(vals[2]):>8}")


def compare(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"metrics_{args.split}_student_*_{args.hz}hz_*.json"
    json_files = sorted(results_dir.glob(pattern))

    records = []
    for jf in json_files:
        rec = _load_metrics(jf, args.split)
        if rec:
            records.append(rec)

    if not records:
        print(f"No metrics files matching '{pattern}' in {results_dir}")
        return

    print(f"Loaded {len(records)} records for {args.hz}Hz, split={args.split}")

    # Identify baselines — prefer seed=0 if multiple seeds
    def _find_baseline(losses_str: str) -> dict | None:
        matches = [r for r in records if r["losses"] == losses_str]
        if not matches:
            return None
        seed0 = [r for r in matches if r.get("seed", 0) == 0]
        return seed0[0] if seed0 else matches[0]

    bce_baseline  = _find_baseline("bce")
    ta_mkd_base   = _find_baseline("bce,ta_mkd")
    best_baseline = _find_baseline("bce,ta_mkd,ta_crf,feature")

    print(f"\n{'='*70}")
    print(f"DELTA TABLES  {args.hz}Hz  split={args.split}")
    print(f"{'='*70}")

    delta_bce   = _make_delta_table(records, bce_baseline,  "bce")
    delta_ta    = _make_delta_table(records, ta_mkd_base,   "bce,ta_mkd")
    delta_best  = _make_delta_table(records, best_baseline, "bce,ta_mkd,ta_crf,feature")

    _print_macro_delta(delta_bce,  "vs BCE baseline")
    _print_macro_delta(delta_ta,   "vs TA-MKD baseline")
    _print_macro_delta(delta_best, "vs best (TA-MKD+CRF+Feature)")

    hz = args.hz
    if not delta_bce.empty:
        p = out_dir / f"delta_vs_bce_{hz}hz.csv"
        delta_bce.to_csv(p, index=False)
        print(f"\nSaved: {p}")
    if not delta_ta.empty:
        p = out_dir / f"delta_vs_ta_mkd_{hz}hz.csv"
        delta_ta.to_csv(p, index=False)
        print(f"Saved: {p}")
    if not delta_best.empty:
        p = out_dir / f"delta_vs_best_{hz}hz.csv"
        delta_best.to_csv(p, index=False)
        print(f"Saved: {p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="dafd_mvkt/outputs")
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--hz",          type=int, default=100)
    p.add_argument("--out_dir",     default="dafd_mvkt/outputs/analysis")
    return p.parse_args()


if __name__ == "__main__":
    compare(parse_args())
