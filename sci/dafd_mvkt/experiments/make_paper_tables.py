"""
Generate paper-ready tables from experiment results.

Tables produced:
  Table 1: MVKT comparison — Lead I 100Hz, our methods vs MVKT-ECG (main result)
  Table 2: Low-Hz trade-off — 100Hz vs 50Hz for Lead II
  Table 3: Ablation — loss component ablation (Lead II 100Hz)
  Table 4: Class-wise breakdown — best method vs MVKT-ECG

Usage:
    python dafd_mvkt/experiments/make_paper_tables.py \\
        --results_dirs dafd_mvkt/outputs \\
        --out_dir dafd_mvkt/outputs/paper_tables \\
        --split test
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dafd_mvkt.experiments.aggregate_results import load_metrics_file, aggregate

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

MVKT_ECG = {
    "macro_auc":      0.843,
    "macro_f1_tuned": 0.626,
    "auc_NORM": float("nan"),
    "auc_MI":   float("nan"),
    "auc_STTC": float("nan"),
    "auc_CD":   float("nan"),
    "auc_HYP":  float("nan"),
}


def _fmt(mean: float, std: float | None = None, n: int = 1) -> str:
    if math.isnan(mean):
        return "—"
    if n > 1 and std is not None and not math.isnan(std):
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def _row(df: pd.DataFrame, method_name: str) -> pd.Series | None:
    rows = df[df["method_name"] == method_name]
    return rows.iloc[0] if len(rows) else None


def _best_row(df: pd.DataFrame, lead: str, hz: int,
              model_type_prefix: str = "student") -> pd.Series | None:
    """Return student row with highest mean_macro_auc matching lead and hz."""
    mask = (df["lead"].astype(str) == str(lead)) & (df["hz"].astype(int) == int(hz))
    # Filter to student models only (method_name starts with {lead}_{hz}hz)
    mask &= df["method_name"].str.startswith(f"{lead}_{hz}hz", na=False)
    if model_type_prefix:
        pass  # already filtered to student by name prefix
    sub = df[mask].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=["mean_macro_auc"])
    if sub.empty:
        return None
    return sub.loc[sub["mean_macro_auc"].idxmax()]


# ── Table 1: MVKT comparison (Lead I 100Hz) ───────────────────────────────────
def _build_table1(df: pd.DataFrame, cw: pd.DataFrame) -> str:
    lines = [
        "## Table 1: Lead I 100Hz — Comparison with MVKT-ECG",
        "",
        "| Method | AUC | F1_tuned | ΔAUC | ΔF1 | n |",
        "|---|---:|---:|---:|---:|---:|",
        f"| **MVKT-ECG (reported)** | **{MVKT_ECG['macro_auc']:.4f}** "
        f"| **{MVKT_ECG['macro_f1_tuned']:.4f}** | — | — | 1 |",
    ]

    # method_names in aggregate use comma-separated losses
    methods = [
        ("I_100hz_bce",                        "BCE only"),
        ("I_100hz_bce,ta_mkd",                 "BCE + MKD"),
        ("I_100hz_bce,ta_mkd,ta_crf,feature",  "HST-KD"),
        ("I_100hz_bce,ta_mkd,ta_crf,feature_clecg",         "HST-KD + CLECG"),
        ("I_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg", "HST-KD + CLECG-TA + CLECG"),
    ]
    for mname, label in methods:
        row = _row(df, mname)
        if row is None:
            continue
        n    = int(row.get("n_seeds", 1))
        auc  = row["mean_macro_auc"]
        f1   = row["mean_macro_f1_tuned"]
        dauc = auc  - MVKT_ECG["macro_auc"]
        df1  = f1   - MVKT_ECG["macro_f1_tuned"]
        bold = "**" if (auc >= MVKT_ECG["macro_auc"] and f1 >= MVKT_ECG["macro_f1_tuned"]) else ""
        lines.append(
            f"| {bold}{label}{bold} "
            f"| {bold}{_fmt(auc, row['std_macro_auc'], n)}{bold} "
            f"| {bold}{_fmt(f1, row['std_macro_f1_tuned'], n)}{bold} "
            f"| {dauc:+.4f} | {df1:+.4f} | {n} |"
        )
    return "\n".join(lines)


# ── Table 2: Low-Hz trade-off (Lead II 100Hz vs 50Hz) ────────────────────────
def _build_table2(df: pd.DataFrame) -> str:
    lines = [
        "## Table 2: Sampling Rate Trade-off (Lead II)",
        "",
        "| Model | Hz | AUC | F1_tuned | ΔAUC vs 100Hz | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    ref_auc = float("nan")
    entries = [
        ("II_100hz_bce",                        "BCE 100Hz",            100),
        ("II_100hz_bce,ta_mkd,ta_crf,feature",  "HST-KD 100Hz",         100),
        ("II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg",
                                                "HST-KD+CLECG 100Hz",   100),
        ("II_50hz_bce",                         "BCE 50Hz",              50),
        ("II_50hz_bce,ta_mkd,ta_crf,feature",   "HST-KD 50Hz",           50),
        ("II_50hz_bce,ta_mkd,ta_crf,feature_clecg",
                                                "HST-KD+CLECG 50Hz",     50),
    ]
    best_100hz_auc = float("nan")
    for mname, label, hz in entries:
        row = _row(df, mname)
        if row is None:
            continue
        n   = int(row.get("n_seeds", 1))
        auc = row["mean_macro_auc"]
        f1  = row["mean_macro_f1_tuned"]
        if hz == 100 and not math.isnan(auc):
            if math.isnan(best_100hz_auc) or auc > best_100hz_auc:
                best_100hz_auc = auc
        delta = f"{auc - best_100hz_auc:+.4f}" if hz == 50 and not math.isnan(best_100hz_auc) else "—"
        lines.append(
            f"| {label} | {hz} "
            f"| {_fmt(auc, row['std_macro_auc'], n)} "
            f"| {_fmt(f1, row['std_macro_f1_tuned'], n)} "
            f"| {delta} | {n} |"
        )
    return "\n".join(lines)


# ── Table 3: Ablation (Lead II 100Hz) ─────────────────────────────────────────
def _build_table3(df: pd.DataFrame) -> str:
    lines = [
        "## Table 3: Ablation — Loss Components (Lead II 100Hz)",
        "",
        "| Losses / Method | AUC | F1@0.5 | F1_tuned | ΔAUC | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    methods = [
        ("II_100hz_bce",                                              "BCE only"),
        ("II_100hz_bce,ta_mkd",                                       "+ ta_mkd"),
        ("II_100hz_bce,ta_mkd,ta_crf,feature",                        "+ ta_crf + feature (HST-KD)"),
        ("II_100hz_bce,ta_mkd,ta_crf,feature_clecg",                  "HST-KD + CLECG student"),
        ("II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg",          "HST-KD + CLECG-TA + CLECG (full)"),
    ]
    base_auc = float("nan")
    for mname, label in methods:
        row = _row(df, mname)
        if row is None:
            continue
        n   = int(row.get("n_seeds", 1))
        auc = row["mean_macro_auc"]
        f1  = row["mean_macro_f1_tuned"]
        f105 = row.get("mean_macro_f1_0_5", float("nan"))
        if math.isnan(base_auc):
            base_auc = auc
        delta = f"{auc - base_auc:+.4f}" if not math.isnan(base_auc) else "—"
        lines.append(
            f"| {label} "
            f"| {_fmt(auc, row['std_macro_auc'], n)} "
            f"| {_fmt(f105, row.get('std_macro_f1_0_5', float('nan')), n)} "
            f"| {_fmt(f1, row['std_macro_f1_tuned'], n)} "
            f"| {delta} | {n} |"
        )
    return "\n".join(lines)


# ── Table 4: Class-wise breakdown ─────────────────────────────────────────────
def _build_table4(df: pd.DataFrame, cw: pd.DataFrame) -> str:
    lines = [
        "## Table 4: Class-wise AUC — Best Method vs MVKT-ECG",
        "",
        "| Method | NORM | MI | STTC | CD | HYP | Macro |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| MVKT-ECG | — | — | — | — | — "
        f"| {MVKT_ECG['macro_auc']:.4f} |",
    ]

    # Best Lead I 100Hz (highest macro AUC)
    best_I = _best_row(df, "I", 100)
    # Best Lead II 100Hz
    best_II = _best_row(df, "II", 100)

    for row, tag in [(best_I, "Best Lead I"), (best_II, "Best Lead II")]:
        if row is None:
            continue
        n    = int(row.get("n_seeds", 1))
        meth = row["method_name"]
        auc  = row["mean_macro_auc"]

        # Per-class AUC from classwise dataframe
        cw_row = cw[cw["method_name"] == meth] if cw is not None and not cw.empty else None
        class_aucs = []
        for c in CLASSES:
            if cw_row is not None and not cw_row.empty:
                c_row = cw_row[cw_row["class"] == c]
                val = c_row["auc_mean"].iloc[0] if not c_row.empty else float("nan")
            else:
                val = row.get(f"mean_auc_{c}", float("nan"))
            class_aucs.append(_fmt(val))

        lines.append(
            f"| **{tag}** ({meth}) | "
            + " | ".join(class_aucs)
            + f" | **{_fmt(auc, row['std_macro_auc'], n)}** |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+",
                   default=["dafd_mvkt/outputs", "dafd_mvkt/outputs/tuning"])
    p.add_argument("--out_dir", default="dafd_mvkt/outputs/paper_tables")
    p.add_argument("--split",   default="test")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for d in args.results_dirs:
        rd = Path(d)
        if not rd.exists():
            continue
        for f in sorted(rd.glob(f"metrics_{args.split}_*.json")):
            rec = load_metrics_file(f, args.split)
            if rec:
                records.append(rec)

    if not records:
        print(f"No results found in {args.results_dirs}. Run experiments first.")
        return

    print(f"Loaded {len(records)} result files.")
    summary, classwise = aggregate(records)

    tables = [
        _build_table1(summary, classwise),
        _build_table2(summary),
        _build_table3(summary),
        _build_table4(summary, classwise),
    ]

    full_md = "\n\n".join(tables)
    out_path = out_dir / "paper_tables.md"
    out_path.write_text(full_md)
    print(f"Saved: {out_path}")
    print()
    print(full_md)


if __name__ == "__main__":
    main()
