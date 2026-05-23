"""
Aggregate multi-seed experiment results from metrics JSON files.

Scans --results_dir for metrics_{split}_{model_name}.json files,
parses model metadata from the filename, groups by method, and
computes mean ± std across seeds.

Outputs:
  {out_dir}/summary_results.csv
  {out_dir}/summary_results.md
  {out_dir}/classwise_summary.csv

Usage:
    python dafd_mvkt/experiments/aggregate_results.py \\
        --results_dir dafd_mvkt/outputs \\
        --out_dir dafd_mvkt/outputs \\
        --split test
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Longer terms must come first for greedy matching
_LOSS_TERMS = [
    "teacher_mkd", "gated_kd", "segment_feature",
    "ta_mkd", "ta_crf", "feature", "bce",
]


def _parse_losses_fname(s: str) -> str | None:
    """
    Greedily parse an underscore-joined losses string back to comma form.
    'bce_ta_mkd_ta_crf_feature' -> 'bce,ta_mkd,ta_crf,feature'
    Returns None if string cannot be fully parsed.
    """
    terms, remaining = [], s
    while remaining:
        matched = False
        for t in _LOSS_TERMS:
            if remaining == t:
                terms.append(t)
                remaining = ""
                matched = True
                break
            if remaining.startswith(t + "_"):
                terms.append(t)
                remaining = remaining[len(t) + 1:]
                matched = True
                break
        if not matched:
            return None
    return ",".join(terms)


def parse_model_name(name: str) -> dict | None:
    """
    Parse a model_name string into metadata dict.

    Supported formats:
      student_{LEAD}_{HZ}hz_{LOSSES}_seed{SEED}
      ta_{LEAD}_{HZ}hz[_seed{SEED}]
      teacher[_12lead_500hz]
    """
    # skip-fork: student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_simclr_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_skipfork_(SF\d+)_(\w+)_to_(\w+)_(\w+)_(.+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, sf_id, t1, t2, t3, variant, seed = m.groups()
        method_name = f"skipfork_{sf_id}_{t1}_to_{t2}_{t3}_{variant}"
        return {
            "model_type":  "student_skipfork",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      variant,
            "seed":        int(seed),
            "method_name": method_name,
        }

    # fork-join: student_II_50hz_forkjoin_F02_12L100_to_1L500_1L100_simclr_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_forkjoin_(F\d+)_(\w+)_to_(\w+)_(\w+)_(.+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, fk_id, t1, t2, t3, variant, seed = m.groups()
        method_name = f"forkjoin_{fk_id}_{t1}_to_{t2}_{t3}_{variant}"
        return {
            "model_type":  "student_forkjoin",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      variant,
            "seed":        int(seed),
            "method_name": method_name,
        }

    # hierarchical chain: student_II_50hz_hierchain_H02_...
    m = re.match(
        r"^student_(\w+)_(\d+)hz_hierchain_(H\d+)_(.+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, h_id, spec, seed = m.groups()
        method_name = f"hierchain_{h_id}_{spec}"
        return {
            "model_type":  "student_hierchain",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      "bce,mkd_chain",
            "seed":        int(seed),
            "method_name": method_name,
        }

    # 3-teacher EMA: student_II_50hz_3T_C14_ema_lam005_decay099_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_3T_(\w+)_ema_lam(\w+)_decay(\w+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, combo, lam, decay, seed = m.groups()
        method_name = f"3T_ema_{combo}_lam{lam}_decay{decay}"
        return {
            "model_type":  "student_3T_ema",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      f"bce,mkd_{combo},ema",
            "seed":        int(seed),
            "method_name": method_name,
        }

    # progressive EMA: student_II_50hz_prog_1l100_ema_lam005_decay099_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_prog_1l100_ema_lam(\w+)_decay(\w+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, lam, decay, seed = m.groups()
        method_name = f"prog_ema_lam{lam}_decay{decay}"
        return {
            "model_type":  "student_prog_ema",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      "bce,mkd_1l100,ema",
            "seed":        int(seed),
            "method_name": method_name,
        }

    # mutual learning student: student_II_50hz_mut_12l100_lam005_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_mut_(\w+)_lam(\w+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, upstream, lam, seed = m.groups()
        method_name = f"mut_{upstream}_lam{lam}"
        return {
            "model_type":  "student_mutual",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      f"bce,mkd_mutual_{upstream}",
            "seed":        int(seed),
            "method_name": method_name,
        }

    # student-aware skip-fork: student_II_50hz_saf_sf02_12l100_to_1l500_1l100_sa005_seed0
    m = re.match(
        r"^student_(\w+)_(\d+)hz_saf_(\w+)_(\w+)_to_(\w+)_(\w+)_sa(\w+)_seed(\d+)$",
        name,
    )
    if m:
        lead, hz, sf_id, t1, t2, t3, lam_sa, seed = m.groups()
        method_name = f"saf_{sf_id}_{t1}_to_{t2}_{t3}_sa{lam_sa}"
        return {
            "model_type":  "student_saf",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      f"bce,mkd_saf_{sf_id}",
            "seed":        int(seed),
            "method_name": method_name,
        }

    # F02 weight tuning: student_II_50hz_f02_F02Wxx_seed0
    m = re.match(r"^student_(\w+)_(\d+)hz_f02_(F02W\d+)_seed(\d+)$", name)
    if m:
        lead, hz, vid, seed = m.groups()
        return {"model_type": "f02_opt_weight", "lead": lead, "hz": int(hz),
                "losses": "bce,mkd_f02w", "seed": int(seed), "method_name": f"f02_{vid}"}

    # F02 CRF/FeatureKD: student_II_50hz_f02_F02Fxx_seed0
    m = re.match(r"^student_(\w+)_(\d+)hz_f02_(F02F\d+)_seed(\d+)$", name)
    if m:
        lead, hz, vid, seed = m.groups()
        return {"model_type": "f02_opt_crf_feat", "lead": lead, "hz": int(hz),
                "losses": "bce,mkd,crf_feat", "seed": int(seed), "method_name": f"f02_{vid}"}

    # F02 temperature: student_II_50hz_f02_F02Txx_seed0
    m = re.match(r"^student_(\w+)_(\d+)hz_f02_(F02T\d+)_seed(\d+)$", name)
    if m:
        lead, hz, vid, seed = m.groups()
        return {"model_type": "f02_opt_temp", "lead": lead, "hz": int(hz),
                "losses": "bce,mkd_f02t", "seed": int(seed), "method_name": f"f02_{vid}"}

    # F02 branch enhance: student_II_50hz_f02_F02BRxx_seed0
    m = re.match(r"^student_(\w+)_(\d+)hz_f02_(F02BR\d+)_seed(\d+)$", name)
    if m:
        lead, hz, vid, seed = m.groups()
        return {"model_type": "f02_opt_branch", "lead": lead, "hz": int(hz),
                "losses": "bce,mkd,branch_crf", "seed": int(seed), "method_name": f"f02_{vid}"}

    # F02 confidence-weighted: student_II_50hz_f02_F02CWxx_seed0
    m = re.match(r"^student_(\w+)_(\d+)hz_f02_(F02CW\d+)_seed(\d+)$", name)
    if m:
        lead, hz, vid, seed = m.groups()
        return {"model_type": "f02_opt_cw", "lead": lead, "hz": int(hz),
                "losses": "bce,mkd_f02cw", "seed": int(seed), "method_name": f"f02_{vid}"}

    # student_II_100hz_bce_ta_mkd_ta_crf_feature[_clecg][_prog][_clecgta_clecg]_seed0
    # Extract optional known suffixes before _seed{N}
    m = re.match(r"^student_(\w+)_(\d+)hz_(.+)_seed(\d+)$", name)
    if m:
        lead, hz, losses_part, seed = m.groups()
        # Strip optional variant suffixes: _clecgta_clecg, _prog_clecg, _clecg, _prog
        # Order matters: longer first
        variant_suffixes = [
            "_prog_clecgta_clecg", "_prog_clecg", "_clecgta_clecg",
            "_prog_simclrta_simclr", "_prog_simclr",
            "_simclrta_simclr", "_simclr",
            "_clecg", "_prog",
        ]
        variant_tag = ""
        for vs in variant_suffixes:
            if losses_part.endswith(vs):
                variant_tag = vs
                losses_part = losses_part[: -len(vs)]
                break
        losses = _parse_losses_fname(losses_part) or losses_part
        method_name = f"{lead}_{hz}hz_{losses}{variant_tag}"
        return {
            "model_type": "student",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      losses,
            "seed":        int(seed),
            "method_name": method_name,
        }

    # ta_II_500hz or ta_II_500hz_seed0
    m = re.match(r"^ta_(\w+)_(\d+)hz(?:_seed(\d+))?$", name)
    if m:
        lead, hz, seed = m.groups()
        return {
            "model_type": "ta",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      "bce,mkd,crf",
            "seed":        int(seed) if seed else 0,
            "method_name": f"ta_{lead}_{hz}hz",
        }

    # ta_II_500hz_clecg or ta_II_500hz_clecg_seed0  (CLECG-initialized TA)
    m = re.match(r"^ta_(\w+)_(\d+)hz_clecg(?:_seed(\d+))?$", name)
    if m:
        lead, hz, seed = m.groups()
        return {
            "model_type":  "ta_clecg",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      "bce,mkd,crf+clecg",
            "seed":        int(seed) if seed else 0,
            "method_name": f"ta_{lead}_{hz}hz_clecg",
        }

    # ta_II_500hz_simclr or ta_II_500hz_simclr_seed0  (SimCLR-initialized TA)
    m = re.match(r"^ta_(\w+)_(\d+)hz_simclr(?:_seed(\d+))?$", name)
    if m:
        lead, hz, seed = m.groups()
        return {
            "model_type":  "ta_simclr",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      "bce,mkd,crf+simclr",
            "seed":        int(seed) if seed else 0,
            "method_name": f"ta_{lead}_{hz}hz_simclr",
        }

    # ta_{LEAD}_{HZ}hz_from_ta{HIGH_HZ}_seed{N}  (progressive TA)
    m = re.match(r"^ta_(\w+)_(\d+)hz_from_ta(\d+)_seed(\d+)$", name)
    if m:
        lead, hz, high_hz, seed = m.groups()
        return {
            "model_type":  "ta_progressive",
            "lead":        lead,
            "hz":          int(hz),
            "losses":      f"bce,mkd,crf,feat_from_ta{high_hz}",
            "seed":        int(seed),
            "method_name": f"ta_{lead}_{hz}hz_from_ta{high_hz}",
        }

    # mutual learning 1L100 teacher: teacher_1l100_mut_12l100_lam005_seed0
    m = re.match(r"^teacher_1l100_mut_(\w+)_lam(\w+)_seed(\d+)$", name)
    if m:
        upstream, lam, seed = m.groups()
        return {
            "model_type":  "teacher_mutual",
            "lead":        "II",
            "hz":          100,
            "losses":      f"bce,mkd_mut_{upstream}",
            "seed":        int(seed),
            "method_name": f"teacher_1l100_mut_{upstream}_lam{lam}",
        }

    # teacher variants
    if re.match(r"^teacher", name):
        return {
            "model_type": "teacher",
            "lead":        "12",
            "hz":          500,
            "losses":      "bce",
            "seed":        0,
            "method_name": "teacher_12lead_500hz",
        }

    return None


def load_metrics_file(path: Path, split: str) -> dict | None:
    model_name = path.stem.replace(f"metrics_{split}_", "", 1)
    meta = parse_model_name(model_name)
    if meta is None:
        return None

    with open(path) as f:
        d = json.load(f)

    # Prefer _meta embedded by evaluate.py if present
    if "_meta" in d:
        em = d["_meta"]
        meta.setdefault("lead", em.get("lead", meta["lead"]))
        meta.setdefault("hz",   em.get("hz",   meta["hz"]))

    rec = {**meta, "source_file": str(path)}
    rec["macro_auc"]        = d.get("macro_auc", float("nan"))
    rec["macro_f1_0_5"]     = d.get("macro_f1_0_5", d.get("macro_f1", float("nan")))
    rec["macro_f1_tuned"]   = d.get("macro_f1_tuned", d.get("macro_f1", float("nan")))

    for c in CLASSES:
        rec[f"auc_{c}"]       = d.get(f"auc_{c}", float("nan"))
        rec[f"f1_0_5_{c}"]    = d.get(f"f1_0_5_{c}", d.get(f"f1_{c}", float("nan")))
        rec[f"f1_tuned_{c}"]  = d.get(f"f1_tuned_{c}", d.get(f"f1_{c}", float("nan")))

    return rec


def _fmt(mean: float, std: float, n: int) -> str:
    if np.isnan(mean):
        return "N/A"
    if n > 1:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def aggregate(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(records)

    group_cols   = ["method_name", "lead", "hz", "losses"]
    metric_cols  = ["macro_auc", "macro_f1_0_5", "macro_f1_tuned"]

    rows = []
    for key, grp in df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
        row["n_seeds"] = len(grp)
        for col in metric_cols:
            vals = grp[col].dropna().values
            row[f"mean_{col}"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"std_{col}"]  = float(np.std(vals))  if len(vals) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sort: teacher first, then by hz desc, then by method
    summary["_sort_key"] = summary["method_name"].apply(
        lambda x: (0 if "teacher" in x else (1 if x.startswith("ta_") else 2),
                   -summary.loc[summary["method_name"] == x, "hz"].iloc[0]
                   if len(summary[summary["method_name"] == x]) else 0,
                   x)
    )
    summary = summary.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    # Class-wise summary
    cw_rows = []
    for key, grp in df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
        base["n_seeds"] = len(grp)
        for c in CLASSES:
            row = {**base, "class": c}
            for metric in ["auc", "f1_0_5", "f1_tuned"]:
                col = f"{metric}_{c}"
                vals = grp[col].dropna().values
                row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
                row[f"{metric}_std"]  = float(np.std(vals))  if len(vals) > 1 else 0.0
            cw_rows.append(row)
    classwise = pd.DataFrame(cw_rows)

    return summary, classwise


def to_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "| Method | Lead | Hz | Losses | AUC mean±std | F1@0.5 mean±std | F1_tuned mean±std | n |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        n = int(row["n_seeds"])
        lines.append(
            f"| {row['method_name']} "
            f"| {row['lead']} "
            f"| {int(row['hz'])} "
            f"| {row['losses']} "
            f"| {_fmt(row['mean_macro_auc'], row['std_macro_auc'], n)} "
            f"| {_fmt(row['mean_macro_f1_0_5'], row['std_macro_f1_0_5'], n)} "
            f"| {_fmt(row['mean_macro_f1_tuned'], row['std_macro_f1_tuned'], n)} "
            f"| {n} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="dafd_mvkt/outputs",
                   help="Directory containing metrics_*.json files")
    p.add_argument("--out_dir",     default="dafd_mvkt/outputs",
                   help="Directory to write summary files")
    p.add_argument("--split",       default="test", choices=["val", "test"])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []
    for f in sorted(results_dir.glob(f"metrics_{args.split}_*.json")):
        rec = load_metrics_file(f, args.split)
        if rec is None:
            skipped.append(f.name)
        else:
            records.append(rec)

    if skipped:
        print(f"Skipped (unparseable): {skipped}")

    if not records:
        print(f"No parseable results found in {results_dir}")
        return

    print(f"Loaded {len(records)} result files.")
    summary, classwise = aggregate(records)

    # Save
    csv_path = out_dir / "summary_results.csv"
    md_path  = out_dir / "summary_results.md"
    cw_path  = out_dir / "classwise_summary.csv"

    summary.to_csv(csv_path, index=False)
    classwise.to_csv(cw_path, index=False)

    md_text = to_markdown(summary)
    md_path.write_text(md_text)

    print(f"\nSaved:\n  {csv_path}\n  {md_path}\n  {cw_path}")
    print(f"\n{md_text}")


if __name__ == "__main__":
    main()
