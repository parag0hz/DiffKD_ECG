"""
Pretraining comparison report: Random init vs CLECG vs SimCLR.

Reads result files, compares pretraining methods across leads/Hz,
and generates a markdown report with recommendations.

Output: dafd_mvkt/outputs/auto/pretraining_comparison_report.md

Usage:
    python dafd_mvkt/experiments/make_pretraining_report.py \\
        --results_dirs dafd_mvkt/outputs \\
        --out dafd_mvkt/outputs/auto/pretraining_comparison_report.md \\
        --split test
"""
from __future__ import annotations
import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dafd_mvkt.experiments.aggregate_results import load_metrics_file, aggregate

MVKT_AUC = 0.843
MVKT_F1  = 0.626


def _fmt(v: float, n: int = 1) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.4f}"


def _get(df, method_name: str, col: str) -> float:
    rows = df[df["method_name"] == method_name]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0].get(col, float("nan")))


def _best_student(df, lead: str, hz: int) -> tuple[str, float, float]:
    """Return (method_name, auc, f1) for best student model by AUC."""
    mask = (
        df["lead"].astype(str) == str(lead)
    ) & (
        df["hz"].astype(int) == int(hz)
    ) & (
        df["method_name"].str.startswith(f"{lead}_{hz}hz", na=False)
    )
    sub = df[mask].dropna(subset=["mean_macro_auc"])
    if sub.empty:
        return ("—", float("nan"), float("nan"))
    best = sub.loc[sub["mean_macro_auc"].idxmax()]
    return (str(best["method_name"]),
            float(best["mean_macro_auc"]),
            float(best["mean_macro_f1_tuned"]))


def build_report(df, split: str) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Pretraining Comparison Report — {ts}",
        "",
        f"Split: `{split}` | MVKT target: AUC {MVKT_AUC}, F1 {MVKT_F1}",
        "",
    ]

    # ── 1. SimCLR source ──────────────────────────────────────────────────────
    lines += [
        "## 1. SimCLR Source",
        "",
        "| Item | Value |",
        "|---|---|",
        "| comper_repo encoder | ConvNeXt1d (12-lead, `downsample_layers.*`) |",
        "| dafd_mvkt encoder | ResNet1d (1-lead, `stem.*`, `layer_blocks.*`) |",
        "| Architecture compatible | **NO** — different backbone, weight transfer impossible |",
        "| SimCLR implementation | `dafd_mvkt/pretrain_simclr.py` (native ResNet1d) |",
        "| Checkpoint format | `{\"state_dict\": ..., \"hz\": ..., \"lead\": ...}` (CLECG-compatible) |",
        "",
    ]

    # ── 2. 100Hz Lead II comparison ───────────────────────────────────────────
    lines += ["## 2. Lead II 100Hz — Pretraining Comparison", ""]
    lines += [
        "| Method | AUC | F1_tuned | ΔAUC vs random | ΔF1 vs random | Beats MVKT? |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    base_m   = "II_100hz_bce,ta_mkd,ta_crf,feature"
    clecg_m  = "II_100hz_bce,ta_mkd,ta_crf,feature_clecg"
    clecgta_m = "II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg"
    simclr_m  = "II_100hz_bce,ta_mkd,ta_crf,feature_simclr"
    simclrta_m = "II_100hz_bce,ta_mkd,ta_crf,feature_simclrta_simclr"

    entries = [
        ("MVKT-ECG (ref)", None),
        ("Random init",    base_m),
        ("CLECG student",  clecg_m),
        ("CLECG TA+Student", clecgta_m),
        ("SimCLR student",  simclr_m),
        ("SimCLR TA+Student", simclrta_m),
    ]
    rand_auc = _get(df, base_m, "mean_macro_auc")
    rand_f1  = _get(df, base_m, "mean_macro_f1_tuned")

    for label, mname in entries:
        if mname is None:
            lines.append(
                f"| **{label}** | **{MVKT_AUC:.4f}** | **{MVKT_F1:.4f}** | — | — | — |"
            )
            continue
        auc = _get(df, mname, "mean_macro_auc")
        f1  = _get(df, mname, "mean_macro_f1_tuned")
        if math.isnan(auc):
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        dauc = auc - rand_auc if not math.isnan(rand_auc) else float("nan")
        df1  = f1  - rand_f1  if not math.isnan(rand_f1)  else float("nan")
        beats = "✓" if auc >= MVKT_AUC and f1 >= MVKT_F1 else ""
        bold = "**" if beats else ""
        lines.append(
            f"| {bold}{label}{bold} "
            f"| {bold}{_fmt(auc)}{bold} "
            f"| {bold}{_fmt(f1)}{bold} "
            f"| {f'{dauc:+.4f}' if not math.isnan(dauc) else '—'} "
            f"| {f'{df1:+.4f}' if not math.isnan(df1) else '—'} "
            f"| {beats} |"
        )
    lines.append("")

    # ── 3. 100Hz Lead I comparison ────────────────────────────────────────────
    lines += ["## 3. Lead I 100Hz — Pretraining Comparison", ""]
    lines += [
        "| Method | AUC | F1_tuned | ΔAUC vs random | Beats MVKT? |",
        "|---|---:|---:|---:|:---:|",
    ]
    base_I     = "I_100hz_bce,ta_mkd,ta_crf,feature"
    clecg_I    = "I_100hz_bce,ta_mkd,ta_crf,feature_clecg"
    clecgta_I  = "I_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg"
    simclr_I   = "I_100hz_bce,ta_mkd,ta_crf,feature_simclr"
    simclrta_I = "I_100hz_bce,ta_mkd,ta_crf,feature_simclrta_simclr"

    rand_auc_I = _get(df, base_I, "mean_macro_auc")
    for label, mname in [
        ("MVKT-ECG (ref)", None),
        ("Random init",    base_I),
        ("CLECG student",  clecg_I),
        ("CLECG TA+Student", clecgta_I),
        ("SimCLR student",  simclr_I),
        ("SimCLR TA+Student", simclrta_I),
    ]:
        if mname is None:
            lines.append(f"| **MVKT-ECG (ref)** | **{MVKT_AUC:.4f}** | **{MVKT_F1:.4f}** | — | — |")
            continue
        auc = _get(df, mname, "mean_macro_auc")
        f1  = _get(df, mname, "mean_macro_f1_tuned")
        if math.isnan(auc):
            lines.append(f"| {label} | — | — | — | — |")
            continue
        dauc = auc - rand_auc_I if not math.isnan(rand_auc_I) else float("nan")
        beats = "✓" if auc >= MVKT_AUC and f1 >= MVKT_F1 else ""
        lines.append(
            f"| {label} | {_fmt(auc)} | {_fmt(f1)} "
            f"| {f'{dauc:+.4f}' if not math.isnan(dauc) else '—'} "
            f"| {beats} |"
        )
    lines.append("")

    # ── 4. 50Hz trade-off comparison ──────────────────────────────────────────
    lines += ["## 4. Lead II 50Hz — Progressive HST-KD Comparison", ""]
    lines += [
        "| Method | AUC | F1_tuned | ΔAUC vs BCE 50Hz |",
        "|---|---:|---:|---:|",
    ]
    bce_50     = "II_50hz_bce"
    prog_50    = "II_50hz_bce,ta_mkd,ta_crf,feature_prog"
    clecg_50   = "II_50hz_bce,ta_mkd,ta_crf,feature_prog_clecg"
    simclr_50  = "II_50hz_bce,ta_mkd,ta_crf,feature_prog_simclr"
    bce50_auc  = _get(df, bce_50, "mean_macro_auc")
    for label, mname in [
        ("BCE 50Hz (baseline)",         bce_50),
        ("Progressive HST-KD",          prog_50),
        ("Progressive HST-KD + CLECG",  clecg_50),
        ("Progressive HST-KD + SimCLR", simclr_50),
    ]:
        auc = _get(df, mname, "mean_macro_auc")
        f1  = _get(df, mname, "mean_macro_f1_tuned")
        if math.isnan(auc):
            lines.append(f"| {label} | — | — | — |")
            continue
        dauc = auc - bce50_auc if not math.isnan(bce50_auc) else float("nan")
        lines.append(
            f"| {label} | {_fmt(auc)} | {_fmt(f1)} "
            f"| {f'{dauc:+.4f}' if not math.isnan(dauc) else '—'} |"
        )
    lines.append("")

    # ── 5. Recommendation ─────────────────────────────────────────────────────
    lines += ["## 5. Recommendation", ""]

    best_100hz_m, best_100hz_auc, best_100hz_f1 = _best_student(df, "II", 100)
    best_method_tag = best_100hz_m.replace("II_100hz_bce,ta_mkd,ta_crf,feature", "")
    if "_simclrta_simclr" in best_method_tag:
        rec = "SimCLR full pipeline (TA+Student init) is best. Recommend replacing CLECG."
    elif "_simclr" in best_method_tag and "_clecg" not in best_method_tag:
        rec = "SimCLR student init beats CLECG. Consider SimCLR as primary pretraining."
    elif "_clecgta_clecg" in best_method_tag:
        rec = "CLECG full pipeline remains best. SimCLR is complementary ablation."
    elif "_clecg" in best_method_tag:
        rec = "CLECG student init is best. SimCLR does not improve further."
    elif best_method_tag == "":
        rec = "No pretraining helps vs random init at this point."
    else:
        rec = f"Best method: {best_100hz_m}. Check tables above."

    beats_mvkt = best_100hz_auc >= MVKT_AUC and best_100hz_f1 >= MVKT_F1
    lines += [
        f"- **Best Lead II 100Hz method**: `{best_100hz_m}`",
        f"  AUC {_fmt(best_100hz_auc)}  F1 {_fmt(best_100hz_f1)}  "
        f"({'beats MVKT ✓' if beats_mvkt else 'does not beat MVKT'})",
        f"- **Recommendation**: {rec}",
        "",
    ]

    # ── 6. Warnings ───────────────────────────────────────────────────────────
    lines += ["## 6. Warnings", ""]
    warnings = []

    # Check if all results are seed=0 only
    if "n_seeds" in df.columns and df["n_seeds"].max() <= 1:
        warnings.append("All results are seed=0 only. Multi-seed validation recommended before final claims.")

    # Check SimCLR load warnings
    simclr_methods = [m for m in df["method_name"].tolist() if "simclr" in str(m)]
    if not simclr_methods:
        warnings.append("No SimCLR downstream results found. SimCLR pretraining may not have completed.")

    # Architecture mismatch warning
    warnings.append(
        "comper_repo SimCLR uses ConvNeXt (not ResNet1d). "
        "dafd_mvkt SimCLR is independently pretrained with ResNet1d backbone. "
        "Architectures are NOT interchangeable."
    )

    # Check if SimCLR improves over CLECG
    clecg_auc = _get(df, "II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg", "mean_macro_auc")
    simclr_auc = _get(df, "II_100hz_bce,ta_mkd,ta_crf,feature_simclrta_simclr", "mean_macro_auc")
    if not math.isnan(clecg_auc) and not math.isnan(simclr_auc):
        if simclr_auc > clecg_auc + 0.002:
            warnings.append(f"SimCLR improves over CLECG by {simclr_auc - clecg_auc:+.4f} AUC on Lead II 100Hz.")
        elif clecg_auc > simclr_auc + 0.002:
            warnings.append(f"CLECG outperforms SimCLR by {clecg_auc - simclr_auc:+.4f} AUC on Lead II 100Hz. Keep CLECG.")

    for w in warnings:
        lines.append(f"- ⚠ {w}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+",
                   default=["dafd_mvkt/outputs", "dafd_mvkt/outputs/tuning"])
    p.add_argument("--out",   default="dafd_mvkt/outputs/auto/pretraining_comparison_report.md")
    p.add_argument("--split", default="test")
    args = p.parse_args()

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
        print(f"No results found in {args.results_dirs}.")
        summary = None
    else:
        print(f"Loaded {len(records)} result files.")
        import pandas as pd
        summary, _ = aggregate(records)

    if summary is None:
        import pandas as pd
        summary = pd.DataFrame()

    report = build_report(summary, args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Report saved: {out}")
    print()
    print(report)


if __name__ == "__main__":
    main()
