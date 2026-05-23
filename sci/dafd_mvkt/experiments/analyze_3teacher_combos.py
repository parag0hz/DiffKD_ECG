"""
Analyze 3-teacher combination effects after 20-combo screening.

Usage:
    python dafd_mvkt/experiments/analyze_3teacher_combos.py \
        --seed 0 \
        --output_dir dafd_mvkt/outputs/multiteacher
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

COMBOS = [f"C{i:02d}" for i in range(1, 21)]
COMBO_TEACHERS: dict[str, list[str]] = {
    "C01":["T1","T2","T3"], "C02":["T1","T2","T4"], "C03":["T1","T2","T5"],
    "C04":["T1","T2","T6"], "C05":["T1","T3","T4"], "C06":["T1","T3","T5"],
    "C07":["T1","T3","T6"], "C08":["T1","T4","T5"], "C09":["T1","T4","T6"],
    "C10":["T1","T5","T6"], "C11":["T2","T3","T4"], "C12":["T2","T3","T5"],
    "C13":["T2","T3","T6"], "C14":["T2","T4","T5"], "C15":["T2","T4","T6"],
    "C16":["T2","T5","T6"], "C17":["T3","T4","T5"], "C18":["T3","T4","T6"],
    "C19":["T3","T5","T6"], "C20":["T4","T5","T6"],
}
TEACHER_LABELS = {
    "T1":"12L500","T2":"12L100","T3":"12L50",
    "T4":"1L500","T5":"1L100","T6":"1L50",
}
_12L_TEACHERS = {"T1","T2","T3"}
_1L_TEACHERS  = {"T4","T5","T6"}
_50HZ_TEACHERS = {"T3","T6"}

BCE_AUC  = 0.8061; BCE_F1  = 0.5740
BEST_AUC = 0.8396; BEST_F1 = 0.6176
MVKT_AUC = 0.843;  MVKT_F1 = 0.626


def load_results(out_dir: Path, seed: int) -> dict[str, dict]:
    results = {}
    for combo in COMBOS:
        name = f"student_II_50hz_3T_{combo}_equal_simclr_seed{seed}"
        mf   = out_dir.parent / f"metrics_test_{name}.json"
        if mf.exists():
            d = json.load(open(mf))
            results[combo] = {
                "auc":  d["macro_auc"],
                "f1t":  d.get("macro_f1_tuned", d.get("macro_f1", 0)),
                "f10":  d.get("macro_f1_0_5", 0),
                "teachers": COMBO_TEACHERS[combo],
            }
    return results


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(out_dir, args.seed)

    n_done = len(results)
    print(f"Loaded {n_done}/{len(COMBOS)} combo results.")
    if n_done == 0:
        print("No results found. Run run_3teacher_20combos.sh first.")
        return

    sorted_by_auc = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)
    top5   = [c for c, _ in sorted_by_auc[:5]]
    bot5   = [c for c, _ in sorted_by_auc[-5:]]

    lines = []

    # ── 1. Best overall ───────────────────────────────────────────────────────
    best_c, best_v = sorted_by_auc[0]
    best_f1_c, best_f1_v = max(results.items(), key=lambda x: x[1]["f1t"])

    lines.append("# 3-Teacher Combination Analysis\n")
    lines.append(f"Results loaded: {n_done}/{len(COMBOS)}\n\n")
    lines.append("## Best Overall\n")
    lines.append(f"- Best by AUC:    {best_c} ({'+'.join(TEACHER_LABELS[t] for t in best_v['teachers'])}) "
                 f"AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}\n")
    lines.append(f"- Best by F1_tuned: {best_f1_c} ({'+'.join(TEACHER_LABELS[t] for t in best_f1_v['teachers'])}) "
                 f"AUC={best_f1_v['auc']:.4f}  F1={best_f1_v['f1t']:.4f}\n")
    lines.append(f"\nReferences:\n")
    lines.append(f"- BCE 50Hz:     AUC={BCE_AUC}  F1={BCE_F1}\n")
    lines.append(f"- Current best: AUC={BEST_AUC} F1={BEST_F1}\n")
    lines.append(f"- MVKT target:  AUC={MVKT_AUC} F1={MVKT_F1}\n\n")

    # ── 2. Full ranking ───────────────────────────────────────────────────────
    lines.append("## Full Ranking by AUC\n\n")
    lines.append("| Rank | Combo | Teachers | AUC | F1_tuned | ΔAUC vs BCE | Beats MVKT |\n")
    lines.append("|------|-------|----------|-----|----------|-------------|------------|\n")
    for rank, (c, v) in enumerate(sorted_by_auc, 1):
        tstr  = "+".join(TEACHER_LABELS[t] for t in v["teachers"])
        mvkt  = "✅" if v["auc"] > MVKT_AUC else ""
        lines.append(f"| {rank} | {c} | {tstr} | {v['auc']:.4f} | "
                     f"{v['f1t']:.4f} | {v['auc']-BCE_AUC:+.4f} | {mvkt} |\n")
    lines.append("\n")

    # ── 3. Per-teacher marginal effect ────────────────────────────────────────
    lines.append("## Per-Teacher Marginal Effect\n\n")
    teacher_effects = []
    for tid in ["T1","T2","T3","T4","T5","T6"]:
        with_t = [(c, v) for c, v in results.items() if tid in v["teachers"]]
        without = [(c, v) for c, v in results.items() if tid not in v["teachers"]]
        if not with_t:
            continue
        mean_auc_with = sum(v["auc"] for _, v in with_t) / len(with_t)
        mean_f1_with  = sum(v["f1t"] for _, v in with_t) / len(with_t)
        mean_auc_wo   = sum(v["auc"] for _, v in without) / len(without) if without else float("nan")
        top5_cnt = sum(1 for c, _ in with_t if c in top5)
        bot5_cnt = sum(1 for c, _ in with_t if c in bot5)
        teacher_effects.append({
            "teacher_id": tid, "label": TEACHER_LABELS[tid],
            "n_combos": len(with_t),
            "mean_auc": round(mean_auc_with, 4),
            "mean_f1":  round(mean_f1_with, 4),
            "mean_auc_without": round(mean_auc_wo, 4),
            "delta_auc": round(mean_auc_with - mean_auc_wo, 4),
            "top5_count": top5_cnt,
            "bot5_count": bot5_cnt,
        })

    lines.append("| Teacher | Label | Combos | Mean AUC (with) | Mean AUC (without) | Δ AUC | Top5 | Bot5 |\n")
    lines.append("|---------|-------|--------|------------------|--------------------|-------|------|------|\n")
    for r in sorted(teacher_effects, key=lambda x: x["delta_auc"], reverse=True):
        lines.append(f"| {r['teacher_id']} | {r['label']} | {r['n_combos']} | "
                     f"{r['mean_auc']} | {r['mean_auc_without']} | "
                     f"{r['delta_auc']:+.4f} | {r['top5_count']} | {r['bot5_count']} |\n")
    lines.append("\n")

    # ── 4. Pairwise interaction ───────────────────────────────────────────────
    lines.append("## Teacher Pair Effects\n\n")
    pair_effects = []
    all_tids = list(TEACHER_LABELS.keys())
    for t1, t2 in combinations(all_tids, 2):
        with_pair = [(c, v) for c, v in results.items()
                     if t1 in v["teachers"] and t2 in v["teachers"]]
        if not with_pair:
            continue
        mean_auc = sum(v["auc"] for _, v in with_pair) / len(with_pair)
        mean_f1  = sum(v["f1t"] for _, v in with_pair) / len(with_pair)
        pair_effects.append({
            "pair": f"{t1}+{t2}",
            "labels": f"{TEACHER_LABELS[t1]}+{TEACHER_LABELS[t2]}",
            "n": len(with_pair),
            "mean_auc": round(mean_auc, 4),
            "mean_f1":  round(mean_f1, 4),
        })
    pair_effects.sort(key=lambda x: x["mean_auc"], reverse=True)
    lines.append("| Pair | Labels | N | Mean AUC | Mean F1 |\n")
    lines.append("|------|--------|---|----------|---------|\n")
    for r in pair_effects:
        lines.append(f"| {r['pair']} | {r['labels']} | {r['n']} | {r['mean_auc']} | {r['mean_f1']} |\n")
    lines.append("\n")

    # ── 5. Category comparison ────────────────────────────────────────────────
    lines.append("## Category Comparison\n\n")
    categories = {
        "12L-only (T1/T2/T3)": [c for c,v in results.items() if set(v["teachers"]) <= _12L_TEACHERS],
        "1L-only (T4/T5/T6)":  [c for c,v in results.items() if set(v["teachers"]) <= _1L_TEACHERS],
        "Mixed 12L+1L":         [c for c,v in results.items()
                                  if set(v["teachers"]) & _12L_TEACHERS and set(v["teachers"]) & _1L_TEACHERS],
        "Has 50Hz teacher":     [c for c,v in results.items() if set(v["teachers"]) & _50HZ_TEACHERS],
        "No 50Hz teacher":      [c for c,v in results.items() if not (set(v["teachers"]) & _50HZ_TEACHERS)],
        "Has T6 (same-input)":  [c for c,v in results.items() if "T6" in v["teachers"]],
        "No T6":                [c for c,v in results.items() if "T6" not in v["teachers"]],
    }
    lines.append("| Category | N | Mean AUC | Mean F1 | Best AUC |\n")
    lines.append("|----------|---|----------|---------|----------|\n")
    for cat_name, combos_in in categories.items():
        if not combos_in:
            continue
        aucs = [results[c]["auc"] for c in combos_in]
        f1s  = [results[c]["f1t"] for c in combos_in]
        lines.append(f"| {cat_name} | {len(combos_in)} | "
                     f"{sum(aucs)/len(aucs):.4f} | "
                     f"{sum(f1s)/len(f1s):.4f} | "
                     f"{max(aucs):.4f} |\n")
    lines.append("\n")

    # ── write files ───────────────────────────────────────────────────────────
    md_path = out_dir / "3teacher_analysis.md"
    with open(md_path, "w") as f:
        f.writelines(lines)

    effects_csv = out_dir / "3teacher_teacher_effects.csv"
    if teacher_effects:
        with open(effects_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=teacher_effects[0].keys())
            w.writeheader(); w.writerows(teacher_effects)

    pair_csv = out_dir / "3teacher_pair_effects.csv"
    if pair_effects:
        with open(pair_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pair_effects[0].keys())
            w.writeheader(); w.writerows(pair_effects)

    print(f"Saved: {md_path}")
    print(f"Saved: {effects_csv}")
    print(f"Saved: {pair_csv}")
    print(f"\nBest combo: {best_c}  AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output_dir", default="dafd_mvkt/outputs/multiteacher")
    analyze(p.parse_args())
