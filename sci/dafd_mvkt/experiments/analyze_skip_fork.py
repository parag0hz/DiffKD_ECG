"""
Analyze skip-fork-join dual distillation results.

Usage:
    python dafd_mvkt/experiments/analyze_skip_fork.py \\
        --seed 0 \\
        --output_dir dafd_mvkt/outputs/skipfork \\
        --forkjoin_dir dafd_mvkt/outputs/forkjoin
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

BCE_AUC  = 0.8061;  BCE_F1  = 0.5740
PROG_AUC = 0.8396;  PROG_F1 = 0.6176
C14_AUC  = 0.8425;  C14_F1  = 0.6243
MVKT_AUC = 0.843;   MVKT_F1 = 0.626

SFS = {
    "SF01": {"t1": "12L100", "t2": "12L50",  "t3": "1L100", "weights": [0.20, 0.40, 0.40]},
    "SF02": {"t1": "12L100", "t2": "1L500",  "t3": "1L100", "weights": [0.20, 0.40, 0.40]},
    "SF03": {"t1": "12L500", "t2": "12L50",  "t3": "1L100", "weights": [0.10, 0.45, 0.45]},
    "SF04": {"t1": "12L500", "t2": "1L500",  "t3": "1L100", "weights": [0.10, 0.45, 0.45]},
    "SF05": {"t1": "12L100", "t2": "12L50",  "t3": "1L50",  "weights": [0.20, 0.40, 0.40]},
    "SF06": {"t1": "1L500",  "t2": "1L100",  "t3": "1L50",  "weights": [0.20, 0.40, 0.40]},
}
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_results(skipfork_dir: Path, seed: int) -> dict[str, dict]:
    results = {}
    for sid, spec in SFS.items():
        t1, t2, t3 = spec["t1"], spec["t2"], spec["t3"]
        name = (f"student_II_50hz_skipfork_{sid}_"
                f"{t1}_to_{t2}_{t3}_simclr_seed{seed}")
        mf = skipfork_dir / f"metrics_test_{name}.json"
        if mf.exists():
            d = json.load(open(mf))
            results[sid] = {
                "auc":  d["macro_auc"],
                "f1t":  d.get("macro_f1_tuned", d.get("macro_f1", 0)),
                "f10":  d.get("macro_f1_0_5", 0),
                "t1": t1, "t2": t2, "t3": t3,
                "weights": spec["weights"],
                "structure": f"{t1}→{{{t2},{t3}}}→S+skip",
                "per_class": {c: d.get(f"auc_{c}", float("nan"))
                              for c in SUPERCLASSES},
            }
    return results


def load_forkjoin_result(forkjoin_dir: Path, fid: str, seed: int) -> dict | None:
    """Load ordinary fork-join result for comparison if available."""
    FORK_SPEC = {
        "F01": ("12L100", "12L50",  "1L100"),
        "F02": ("12L100", "1L500",  "1L100"),
        "F03": ("12L500", "12L50",  "1L100"),
        "F04": ("12L500", "1L500",  "1L100"),
        "F05": ("12L100", "12L50",  "1L50"),
        "F06": ("1L500",  "1L100",  "1L50"),
    }
    if forkjoin_dir is None or not forkjoin_dir.exists():
        return None
    if fid not in FORK_SPEC:
        return None
    t1, t2, t3 = FORK_SPEC[fid]
    name = f"student_II_50hz_forkjoin_{fid}_{t1}_to_{t2}_{t3}_simclr_seed{seed}"
    mf = forkjoin_dir / f"metrics_test_{name}.json"
    if not mf.exists():
        return None
    d = json.load(open(mf))
    return {
        "auc": d["macro_auc"],
        "f1t": d.get("macro_f1_tuned", d.get("macro_f1", 0)),
        "structure": f"{t1}→{{{t2},{t3}}}→S",
    }


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fj_dir  = Path(args.forkjoin_dir) if args.forkjoin_dir else None

    results = load_results(out_dir, args.seed)
    n_done  = len(results)
    print(f"Loaded {n_done}/{len(SFS)} skip-fork results.")

    if n_done == 0:
        print("No results found. Run run_skip_fork_6.sh first.")
        return

    sorted_by_auc = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)

    # ── main table ─────────────────────────────────────────────────────────────
    rows = []
    for rank, (sid, v) in enumerate(sorted_by_auc, 1):
        w = v["weights"]
        rows.append({
            "rank":           rank,
            "skipfork_id":    sid,
            "structure":      v["structure"],
            "t1": v["t1"], "t2": v["t2"], "t3": v["t3"],
            "weights":        f"{w[0]}/{w[1]}/{w[2]}",
            "auc":            round(v["auc"], 4),
            "f1_tuned":       round(v["f1t"], 4),
            "f1_0_5":         round(v["f10"], 4),
            "delta_auc_vs_bce_50hz":           round(v["auc"] - BCE_AUC,  4),
            "delta_f1_vs_bce_50hz":            round(v["f1t"] - BCE_F1,   4),
            "delta_auc_vs_progressive_simclr": round(v["auc"] - PROG_AUC, 4),
            "delta_f1_vs_progressive_simclr":  round(v["f1t"] - PROG_F1,  4),
            "delta_auc_vs_parallel_c14":       round(v["auc"] - C14_AUC,  4),
            "delta_f1_vs_parallel_c14":        round(v["f1t"] - C14_F1,   4),
            "beats_progressive_simclr": "YES" if v["auc"] > PROG_AUC else "no",
            "beats_parallel_c14":       "YES" if v["auc"] > C14_AUC  else "no",
            "beats_mvkt_auc":           "YES" if v["auc"] > MVKT_AUC else "no",
            "beats_mvkt_f1":            "YES" if v["f1t"] > MVKT_F1  else "no",
        })

    csv_path = out_dir / "skip_fork_6_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {csv_path}")

    md_path = out_dir / "skip_fork_6_results.md"
    with open(md_path, "w") as f:
        f.write("# Skip-Fork-Join Dual Distillation Results\n\n")
        f.write(f"Results: {n_done}/{len(SFS)} skip-forks complete\n\n")
        f.write("**References:**\n")
        f.write(f"- BCE 50Hz:           AUC={BCE_AUC}  F1={BCE_F1}\n")
        f.write(f"- Progressive+SimCLR: AUC={PROG_AUC} F1={PROG_F1}\n")
        f.write(f"- Parallel C14:       AUC={C14_AUC}  F1={C14_F1}\n")
        f.write(f"- MVKT target:        AUC={MVKT_AUC} F1={MVKT_F1}\n\n")
        f.write("| Rank | SF | Structure | Weights | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |\n")
        f.write("|------|----|-----------|---------|----|----------|-------------|-------------|-------|------|-------|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['skipfork_id']} | {r['structure']} | "
                    f"{r['weights']} | {r['auc']} | {r['f1_tuned']} | "
                    f"{r['delta_auc_vs_bce_50hz']:+.4f} | "
                    f"{r['delta_auc_vs_parallel_c14']:+.4f} | "
                    f"{r['beats_progressive_simclr']} | {r['beats_parallel_c14']} | "
                    f"{r['beats_mvkt_auc']} |\n")
        if n_done < len(SFS):
            missing = [h for h in SFS if h not in results]
            f.write(f"\n**Pending ({len(missing)}):** {', '.join(missing)}\n")
    print(f"Saved: {md_path}")

    # ── per-class CSV ──────────────────────────────────────────────────────────
    cw_path = out_dir / "skip_fork_classwise.csv"
    cw_rows = []
    for sid, v in sorted_by_auc:
        row = {"skipfork_id": sid, "structure": v["structure"],
               "macro_auc": round(v["auc"], 4)}
        for c in SUPERCLASSES:
            row[f"auc_{c}"] = round(v["per_class"][c], 4)
        cw_rows.append(row)
    with open(cw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cw_rows[0].keys())
        w.writeheader(); w.writerows(cw_rows)
    print(f"Saved: {cw_path}")

    # ── analysis markdown ──────────────────────────────────────────────────────
    best_id, best_v = sorted_by_auc[0]
    best_f1_id, best_f1_v = max(results.items(), key=lambda x: x[1]["f1t"])

    ana_path = out_dir / "skip_fork_analysis.md"
    lines = []
    lines.append("# Skip-Fork-Join Dual Distillation Analysis\n\n")
    lines.append(f"Loaded {n_done}/{len(SFS)} results.\n\n")

    lines.append("## 1. Best Overall\n\n")
    lines.append(f"- Best AUC: **{best_id}** ({best_v['structure']}) "
                 f"AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}\n")
    lines.append(f"- Best F1:  **{best_f1_id}** ({best_f1_v['structure']}) "
                 f"AUC={best_f1_v['auc']:.4f}  F1={best_f1_v['f1t']:.4f}\n\n")

    lines.append("## 2. Reference Comparison\n\n")
    lines.append("| SF | Structure | AUC | F1_tuned | >Prog | >C14 | >MVKT AUC | >MVKT F1 |\n")
    lines.append("|----|-----------|-----|----------|-------|------|-----------|----------|\n")
    for sid, v in sorted_by_auc:
        lines.append(
            f"| {sid} | {v['structure']} | {v['auc']:.4f} | {v['f1t']:.4f} | "
            f"{'✅' if v['auc']>PROG_AUC else '❌'} | "
            f"{'✅' if v['auc']>C14_AUC  else '❌'} | "
            f"{'✅' if v['auc']>MVKT_AUC else '❌'} | "
            f"{'✅' if v['f1t']>MVKT_F1  else '❌'} |\n"
        )
    lines.append("\n")

    # SF02 vs parallel C14
    lines.append("## 3. SF02 vs Parallel C14 (same teacher set)\n\n")
    lines.append("SF02 = 12L100 → {1L500, 1L100} → S + skip 12L100 → S\n")
    lines.append("C14  = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)\n\n")
    if "SF02" in results:
        v = results["SF02"]
        lines.append("| Method | AUC | F1_tuned | ΔAUC |\n")
        lines.append("|--------|-----|----------|------|\n")
        lines.append(f"| C14 (parallel) | {C14_AUC} | {C14_F1} | ref |\n")
        lines.append(f"| SF02 (skip-fork) | {v['auc']:.4f} | {v['f1t']:.4f} | "
                     f"{v['auc']-C14_AUC:+.4f} |\n\n")
        if v["auc"] > C14_AUC:
            lines.append("→ Skip-fork-join **improves** over parallel C14 with same teachers.\n\n")
        else:
            lines.append("→ Skip-fork-join does **not** improve over parallel C14.\n\n")
    else:
        lines.append("SF02 not yet available.\n\n")

    # SF02 vs ordinary F02
    lines.append("## 4. SF02 vs Ordinary Fork-Join F02\n\n")
    lines.append("SF02 = 12L100 → {1L500, 1L100} → S  + skip 12L100 → S\n")
    lines.append("F02  = 12L100 → {1L500, 1L100} → S  (no skip)\n\n")
    f02 = load_forkjoin_result(fj_dir, "F02", args.seed)
    if "SF02" in results and f02 is not None:
        sf02 = results["SF02"]
        lines.append("| Method | AUC | F1_tuned | ΔAUC |\n")
        lines.append("|--------|-----|----------|------|\n")
        lines.append(f"| F02  (fork, no skip) | {f02['auc']:.4f} | {f02['f1t']:.4f} | ref |\n")
        lines.append(f"| SF02 (skip-fork)     | {sf02['auc']:.4f} | {sf02['f1t']:.4f} | "
                     f"{sf02['auc']-f02['auc']:+.4f} |\n\n")
        delta = sf02["auc"] - f02["auc"]
        if delta > 0:
            lines.append(f"→ Skip connection from t1 **helps** F02: +{delta:.4f} AUC.\n\n")
        else:
            lines.append(f"→ Skip connection from t1 **does not help** F02: {delta:.4f} AUC.\n\n")
    elif "SF02" in results:
        lines.append("F02 ordinary fork not available for comparison.\n\n")
    else:
        lines.append("SF02 not yet available.\n\n")

    # SF01 vs ordinary F01
    lines.append("## 5. SF01 vs Ordinary Fork-Join F01\n\n")
    lines.append("SF01 = 12L100 → {12L50, 1L100} → S  + skip 12L100 → S\n")
    lines.append("F01  = 12L100 → {12L50, 1L100} → S  (no skip)\n\n")
    f01 = load_forkjoin_result(fj_dir, "F01", args.seed)
    if "SF01" in results and f01 is not None:
        sf01 = results["SF01"]
        lines.append("| Method | AUC | F1_tuned | ΔAUC |\n")
        lines.append("|--------|-----|----------|------|\n")
        lines.append(f"| F01  (fork, no skip) | {f01['auc']:.4f} | {f01['f1t']:.4f} | ref |\n")
        lines.append(f"| SF01 (skip-fork)     | {sf01['auc']:.4f} | {sf01['f1t']:.4f} | "
                     f"{sf01['auc']-f01['auc']:+.4f} |\n\n")
        delta = sf01["auc"] - f01["auc"]
        if delta > 0:
            lines.append(f"→ Skip connection from t1 **helps** F01: +{delta:.4f} AUC.\n\n")
        else:
            lines.append(f"→ Skip connection from t1 **does not help** F01: {delta:.4f} AUC.\n\n")
    elif "SF01" in results:
        lines.append("F01 ordinary fork not available for comparison.\n\n")
    else:
        lines.append("SF01 not yet available.\n\n")

    # Skip from 12L100 (SF01/SF02/SF05)
    lines.append("## 6. Skip from 12L100 (SF01, SF02, SF05)\n\n")
    skip_12l100 = [(sid, v) for sid, v in results.items()
                   if v["t1"] == "12L100"]
    if skip_12l100:
        avg_auc = sum(v["auc"] for _, v in skip_12l100) / len(skip_12l100)
        lines.append(f"- n={len(skip_12l100)}: mean AUC={avg_auc:.4f} "
                     f"({', '.join(s for s,_ in skip_12l100)})\n")
        for sid, v in skip_12l100:
            lines.append(f"  - {sid}: AUC={v['auc']:.4f}  F1={v['f1t']:.4f}  "
                         f"weights={v['weights']}\n")
    else:
        lines.append("No results with t1=12L100 yet.\n")
    lines.append("\n")

    # Skip from 12L500 (SF03/SF04)
    lines.append("## 7. Skip from 12L500 (SF03, SF04)\n\n")
    skip_12l500 = [(sid, v) for sid, v in results.items()
                   if v["t1"] == "12L500"]
    if skip_12l500:
        avg_auc = sum(v["auc"] for _, v in skip_12l500) / len(skip_12l500)
        lines.append(f"- n={len(skip_12l500)}: mean AUC={avg_auc:.4f} "
                     f"({', '.join(s for s,_ in skip_12l500)})\n")
        for sid, v in skip_12l500:
            lines.append(f"  - {sid}: AUC={v['auc']:.4f}  F1={v['f1t']:.4f}  "
                         f"weights={v['weights']}\n")
    else:
        lines.append("No results with t1=12L500 yet.\n")
    lines.append("\n")

    # 1L50 branch effect (SF05/SF06)
    lines.append("## 8. Branch with 1L50 vs without\n\n")
    with_1l50    = [(sid, v) for sid, v in results.items()
                    if v["t2"] == "1L50" or v["t3"] == "1L50"]
    without_1l50 = [(sid, v) for sid, v in results.items()
                    if v["t2"] != "1L50" and v["t3"] != "1L50"]
    if with_1l50 and without_1l50:
        avg_with    = sum(v["auc"] for _, v in with_1l50) / len(with_1l50)
        avg_without = sum(v["auc"] for _, v in without_1l50) / len(without_1l50)
        lines.append(f"- With 1L50 branch (n={len(with_1l50)}): mean AUC={avg_with:.4f} "
                     f"({', '.join(s for s,_ in with_1l50)})\n")
        lines.append(f"- Without 1L50 branch (n={len(without_1l50)}): mean AUC={avg_without:.4f} "
                     f"({', '.join(s for s,_ in without_1l50)})\n")
        lines.append(f"→ {'1L50 branch helps.' if avg_with > avg_without else '1L50 branch hurts.'}\n\n")
    else:
        lines.append("Insufficient data for comparison.\n\n")

    # recommendation
    lines.append("## 9. Recommendation\n\n")
    lines.append(f"Best skip-fork by AUC: **{best_id}** = {best_v['structure']}\n")
    lines.append(f"  AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}\n")
    lines.append(f"  weights (skip/b2/b3)={best_v['weights']}\n\n")
    if best_v["auc"] > MVKT_AUC:
        lines.append("✅ **Strong success**: beats MVKT target AUC.\n")
    elif best_v["auc"] > C14_AUC:
        lines.append("✅ **Success 2**: beats parallel C14.\n")
    elif best_v["auc"] > PROG_AUC:
        lines.append("✅ **Success 1**: beats Progressive+SimCLR.\n")
    else:
        lines.append("❌ Does not beat Progressive+SimCLR.\n")
    lines.append(f"\nNext step: run scripts/tune_best_skip_fork_weights.sh "
                 f"with SF_ID={best_id}\n")

    with open(ana_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {ana_path}")
    print(f"\nBest: {best_id}  AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--output_dir",    default="dafd_mvkt/outputs/skipfork")
    p.add_argument("--forkjoin_dir",  default="dafd_mvkt/outputs/forkjoin",
                   help="Path to ordinary fork-join outputs for comparison (optional)")
    analyze(p.parse_args())
