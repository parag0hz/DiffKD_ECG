"""
Analyze fork-join dual distillation results.

Usage:
    python dafd_mvkt/experiments/analyze_fork_join.py \\
        --seed 0 \\
        --output_dir dafd_mvkt/outputs/forkjoin
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

FORKS = {
    "F01": {"t1": "12L100", "t2": "12L50",  "t3": "1L100"},
    "F02": {"t1": "12L100", "t2": "1L500",  "t3": "1L100"},
    "F03": {"t1": "12L500", "t2": "12L50",  "t3": "1L100"},
    "F04": {"t1": "12L500", "t2": "1L500",  "t3": "1L100"},
    "F05": {"t1": "12L100", "t2": "12L50",  "t3": "1L50"},
    "F06": {"t1": "1L500",  "t2": "1L100",  "t3": "1L50"},
}
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_results(forkjoin_dir: Path, seed: int) -> dict[str, dict]:
    results = {}
    for fid, spec in FORKS.items():
        t1, t2, t3 = spec["t1"], spec["t2"], spec["t3"]
        name = (f"student_II_50hz_forkjoin_{fid}_"
                f"{t1}_to_{t2}_{t3}_simclr_seed{seed}")
        mf = forkjoin_dir / f"metrics_test_{name}.json"
        if mf.exists():
            d = json.load(open(mf))
            results[fid] = {
                "auc":  d["macro_auc"],
                "f1t":  d.get("macro_f1_tuned", d.get("macro_f1", 0)),
                "f10":  d.get("macro_f1_0_5", 0),
                "t1": t1, "t2": t2, "t3": t3,
                "structure": f"{t1}→{{{t2},{t3}}}→S",
                "per_class": {c: d.get(f"auc_{c}", float("nan"))
                              for c in SUPERCLASSES},
            }
    return results


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(out_dir, args.seed)
    n_done  = len(results)
    print(f"Loaded {n_done}/{len(FORKS)} fork-join results.")

    if n_done == 0:
        print("No results found. Run run_fork_join_6.sh first.")
        return

    sorted_by_auc = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)

    # ── main table ────────────────────────────────────────────────────────────
    rows = []
    for rank, (fid, v) in enumerate(sorted_by_auc, 1):
        rows.append({
            "rank":                            rank,
            "fork_id":                         fid,
            "structure":                       v["structure"],
            "t1":                              v["t1"],
            "t2":                              v["t2"],
            "t3":                              v["t3"],
            "auc":                             round(v["auc"], 4),
            "f1_tuned":                        round(v["f1t"], 4),
            "f1_0_5":                          round(v["f10"], 4),
            "delta_auc_vs_bce_50hz":           round(v["auc"] - BCE_AUC,  4),
            "delta_f1_vs_bce_50hz":            round(v["f1t"] - BCE_F1,   4),
            "delta_auc_vs_progressive_simclr": round(v["auc"] - PROG_AUC, 4),
            "delta_f1_vs_progressive_simclr":  round(v["f1t"] - PROG_F1,  4),
            "delta_auc_vs_parallel_c14":       round(v["auc"] - C14_AUC,  4),
            "delta_f1_vs_parallel_c14":        round(v["f1t"] - C14_F1,   4),
            "beats_progressive_simclr":        "YES" if v["auc"] > PROG_AUC else "no",
            "beats_parallel_c14":              "YES" if v["auc"] > C14_AUC  else "no",
            "beats_mvkt_auc":                  "YES" if v["auc"] > MVKT_AUC else "no",
            "beats_mvkt_f1":                   "YES" if v["f1t"] > MVKT_F1  else "no",
        })

    csv_path = out_dir / "fork_join_6_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {csv_path}")

    md_path = out_dir / "fork_join_6_results.md"
    with open(md_path, "w") as f:
        f.write("# Fork-Join Dual Distillation Results\n\n")
        f.write(f"Results: {n_done}/{len(FORKS)} forks complete\n\n")
        f.write("**References:**\n")
        f.write(f"- BCE 50Hz:           AUC={BCE_AUC}  F1={BCE_F1}\n")
        f.write(f"- Progressive+SimCLR: AUC={PROG_AUC} F1={PROG_F1}\n")
        f.write(f"- Parallel C14:       AUC={C14_AUC} F1={C14_F1}\n")
        f.write(f"- MVKT target:        AUC={MVKT_AUC}  F1={MVKT_F1}\n\n")
        f.write("| Rank | Fork | Structure | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |\n")
        f.write("|------|------|-----------|-----|----------|-------------|-------------|-------|------|-------|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['fork_id']} | {r['structure']} | {r['auc']} | "
                    f"{r['f1_tuned']} | {r['delta_auc_vs_bce_50hz']:+.4f} | "
                    f"{r['delta_auc_vs_parallel_c14']:+.4f} | "
                    f"{r['beats_progressive_simclr']} | {r['beats_parallel_c14']} | "
                    f"{r['beats_mvkt_auc']} |\n")
        if n_done < len(FORKS):
            missing = [h for h in FORKS if h not in results]
            f.write(f"\n**Pending ({len(missing)}):** {', '.join(missing)}\n")
    print(f"Saved: {md_path}")

    # ── per-class CSV ─────────────────────────────────────────────────────────
    cw_path = out_dir / "fork_join_classwise.csv"
    cw_rows = []
    for fid, v in sorted_by_auc:
        row = {"fork_id": fid, "structure": v["structure"],
               "macro_auc": round(v["auc"], 4)}
        for c in SUPERCLASSES:
            row[f"auc_{c}"] = round(v["per_class"][c], 4)
        cw_rows.append(row)
    with open(cw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cw_rows[0].keys())
        w.writeheader(); w.writerows(cw_rows)
    print(f"Saved: {cw_path}")

    # ── analysis markdown ─────────────────────────────────────────────────────
    best_id, best_v = sorted_by_auc[0]
    best_f1_id, best_f1_v = max(results.items(), key=lambda x: x[1]["f1t"])

    ana_path = out_dir / "fork_join_analysis.md"
    lines = []
    lines.append("# Fork-Join Dual Distillation Analysis\n\n")
    lines.append(f"Loaded {n_done}/{len(FORKS)} results.\n\n")

    lines.append("## 1. Best Overall\n\n")
    lines.append(f"- Best AUC: **{best_id}** ({best_v['structure']}) "
                 f"AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}\n")
    lines.append(f"- Best F1:  **{best_f1_id}** ({best_f1_v['structure']}) "
                 f"AUC={best_f1_v['auc']:.4f}  F1={best_f1_v['f1t']:.4f}\n\n")

    lines.append("## 2. Reference Comparison\n\n")
    lines.append("| Fork | Structure | AUC | F1_tuned | >Prog | >C14 | >MVKT AUC | >MVKT F1 |\n")
    lines.append("|------|-----------|-----|----------|-------|------|-----------|----------|\n")
    for fid, v in sorted_by_auc:
        lines.append(
            f"| {fid} | {v['structure']} | {v['auc']:.4f} | {v['f1t']:.4f} | "
            f"{'✅' if v['auc']>PROG_AUC else '❌'} | "
            f"{'✅' if v['auc']>C14_AUC  else '❌'} | "
            f"{'✅' if v['auc']>MVKT_AUC else '❌'} | "
            f"{'✅' if v['f1t']>MVKT_F1  else '❌'} |\n"
        )
    lines.append("\n")

    # F02 vs C14
    lines.append("## 3. F02 vs Parallel C14 (same teacher set)\n\n")
    lines.append("F02 = 12L100 → {1L500, 1L100} → S  (fork-join)\n")
    lines.append("C14 = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)\n\n")
    if "F02" in results:
        v = results["F02"]
        lines.append("| Method | AUC | F1_tuned | ΔAUC |\n")
        lines.append("|--------|-----|----------|------|\n")
        lines.append(f"| C14 (parallel) | {C14_AUC} | {C14_F1} | ref |\n")
        lines.append(f"| F02 (fork-join) | {v['auc']:.4f} | {v['f1t']:.4f} | "
                     f"{v['auc']-C14_AUC:+.4f} |\n")
        verdict = ("Fork-join **improves** over parallel with same teachers."
                   if v["auc"] > C14_AUC
                   else "Fork-join does **not** improve over parallel with same teachers.")
        lines.append(f"\n→ {verdict}\n\n")
    else:
        lines.append("F02 not yet available.\n\n")

    # F01 spatial/rate
    lines.append("## 4. F01: Spatial (12L) + Rate (1L) Factorization\n\n")
    lines.append("F01 = 12L100 → {12L50, 1L100} → S\n")
    lines.append("Spatial branch: 12L50 (same leads, lower rate)\n")
    lines.append("Rate branch:    1L100 (fewer leads, same rate)\n\n")
    if "F01" in results:
        v = results["F01"]
        lines.append(f"F01 AUC={v['auc']:.4f}  F1={v['f1t']:.4f}  "
                     f"ΔvsC14={v['auc']-C14_AUC:+.4f}\n\n")
    else:
        lines.append("F01 not yet available.\n\n")

    # 12L500 vs 12L100 as t1
    lines.append("## 5. Upstream t1: 12L500 vs 12L100\n\n")
    with_500  = [(fid, v) for fid, v in results.items() if v["t1"] == "12L500"]
    with_100  = [(fid, v) for fid, v in results.items() if v["t1"] == "12L100"]
    with_1l   = [(fid, v) for fid, v in results.items() if v["t1"] == "1L500"]
    if with_500 and with_100:
        avg_500 = sum(v["auc"] for _, v in with_500) / len(with_500)
        avg_100 = sum(v["auc"] for _, v in with_100) / len(with_100)
        lines.append(f"- t1=12L500 (n={len(with_500)}): mean AUC={avg_500:.4f} "
                     f"({', '.join(f for f,_ in with_500)})\n")
        lines.append(f"- t1=12L100 (n={len(with_100)}): mean AUC={avg_100:.4f} "
                     f"({', '.join(f for f,_ in with_100)})\n")
        lines.append(f"→ {'12L500 helps.' if avg_500 > avg_100 else '12L100 is sufficient.'}\n\n")

    # 1L50 branch effect
    lines.append("## 6. Branch with 1L50 vs without\n\n")
    with_1l50    = [(fid, v) for fid, v in results.items()
                    if v["t2"] == "1L50" or v["t3"] == "1L50"]
    without_1l50 = [(fid, v) for fid, v in results.items()
                    if v["t2"] != "1L50" and v["t3"] != "1L50"]
    if with_1l50 and without_1l50:
        avg_with    = sum(v["auc"] for _, v in with_1l50) / len(with_1l50)
        avg_without = sum(v["auc"] for _, v in without_1l50) / len(without_1l50)
        lines.append(f"- With 1L50 branch (n={len(with_1l50)}): mean AUC={avg_with:.4f} "
                     f"({', '.join(f for f,_ in with_1l50)})\n")
        lines.append(f"- Without 1L50 branch (n={len(without_1l50)}): mean AUC={avg_without:.4f} "
                     f"({', '.join(f for f,_ in without_1l50)})\n")
        lines.append(f"→ {'1L50 branch helps.' if avg_with > avg_without else '1L50 branch hurts.'}\n\n")

    # recommendation
    lines.append("## 7. Recommendation\n\n")
    lines.append(f"Best fork by AUC: **{best_id}** = {best_v['structure']}\n")
    lines.append(f"  AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}\n\n")
    if best_v["auc"] > MVKT_AUC:
        lines.append("✅ **Strong success**: beats MVKT target AUC.\n")
    elif best_v["auc"] > C14_AUC:
        lines.append("✅ **Success 2**: beats parallel C14.\n")
    elif best_v["auc"] > PROG_AUC:
        lines.append("✅ **Success 1**: beats Progressive+SimCLR.\n")
    else:
        lines.append("❌ Does not beat Progressive+SimCLR.\n")
    lines.append(f"\nNext step: run scripts/run_best_fork_join_full.sh "
                 f"with fork_id={best_id}\n")

    with open(ana_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {ana_path}")
    print(f"\nBest: {best_id}  AUC={best_v['auc']:.4f}  F1={best_v['f1t']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output_dir", default="dafd_mvkt/outputs/forkjoin")
    analyze(p.parse_args())
