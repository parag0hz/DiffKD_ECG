"""
Analyze hierarchical chain KD results.

Usage:
    python dafd_mvkt/experiments/analyze_hierarchical_chains.py \\
        --seed 0 \\
        --output_dir dafd_mvkt/outputs/hierchain
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

CHAINS = {
    "H01": ["12L500", "1L500",  "1L100"],
    "H02": ["12L100", "1L500",  "1L100"],
    "H03": ["1L500",  "1L100",  "1L50"],
    "H04": ["12L100", "1L100",  "1L50"],
    "H05": ["12L100", "12L50",  "1L50"],
    "H06": ["12L500", "12L100", "1L100"],
}

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_results(hierchain_dir: Path, seed: int) -> dict[str, dict]:
    results = {}
    for hid, views in CHAINS.items():
        t1, t2, t3 = views
        name = f"student_II_50hz_hier_{hid}_{t1}_{t2}_{t3}_simclr_seed{seed}"
        mf   = hierchain_dir / f"metrics_test_{name}.json"
        if mf.exists():
            d = json.load(open(mf))
            results[hid] = {
                "auc":     d["macro_auc"],
                "f1t":     d.get("macro_f1_tuned", d.get("macro_f1", 0)),
                "f10":     d.get("macro_f1_0_5", 0),
                "chain":   f"{t1}→{t2}→{t3}→S",
                "views":   views,
                "per_class": {c: d.get(f"auc_{c}", float("nan")) for c in SUPERCLASSES},
            }
    return results


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(out_dir, args.seed)
    n_done  = len(results)
    print(f"Loaded {n_done}/{len(CHAINS)} chain results.")

    if n_done == 0:
        print("No results found. Run run_hierarchical_6chains.sh first.")
        return

    sorted_by_auc = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)

    # ── main results table ────────────────────────────────────────────────────
    rows = []
    for rank, (hid, v) in enumerate(sorted_by_auc, 1):
        rows.append({
            "rank":                       rank,
            "chain_id":                   hid,
            "chain":                      v["chain"],
            "auc":                        round(v["auc"], 4),
            "f1_tuned":                   round(v["f1t"], 4),
            "f1_0_5":                     round(v["f10"], 4),
            "delta_auc_vs_bce_50hz":      round(v["auc"] - BCE_AUC,  4),
            "delta_f1_vs_bce_50hz":       round(v["f1t"] - BCE_F1,   4),
            "delta_auc_vs_progressive_simclr": round(v["auc"] - PROG_AUC, 4),
            "delta_f1_vs_progressive_simclr":  round(v["f1t"] - PROG_F1,  4),
            "delta_auc_vs_parallel_c14":  round(v["auc"] - C14_AUC,  4),
            "delta_f1_vs_parallel_c14":   round(v["f1t"] - C14_F1,   4),
            "beats_progressive_simclr":   "YES" if v["auc"] > PROG_AUC else "no",
            "beats_parallel_c14":         "YES" if v["auc"] > C14_AUC  else "no",
            "beats_mvkt_auc":             "YES" if v["auc"] > MVKT_AUC  else "no",
            "beats_mvkt_f1":              "YES" if v["f1t"] > MVKT_F1   else "no",
        })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / "hierarchical_6chains_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {csv_path}")

    # ── markdown results ──────────────────────────────────────────────────────
    md_path = out_dir / "hierarchical_6chains_results.md"
    with open(md_path, "w") as f:
        f.write("# Hierarchical 6-Chain KD Results\n\n")
        f.write(f"Results: {n_done}/{len(CHAINS)} chains complete\n\n")
        f.write("**References:**\n")
        f.write(f"- BCE 50Hz:              AUC={BCE_AUC}  F1={BCE_F1}\n")
        f.write(f"- Progressive+SimCLR:    AUC={PROG_AUC} F1={PROG_F1}\n")
        f.write(f"- Parallel 3T C14:       AUC={C14_AUC} F1={C14_F1}\n")
        f.write(f"- MVKT target:           AUC={MVKT_AUC}  F1={MVKT_F1}\n\n")
        f.write("| Rank | Chain | Path | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |\n")
        f.write("|------|-------|------|-----|----------|-------------|-------------|-------|------|-------|\n")
        for r in rows:
            f.write(f"| {r['rank']} | {r['chain_id']} | {r['chain']} | {r['auc']} | "
                    f"{r['f1_tuned']} | {r['delta_auc_vs_bce_50hz']:+.4f} | "
                    f"{r['delta_auc_vs_parallel_c14']:+.4f} | "
                    f"{r['beats_progressive_simclr']} | {r['beats_parallel_c14']} | "
                    f"{r['beats_mvkt_auc']} |\n")
        if n_done < len(CHAINS):
            missing = [h for h in CHAINS if h not in results]
            f.write(f"\n**Pending ({len(missing)}):** {', '.join(missing)}\n")
    print(f"Saved: {md_path}")

    # ── per-class AUC ─────────────────────────────────────────────────────────
    cw_path = out_dir / "hierarchical_6chains_classwise.csv"
    cw_rows = []
    for hid, v in sorted_by_auc:
        row = {"chain_id": hid, "chain": v["chain"], "macro_auc": round(v["auc"], 4)}
        for c in SUPERCLASSES:
            row[f"auc_{c}"] = round(v["per_class"][c], 4)
        cw_rows.append(row)
    with open(cw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cw_rows[0].keys())
        w.writeheader(); w.writerows(cw_rows)
    print(f"Saved: {cw_path}")

    # ── analysis markdown ─────────────────────────────────────────────────────
    best_auc_id, best_auc_v = sorted_by_auc[0]
    best_f1_id, best_f1_v   = max(results.items(), key=lambda x: x[1]["f1t"])

    ana_path = out_dir / "hierarchical_chain_analysis.md"
    lines = []
    lines.append("# Hierarchical Chain KD Analysis\n\n")
    lines.append(f"Loaded {n_done}/{len(CHAINS)} results.\n\n")

    # 1. Best overall
    lines.append("## 1. Best Overall\n\n")
    lines.append(f"- Best AUC:    **{best_auc_id}** ({best_auc_v['chain']}) "
                 f"AUC={best_auc_v['auc']:.4f}  F1={best_auc_v['f1t']:.4f}\n")
    lines.append(f"- Best F1:     **{best_f1_id}** ({best_f1_v['chain']}) "
                 f"AUC={best_f1_v['auc']:.4f}  F1={best_f1_v['f1t']:.4f}\n\n")

    # 2. Beats references?
    lines.append("## 2. Reference Comparison\n\n")
    lines.append("| Chain | AUC | F1_tuned | >Prog+SimCLR | >Parallel C14 | >MVKT AUC | >MVKT F1 |\n")
    lines.append("|-------|-----|----------|-------------|---------------|-----------|----------|\n")
    for hid, v in sorted_by_auc:
        lines.append(f"| {hid} ({v['chain']}) | {v['auc']:.4f} | {v['f1t']:.4f} | "
                     f"{'✅' if v['auc']>PROG_AUC else '❌'} | "
                     f"{'✅' if v['auc']>C14_AUC else '❌'} | "
                     f"{'✅' if v['auc']>MVKT_AUC else '❌'} | "
                     f"{'✅' if v['f1t']>MVKT_F1 else '❌'} |\n")
    lines.append("\n")

    # 3. H02 vs C14 (same teacher set)
    lines.append("## 3. H02 vs Parallel C14 (same teacher set)\n\n")
    lines.append("H02 = 12L100 → 1L500 → 1L100 → S  (hierarchical)\n")
    lines.append("C14 = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)\n\n")
    if "H02" in results:
        h02 = results["H02"]
        lines.append(f"| | AUC | F1_tuned | ΔAUC |\n")
        lines.append(f"|--|-----|----------|------|\n")
        lines.append(f"| C14 (parallel) | {C14_AUC} | {C14_F1} | ref |\n")
        lines.append(f"| H02 (hier) | {h02['auc']:.4f} | {h02['f1t']:.4f} | "
                     f"{h02['auc']-C14_AUC:+.4f} |\n")
        verdict = "Hierarchy **improves** over parallel KD." if h02["auc"] > C14_AUC \
            else "Hierarchy does **not** improve over parallel KD."
        lines.append(f"\n→ {verdict}\n\n")
    else:
        lines.append("H02 result not yet available.\n\n")

    # 4. 12L500 chains analysis
    lines.append("## 4. Chains Starting with 12L500\n\n")
    with_12l500  = [(h, v) for h, v in results.items() if v["views"][0] == "12L500"]
    without_12l500 = [(h, v) for h, v in results.items() if v["views"][0] != "12L500"]
    if with_12l500 and without_12l500:
        avg_with    = sum(v["auc"] for _, v in with_12l500) / len(with_12l500)
        avg_without = sum(v["auc"] for _, v in without_12l500) / len(without_12l500)
        lines.append(f"- With 12L500 t1 (n={len(with_12l500)}): mean AUC={avg_with:.4f}\n")
        lines.append(f"- Without 12L500 t1 (n={len(without_12l500)}): mean AUC={avg_without:.4f}\n")
        verdict = "12L500 as t1 helps." if avg_with > avg_without else "12L500 as t1 does not help."
        lines.append(f"→ {verdict}\n\n")

    # 5. Chains ending with 1L100 vs 1L50 before student
    lines.append("## 5. t3 View Before Student: 1L100 vs 1L50\n\n")
    end_1l100 = [(h, v) for h, v in results.items() if v["views"][2] == "1L100"]
    end_1l50  = [(h, v) for h, v in results.items() if v["views"][2] == "1L50"]
    if end_1l100 and end_1l50:
        avg_1l100 = sum(v["auc"] for _, v in end_1l100) / len(end_1l100)
        avg_1l50  = sum(v["auc"] for _, v in end_1l50)  / len(end_1l50)
        lines.append(f"- t3=1L100 chains (n={len(end_1l100)}): mean AUC={avg_1l100:.4f} "
                     f"({', '.join(h for h, _ in end_1l100)})\n")
        lines.append(f"- t3=1L50 chains  (n={len(end_1l50)}):  mean AUC={avg_1l50:.4f} "
                     f"({', '.join(h for h, _ in end_1l50)})\n")
        verdict = "1L100 as t3 is better." if avg_1l100 > avg_1l50 else "1L50 as t3 is better."
        lines.append(f"→ {verdict}\n\n")

    # 6. Recommendation
    lines.append("## 6. Recommendation\n\n")
    lines.append(f"Best chain by AUC: **{best_auc_id}** = {best_auc_v['chain']}\n")
    lines.append(f"  AUC={best_auc_v['auc']:.4f}  F1={best_auc_v['f1t']:.4f}\n\n")
    if best_auc_v["auc"] > MVKT_AUC:
        lines.append("✅ **Strong success**: beats MVKT target AUC.\n")
    elif best_auc_v["auc"] > C14_AUC:
        lines.append("✅ **Success 2**: beats parallel C14.\n")
    elif best_auc_v["auc"] > PROG_AUC:
        lines.append("✅ **Success 1**: beats Progressive+SimCLR.\n")
    else:
        lines.append("❌ Does not beat Progressive+SimCLR. Consider adding CRF/FeatureKD.\n")
    lines.append(f"\nNext step: run scripts/run_best_hierarchical_chain_full.sh "
                 f"with chain_id={best_auc_id}\n")

    with open(ana_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {ana_path}")

    # summary print
    print(f"\nBest: {best_auc_id}  AUC={best_auc_v['auc']:.4f}  F1={best_auc_v['f1t']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--output_dir", default="dafd_mvkt/outputs/hierchain")
    analyze(p.parse_args())
