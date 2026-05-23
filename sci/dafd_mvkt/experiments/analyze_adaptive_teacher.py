"""
Analyze adaptive teacher-student distillation results.

Parses all metrics_test_*.json files from outputs/adaptive/ and generates:
    - adaptive_results.csv   — all methods ranked by AUC
    - adaptive_results.md    — markdown table
    - adaptive_analysis.md   — method-group comparisons

Usage:
    python dafd_mvkt/experiments/analyze_adaptive_teacher.py \\
        --output_dir dafd_mvkt/outputs/adaptive \\
        --seed 0
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

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ── naming helpers ────────────────────────────────────────────────────────────

def _id_from_name(name: str) -> str | None:
    """Return a short experiment ID from the output_name string, or None."""
    # 3-teacher EMA: student_II_50hz_3T_C14_ema_lam005_decay099_seed0
    if "_3T_" in name and "_ema_" in name:
        parts = name.split("_")
        try:
            lam  = parts[parts.index("ema") + 1].replace("lam", "")
            dec  = parts[parts.index("ema") + 2].replace("decay", "")
            return f"3T-EMA lam={lam} dec={dec}"
        except (ValueError, IndexError):
            return name

    # progressive EMA: student_II_50hz_prog_1l100_ema_lam005_decay099_seed0
    if "_prog_1l100_ema_" in name:
        parts = name.split("_")
        try:
            lam = parts[parts.index("ema") + 1].replace("lam", "")
            dec = parts[parts.index("ema") + 2].replace("decay", "")
            return f"PROG-EMA lam={lam} dec={dec}"
        except (ValueError, IndexError):
            return name

    # mutual: student_II_50hz_mut_12l100_lam005_seed0
    if "_mut_" in name:
        parts = name.split("_")
        try:
            idx = parts.index("mut")
            up  = parts[idx + 1]
            lam = parts[idx + 2].replace("lam", "")
            return f"MUT {up} lam={lam}"
        except (ValueError, IndexError):
            return name

    # mutual teacher: teacher_1l100_mut_12l100_lam005_seed0
    if name.startswith("teacher_1l100_mut_"):
        parts = name.split("_")
        try:
            idx = parts.index("mut")
            up  = parts[idx + 1]
            lam = parts[idx + 2].replace("lam", "")
            return f"MUT-T1L100 {up} lam={lam}"
        except (ValueError, IndexError):
            return name

    # student-aware: student_II_50hz_saf_sf02_12l100_to_1l500_1l100_sa005_seed0
    if "_saf_" in name:
        parts = name.split("_")
        try:
            idx = parts.index("saf")
            sf  = parts[idx + 1].upper()
            sa_part = next(p for p in parts if p.startswith("sa") and p[2:].isdigit())
            lam = sa_part.replace("sa", "")
            return f"SAF {sf} sa={lam}"
        except (ValueError, IndexError, StopIteration):
            return name

    return None


def _method_group(name: str) -> str:
    if "_3T_" in name and "_ema_" in name:
        return "3T-EMA"
    if "_prog_1l100_ema_" in name:
        return "PROG-EMA"
    if name.startswith("teacher_1l100_mut_"):
        return "MUT-teacher"
    if "_mut_" in name:
        return "MUT-student"
    if "_saf_" in name and "_t2" not in name and "_t3" not in name:
        return "SAF-student"
    if "_saf_" in name and "_t2" in name:
        return "SAF-t2"
    if "_saf_" in name and "_t3" in name:
        return "SAF-t3"
    return "other"


def load_results(out_dir: Path) -> list[dict]:
    rows = []
    for mf in sorted(out_dir.glob("metrics_test_*.json")):
        try:
            d = json.load(open(mf))
        except (json.JSONDecodeError, OSError):
            continue
        name  = d.get("output_name", mf.stem.replace("metrics_test_", ""))
        group = _method_group(name)
        short = _id_from_name(name) or name
        rows.append({
            "name":  name,
            "short": short,
            "group": group,
            "auc":   d.get("macro_auc", float("nan")),
            "f1t":   d.get("macro_f1_tuned", d.get("macro_f1", 0)),
            "f10":   d.get("macro_f1_0_5", 0),
            "val_auc": d.get("val_auc_best", float("nan")),
            **{f"auc_{c}": d.get(f"auc_{c}", float("nan")) for c in SUPERCLASSES},
        })
    return rows


def check(auc: float, ref: float) -> str:
    return "✅" if auc > ref else "❌"


def fmt(auc: float, ref: float) -> str:
    return f"{auc - ref:+.4f}"


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"[WARN] output_dir not found: {out_dir}")
        return

    rows = load_results(out_dir)
    n_total = len(rows)

    # filter to student-facing rows for ranking
    student_rows = [r for r in rows if r["group"] in ("3T-EMA", "PROG-EMA",
                                                        "MUT-student", "SAF-student")]
    student_rows.sort(key=lambda r: r["auc"], reverse=True)

    print(f"\nLoaded {n_total} results ({len(student_rows)} student-facing).")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / "adaptive_results.csv"
    fieldnames = ["rank", "group", "short", "auc", "f1t", "f10", "val_auc",
                  ">Prog", ">C14", ">MVKT_AUC", ">MVKT_F1", "name"]
    all_rows_sorted = sorted(rows, key=lambda r: r["auc"], reverse=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(all_rows_sorted, 1):
            w.writerow({
                "rank":      rank,
                "group":     r["group"],
                "short":     r["short"],
                "auc":       r["auc"],
                "f1t":       r["f1t"],
                "f10":       r["f10"],
                "val_auc":   r["val_auc"],
                ">Prog":     "YES" if r["auc"] > PROG_AUC else "no",
                ">C14":      "YES" if r["auc"] > C14_AUC  else "no",
                ">MVKT_AUC": "YES" if r["auc"] > MVKT_AUC else "no",
                ">MVKT_F1":  "YES" if r["f1t"] > MVKT_F1  else "no",
                "name":      r["name"],
            })
    print(f"Saved: {csv_path}")

    # ── results MD ───────────────────────────────────────────────────────────
    md_results = out_dir / "adaptive_results.md"
    lines = [
        "# Adaptive Teacher-Student Distillation Results",
        "",
        f"Results: {len(student_rows)} student-facing experiments",
        "",
        "**References:**",
        f"- BCE 50Hz:           AUC={BCE_AUC}  F1={BCE_F1}",
        f"- Progressive+SimCLR: AUC={PROG_AUC} F1={PROG_F1}",
        f"- Parallel C14:       AUC={C14_AUC}  F1={C14_F1}",
        f"- MVKT target:        AUC={MVKT_AUC} F1={MVKT_F1}",
        "",
        "| Rank | Group | Method | AUC | F1_tuned | >Prog | >C14 | >MVKT |",
        "|------|-------|--------|-----|----------|-------|------|-------|",
    ]
    for rank, r in enumerate(student_rows, 1):
        lines.append(
            f"| {rank} | {r['group']} | {r['short']} | "
            f"{r['auc']:.4f} | {r['f1t']:.4f} | "
            f"{'YES' if r['auc']>PROG_AUC else 'no'} | "
            f"{'YES' if r['auc']>C14_AUC  else 'no'} | "
            f"{'YES' if r['auc']>MVKT_AUC else 'no'} |"
        )

    md_results.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_results}")

    # ── analysis MD ──────────────────────────────────────────────────────────
    md_analysis = out_dir / "adaptive_analysis.md"
    alines = [
        "# Adaptive Teacher-Student Distillation Analysis",
        "",
    ]

    # best overall
    if student_rows:
        best = student_rows[0]
        alines += [
            "## 1. Best Overall (student-facing)",
            "",
            f"- Best AUC: **{best['short']}**  AUC={best['auc']:.4f}  F1={best['f1t']:.4f}",
            f"  ({'beats MVKT AUC' if best['auc']>MVKT_AUC else 'does not beat MVKT AUC'})",
            "",
        ]

    # group summaries
    groups = ["3T-EMA", "PROG-EMA", "MUT-student", "SAF-student"]
    alines.append("## 2. Group Summaries")
    alines.append("")
    for g in groups:
        g_rows = [r for r in rows if r["group"] == g]
        if not g_rows:
            alines.append(f"### {g}: no results yet")
            alines.append("")
            continue
        g_rows_s = sorted(g_rows, key=lambda r: r["auc"], reverse=True)
        best_g = g_rows_s[0]
        mean_auc = sum(r["auc"] for r in g_rows) / len(g_rows)
        alines.append(f"### {g} (n={len(g_rows)})")
        alines.append(f"- Best AUC: {best_g['auc']:.4f}  F1={best_g['f1t']:.4f}  ({best_g['short']})")
        alines.append(f"- Mean AUC: {mean_auc:.4f}")
        alines.append("")
        alines.append(f"| Method | AUC | F1_tuned | >Prog | >C14 | >MVKT |")
        alines.append(f"|--------|-----|----------|-------|------|-------|")
        for r in g_rows_s:
            alines.append(
                f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} | "
                f"{'✅' if r['auc']>PROG_AUC else '❌'} | "
                f"{'✅' if r['auc']>C14_AUC  else '❌'} | "
                f"{'✅' if r['auc']>MVKT_AUC else '❌'} |"
            )
        alines.append("")

    # EMA decay comparison within 3T-EMA
    alines.append("## 3. EMA Regularization Effect (3T-EMA vs C14 baseline)")
    alines.append("")
    alines.append(f"Reference — C14 (parallel 3-teacher): AUC={C14_AUC}  F1={C14_F1}")
    alines.append("")
    t3e_rows = [r for r in rows if r["group"] == "3T-EMA"]
    if t3e_rows:
        t3e_rows.sort(key=lambda r: r["auc"], reverse=True)
        alines.append("| Method | AUC | F1_tuned | ΔAUC vs C14 |")
        alines.append("|--------|-----|----------|-------------|")
        for r in t3e_rows:
            alines.append(f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} "
                          f"| {r['auc']-C14_AUC:+.4f} |")
        alines.append("")
        best_t3e = t3e_rows[0]
        if best_t3e["auc"] > C14_AUC:
            alines.append(f"→ EMA regularization **improves** over C14: +{best_t3e['auc']-C14_AUC:.4f} AUC")
        else:
            alines.append(f"→ EMA regularization **does not improve** over C14: "
                          f"{best_t3e['auc']-C14_AUC:.4f} AUC")
        alines.append("")
    else:
        alines.append("No 3T-EMA results yet.\n")

    # PROG-EMA vs PROG baseline
    alines.append("## 4. PROG-EMA vs Progressive Baseline")
    alines.append("")
    alines.append(f"Reference — Progressive+SimCLR: AUC={PROG_AUC}  F1={PROG_F1}")
    alines.append("")
    pe_rows = [r for r in rows if r["group"] == "PROG-EMA"]
    if pe_rows:
        pe_rows.sort(key=lambda r: r["auc"], reverse=True)
        alines.append("| Method | AUC | F1_tuned | ΔAUC vs Prog |")
        alines.append("|--------|-----|----------|--------------|")
        for r in pe_rows:
            alines.append(f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} "
                          f"| {r['auc']-PROG_AUC:+.4f} |")
        best_pe = pe_rows[0]
        alines.append("")
        if best_pe["auc"] > PROG_AUC:
            alines.append(f"→ EMA regularization **improves** progressive KD: "
                          f"+{best_pe['auc']-PROG_AUC:.4f} AUC")
        else:
            alines.append(f"→ EMA regularization **does not improve** progressive KD: "
                          f"{best_pe['auc']-PROG_AUC:.4f} AUC")
        alines.append("")
    else:
        alines.append("No PROG-EMA results yet.\n")

    # Mutual learning analysis
    alines.append("## 5. Mutual Learning Analysis")
    alines.append("")
    mut_s = [r for r in rows if r["group"] == "MUT-student"]
    mut_t = [r for r in rows if r["group"] == "MUT-teacher"]
    if mut_s:
        mut_s.sort(key=lambda r: r["auc"], reverse=True)
        alines.append("### Student (1L50):")
        alines.append("| Method | AUC | F1_tuned | ΔAUC vs Prog |")
        alines.append("|--------|-----|----------|--------------|")
        for r in mut_s:
            alines.append(f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} "
                          f"| {r['auc']-PROG_AUC:+.4f} |")
        alines.append("")
    if mut_t:
        mut_t.sort(key=lambda r: r["auc"], reverse=True)
        alines.append("### Jointly-trained 1L100 teacher:")
        alines.append("| Method | AUC | F1_tuned |")
        alines.append("|--------|-----|----------|")
        for r in mut_t:
            alines.append(f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} |")
        alines.append("")
    if not mut_s and not mut_t:
        alines.append("No mutual learning results yet.\n")

    # Student-aware SAF analysis
    alines.append("## 6. Student-Aware Branch Distillation (SAF)")
    alines.append("")
    saf_s = [r for r in rows if r["group"] == "SAF-student"]
    if saf_s:
        saf_s.sort(key=lambda r: r["auc"], reverse=True)
        alines.append("| Method | AUC | F1_tuned | ΔAUC vs MVKT |")
        alines.append("|--------|-----|----------|--------------|")
        for r in saf_s:
            alines.append(f"| {r['short']} | {r['auc']:.4f} | {r['f1t']:.4f} "
                          f"| {r['auc']-MVKT_AUC:+.4f} |")
        alines.append("")
    else:
        alines.append("No SAF results yet.\n")

    # final recommendation
    alines.append("## 7. Recommendation")
    alines.append("")
    if student_rows:
        best = student_rows[0]
        mvkt_ok = best["auc"] > MVKT_AUC
        alines.append(f"Best adaptive method: **{best['short']}**")
        alines.append(f"  AUC={best['auc']:.4f}  F1_tuned={best['f1t']:.4f}")
        alines.append(f"  ΔAUC vs Prog={best['auc']-PROG_AUC:+.4f}  "
                      f"ΔAUC vs MVKT={best['auc']-MVKT_AUC:+.4f}")
        if mvkt_ok:
            alines.append("")
            alines.append("✅ **Strong success**: beats MVKT AUC target.")
        else:
            alines.append("")
            alines.append("❌ Does not beat MVKT AUC target.")
    else:
        alines.append("No results to rank yet.")

    md_analysis.write_text("\n".join(alines) + "\n")
    print(f"Saved: {md_analysis}")

    # console summary
    print("\n" + "="*60)
    print("Adaptive Experiments Summary")
    print("="*60)
    if student_rows:
        print(f"{'Rank':<5} {'Group':<15} {'Method':<35} {'AUC':<7} {'F1t':<7} {'MVKT?'}")
        print("-"*80)
        for rank, r in enumerate(student_rows[:10], 1):
            mvkt = "YES ✓" if r["auc"] > MVKT_AUC else "no"
            print(f"{rank:<5} {r['group']:<15} {r['short']:<35} "
                  f"{r['auc']:.4f} {r['f1t']:.4f} {mvkt}")
    else:
        print("No student-facing results found yet.")
    print("="*60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="dafd_mvkt/outputs/adaptive")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
