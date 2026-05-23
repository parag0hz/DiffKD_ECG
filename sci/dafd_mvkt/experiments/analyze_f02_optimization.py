"""
Analyze F02 optimization experiment results.

Usage:
    python dafd_mvkt/experiments/analyze_f02_optimization.py \\
        --output_dir dafd_mvkt/outputs/f02_opt
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

F02_AUC  = 0.8447;  F02_F1  = 0.6275
C14_AUC  = 0.8425;  C14_F1  = 0.6243
PROG_AUC = 0.8396;  PROG_F1 = 0.6176
MVKT_AUC = 0.843;   MVKT_F1 = 0.626
BCE_AUC  = 0.8061;  BCE_F1  = 0.5740

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _variant_type(vid: str) -> str:
    if re.match(r"F02W\d+", vid):  return "weight"
    if re.match(r"F02F\d+", vid):  return "crf_feature"
    if re.match(r"F02T\d+", vid):  return "temperature"
    if re.match(r"F02BR\d+", vid): return "branch_enhance"
    if re.match(r"F02CW\d+", vid): return "confidence_weighted"
    return "other"


def load_results(out_dir: Path) -> list[dict]:
    rows = []
    for mf in sorted(out_dir.glob("metrics_test_*.json")):
        try:
            d = json.load(open(mf))
        except Exception:
            continue
        name = d.get("output_name", mf.stem.replace("metrics_test_", ""))
        # extract variant_id from name or json
        vid = d.get("variant_id", "")
        if not vid:
            m = re.search(r"f02_(F02\w+)_seed", name)
            vid = m.group(1) if m else name
        rows.append({
            "name":         name,
            "variant_id":   vid,
            "variant_type": _variant_type(vid),
            "w500":         d.get("w500", float("nan")),
            "w100":         d.get("w100", float("nan")),
            "temperature":  d.get("temperature", 2.0),
            "use_crf":      d.get("use_crf", False),
            "beta_crf":     d.get("beta_crf", 0.0),
            "use_feat":     d.get("use_feature_kd", False),
            "gamma_feat":   d.get("gamma_feature", 0.0),
            "tw_mode":      d.get("tw_mode", "fixed"),
            "auc":          d.get("macro_auc", float("nan")),
            "f1t":          d.get("macro_f1_tuned", d.get("macro_f1", 0)),
            "f10":          d.get("macro_f1_0_5", 0),
            "val_auc":      d.get("val_auc_best", float("nan")),
            **{f"auc_{c}":    d.get(f"auc_{c}", float("nan"))    for c in SUPERCLASSES},
            **{f"f1t_{c}":    d.get(f"f1_tuned_{c}", float("nan")) for c in SUPERCLASSES},
        })
    return rows


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"[WARN] {out_dir} not found"); return

    rows = load_results(out_dir)
    if not rows:
        print("No results found."); return

    rows.sort(key=lambda r: r["auc"], reverse=True)
    print(f"Loaded {len(rows)} results.")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_cols = ["rank", "variant_id", "variant_type",
                "w500", "w100", "temperature", "beta_crf", "gamma_feat", "tw_mode",
                "auc", "f1t", "f10", "val_auc",
                "delta_auc_f02", "delta_f1_f02", "delta_auc_c14",
                "beats_f02", "beats_c14", "beats_mvkt_auc", "beats_mvkt_f1",
                *[f"auc_{c}" for c in SUPERCLASSES],
                "name"]
    csv_rows = []
    for rank, r in enumerate(rows, 1):
        csv_rows.append({
            "rank":           rank,
            "variant_id":     r["variant_id"],
            "variant_type":   r["variant_type"],
            "w500":           round(r["w500"], 3) if r["w500"] == r["w500"] else "",
            "w100":           round(r["w100"], 3) if r["w100"] == r["w100"] else "",
            "temperature":    r["temperature"],
            "beta_crf":       r["beta_crf"],
            "gamma_feat":     r["gamma_feat"],
            "tw_mode":        r["tw_mode"],
            "auc":            r["auc"],
            "f1t":            r["f1t"],
            "f10":            r["f10"],
            "val_auc":        r["val_auc"],
            "delta_auc_f02":  round(r["auc"] - F02_AUC, 5),
            "delta_f1_f02":   round(r["f1t"] - F02_F1, 5),
            "delta_auc_c14":  round(r["auc"] - C14_AUC, 5),
            "beats_f02":      "YES" if r["auc"] > F02_AUC else "no",
            "beats_c14":      "YES" if r["auc"] > C14_AUC else "no",
            "beats_mvkt_auc": "YES" if r["auc"] > MVKT_AUC else "no",
            "beats_mvkt_f1":  "YES" if r["f1t"] > MVKT_F1  else "no",
            **{f"auc_{c}": r[f"auc_{c}"] for c in SUPERCLASSES},
            "name":           r["name"],
        })
    csv_path = out_dir / "f02_optimization_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader(); w.writerows(csv_rows)
    print(f"Saved: {csv_path}")

    # ── classwise CSV ─────────────────────────────────────────────────────────
    cw_cols = ["variant_id"] + [f"auc_{c}" for c in SUPERCLASSES] + [f"f1t_{c}" for c in SUPERCLASSES]
    cw_path = out_dir / "f02_optimization_classwise.csv"
    with open(cw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cw_cols)
        w.writeheader()
        for r in rows:
            w.writerow({"variant_id": r["variant_id"],
                        **{f"auc_{c}": round(r[f"auc_{c}"], 5) for c in SUPERCLASSES},
                        **{f"f1t_{c}": round(r[f"f1t_{c}"], 5) for c in SUPERCLASSES}})
    print(f"Saved: {cw_path}")

    # ── results MD ───────────────────────────────────────────────────────────
    md_lines = [
        "# F02 Optimization Results",
        "",
        f"Results: {len(rows)} variants",
        "",
        "**References:**",
        f"- BCE 50Hz:            AUC={BCE_AUC}  F1={BCE_F1}",
        f"- Progressive+SimCLR:  AUC={PROG_AUC} F1={PROG_F1}",
        f"- Parallel C14:        AUC={C14_AUC}  F1={C14_F1}",
        f"- **F02 original**:    AUC={F02_AUC}  F1={F02_F1}",
        f"- MVKT target:         AUC={MVKT_AUC} F1={MVKT_F1}",
        "",
        "| Rank | ID | Type | w500 | w100 | T | β_crf | γ_feat | AUC | F1_tuned | ΔvF02 | >F02 | >MVKT |",
        "|------|-----|------|------|------|---|-------|--------|-----|----------|-------|------|-------|",
    ]
    for rank, r in enumerate(rows, 1):
        w5 = f"{r['w500']:.2f}" if r["w500"] == r["w500"] else "-"
        w1 = f"{r['w100']:.2f}" if r["w100"] == r["w100"] else "-"
        md_lines.append(
            f"| {rank} | {r['variant_id']} | {r['variant_type']} | "
            f"{w5} | {w1} | {r['temperature']} | {r['beta_crf']} | {r['gamma_feat']} | "
            f"{r['auc']:.4f} | {r['f1t']:.4f} | "
            f"{r['auc']-F02_AUC:+.4f} | "
            f"{'✅' if r['auc']>F02_AUC else '❌'} | "
            f"{'✅' if r['auc']>MVKT_AUC else '❌'} |"
        )
    (out_dir / "f02_optimization_results.md").write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {out_dir}/f02_optimization_results.md")

    # ── analysis MD ──────────────────────────────────────────────────────────
    best = rows[0]
    al = [
        "# F02 Optimization Analysis",
        "",
        f"## 1. Best Overall",
        "",
        f"- **{best['variant_id']}** ({best['variant_type']})",
        f"  AUC={best['auc']:.4f}  F1_tuned={best['f1t']:.4f}",
        f"  ΔAUC vs F02 = {best['auc']-F02_AUC:+.4f}",
        f"  Beats F02:  {'YES ✓' if best['auc']>F02_AUC else 'no'}",
        f"  Beats MVKT: {'YES ✓' if best['auc']>MVKT_AUC else 'no'}",
        "",
    ]

    # Q1: best weight ratio
    w_rows = [r for r in rows if r["variant_type"] == "weight"]
    al.append("## 2. Best Teacher Weight Ratio (Phase 2)")
    al.append("")
    if w_rows:
        w_rows.sort(key=lambda r: r["auc"], reverse=True)
        al.append("| ID | w500 | w100 | AUC | F1_tuned | ΔvF02 |")
        al.append("|-----|------|------|-----|----------|-------|")
        for r in w_rows:
            al.append(f"| {r['variant_id']} | {r['w500']:.2f} | {r['w100']:.2f} | "
                      f"{r['auc']:.4f} | {r['f1t']:.4f} | {r['auc']-F02_AUC:+.4f} |")
        bw = w_rows[0]
        al.append("")
        al.append(f"→ Best weight: **{bw['variant_id']}** w500={bw['w500']:.2f} w100={bw['w100']:.2f}")
    else:
        al.append("No weight tuning results yet.")
    al.append("")

    # Q2/Q3: CRF/FeatureKD
    f_rows = [r for r in rows if r["variant_type"] == "crf_feature"]
    al.append("## 3. CRF / FeatureKD Effect (Phase 3)")
    al.append("")
    if f_rows:
        f_rows.sort(key=lambda r: r["auc"], reverse=True)
        al.append("| ID | CRF | β_crf | Feat | γ_feat | AUC | F1_tuned | ΔvF02 |")
        al.append("|-----|-----|-------|------|--------|-----|----------|-------|")
        for r in f_rows:
            al.append(f"| {r['variant_id']} | {'✓' if r['use_crf'] else '-'} | "
                      f"{r['beta_crf']} | {'✓' if r['use_feat'] else '-'} | "
                      f"{r['gamma_feat']} | {r['auc']:.4f} | {r['f1t']:.4f} | "
                      f"{r['auc']-F02_AUC:+.4f} |")
        bf = f_rows[0]
        improves = bf["auc"] > F02_AUC
        al.append("")
        al.append(f"→ Best CRF/Feat variant: **{bf['variant_id']}**  "
                  f"AUC={bf['auc']:.4f}  "
                  f"{'improves over F02 ✓' if improves else 'does not improve over F02'}")
    else:
        al.append("No CRF/Feature results yet.")
    al.append("")

    # Q4: temperature
    t_rows = [r for r in rows if r["variant_type"] == "temperature"]
    al.append("## 4. Temperature Effect (Phase 4)")
    al.append("")
    if t_rows:
        t_rows.sort(key=lambda r: r["auc"], reverse=True)
        al.append("| ID | T | AUC | F1_tuned | ΔvF02 |")
        al.append("|-----|---|-----|----------|-------|")
        for r in t_rows:
            al.append(f"| {r['variant_id']} | {r['temperature']} | "
                      f"{r['auc']:.4f} | {r['f1t']:.4f} | {r['auc']-F02_AUC:+.4f} |")
        bt = t_rows[0]
        al.append(f"\n→ Best temperature: **T={bt['temperature']}** ({bt['variant_id']})")
    else:
        al.append("No temperature tuning results yet.")
    al.append("")

    # Q5: branch enhancement
    br_rows = [r for r in rows if r["variant_type"] == "branch_enhance"]
    al.append("## 5. Branch CRF Enhancement (Phase 5)")
    al.append("")
    if br_rows:
        br_rows.sort(key=lambda r: r["auc"], reverse=True)
        al.append("| ID | AUC | F1_tuned | ΔvF02 |")
        al.append("|-----|-----|----------|-------|")
        for r in br_rows:
            al.append(f"| {r['variant_id']} | {r['auc']:.4f} | {r['f1t']:.4f} | "
                      f"{r['auc']-F02_AUC:+.4f} |")
    else:
        al.append("No branch enhancement results yet.")
    al.append("")

    # Q6: confidence weighting
    cw_rows = [r for r in rows if r["variant_type"] == "confidence_weighted"]
    al.append("## 6. Confidence-Weighted KD (Phase 6)")
    al.append("")
    if cw_rows:
        cw_rows.sort(key=lambda r: r["auc"], reverse=True)
        for r in cw_rows:
            al.append(f"- **{r['variant_id']}** mode={r['tw_mode']}  "
                      f"AUC={r['auc']:.4f}  F1={r['f1t']:.4f}  ΔAUC={r['auc']-F02_AUC:+.4f}")
    else:
        al.append("No confidence-weighted results yet.")
    al.append("")

    # Q10: per-class improvement
    al.append("## 7. Per-Class Analysis (vs F02)")
    al.append("")
    al.append(f"Reference F02 per-class AUC not stored — showing best variant per-class AUC:")
    al.append("")
    al.append("| Variant | " + " | ".join(SUPERCLASSES) + " |")
    al.append("|---------|" + "---|" * len(SUPERCLASSES))
    for r in rows[:5]:
        vals = " | ".join(f"{r[f'auc_{c}']:.4f}" for c in SUPERCLASSES)
        al.append(f"| {r['variant_id']} | {vals} |")
    al.append("")

    # final recommendation
    al.append("## 8. Final Recommendation")
    al.append("")
    if best["auc"] > F02_AUC:
        al.append(f"✅ **{best['variant_id']}** improves over F02 by "
                  f"{best['auc']-F02_AUC:+.4f} AUC, {best['f1t']-F02_F1:+.4f} F1")
        al.append(f"   AUC={best['auc']:.4f}  F1_tuned={best['f1t']:.4f}")
        if best["auc"] > MVKT_AUC:
            al.append("   ✅ Also beats MVKT AUC target.")
    else:
        al.append(f"❌ No variant improves over F02 (best: {best['variant_id']} "
                  f"AUC={best['auc']:.4f} vs F02={F02_AUC}).")
        al.append("   Report F02 original as final best.")
    al.append("")

    (out_dir / "f02_optimization_analysis.md").write_text("\n".join(al) + "\n")
    print(f"Saved: {out_dir}/f02_optimization_analysis.md")

    # console summary
    print("\n" + "="*60)
    print("F02 Optimization Summary")
    print("="*60)
    print(f"{'Rank':<5} {'ID':<10} {'Type':<18} {'AUC':<8} {'F1t':<8} {'ΔvF02':<8} {'Beats F02'}")
    print("-"*70)
    for rank, r in enumerate(rows[:10], 1):
        print(f"{rank:<5} {r['variant_id']:<10} {r['variant_type']:<18} "
              f"{r['auc']:.4f}  {r['f1t']:.4f}  {r['auc']-F02_AUC:+.4f}  "
              f"{'YES ✓' if r['auc']>F02_AUC else 'no'}")
    print("="*60)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="dafd_mvkt/outputs/f02_opt")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
