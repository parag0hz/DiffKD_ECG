"""
Generate outputs/auto/latest_summary.md from status.json and key_results.json.
"""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

AUTO_DIR = Path("dafd_mvkt/outputs/auto")
MVKT_AUC = 0.843
MVKT_F1  = 0.626


def _fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}" if not math.isnan(v) else "N/A"


def _delta(v: float, ref: float) -> str:
    if math.isnan(v) or math.isnan(ref):
        return ""
    d = v - ref
    return f" ({d:+.4f})"


def make_report(status: dict, records: list[dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Pipeline Status Report — {ts}",
        "",
        "## 1. MVKT Target",
        f"- Target: AUC **{MVKT_AUC}**, F1_tuned **{MVKT_F1}**",
        f"- **Beaten: {'YES ✓' if status.get('mvkt_beaten') else 'NO'}**",
        "",
    ]

    # Best 100Hz
    lines += ["## 2. Best 100Hz Result"]
    best_100 = _best_rec(records, hz=100, model_type="student")
    if best_100:
        auc = best_100["macro_auc"]
        f1  = best_100["macro_f1_tuned"]
        lines += [
            f"| | Value | Gap to MVKT |",
            f"|---|---|---|",
            f"| Method | {best_100['method_name']} | — |",
            f"| Lead | {best_100.get('lead','?')} | — |",
            f"| AUC | **{_fmt(auc)}** | {_fmt(MVKT_AUC - auc)} |",
            f"| F1_tuned | **{_fmt(f1)}** | {_fmt(MVKT_F1 - f1)} |",
            "",
        ]
    else:
        lines += ["*(no 100Hz student results yet)*", ""]

    # Lead breakdown
    lines += ["## 3. Lead Comparison (100Hz)"]
    for lead in ["I", "II"]:
        best = _best_rec(records, hz=100, model_type="student", lead=lead)
        if best:
            lines.append(f"- **Lead {lead}**: AUC={_fmt(best['macro_auc'])}  F1={_fmt(best['macro_f1_tuned'])}  [{best['method_name']}]")
        else:
            lines.append(f"- **Lead {lead}**: N/A")
    lines.append("")

    # Best 50Hz
    lines += ["## 4. Best 50Hz Result (trade-off)"]
    best_50 = _best_rec(records, hz=50, model_type="student")
    bce_50  = _best_bce(records, hz=50)
    if best_50:
        auc_gain = (best_50["macro_auc"] - bce_50["macro_auc"]) if bce_50 else float("nan")
        f1_gain  = (best_50["macro_f1_tuned"] - bce_50["macro_f1_tuned"]) if bce_50 else float("nan")
        lines += [
            f"- Method: {best_50['method_name']}",
            f"- AUC: {_fmt(best_50['macro_auc'])}{_delta(best_50['macro_auc'], bce_50['macro_auc']) if bce_50 else ''}",
            f"- F1_tuned: {_fmt(best_50['macro_f1_tuned'])}{_delta(best_50['macro_f1_tuned'], bce_50['macro_f1_tuned']) if bce_50 else ''}",
            f"- Gap to MVKT AUC: {_fmt(MVKT_AUC - best_50['macro_auc'])}",
            f"- Gain vs BCE 50Hz AUC: {_fmt(auc_gain)}",
            "",
        ]
    else:
        lines += ["*(no 50Hz results yet)*", ""]

    # Status and next action
    lines += [
        "## 5. Pipeline Status",
        f"- Status: `{status.get('status', 'unknown')}`",
        f"- Next recommended: `{status.get('next_stage', 'unknown')}`",
        "",
        "## 6. Warnings",
    ]
    warnings = []
    # Single seed warning
    seed_counts = {}
    for r in records:
        key = r.get("method_name","")
        seed_counts[key] = seed_counts.get(key, 0) + 1
    if all(v == 1 for v in seed_counts.values()):
        warnings.append("All results are single-seed (seed=0). Run multi-seed for final results.")

    if not status.get("leadI_beaten") and not math.isnan(status.get("leadI_best_auc", float("nan"))):
        if status.get("leadI_best_auc", 0) < 0.835:
            warnings.append(f"Lead I is weak (AUC={_fmt(status['leadI_best_auc'])}). Prioritize Lead II.")

    if not math.isnan(status.get("best_f1_100hz", float("nan"))):
        if status["best_auc_100hz"] >= MVKT_AUC and status["best_f1_100hz"] < MVKT_F1:
            warnings.append("AUC target beaten but F1 target not met. Try teacher_mkd or threshold tuning.")

    # TA vs student check
    best_ta_I = _best_rec(records, hz=500, model_type="ta", lead="I")
    best_s_I  = _best_rec(records, hz=100, model_type="student", lead="I")
    if best_ta_I and best_s_I and best_ta_I["macro_auc"] < best_s_I["macro_auc"]:
        warnings.append("TA-I-500Hz AUC is weaker than student. CLECG TA strengthening is important.")

    for w in warnings:
        lines.append(f"- ⚠ {w}")
    if not warnings:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines)


def _best_rec(records: list[dict], hz: int, model_type: str,
              lead: str | None = None) -> dict | None:
    cands = [
        r for r in records
        if int(r.get("hz", 0)) == hz
        and r.get("model_type", "").startswith(model_type)
        and (lead is None or str(r.get("lead","")) == lead)
        and not math.isnan(r.get("macro_auc", float("nan")))
    ]
    return max(cands, key=lambda r: r["macro_auc"]) if cands else None


def _best_bce(records: list[dict], hz: int) -> dict | None:
    cands = [
        r for r in records
        if int(r.get("hz", 0)) == hz
        and r.get("losses", "").strip() == "bce"
        and not math.isnan(r.get("macro_auc", float("nan")))
    ]
    return max(cands, key=lambda r: r["macro_auc"]) if cands else None


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--status",  default="dafd_mvkt/outputs/auto/status.json")
    p.add_argument("--results", default="dafd_mvkt/outputs/auto/key_results.json")
    p.add_argument("--out",     default="dafd_mvkt/outputs/auto/latest_summary.md")
    args = p.parse_args()

    status  = json.loads(Path(args.status).read_text()) if Path(args.status).exists() else {}
    records = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else []

    report = make_report(status, records)
    Path(args.out).write_text(report)
    print(report)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
