"""
Decision logic for the autonomous pipeline.

Reads key_results.json, applies decision rules, writes:
  outputs/auto/status.json
  outputs/auto/decision.env   (shell-sourceable)
  outputs/auto/decision_log.md
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

AUTO_DIR = Path("dafd_mvkt/outputs/auto")


def best_for(records: list[dict], lead: str, hz: int,
             model_type_prefix: str = "student") -> dict | None:
    cands = [
        r for r in records
        if str(r.get("lead", "")) == lead
        and int(r.get("hz", 0)) == hz
        and r.get("model_type", "").startswith(model_type_prefix)
        and not math.isnan(r.get("macro_auc", float("nan")))
    ]
    if not cands:
        return None
    return max(cands, key=lambda r: r["macro_auc"])


def best_100hz(records: list[dict]) -> dict | None:
    cands = [
        r for r in records
        if int(r.get("hz", 0)) == 100
        and r.get("model_type", "").startswith("student")
        and not math.isnan(r.get("macro_auc", float("nan")))
    ]
    return max(cands, key=lambda r: r["macro_auc"]) if cands else None


def best_50hz(records: list[dict]) -> dict | None:
    cands = [
        r for r in records
        if int(r.get("hz", 0)) == 50
        and r.get("model_type", "").startswith("student")
        and not math.isnan(r.get("macro_auc", float("nan")))
    ]
    return max(cands, key=lambda r: r["macro_auc"]) if cands else None


def decide(records: list[dict], target_auc: float, target_f1: float) -> dict:
    best100 = best_100hz(records)
    best50  = best_50hz(records)
    bI      = best_for(records, "I",  100)
    bII     = best_for(records, "II", 100)
    bce50   = best_for(records, "II", 50)  # approximate BCE baseline

    status: dict = {
        "target_auc": target_auc,
        "target_f1":  target_f1,
        "mvkt_beaten": False,
        "status": "below_target",
        "next_stage": "leadI_clecg",
        "best_method_100hz": "",
        "best_auc_100hz": float("nan"),
        "best_f1_100hz":  float("nan"),
        "best_lead_100hz": "",
        "gap_auc": float("nan"),
        "gap_f1":  float("nan"),
        "leadI_beaten": False,
        "leadII_beaten": False,
        "leadI_best_auc":  float("nan"),
        "leadII_best_auc": float("nan"),
        "tradeoff_status": "pending",
        "best_50hz_auc": float("nan"),
        "best_50hz_f1":  float("nan"),
        "bce_50hz_auc":  float("nan"),
    }

    if bI:
        status["leadI_best_auc"] = bI["macro_auc"]
        if bI["macro_auc"] >= target_auc and bI["macro_f1_tuned"] >= target_f1:
            status["leadI_beaten"] = True

    if bII:
        status["leadII_best_auc"] = bII["macro_auc"]
        if bII["macro_auc"] >= target_auc and bII["macro_f1_tuned"] >= target_f1:
            status["leadII_beaten"] = True

    if best100:
        auc = best100["macro_auc"]
        f1  = best100["macro_f1_tuned"]
        status["best_method_100hz"] = best100["method_name"]
        status["best_auc_100hz"]    = auc
        status["best_f1_100hz"]     = f1
        status["best_lead_100hz"]   = str(best100.get("lead", ""))
        status["gap_auc"] = target_auc - auc
        status["gap_f1"]  = target_f1  - f1

        # Rule A: fully beaten
        if auc >= target_auc and f1 >= target_f1:
            status["mvkt_beaten"] = True
            status["status"]      = "mvkt_beaten"
            status["next_stage"]  = "50hz_tradeoff"

        # Rule B: AUC beaten but F1 short
        elif auc >= target_auc and f1 < target_f1:
            status["status"]     = "auc_beaten_f1_short"
            status["next_stage"] = "threshold_or_teacher_mkd"

        # Rule C: near target (within 0.005)
        elif target_auc - auc <= 0.005:
            status["status"]     = "near_mvkt"
            status["next_stage"] = "tuning"

        # Rule D: Lead I weak
        elif bI and bI["macro_auc"] < 0.835:
            status["status"]     = "leadI_weak"
            status["next_stage"] = "leadII_priority"

        else:
            status["next_stage"] = "leadII_clecg"

    # Rule E: Lead II closer than Lead I
    if bI and bII:
        if bII["macro_auc"] > bI["macro_auc"] and not status["mvkt_beaten"]:
            if status["next_stage"] not in ("50hz_tradeoff", "tuning"):
                status["next_stage"] = "leadII_clecg"

    # 50Hz
    if best50:
        status["best_50hz_auc"] = best50["macro_auc"]
        status["best_50hz_f1"]  = best50["macro_f1_tuned"]
        bce50_rec = best_for(records, str(best50.get("lead", "II")), 50)
        bce50_recs = [r for r in records
                      if str(r.get("lead","")) == str(best50.get("lead","II"))
                      and int(r.get("hz",0)) == 50
                      and "bce" == r.get("losses","").strip()]
        if bce50_recs:
            bce50_auc = max(r["macro_auc"] for r in bce50_recs)
            status["bce_50hz_auc"] = bce50_auc
            if best50["macro_auc"] > bce50_auc + 0.002:
                status["tradeoff_status"] = "improved"
            else:
                status["tradeoff_status"] = "no_gain"

    # Rules F/G for 50Hz
    if best50 and not math.isnan(status["bce_50hz_auc"]):
        clecg_50_recs = [r for r in records
                         if int(r.get("hz",0)) == 50
                         and "clecg" in r.get("method_name", "")]
        if clecg_50_recs:
            best_clecg_50 = max(clecg_50_recs, key=lambda r: r["macro_auc"])
            noclecg_50_recs = [r for r in records
                                if int(r.get("hz",0)) == 50
                                and "clecg" not in r.get("method_name","")
                                and r.get("model_type","").startswith("student")]
            if noclecg_50_recs:
                best_noclecg = max(noclecg_50_recs, key=lambda r: r["macro_auc"])
                if best_clecg_50["macro_auc"] > best_noclecg["macro_auc"]:
                    status["tradeoff_status"] = "clecg_improved"
                else:
                    status["tradeoff_status"] = "no_clecg_gain"

    return status


def write_env(status: dict, path: Path) -> None:
    lines = [
        f"MVKT_BEATEN={1 if status['mvkt_beaten'] else 0}",
        f"PIPELINE_STATUS={status['status']}",
        f"BEST_AUC_100HZ={status['best_auc_100hz']:.4f}" if not math.isnan(status['best_auc_100hz']) else "BEST_AUC_100HZ=nan",
        f"BEST_F1_100HZ={status['best_f1_100hz']:.4f}"   if not math.isnan(status['best_f1_100hz'])  else "BEST_F1_100HZ=nan",
        f"BEST_LEAD_100HZ={status['best_lead_100hz']}",
        f"NEXT_STAGE={status['next_stage']}",
        f"LEADII_BEATEN={1 if status['leadII_beaten'] else 0}",
        f"LEADI_BEATEN={1 if status['leadI_beaten'] else 0}",
        f"BEST_50HZ_AUC={status['best_50hz_auc']:.4f}" if not math.isnan(status['best_50hz_auc']) else "BEST_50HZ_AUC=nan",
        f"TRADEOFF_STATUS={status['tradeoff_status']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_log(status: dict, path: Path) -> None:
    from datetime import datetime
    lines = [
        f"# Decision Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Status**: `{status['status']}`",
        f"**MVKT beaten**: {status['mvkt_beaten']}",
        f"**Next stage**: `{status['next_stage']}`",
        "",
        "## 100Hz Results",
        f"- Best method: {status['best_method_100hz']}",
        f"- AUC: {status['best_auc_100hz']:.4f}  (target {status['target_auc']:.3f}, gap {status['gap_auc']:.4f})"
            if not math.isnan(status['best_auc_100hz']) else "- AUC: N/A",
        f"- F1_tuned: {status['best_f1_100hz']:.4f}  (target {status['target_f1']:.3f}, gap {status['gap_f1']:.4f})"
            if not math.isnan(status['best_f1_100hz']) else "- F1_tuned: N/A",
        f"- Lead I best AUC: {status['leadI_best_auc']:.4f}" if not math.isnan(status['leadI_best_auc']) else "- Lead I: N/A",
        f"- Lead II best AUC: {status['leadII_best_auc']:.4f}" if not math.isnan(status['leadII_best_auc']) else "- Lead II: N/A",
        "",
        "## 50Hz Results",
        f"- Best 50Hz AUC: {status['best_50hz_auc']:.4f}" if not math.isnan(status['best_50hz_auc']) else "- Best 50Hz: N/A",
        f"- Trade-off status: {status['tradeoff_status']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results",    default="dafd_mvkt/outputs/auto/key_results.json")
    p.add_argument("--target_auc", type=float, default=0.843)
    p.add_argument("--target_f1",  type=float, default=0.626)
    args = p.parse_args()

    AUTO_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        records = json.load(f)

    status = decide(records, args.target_auc, args.target_f1)

    # Save
    (AUTO_DIR / "status.json").write_text(json.dumps(status, indent=2, default=str))
    write_env(status, AUTO_DIR / "decision.env")
    write_log(status, AUTO_DIR / "decision_log.md")

    print(f"Status      : {status['status']}")
    print(f"MVKT beaten : {status['mvkt_beaten']}")
    print(f"Best 100Hz  : AUC={status['best_auc_100hz']:.4f}  F1={status['best_f1_100hz']:.4f}  [{status['best_method_100hz']}]"
          if not math.isnan(status['best_auc_100hz']) else "Best 100Hz  : N/A")
    print(f"Next stage  : {status['next_stage']}")
    print(f"Saved: {AUTO_DIR}/status.json")


if __name__ == "__main__":
    main()
