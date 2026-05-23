"""
Collect key results from all metrics JSON files.

Scans dafd_mvkt/outputs/ for metrics_test_*.json, parses model metadata,
and saves key_results.csv and key_results.json.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dafd_mvkt.experiments.aggregate_results import parse_model_name

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
AUTO_DIR = Path("dafd_mvkt/outputs/auto")


def collect(results_dirs: list[str], split: str = "test") -> list[dict]:
    seen = set()
    records = []
    for d in results_dirs:
        for f in sorted(Path(d).glob(f"metrics_{split}_*.json")):
            if f in seen:
                continue
            seen.add(f)
            try:
                _load_one(f, split, records)
            except Exception as e:
                print(f"  [skip] {f.name}: {e}", file=sys.stderr)
    return records


def _load_one(path: Path, split: str, records: list) -> None:
    with open(path) as fh:
        d = json.load(fh)

    model_name = path.stem.replace(f"metrics_{split}_", "", 1)
    meta = parse_model_name(model_name)
    if meta is None:
        return

    rec: dict = {
        "model_name":   model_name,
        "method_name":  meta["method_name"],
        "model_type":   meta["model_type"],
        "lead":         meta["lead"],
        "hz":           meta["hz"],
        "losses":       meta["losses"],
        "seed":         meta["seed"],
        "macro_auc":    d.get("macro_auc", float("nan")),
        "macro_f1_0_5": d.get("macro_f1_0_5", float("nan")),
        "macro_f1_tuned": d.get("macro_f1_tuned", d.get("macro_f1", float("nan"))),
        "metrics_json": str(path),
    }

    # per-class AUC
    for c in CLASSES:
        rec[f"auc_{c}"] = d.get(f"auc_{c}", float("nan"))

    # per-class F1 tuned
    for c in CLASSES:
        rec[f"f1_tuned_{c}"] = d.get(f"f1_tuned_{c}", d.get(f"f1_{c}", float("nan")))

    # checkpoint + prediction CSV
    stem = path.parent / model_name
    rec["checkpoint"] = str(stem.parent / f"{model_name}_best.pt") if (stem.parent / f"{model_name}_best.pt").exists() else ""
    pred_csv = path.parent / f"predictions_{split}_{model_name}.csv"
    rec["predictions_csv"] = str(pred_csv) if pred_csv.exists() else ""

    # embedded meta from evaluate.py
    if "_meta" in d:
        em = d["_meta"]
        rec["lead"] = em.get("lead", rec["lead"])
        rec["hz"]   = em.get("hz",   rec["hz"])

    records.append(rec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+",
                   default=["dafd_mvkt/outputs",
                             "dafd_mvkt/outputs/summary",
                             "dafd_mvkt/outputs/tuning"])
    p.add_argument("--split", default="test")
    args = p.parse_args()

    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    records = collect(args.results_dirs, args.split)
    print(f"Collected {len(records)} results.")

    # Save JSON
    json_path = AUTO_DIR / "key_results.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Save CSV
    import pandas as pd
    if records:
        pd.DataFrame(records).to_csv(AUTO_DIR / "key_results.csv", index=False)

    print(f"Saved: {json_path}")

    # Quick summary
    by_lead_hz: dict[tuple, list] = {}
    for r in records:
        key = (str(r["lead"]), int(r["hz"] or 0))
        by_lead_hz.setdefault(key, []).append(r)

    for (lead, hz), recs in sorted(by_lead_hz.items(), key=lambda x: (-x[0][1], x[0][0])):
        best = max(recs, key=lambda r: r["macro_auc"] if r["macro_auc"] == r["macro_auc"] else -1)
        print(f"  Lead={lead} {hz}Hz  best AUC={best['macro_auc']:.4f} "
              f"F1={best['macro_f1_tuned']:.4f}  [{best['method_name']}]")


if __name__ == "__main__":
    main()
