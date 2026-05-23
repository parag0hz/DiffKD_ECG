#!/usr/bin/env python3
"""Re-evaluate F02W best checkpoints with corrected sigmoid in _run_eval.

Usage (from /home/kwy00/sci):
  python dafd_mvkt/scripts/reeval_f02w.py \
      --data_dir comper_repo/ptb_xl \
      --output_dir dafd_mvkt/outputs/f02_opt \
      --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # dafd_mvkt/

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset, SUPERCLASSES
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds

STUDENT_KEY = "student_x"
F02_AUC = 0.84466
F02_F1  = 0.6275


def _build_model():
    return ResNet1d(in_channels=1, num_classes=5,
                    layers=[3, 4, 6, 3], base_channels=64,
                    proj_dim=128, dropout=0.0)


@torch.no_grad()
def _run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x = batch[STUDENT_KEY].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return probs, torch.cat(all_labels).numpy(), all_ids


def reeval_variant(ckpt_path: Path, metrics_path: Path,
                   val_loader, test_loader, device):
    existing = json.load(open(metrics_path))
    out_name = existing["output_name"]

    model = _build_model().to(device)
    load_checkpoint(str(ckpt_path), model, device)

    val_probs, val_labels, _ = _run_eval(model, val_loader, device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(model, test_loader, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    existing["macro_auc"]      = round(m05["macro_auc"], 5)
    existing["macro_f1_0_5"]   = round(m05.get("macro_f1", 0), 5)
    existing["macro_f1_tuned"] = round(m_tune.get("macro_f1", 0), 5)
    existing["per_class_thresholds"] = {
        SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
    }
    for c in SUPERCLASSES:
        existing[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        existing[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    auc = existing["macro_auc"]
    f1t = existing["macro_f1_tuned"]
    print(f"  {out_name}: AUC={auc:.4f}  F1_tuned={f1t:.4f}  "
          f"(vs F02: {auc-F02_AUC:+.4f} AUC  {f1t-F02_F1:+.4f} F1)")
    return existing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="comper_repo/ptb_xl")
    p.add_argument("--output_dir",  default="dafd_mvkt/outputs/f02_opt")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--lead",        default="II")
    p.add_argument("--variants",    nargs="+", default=None,
                   help="e.g. F02W01 F02W02 … (default: all completed)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)

    val_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="val",  lead=args.lead)
    test_ds = PTBXLMultiTeacherDataset(args.data_dir, split="test", lead=args.lead)
    val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False, num_workers=4,
                             persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4,
                             persistent_workers=True)

    # find completed variants
    pattern = f"student_II_50hz_f02_F02W*_seed{args.seed}"
    if args.variants:
        names = [f"student_II_50hz_f02_{v}_seed{args.seed}" for v in args.variants]
    else:
        names = sorted([p.stem.replace("metrics_test_", "")
                        for p in out_dir.glob(f"metrics_test_{pattern}.json")])

    print(f"Re-evaluating {len(names)} variants with corrected sigmoid…")
    results = []
    for name in names:
        ckpt_path    = out_dir / f"{name}_best.pt"
        metrics_path = out_dir / f"metrics_test_{name}.json"
        if not ckpt_path.exists():
            print(f"  [skip] {name}: best.pt not found")
            continue
        if not metrics_path.exists():
            print(f"  [skip] {name}: metrics json not found")
            continue
        print(f"  {name} …")
        r = reeval_variant(ckpt_path, metrics_path, val_loader, test_loader, device)
        results.append(r)

    if results:
        best = max(results, key=lambda d: d["macro_auc"])
        print(f"\nBest: {best['variant_id']}  AUC={best['macro_auc']:.4f}  "
              f"F1_tuned={best['macro_f1_tuned']:.4f}")
        print("Done.")


if __name__ == "__main__":
    main()
