"""
Classification metrics for multi-label ECG.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def compute_metrics(
    labels: np.ndarray,          # (N, C) multi-hot float
    probs: np.ndarray,           # (N, C) sigmoid probabilities
    threshold: float | np.ndarray = 0.5,
) -> dict:
    """
    Compute classification metrics for multi-label ECG classification.

    Returns dict with macro and per-class AUC, F1, precision, recall,
    sensitivity, specificity.
    """
    if isinstance(threshold, float):
        thresh = np.full(probs.shape[1], threshold)
    else:
        thresh = np.asarray(threshold)

    preds = (probs >= thresh[None]).astype(int)   # (N, C)
    C = labels.shape[1]

    result: dict = {}

    # ── macro AUC ─────────────────────────────────────────────────────────────
    try:
        result["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
    except Exception:
        result["macro_auc"] = float("nan")

    # ── macro F1 ──────────────────────────────────────────────────────────────
    result["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))

    # ── per-class metrics ─────────────────────────────────────────────────────
    per_auc, per_f1, per_prec, per_rec, per_spec = [], [], [], [], []
    for c in range(C):
        try:
            auc_c = float(roc_auc_score(labels[:, c], probs[:, c]))
        except Exception:
            auc_c = float("nan")
        per_auc.append(auc_c)
        per_f1.append(float(f1_score(labels[:, c], preds[:, c], zero_division=0)))
        per_prec.append(float(precision_score(labels[:, c], preds[:, c], zero_division=0)))
        per_rec.append(float(recall_score(labels[:, c], preds[:, c], zero_division=0)))

        # specificity = TN / (TN + FP)
        try:
            tn, fp, fn, tp = confusion_matrix(labels[:, c], preds[:, c],
                                               labels=[0, 1]).ravel()
            per_spec.append(float(tn) / (tn + fp + 1e-8))
        except Exception:
            per_spec.append(float("nan"))

    for i, name in enumerate(CLASSES):
        result[f"auc_{name}"]  = per_auc[i]
        result[f"f1_{name}"]   = per_f1[i]
        result[f"prec_{name}"] = per_prec[i]
        result[f"rec_{name}"]  = per_rec[i]
        result[f"spec_{name}"] = per_spec[i]

    result["macro_precision"]   = float(np.nanmean(per_prec))
    result["macro_recall"]      = float(np.nanmean(per_rec))
    result["macro_sensitivity"] = result["macro_recall"]
    result["macro_specificity"] = float(np.nanmean(per_spec))

    return result


def find_best_thresholds(
    labels: np.ndarray, probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-class threshold search maximizing F1 on validation set.

    Returns thresholds array of shape (C,).
    """
    if grid is None:
        grid = np.arange(0.1, 0.9, 0.05)

    C = labels.shape[1]
    best_thresh = np.full(C, 0.5)
    for c in range(C):
        best_f1 = -1.0
        for t in grid:
            f1 = f1_score(labels[:, c], (probs[:, c] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh[c] = t
    return best_thresh
