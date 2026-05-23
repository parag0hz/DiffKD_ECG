"""
PTB-XL dataset for DAFD-MVKT.

Modes:
  "teacher"     → {"teacher_x": [12, 5000], "y": [5], "record_id"}
  "student"     → {"teacher_x": [12, 5000], "student_x": [1, L], "y", "record_id"}
  "ta"          → {"teacher_x": [12, 5000], "ta_x": [1, 5000], "y", "record_id"}
  "hier"        → {"teacher_x": [12, 5000], "ta_x": [1, ta_L], "student_x": [1, L],
                    "y", "record_id"}
  "progressive" → {"ta_high_x": [1, 5000], "ta_low_x": [1, L], "y", "record_id"}

where L = 1000 (100 Hz) or 500 (50 Hz), ta_L depends on ta_hz parameter.

Label convention:
  Multi-hot [5] float32 for superclasses NORM, MI, STTC, CD, HYP.

Normalization:
  Per-lead z-score applied to teacher_x after loading.
  ta_x is extracted from the already-normalized teacher_x (no extra step).
  If ta_hz != 500 in "hier" mode, ta_x is further downsampled and re-normalized.
  student_x and ta_low_x are re-normalized after anti-aliased downsampling.
"""
from __future__ import annotations
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from utils.signal import LEAD_IDX, downsample_ecg, zscore_normalize

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLS2IDX = {c: i for i, c in enumerate(SUPERCLASSES)}

_TRAIN_FOLDS = set(range(1, 9))
_VAL_FOLDS   = {9}
_TEST_FOLDS  = {10}

_VALID_MODES = ("teacher", "student", "ta", "hier", "progressive")


def _build_multihot(superclasses: list[str]) -> np.ndarray:
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for c in superclasses:
        if c in CLS2IDX:
            vec[CLS2IDX[c]] = 1.0
    return vec


def _aggregate_superclass(scp_codes: dict, agg_df: pd.DataFrame) -> list[str]:
    classes = []
    for key in scp_codes:
        if key in agg_df.index:
            cls = agg_df.loc[key, "diagnostic_class"]
            if pd.notna(cls) and cls in CLS2IDX:
                classes.append(cls)
    return sorted(set(classes))


def _build_metadata(ptbxl_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    agg = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    agg = agg[agg["diagnostic"] == 1]

    meta["superclasses"] = meta["scp_codes"].apply(
        lambda d: _aggregate_superclass(d, agg)
    )
    meta["label_vec"] = meta["superclasses"].apply(_build_multihot)

    # keep only records with at least one superclass label
    meta = meta[meta["superclasses"].map(len) > 0].copy()
    return meta


class PTBXLDataset(Dataset):
    """
    PTB-XL multi-label ECG dataset.

    Args:
        ptbxl_root:     path to PTB-XL root (must contain ptbxl_database.csv,
                        scp_statements.csv, records500/)
        split:          "train" | "val" | "test"
        mode:           "teacher" | "student" | "ta" | "hier"
        sampling_rate:  target Hz for student input (100 or 50)
        lead:           lead name for single-lead input (e.g. "II"); takes
                        priority over student_lead
        student_lead:   legacy alias for lead
        normalize:      "zscore" (only option currently)
    """

    def __init__(
        self,
        ptbxl_root: str | Path,
        split: str,
        mode: str = "teacher",
        sampling_rate: int = 100,
        lead: str | None = None,
        student_lead: str = "II",
        normalize: str = "zscore",
        ta_hz: int = 500,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert mode in _VALID_MODES, f"Unknown mode: {mode}"

        lead_name = lead if lead is not None else student_lead
        assert lead_name in LEAD_IDX, f"Unknown lead: {lead_name}"

        self.root          = Path(ptbxl_root)
        self.mode          = mode
        self.sampling_rate = sampling_rate
        self.lead_idx      = LEAD_IDX[lead_name]
        self.normalize     = normalize
        self.ta_hz         = ta_hz

        meta = _build_metadata(self.root)

        fold_map = {"train": _TRAIN_FOLDS, "val": _VAL_FOLDS, "test": _TEST_FOLDS}
        self.meta = meta[meta["strat_fold"].isin(fold_map[split])].reset_index()

    def __len__(self) -> int:
        return len(self.meta)

    def _load_500hz(self, row: pd.Series) -> np.ndarray:
        """Load 500 Hz 12-lead ECG. Returns (12, 5000) float32."""
        sig, _ = wfdb.rdsamp(str(self.root / row["filename_hr"]))
        return sig.T.astype(np.float32)   # (12, 5000)

    def _load_100hz(self, row: pd.Series) -> np.ndarray:
        """Load official 100 Hz 12-lead ECG from records100. Returns (12, 1000) float32."""
        sig, _ = wfdb.rdsamp(str(self.root / row["filename_lr"]))
        return sig.T.astype(np.float32)   # (12, 1000)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]

        y = row["label_vec"].astype(np.float32)  # (5,)
        out = {
            "y":         torch.from_numpy(y),
            "record_id": int(row["ecg_id"]),
        }

        if self.mode == "teacher":
            if self.sampling_rate == 100:
                # Use official PTB-XL records100 directly (matches paper baselines)
                raw = self._load_100hz(row)   # (12, 1000)
            elif self.sampling_rate == 500:
                raw = self._load_500hz(row)   # (12, 5000)
            else:
                # e.g. 50Hz: load 500Hz and downsample
                raw = self._load_500hz(row)   # (12, 5000)
                raw = downsample_ecg(raw, src_hz=500, dst_hz=self.sampling_rate)
            if self.normalize == "zscore":
                raw = zscore_normalize(raw)
            out["teacher_x"] = torch.from_numpy(raw)
            return out

        raw = self._load_500hz(row)   # (12, 5000)

        if self.normalize == "zscore":
            raw = zscore_normalize(raw)

        # All single-lead modes extract the selected lead at 500Hz.
        # raw is already per-lead z-scored, so lead_500hz is already normalized.
        lead_500hz = raw[self.lead_idx]   # (5000,)

        if self.mode == "ta":
            out["teacher_x"] = torch.from_numpy(raw)
            out["ta_x"] = torch.from_numpy(lead_500hz[None].astype(np.float32))
            return out

        if self.mode == "progressive":
            # ta_high_x: 1-lead 500Hz (already z-scored)
            ta_high_x = torch.from_numpy(lead_500hz[None].astype(np.float32))
            # ta_low_x: 1-lead low-Hz (downsampled + re-normalized)
            low_sig = downsample_ecg(lead_500hz, 500, self.sampling_rate)
            if self.normalize == "zscore":
                mu  = low_sig.mean()
                std = max(float(low_sig.std()), 1e-6)
                low_sig = (low_sig - mu) / std
            out["ta_high_x"] = ta_high_x
            out["ta_low_x"]  = torch.from_numpy(low_sig[None].astype(np.float32))
            return out

        # student and hier both need student_x (downsampled + re-normalized)
        student_sig = downsample_ecg(lead_500hz, 500, self.sampling_rate)
        if self.normalize == "zscore":
            mu  = student_sig.mean()
            std = max(float(student_sig.std()), 1e-6)
            student_sig = (student_sig - mu) / std
        student_x = torch.from_numpy(student_sig[None].astype(np.float32))

        if self.mode == "student":
            out["teacher_x"] = torch.from_numpy(raw)
            out["student_x"] = student_x
            return out

        # mode == "hier"
        # ta_hz controls the TA signal sampling rate (default 500, progressive: 100)
        if self.ta_hz == 500:
            ta_x = torch.from_numpy(lead_500hz[None].astype(np.float32))
        else:
            ta_sig = downsample_ecg(lead_500hz, 500, self.ta_hz)
            if self.normalize == "zscore":
                mu  = ta_sig.mean()
                std = max(float(ta_sig.std()), 1e-6)
                ta_sig = (ta_sig - mu) / std
            ta_x = torch.from_numpy(ta_sig[None].astype(np.float32))

        out["teacher_x"] = torch.from_numpy(raw)
        out["ta_x"]      = ta_x
        out["student_x"] = student_x
        return out
