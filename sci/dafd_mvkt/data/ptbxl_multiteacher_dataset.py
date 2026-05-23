"""
PTB-XL multi-teacher dataset.

Returns all 6 teacher views + student view from a single ECG record:

  teacher_12l_500: [12, 5000]   12-lead 500 Hz
  teacher_12l_100: [12, 1000]   12-lead 100 Hz (from records100)
  teacher_12l_50:  [12,  500]   12-lead  50 Hz (downsampled from 500)
  teacher_1l_500:  [ 1, 5000]   1-lead 500 Hz
  teacher_1l_100:  [ 1, 1000]   1-lead 100 Hz (from records100)
  teacher_1l_50:   [ 1,  500]   1-lead  50 Hz (downsampled from 500)
  student_x:       [ 1,  500]   same as teacher_1l_50 (student input)
  y:               [5]          multi-hot label
  record_id:       int
"""
from __future__ import annotations
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.signal import LEAD_IDX, downsample_ecg, zscore_normalize

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
_CLS2IDX     = {c: i for i, c in enumerate(SUPERCLASSES)}

_TRAIN_FOLDS = set(range(1, 9))
_VAL_FOLDS   = {9}
_TEST_FOLDS  = {10}


def _build_multihot(superclasses: list[str]) -> np.ndarray:
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for c in superclasses:
        if c in _CLS2IDX:
            vec[_CLS2IDX[c]] = 1.0
    return vec


def _aggregate_superclass(scp_codes: dict, agg_df: pd.DataFrame) -> list[str]:
    classes = []
    for key in scp_codes:
        if key in agg_df.index:
            cls = agg_df.loc[key, "diagnostic_class"]
            if pd.notna(cls) and cls in _CLS2IDX:
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
    meta = meta[meta["superclasses"].map(len) > 0].copy()
    return meta


class PTBXLMultiTeacherDataset(Dataset):
    """
    PTB-XL dataset that returns all 6 teacher views and the student view
    from the same ECG record.

    All signals are loaded from disk once at init and cached in memory,
    so __getitem__ only does array indexing (no disk I/O per sample).
    """

    def __init__(
        self,
        ptbxl_root: str | Path,
        split: str,
        lead: str = "II",
        normalize: str = "zscore",
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert lead in LEAD_IDX, f"Unknown lead: {lead}"

        self.root      = Path(ptbxl_root)
        self.lead_idx  = LEAD_IDX[lead]
        self.normalize = normalize

        meta = _build_metadata(self.root)
        fold_map = {"train": _TRAIN_FOLDS, "val": _VAL_FOLDS, "test": _TEST_FOLDS}
        self.meta = meta[meta["strat_fold"].isin(fold_map[split])].reset_index()

        n = len(self.meta)
        print(f"[{split}] Loading {n} records into memory …")

        # pre-allocate cache arrays
        self._t12l500 = np.empty((n, 12, 5000), dtype=np.float32)
        self._t12l100 = np.empty((n, 12, 1000), dtype=np.float32)
        self._t12l50  = np.empty((n, 12,  500), dtype=np.float32)
        self._t1l500  = np.empty((n,  1, 5000), dtype=np.float32)
        self._t1l100  = np.empty((n,  1, 1000), dtype=np.float32)
        self._t1l50   = np.empty((n,  1,  500), dtype=np.float32)
        self._y       = np.empty((n, 5),        dtype=np.float32)
        self._ids     = np.empty(n,              dtype=np.int64)

        for i, row in enumerate(tqdm(self.meta.itertuples(), total=n, leave=False)):
            # 500 Hz
            sig500, _ = wfdb.rdsamp(str(self.root / row.filename_hr))
            r500 = sig500.T.astype(np.float32)          # (12, 5000)
            if normalize == "zscore":
                r500 = zscore_normalize(r500)

            # 100 Hz
            sig100, _ = wfdb.rdsamp(str(self.root / row.filename_lr))
            r100 = sig100.T.astype(np.float32)          # (12, 1000)
            if normalize == "zscore":
                r100 = zscore_normalize(r100)

            # 50 Hz (downsample from 500)
            r50 = downsample_ecg(r500, src_hz=500, dst_hz=50)  # (12, 500)
            if normalize == "zscore":
                r50 = zscore_normalize(r50)

            # 1-lead views
            l500 = r500[self.lead_idx]          # (5000,)
            l100 = r100[self.lead_idx]          # (1000,)
            l50  = downsample_ecg(l500, src_hz=500, dst_hz=50)  # (500,)
            if normalize == "zscore":
                mu = l50.mean(); std = max(float(l50.std()), 1e-6)
                l50 = (l50 - mu) / std

            self._t12l500[i] = r500
            self._t12l100[i] = r100
            self._t12l50[i]  = r50
            self._t1l500[i]  = l500[None]
            self._t1l100[i]  = l100[None]
            self._t1l50[i]   = l50[None]
            self._y[i]       = row.label_vec
            self._ids[i]     = row.ecg_id

        print(f"[{split}] Loaded {n} records.")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        return {
            "teacher_12l_500": torch.from_numpy(self._t12l500[idx]),
            "teacher_12l_100": torch.from_numpy(self._t12l100[idx]),
            "teacher_12l_50":  torch.from_numpy(self._t12l50[idx]),
            "teacher_1l_500":  torch.from_numpy(self._t1l500[idx]),
            "teacher_1l_100":  torch.from_numpy(self._t1l100[idx]),
            "teacher_1l_50":   torch.from_numpy(self._t1l50[idx]),
            "student_x":       torch.from_numpy(self._t1l50[idx]),
            "y":               torch.from_numpy(self._y[idx]),
            "record_id":       int(self._ids[idx]),
        }
