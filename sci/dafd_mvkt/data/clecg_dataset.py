"""
CLECG pretraining dataset for MoCo-style contrastive learning.

Returns two independently augmented views of the same ECG signal.
No labels are used during pretraining.

Output keys:
    view1   : torch.Tensor [1, L]   (query view)
    view2   : torch.Tensor [1, L]   (key view)
    record_id: int
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
from utils.clecg_augment import CLECGAugment

_TRAIN_FOLDS = set(range(1, 9))


def _build_metadata(ptbxl_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")
    return meta


class CLECGDataset(Dataset):
    """
    Unlabeled PTB-XL dataset for CLECG-style contrastive pretraining.

    Loads 500Hz ECG records, extracts a single lead, downsamples to target Hz,
    z-score normalizes, then produces two independently augmented views via
    CLECGAugment.

    Args:
        ptbxl_root:      path to PTB-XL root directory
        sampling_rate:   target Hz (100, 50, or 500)
        lead:            ECG lead name (e.g. "II", "I")
        split:           "train" | "val"  (only train folds used by default)
        augment_cfg:     kwargs forwarded to CLECGAugment
        seed:            RNG seed for augmentation
    """

    def __init__(
        self,
        ptbxl_root: str | Path,
        sampling_rate: int = 100,
        lead: str = "II",
        split: str = "train",
        augment_cfg: dict | None = None,
        seed: int | None = None,
    ):
        assert lead in LEAD_IDX, f"Unknown lead: {lead}"
        assert split in ("train", "val"), f"split must be 'train' or 'val'"

        self.root          = Path(ptbxl_root)
        self.sampling_rate = sampling_rate
        self.lead_idx      = LEAD_IDX[lead]

        meta = _build_metadata(self.root)

        if split == "train":
            folds = _TRAIN_FOLDS
        else:
            folds = {9}
        self.meta = meta[meta["strat_fold"].isin(folds)].reset_index()

        aug_kw = augment_cfg or {}
        self.augment = CLECGAugment(seed=seed, **aug_kw)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]

        sig, _ = wfdb.rdsamp(str(self.root / row["filename_hr"]))
        lead_500hz = sig[:, self.lead_idx].astype(np.float32)  # (5000,)

        # Downsample with anti-aliasing
        if self.sampling_rate != 500:
            lead = downsample_ecg(lead_500hz, 500, self.sampling_rate)
        else:
            lead = lead_500hz

        # Per-sample z-score
        mu  = float(lead.mean())
        std = max(float(lead.std()), 1e-6)
        lead = (lead - mu) / std

        # Two independently augmented views
        view1, view2 = self.augment(lead)  # each (L,)

        return {
            "view1":     torch.from_numpy(view1[None].astype(np.float32)),
            "view2":     torch.from_numpy(view2[None].astype(np.float32)),
            "record_id": int(row["ecg_id"]),
        }
