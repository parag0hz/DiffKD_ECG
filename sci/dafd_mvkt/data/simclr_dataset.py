"""
SimCLR pretraining dataset for ECG encoders.

Returns two independently augmented views of the same single-lead ECG.
No labels are used during pretraining.

Augmentations (complementary to CLECG's wavelet/masking approach):
  1. Gaussian noise   — random std in [min_noise_std, max_noise_std]
  2. Amplitude scale  — uniform random in [scale_lo, scale_hi]
  3. Time mask        — zero out n_masks contiguous segments (each 10-50 ms)
  4. Baseline wander  — optional low-frequency sine drift
  5. Temporal shift   — optional circular shift ±50 samples

Output keys:
    view1    : torch.Tensor [1, L]
    view2    : torch.Tensor [1, L]
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

_TRAIN_FOLDS = set(range(1, 9))


class SimCLRAugment:
    """
    ECG augmentation pipeline for SimCLR pretraining.

    Applies independent random augmentation chains to produce two views.
    """

    def __init__(
        self,
        p_noise:       float = 0.8,
        min_noise_std: float = 0.02,
        max_noise_std: float = 0.10,
        p_scale:       float = 0.8,
        scale_lo:      float = 0.8,
        scale_hi:      float = 1.2,
        p_mask:        float = 0.5,
        n_masks:       int   = 2,
        mask_len_frac: float = 0.05,
        p_wander:      float = 0.3,
        wander_amp:    float = 0.1,
        p_shift:       float = 0.3,
        max_shift:     int   = 50,
        seed:          int | None = None,
    ):
        self.p_noise       = p_noise
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.p_scale       = p_scale
        self.scale_lo      = scale_lo
        self.scale_hi      = scale_hi
        self.p_mask        = p_mask
        self.n_masks       = n_masks
        self.mask_len_frac = mask_len_frac
        self.p_wander      = p_wander
        self.wander_amp    = wander_amp
        self.p_shift       = p_shift
        self.max_shift     = max_shift
        self.rng           = np.random.default_rng(seed)

    def _one_view(self, x: np.ndarray) -> np.ndarray:
        """Augment a single (L,) signal."""
        x = x.copy().astype(np.float32)
        L = len(x)

        # 1. Gaussian noise
        if self.rng.random() < self.p_noise:
            std = self.rng.uniform(self.min_noise_std, self.max_noise_std)
            x += self.rng.standard_normal(L).astype(np.float32) * std

        # 2. Amplitude scaling
        if self.rng.random() < self.p_scale:
            scale = self.rng.uniform(self.scale_lo, self.scale_hi)
            x *= scale

        # 3. Time masking (zero out contiguous segments)
        if self.rng.random() < self.p_mask:
            mask_len = max(1, int(L * self.mask_len_frac))
            for _ in range(self.n_masks):
                start = int(self.rng.integers(0, max(1, L - mask_len)))
                x[start: start + mask_len] = 0.0

        # 4. Baseline wander
        if self.rng.random() < self.p_wander:
            freq  = self.rng.uniform(0.02, 0.4)
            phase = self.rng.uniform(0, 2 * np.pi)
            t     = np.linspace(0, 10.0, L, dtype=np.float32)
            amp   = self.rng.uniform(0.02, self.wander_amp)
            x    += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

        # 5. Temporal circular shift
        if self.rng.random() < self.p_shift:
            shift = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
            x = np.roll(x, shift)

        return x

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return two independently augmented views of x (shape: L,)."""
        return self._one_view(x), self._one_view(x)


class SimCLRDataset(Dataset):
    """
    Unlabeled PTB-XL dataset for SimCLR pretraining.

    Loads 500Hz ECG records, extracts a single lead, downsamples to target Hz,
    z-score normalizes, then produces two independently augmented views via
    SimCLRAugment.

    Args:
        ptbxl_root:    path to PTB-XL root directory
        sampling_rate: target Hz (100, 50, or 500)
        lead:          ECG lead name ("II", "I", etc.)
        split:         "train" | "val"
        augment_cfg:   kwargs forwarded to SimCLRAugment
        seed:          RNG seed
    """

    def __init__(
        self,
        ptbxl_root:    str | Path,
        sampling_rate: int = 100,
        lead:          str = "II",
        split:         str = "train",
        augment_cfg:   dict | None = None,
        seed:          int | None = None,
    ):
        assert lead in LEAD_IDX, f"Unknown lead: {lead}"
        assert split in ("train", "val"), f"split must be 'train' or 'val'"

        self.root          = Path(ptbxl_root)
        self.sampling_rate = sampling_rate
        self.lead_idx      = LEAD_IDX[lead]

        meta = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        folds = _TRAIN_FOLDS if split == "train" else {9}
        self.meta = meta[meta["strat_fold"].isin(folds)].reset_index()

        aug_kw = augment_cfg or {}
        self.augment = SimCLRAugment(seed=seed, **aug_kw)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row     = self.meta.iloc[idx]
        rec_id  = int(row["ecg_id"])

        # Load 500 Hz record
        fname  = str(row["filename_hr"])
        signal, _ = wfdb.rdsamp(str(self.root / fname))

        # Extract lead (signal shape: L×12)
        lead_sig = signal[:, self.lead_idx].astype(np.float32)

        # Downsample to target Hz
        if self.sampling_rate != 500:
            lead_sig = downsample_ecg(lead_sig, src_hz=500,
                                      dst_hz=self.sampling_rate)

        # Z-score normalize
        lead_sig = zscore_normalize(lead_sig)

        # Two augmented views
        v1, v2 = self.augment(lead_sig)

        return {
            "view1":     torch.tensor(v1, dtype=torch.float32).unsqueeze(0),
            "view2":     torch.tensor(v2, dtype=torch.float32).unsqueeze(0),
            "record_id": rec_id,
        }
