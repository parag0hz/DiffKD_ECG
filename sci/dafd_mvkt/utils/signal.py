"""
ECG signal utilities: anti-aliased downsampling and normalization.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly


# Lead name → index mapping (standard 12-lead order)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX: dict[str, int] = {n: i for i, n in enumerate(LEAD_NAMES)}


def downsample_ecg(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """
    Anti-aliased downsampling via scipy.signal.resample_poly.

    Args:
        x:      ECG array, shape (C, L) or (L,)
        src_hz: source sampling rate (e.g. 500)
        dst_hz: target sampling rate (e.g. 100 or 50)

    Returns:
        Downsampled array, shape (C, L') or (L',)
    """
    if src_hz == dst_hz:
        return x

    from math import gcd
    g = gcd(dst_hz, src_hz)
    up, down = dst_hz // g, src_hz // g

    if x.ndim == 1:
        return resample_poly(x, up, down).astype(x.dtype)

    return np.stack(
        [resample_poly(x[c], up, down) for c in range(x.shape[0])],
        axis=0,
    ).astype(x.dtype)


def zscore_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-lead z-score normalization.

    Args:
        x:   shape (C, L)
        eps: clamp for std to avoid division by zero

    Returns:
        Normalized array, same shape.
    """
    mu = x.mean(axis=-1, keepdims=True)       # (C, 1)
    std = x.std(axis=-1, keepdims=True)        # (C, 1)
    return (x - mu) / std.clip(eps)
