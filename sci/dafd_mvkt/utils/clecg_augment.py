"""
CLECG-style ECG augmentations for contrastive pretraining.

Augmentations:
  1. Segmented masking: divide signal into k segments, zero p ratio in each.
  2. Wavelet smoothing: DWT with db4/db6, soft-threshold detail coefficients.
  3. Gaussian noise: optional, disabled by default.

Each call to CLECGAugment produces two independently augmented views.
"""
from __future__ import annotations
import numpy as np

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


def segment_mask(x: np.ndarray, num_segments: int = 4, mask_ratio: float = 0.2,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Zero out a contiguous mask_ratio fraction of each segment.

    Args:
        x:            signal (L,) or (1, L)
        num_segments: number of equal segments
        mask_ratio:   fraction of each segment to zero
        rng:          optional NumPy random generator
    Returns:
        Augmented signal, same shape as x.
    """
    if rng is None:
        rng = np.random.default_rng()

    flat = x.ndim == 1
    if flat:
        x = x[None]  # (1, L)

    x = x.copy()
    L = x.shape[-1]
    seg_len = L // num_segments
    mask_len = max(1, int(seg_len * mask_ratio))

    for s in range(num_segments):
        start = s * seg_len
        end = start + seg_len
        max_offset = max(0, (seg_len - mask_len))
        offset = int(rng.integers(0, max_offset + 1))
        m_start = start + offset
        m_end = min(m_start + mask_len, end)
        x[..., m_start:m_end] = 0.0

    return x[0] if flat else x


def wavelet_augment(x: np.ndarray, wavelet: str = "db4", level: int = 4,
                    threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Wavelet soft-thresholding augmentation.

    Decomposes signal with DWT, applies soft threshold to detail coefficients
    (removes minor noise components), then reconstructs.  Creates a mildly
    smoothed view without losing dominant morphological features.

    Falls back to identity transform if pywt is unavailable.
    """
    if not _PYWT_AVAILABLE:
        return x

    flat = x.ndim == 1
    if flat:
        x = x[None]

    out = np.empty_like(x)
    for i in range(x.shape[0]):
        sig = x[i].astype(np.float64)
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        # Soft-threshold detail coefficients
        for j in range(1, len(coeffs)):
            thr = threshold_ratio * (np.max(np.abs(coeffs[j])) + 1e-8)
            coeffs[j] = pywt.threshold(coeffs[j], thr, mode="soft")
        rec = pywt.waverec(coeffs, wavelet)
        # waverec may return slightly longer array due to boundary effects
        out[i] = rec[: sig.shape[0]].astype(x.dtype)

    return out[0] if flat else out


def gaussian_noise(x: np.ndarray, std: float = 0.05,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return x + rng.standard_normal(x.shape).astype(x.dtype) * std


class CLECGAugment:
    """
    Compose CLECG augmentations and generate two independent views.

    Args:
        num_segments:   segments for masking (default 4)
        mask_ratio:     fraction to zero per segment (default 0.2)
        use_wavelet:    apply wavelet augmentation (default True)
        wavelet_prob:   probability of applying each wavelet type (default 0.5)
        wavelet_level:  DWT decomposition levels (default 4)
        threshold_ratio: soft-threshold fraction for wavelet (default 0.05)
        use_noise:      apply Gaussian noise (default False)
        noise_std:      noise std (default 0.05)
        seed:           optional RNG seed
    """

    def __init__(
        self,
        num_segments: int = 4,
        mask_ratio: float = 0.2,
        use_wavelet: bool = True,
        wavelet_prob: float = 0.5,
        wavelet_level: int = 4,
        threshold_ratio: float = 0.05,
        use_noise: bool = False,
        noise_std: float = 0.05,
        seed: int | None = None,
    ):
        self.num_segments = num_segments
        self.mask_ratio = mask_ratio
        self.use_wavelet = use_wavelet and _PYWT_AVAILABLE
        self.wavelet_prob = wavelet_prob
        self.wavelet_level = wavelet_level
        self.threshold_ratio = threshold_ratio
        self.use_noise = use_noise
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        if use_wavelet and not _PYWT_AVAILABLE:
            import warnings
            warnings.warn(
                "pywt not available — wavelet augmentation disabled. "
                "Install PyWavelets: pip install PyWavelets",
                RuntimeWarning,
            )

    def _apply_one(self, x: np.ndarray) -> np.ndarray:
        x = segment_mask(x, self.num_segments, self.mask_ratio, self._rng)

        if self.use_wavelet:
            if self._rng.random() < self.wavelet_prob:
                x = wavelet_augment(x, "db4", self.wavelet_level,
                                    self.threshold_ratio)
            if self._rng.random() < self.wavelet_prob:
                x = wavelet_augment(x, "db6", self.wavelet_level,
                                    self.threshold_ratio)

        if self.use_noise:
            x = gaussian_noise(x, self.noise_std, self._rng)

        return x

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply augmentation twice independently. Returns (view1, view2)."""
        return self._apply_one(x.copy()), self._apply_one(x.copy())
