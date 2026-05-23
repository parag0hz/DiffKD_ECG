"""
Diagnosis-Aware Frequency Distillation (DAF) loss.  ← main novelty

Aligns the frequency-domain representations of teacher and student temporal
feature maps, weighted by a learnable diagnosis-frequency gate.

Gate design:
  gate_logits: [C, n_freq]  learnable (C = num_classes, n_freq ≤ max_freq)
  gate        = sigmoid(gate_logits)              → [C, n_freq]
  weights     = teacher_probs @ gate              → [B, n_freq]
  weights     = weights / (weights.sum(-1) + ε)  (per-sample normalisation)

  L_DAF = mean( weights * (student_spec - teacher_spec)^2 )

Spectrum computation (per forward pass):
  1. Mean over channel dim → [B, T_s] (student) / [B, T_t] (teacher)
  2. Adaptive-pool teacher to T_s (downsampling only → no artifacts)
  3. rfft → magnitude, drop DC bin → log1p → L2-normalise → [B, F]
  where F = T_s // 2  (at most max_freq)
  4. Truncate gate to F bins for the weighted loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosisAwareFrequencyDistillation(nn.Module):
    """
    DAF loss module with learnable diagnosis-frequency gate.

    Args:
        num_classes:    number of diagnostic superclasses (5 for PTB-XL)
        teacher_ch:     teacher feature map channel dim (unused, kept for API)
        student_ch:     student feature map channel dim (unused, kept for API)
        common_dim:     unused, kept for API compatibility
        pool_size:      maximum number of frequency bins in the gate
        eps:            numerical stability constant
    """

    def __init__(
        self,
        num_classes: int = 5,
        teacher_ch: int = 512,
        student_ch: int = 512,
        common_dim: int = 256,
        pool_size: int = 128,     # used as max_freq cap
        eps: float = 1e-6,
    ):
        super().__init__()
        # Store max_freq; actual n_freq is determined per forward call
        # (= student feature-map temporal length // 2, capped at max_freq)
        self.max_freq    = pool_size // 2   # e.g. 64 for pool_size=128
        self.num_classes = num_classes
        self.eps         = eps

        # learnable diagnosis-frequency gate [C, max_freq]
        # At runtime we use only the first n_freq columns
        self.gate_logits = nn.Parameter(
            torch.zeros(num_classes, self.max_freq)
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _spectrum(signal: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Compute log-magnitude spectrum of a 1-D batch signal, DC dropped.

        Args:
            signal: [B, T]  (already pooled to desired length)

        Returns:
            spec:   [B, T//2]  L2-normalised log-magnitude
        """
        fft_out = torch.fft.rfft(signal, dim=-1)     # [B, T//2 + 1]
        mag = fft_out.abs()[:, 1:]                   # drop DC → [B, T//2]
        mag = torch.log1p(mag)
        mag = mag / (mag.norm(dim=-1, keepdim=True) + eps)
        return mag                                    # [B, T//2]

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        student_feature_map: torch.Tensor,   # [B, D_s, T_s]
        teacher_feature_map: torch.Tensor,   # [B, D_t, T_t]
        teacher_logits: torch.Tensor,        # [B, C]
    ) -> torch.Tensor:
        """
        Args:
            student_feature_map: temporal feature map from student [B, D_s, T_s]
            teacher_feature_map: temporal feature map from teacher (detached)
            teacher_logits:      teacher class logits (detached)

        Returns:
            Scalar DAF loss.
        """
        teacher_feature_map = teacher_feature_map.detach()
        teacher_logits      = teacher_logits.detach()

        T_s = student_feature_map.shape[-1]   # e.g. 31 (100 Hz) or 15 (50 Hz)
        T_t = teacher_feature_map.shape[-1]   # e.g. 157

        # 1. Channel average → temporal signal [B, T]
        s_signal = student_feature_map.mean(dim=1)    # [B, T_s]
        t_signal = teacher_feature_map.mean(dim=1)    # [B, T_t]

        # 2. Pool teacher DOWN to student temporal size (never upsample teacher)
        #    student signal needs no pooling
        if T_t != T_s:
            t_signal = F.adaptive_avg_pool1d(
                t_signal.unsqueeze(1), T_s
            ).squeeze(1)                              # [B, T_s]

        # 3. rfft → log-magnitude spectrum, DC dropped → [B, T_s // 2]
        s_spec = self._spectrum(s_signal, self.eps)
        t_spec = self._spectrum(t_signal, self.eps)

        n_freq = s_spec.shape[-1]   # T_s // 2 (e.g. 15 or 7)

        # 4. Squared spectral difference [B, n_freq]
        diff = (s_spec - t_spec) ** 2

        # 5. Diagnosis-aware gate: use first n_freq columns
        n = min(n_freq, self.max_freq)
        gate = torch.sigmoid(self.gate_logits[:, :n])     # [C, n]
        teacher_probs = torch.sigmoid(teacher_logits)      # [B, C]
        weights = teacher_probs @ gate                     # [B, n]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

        # 6. Weighted spectral loss (over first n bins)
        return (weights * diff[:, :n]).mean()
