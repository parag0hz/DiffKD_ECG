"""
Temporal Segment Feature Knowledge Distillation.

Divides aligned feature maps into temporal segments and computes
segment-level MSE, optionally weighted by TA activation magnitude.
This captures local P-QRS-T morphology rather than global pooled statistics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSegmentFeatureKDLoss(nn.Module):
    """
    Segment-level feature KD between student and TA feature maps.

    Args:
        student_channels:    C_s
        ta_channels:         C_t
        num_segments:        number of temporal segments (default 8)
        pool_per_segment:    temporal length per segment after pooling (default 16)
        normalize:           L2-normalize along channel dim before MSE
        attention_weighted:  weight segments by TA activation magnitude
        eps:                 stability epsilon for attention normalisation
    """

    def __init__(
        self,
        student_channels: int,
        ta_channels: int,
        num_segments: int = 8,
        pool_per_segment: int = 16,
        normalize: bool = True,
        attention_weighted: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.pool_per_segment = pool_per_segment
        self.aligned_len = num_segments * pool_per_segment
        self.normalize = normalize
        self.attention_weighted = attention_weighted
        self.eps = eps

        if student_channels != ta_channels:
            self.proj: nn.Module = nn.Conv1d(
                student_channels, ta_channels, kernel_size=1, bias=False
            )
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        student_feature_map: torch.Tensor,   # [B, C_s, T_s]
        ta_feature_map: torch.Tensor,         # [B, C_t, T_t]
    ) -> torch.Tensor:
        ta_feature_map = ta_feature_map.detach()

        B = student_feature_map.size(0)

        # Temporal alignment
        s = F.adaptive_avg_pool1d(student_feature_map, self.aligned_len)  # [B, C_s, L]
        t = F.adaptive_avg_pool1d(ta_feature_map,      self.aligned_len)  # [B, C_t, L]

        # Channel alignment
        s = self.proj(s)   # [B, C_t, L]

        # L2 normalise along channel dim
        if self.normalize:
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)

        # Reshape into segments: [B, C_t, num_segments, pool_per_segment]
        C = s.size(1)
        s = s.view(B, C, self.num_segments, self.pool_per_segment)
        t = t.view(B, C, self.num_segments, self.pool_per_segment)

        # Segment attention weights
        if self.attention_weighted:
            # mean |TA| over channel and pool_per_segment dims → [B, num_segments]
            seg_weight = t.abs().mean(dim=(1, 3))
            # normalize per sample
            seg_weight = seg_weight / (
                seg_weight.sum(dim=-1, keepdim=True) + self.eps
            )  # [B, num_segments]
        else:
            seg_weight = torch.full(
                (B, self.num_segments),
                1.0 / self.num_segments,
                device=s.device, dtype=s.dtype,
            )

        # MSE per segment: mean over (C, pool_per_segment) → [B, num_segments]
        mse_per_seg = (s - t).pow(2).mean(dim=(1, 3))   # [B, num_segments]

        # Weighted mean
        return (seg_weight * mse_per_seg).sum(dim=-1).mean()
