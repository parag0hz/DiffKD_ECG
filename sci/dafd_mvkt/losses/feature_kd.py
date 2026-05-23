"""
Feature-space Knowledge Distillation loss.

Aligns student and TA (teacher assistant) temporal feature maps in a
common spatial representation via:
  1. Temporal alignment  — AdaptiveAvgPool1d to pool_size
  2. Channel alignment   — 1×1 Conv1d projection if channels differ
  3. L2 normalisation    — along channel dimension (optional)
  4. MSE loss            — between aligned feature maps
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    """
    Feature KD between student and TA feature maps.

    Args:
        student_channels: channel dim of student feature map (C_s)
        ta_channels:      channel dim of TA feature map (C_t)
        pool_size:        target temporal length after pooling (K)
        normalize:        if True, L2-normalize along channel dim before MSE
    """

    def __init__(
        self,
        student_channels: int,
        ta_channels: int,
        pool_size: int = 128,
        normalize: bool = True,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.normalize = normalize

        # 1×1 Conv1d maps student channels → TA channels if they differ
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

        # Align temporal dimension
        s = F.adaptive_avg_pool1d(student_feature_map, self.pool_size)   # [B, C_s, K]
        t = F.adaptive_avg_pool1d(ta_feature_map,      self.pool_size)   # [B, C_t, K]

        # Align channel dimension
        s = self.proj(s)   # [B, C_t, K]

        # L2 normalize along channel dim
        if self.normalize:
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)

        return F.mse_loss(s, t)
