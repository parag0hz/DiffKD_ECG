"""
Temporal attention modules for ECG student model.

Modules
-------
SE1D          : Squeeze-and-Excitation (channel-only)
CBAM1D        : CBAM simplified for 1-D signals (channel + temporal)
TGMTAModule   : Teacher-Guided Multi-Scale Temporal Attention

All modules share the same forward signature:
    x            [B, C, L]  →  (enhanced_x [B, C, L],  temporal_att [B, 1, L])

When used in ResNet1d with attention_type="none" the modules are never
instantiated, so there is no overhead.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SE-1D
# ─────────────────────────────────────────────────────────────────────────────
class SE1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                     # [B, C, 1]
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # pseudo temporal attention (constant 1s for API compat)
        self._ones: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        scale = self.se(x)                               # [B, C, 1]
        enhanced = x * scale
        # return a flat temporal map of all-ones for API compatibility
        B, C, L = x.shape
        temporal_att = x.new_ones(B, 1, L)
        return enhanced, temporal_att


# ─────────────────────────────────────────────────────────────────────────────
# CBAM-1D
# ─────────────────────────────────────────────────────────────────────────────
class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module for 1-D feature maps.

    Channel attention: SE-style (avg + max pooling → MLP)
    Temporal attention: avg+max along channel → Conv1d → Sigmoid
    """

    def __init__(self, channels: int, reduction: int = 16,
                 temporal_kernel: int = 7):
        super().__init__()
        mid = max(channels // reduction, 4)

        # channel attention
        self.ca_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

        # temporal attention
        pad = temporal_kernel // 2
        self.ta_conv = nn.Sequential(
            nn.Conv1d(2, 1, temporal_kernel, padding=pad, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def _channel_att(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=-1)                             # [B, C]
        mx  = x.max(dim=-1)[0]                          # [B, C]
        return torch.sigmoid(self.ca_mlp(avg) + self.ca_mlp(mx)).unsqueeze(-1)

    def forward(self, x: torch.Tensor):
        x = x * self._channel_att(x)                    # channel refine
        avg = x.mean(dim=1, keepdim=True)               # [B, 1, L]
        mx  = x.max(dim=1, keepdim=True)[0]             # [B, 1, L]
        temporal_att = self.ta_conv(torch.cat([avg, mx], dim=1))   # [B,1,L]
        enhanced = x * temporal_att
        return enhanced, temporal_att


# ─────────────────────────────────────────────────────────────────────────────
# TG-MTA (Teacher-Guided Multi-Scale Temporal Attention)
# ─────────────────────────────────────────────────────────────────────────────
class TGMTAModule(nn.Module):
    """
    Multi-scale depthwise temporal convolution + channel gate.

    Architecture
    ~~~~~~~~~~~~
    Input x [B, C, L]
      → 3–4 parallel depthwise Conv1d with different kernels/dilations
      → concatenate [B, C*K, L]
      → 1×1 Conv fuse   [B, C, L]
      → temporal attention head  [B, 1, L]
      → channel gate (ECA-style)  [B, C, 1]
      → enhanced_x = x + γ * (x * temporal_att * channel_gate)

    The γ (gamma) residual scale is initialised to 0 so the module
    starts as an identity and gradually learns to modify features.

    Args
    ----
    channels        : input channel count
    reduction       : channel gate bottleneck ratio
    use_wide_kernel : if True, adds (k=31, d=2) branch
    """

    _KERNELS = [(3, 1), (7, 1), (15, 2)]          # (kernel, dilation)
    _WIDE    = (31, 2)

    def __init__(self, channels: int, reduction: int = 16,
                 use_wide_kernel: bool = False):
        super().__init__()
        kernels = list(self._KERNELS)
        if use_wide_kernel:
            kernels.append(self._WIDE)

        self.branches = nn.ModuleList()
        for k, d in kernels:
            pad = d * (k - 1) // 2
            self.branches.append(
                nn.Conv1d(channels, channels, k,
                          dilation=d, padding=pad,
                          groups=channels, bias=False)
            )

        n = len(kernels)
        # fuse multi-scale context
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * n, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # temporal attention head
        self.temporal_head = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # lightweight channel gate
        mid = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                     # [B, C, 1]
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # learnable residual scale — init 0 → identity at start
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # multi-scale depthwise
        branches = [b(x) for b in self.branches]       # each [B, C, L]
        fused = self.fuse(torch.cat(branches, dim=1))   # [B, C, L]

        temporal_att  = self.temporal_head(fused)       # [B, 1, L]
        channel_gate  = self.channel_gate(fused)        # [B, C, 1]

        enhanced = x + self.gamma * (x * temporal_att * channel_gate)
        return enhanced, temporal_att


# ─────────────────────────────────────────────────────────────────────────────
# factory
# ─────────────────────────────────────────────────────────────────────────────
def build_attention(
    attention_type: str,
    channels: int,
    use_wide_kernel: bool = False,
) -> nn.Module:
    """Return the appropriate attention module for the given type."""
    if attention_type == "se":
        return SE1D(channels)
    elif attention_type == "cbam":
        return CBAM1D(channels)
    elif attention_type == "tgmta":
        return TGMTAModule(channels, use_wide_kernel=use_wide_kernel)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type!r}")
