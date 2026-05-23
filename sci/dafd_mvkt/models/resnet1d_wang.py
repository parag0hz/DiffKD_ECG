"""
ResNet1d-Wang: Time series classification baseline (Wang et al., IJCNN 2017).

Architecture:
  3 residual blocks with decreasing→increasing channels (128→256→128).
  Each block: 3 Conv1d layers with kernel sizes [8, 5, 3] + shortcut.
  Global Average Pooling → Linear classifier.

Reference:
  Wang, Z., Yan, W., & Oates, T. (2017). Time series classification from
  scratch with deep neural networks: A strong baseline. IJCNN.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import ProjectionHead


class WangBlock(nn.Module):
    """Single residual block from Wang et al.

    Three conv layers with kernel sizes [8, 5, 3], all same-padding.
    Shortcut: 1×1 conv if channels change, else identity.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=8, padding=4, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # trim/pad to same length after conv (handles even-length edge cases)
        res = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        # align lengths if off-by-one from padding
        if out.size(-1) != res.size(-1):
            out = out[..., :res.size(-1)]

        return F.relu(out + res, inplace=True)


class ResNet1dWang(nn.Module):
    """
    Wang et al. ResNet1d for ECG / time-series classification.

    Channel progression: in_ch → 128 → 256 → 128 → GAP → classifier.

    Args:
        in_channels:  number of input leads (12 for teacher, 1 for student)
        num_classes:  output size
        proj_dim:     contrastive projection head output dim
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        proj_dim: int = 128,
    ):
        super().__init__()

        self.block1 = WangBlock(in_channels, 128)
        self.block2 = WangBlock(128, 256)
        self.block3 = WangBlock(256, 128)

        self.feat_dim = 128

        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.proj_head  = ProjectionHead(self.feat_dim, proj_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:               [B, C, L]
            return_features: if False, returns only logits

        Returns dict:
            logits:      [B, num_classes]
            pooled:      [B, feat_dim]
            feature_map: [B, feat_dim, T']
            proj:        [B, proj_dim]  L2-normalized
        """
        h           = self.block1(x)           # [B, 128, L]
        h           = self.block2(h)           # [B, 256, L]
        feature_map = self.block3(h)           # [B, 128, L]
        pooled      = self.gap(feature_map).squeeze(-1)   # [B, 128]
        logits      = self.classifier(pooled)             # [B, C]

        out: dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            out["pooled"]      = pooled
            out["feature_map"] = feature_map
            out["proj"]        = self.proj_head(pooled)
        return out
