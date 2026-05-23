"""
1D ResNet backbone for ECG classification.

Supports variable input channels and sequence lengths:
  - Teacher: in_channels=12, length=5000 (500 Hz × 10 s)
  - Student: in_channels=1,  length=1000 (100 Hz) or 500 (50 Hz)

Forward output dict (when return_features=True):
  logits:      [B, num_classes]
  pooled:      [B, feat_dim]         after global average pool
  feature_map: [B, feat_dim, T']     before global pool
  proj:        [B, proj_dim]         L2-normalized contrastive embedding

Attention support (attention_type != "none"):
  Pass attention_type and attention_layers to __init__.
  Pass return_attention=True to forward to get:
    attentions:  dict[str, Tensor [B,1,L]]  keyed by layer name
  attention_type="none" (default) is backward-compatible with all
  existing checkpoints and produces identical outputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import ProjectionHead


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet1d(nn.Module):
    """
    1D ResNet-34 style backbone.

    Args
    ----
    in_channels    : number of input leads
    num_classes    : classification output size
    layers         : number of BasicBlocks per stage [3, 4, 6, 3]
    base_channels  : first-stage channel count (64)
    proj_dim       : contrastive projection head output dim (128)
    dropout        : dropout within each block
    attention_type : one of {"none", "se", "cbam", "tgmta"}
                     default "none" — identical to the original model
    attention_layers : list of stage names to attach attention to,
                     e.g. ["layer3", "layer4"].  Ignored when
                     attention_type="none".
    """

    _STAGE_NAMES    = ["layer1", "layer2", "layer3", "layer4"]
    _STAGE_CHANNELS = [64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        layers: list[int] | None = None,
        base_channels: int = 64,
        proj_dim: int = 128,
        dropout: float = 0.0,
        attention_type: str = "none",
        attention_layers: list[str] | None = None,
        use_wide_kernel: bool = False,
    ):
        super().__init__()
        if layers is None:
            layers = [3, 4, 6, 3]

        channels = [base_channels * (2 ** i) for i in range(4)]

        # stem: Conv + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # residual stages — kept as layer_blocks to preserve checkpoint keys
        blocks: list[nn.Module] = []
        in_ch = channels[0]
        for stage, (n_blocks, out_ch) in enumerate(zip(layers, channels)):
            for blk in range(n_blocks):
                stride = 2 if (blk == 0 and stage > 0) else 1
                blocks.append(BasicBlock1d(in_ch, out_ch, stride, dropout))
                in_ch = out_ch
        self.layer_blocks = nn.Sequential(*blocks)

        # stage boundary indices inside layer_blocks
        _ends: list[int] = []
        idx = 0
        for n in layers:
            idx += n
            _ends.append(idx)
        self._stage_ends: list[int] = _ends   # e.g. [3,7,13,16]

        self.feat_dim = channels[-1]   # 512 for base_channels=64

        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.proj_head  = ProjectionHead(self.feat_dim, proj_dim)

        # ── optional attention ────────────────────────────────────────────────
        self.attention_type = attention_type
        _att_layers = set(attention_layers or [])

        self.attentions = nn.ModuleDict()
        if attention_type != "none":
            from models.attention_modules import build_attention
            for i, name in enumerate(self._STAGE_NAMES):
                if name in _att_layers:
                    self.attentions[name] = build_attention(
                        attention_type, channels[i],
                        use_wide_kernel=use_wide_kernel,
                    )
        # ─────────────────────────────────────────────────────────────────────

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
        self,
        x: torch.Tensor,
        return_features: bool = True,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args
        ----
        x               : [B, C, L]
        return_features : if False, returns only logits in dict
        return_attention: if True, adds "attentions" key to output dict
                          (only meaningful when attention_type != "none")

        Returns dict with:
            logits      : [B, num_classes]
            pooled      : [B, feat_dim]          (if return_features)
            feature_map : [B, feat_dim, T']       (if return_features)
            proj        : [B, proj_dim]           (if return_features)
            attentions  : dict[str, [B,1,L]]      (if return_attention)
        """
        h = self.stem(x)                               # [B, 64, L/4]

        # ── fast path — no attention, no per-stage tracking ───────────────
        if self.attention_type == "none":
            feature_map = self.layer_blocks(h)         # [B, 512, T']
            att_dict: dict[str, torch.Tensor] = {}
        else:
            # ── per-stage path ─────────────────────────────────────────────
            att_dict = {}
            start = 0
            for name, end in zip(self._STAGE_NAMES, self._stage_ends):
                for j in range(start, end):
                    h = self.layer_blocks[j](h)
                if name in self.attentions:
                    h, att = self.attentions[name](h)
                    att_dict[name] = att
                start = end
            feature_map = h                            # [B, 512, T']

        pooled = self.gap(feature_map).squeeze(-1)     # [B, 512]
        logits = self.classifier(pooled)               # [B, num_classes]

        out: dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            out["pooled"]      = pooled
            out["feature_map"] = feature_map
            out["proj"]        = self.proj_head(pooled)
        if return_attention:
            out["attentions"]  = att_dict
        return out
