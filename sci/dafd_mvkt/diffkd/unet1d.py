"""
Lightweight 1D U-Net for DiffKD feature distillation.

Operates on feature maps [B, feature_dim, L] extracted from ResNet1d layer4.
Actual student layer4 shape: [B, 512, 16]  (input 500@50Hz → stem/2/2 → layers)
Note: user spec listed [B,512,32] but that is layer3; layer4 = [B,512,16].

Architecture
------------
base_channels=128, channel_multipliers=(1,2,4) → ch=[128,256,512]

  Input:  x_t [B,512,L] (noisy teacher feat)
  Cond:   c   [B,512,L] (student feat, conditioning)
  Time:   t   [B]       (int diffusion timestep)

  InputProj  512→128
  ┌─ Enc0    128→128  + AvgPool → L/2   [skip0 @L   ]
  │  Enc1    128→256  + AvgPool → L/4   [skip1 @L/2 ]
  │  Enc2    256→512  + AvgPool → L/8   [skip2 @L/4 ]
  │  Mid     512→512
  │  Dec0    (512+skip2=512)→256 + Up → L/4→L/2
  │  Dec1    (256+skip1=256)→128 + Up → L/2→L
  └─ Dec2    (128+skip0=128)→128
  OutputProj 128→512

Parameters: ~4.7M  (above 2-3M target due to 512-ch bottleneck;
            reduce channel_multipliers=(1,2,2) to get ~2.5M)

Conditioning modes
------------------
"concat" : project cond to block's channel dim, add elementwise (efficient)
"attn"   : cross-attention  Q=noisy_feat, K/V=student_cond  (more expressive)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── time embedding ────────────────────────────────────────────────────────────

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, shape [B, dim]."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    )                                              # [half]
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1) # [B, dim]


class TimeEmbedding(nn.Module):
    """Sinusoidal PE → MLP → [B, time_emb_dim]."""

    def __init__(self, time_emb_dim: int = 256):
        super().__init__()
        sin_dim = max(time_emb_dim // 4, 32)
        self._sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self._sin_dim))


# ── building blocks ───────────────────────────────────────────────────────────

def _groups(ch: int, max_g: int = 8) -> int:
    """Largest divisor of ch that is ≤ max_g (for GroupNorm)."""
    for g in range(max_g, 0, -1):
        if ch % g == 0:
            return g
    return 1


class ResBlock1D(nn.Module):
    """
    Residual block: GN→SiLU→Conv → +time_proj → GN→SiLU→Conv + skip.
    No spatial change; time conditioning via learned shift.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1)  # [B,out_ch,1] broadcast
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    """
    Cross-attention conditioning.
    Query  = noisy feature  [B, C, L]
    Key/V  = student cond   [B, C_cond, L_cond]
    Output = [B, C, L]
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4):
        super().__init__()
        head_dim = max(q_dim // n_heads, 8)
        inner    = head_dim * n_heads
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.scale    = head_dim ** -0.5
        self.norm = nn.GroupNorm(_groups(q_dim), q_dim)
        self.to_q  = nn.Conv1d(q_dim,  inner, 1, bias=False)
        self.to_k  = nn.Conv1d(kv_dim, inner, 1, bias=False)
        self.to_v  = nn.Conv1d(kv_dim, inner, 1, bias=False)
        self.to_out = nn.Conv1d(inner, q_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        L_c = cond.shape[-1]
        H, d = self.n_heads, self.head_dim

        q = self.to_q(self.norm(x)).reshape(B, H, d, L).permute(0, 1, 3, 2)    # [B,H,L,d]
        k = self.to_k(cond).reshape(B, H, d, L_c).permute(0, 1, 3, 2)          # [B,H,L_c,d]
        v = self.to_v(cond).reshape(B, H, d, L_c).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale                           # [B,H,L,L_c]
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).permute(0, 1, 3, 2).reshape(B, H * d, L)
        return self.to_out(out) + x                                              # residual


def _cond_resize(cond: torch.Tensor, target_L: int) -> torch.Tensor:
    if cond.shape[-1] == target_L:
        return cond
    return F.adaptive_avg_pool1d(cond, target_L)


# ── encoder / decoder blocks ──────────────────────────────────────────────────

class EncBlock(nn.Module):
    """ResBlock + cond injection + optional AvgPool downsample."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 feat_dim: int, cond_type: str, downsample: bool = True):
        super().__init__()
        self.res   = ResBlock1D(in_ch, out_ch, time_emb_dim)
        self.cond_type = cond_type
        if cond_type == "concat":
            self.cond_proj = nn.Conv1d(feat_dim, out_ch, 1, bias=False)
        else:  # "attn"
            self.cond_attn = CrossAttention1D(out_ch, feat_dim)
        self.pool = nn.AvgPool1d(2) if downsample else nn.Identity()
        self._down = downsample

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_out, skip_before_pool)."""
        h = self.res(x, t_emb)
        c = _cond_resize(cond, h.shape[-1])
        if self.cond_type == "concat":
            h = h + self.cond_proj(c)
        else:
            h = self.cond_attn(h, c)
        skip = h
        return self.pool(h), skip


class DecBlock(nn.Module):
    """1×1 cat-proj + ResBlock + cond injection + optional Upsample."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 time_emb_dim: int, feat_dim: int, cond_type: str,
                 upsample: bool = True):
        super().__init__()
        self.up = (nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_ch, in_ch, 3, padding=1, bias=False),
        ) if upsample else nn.Identity())
        self._upsample = upsample

        # cheap projection before ResBlock (avoids huge ResBlock on cat channels)
        self.cat_proj = nn.Conv1d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.res      = ResBlock1D(out_ch, out_ch, time_emb_dim)

        self.cond_type = cond_type
        if cond_type == "concat":
            self.cond_proj = nn.Conv1d(feat_dim, out_ch, 1, bias=False)
        else:
            self.cond_attn = CrossAttention1D(out_ch, feat_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if h.shape[-1] != skip.shape[-1]:                    # handle odd-dim mismatch
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
        h = self.cat_proj(torch.cat([h, skip], dim=1))
        h = self.res(h, t_emb)
        c = _cond_resize(cond, h.shape[-1])
        if self.cond_type == "concat":
            h = h + self.cond_proj(c)
        else:
            h = self.cond_attn(h, c)
        return h


# ── main U-Net ────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    Lightweight 1D U-Net for DiffKD.

    Parameters
    ----------
    feature_dim          : channel dim of feature maps (default 512 for layer4)
    base_channels        : first encoder channel width (default 128)
    channel_multipliers  : per-level multipliers; len == num_levels (default (1,2,4))
    time_emb_dim         : time embedding MLP output dim (default 256)
    cond_type            : "concat" | "attn"

    Forward
    -------
    x_t:   [B, feature_dim, L]   noisy teacher feature at timestep t
    t:     [B]                   int diffusion timestep in [0, T-1]
    cond:  [B, feature_dim, L]   student feature (conditioning)
    → out: [B, feature_dim, L]   v-prediction (or ε-prediction)
    """

    def __init__(
        self,
        feature_dim:         int   = 512,
        base_channels:       int   = 128,
        channel_multipliers: tuple = (1, 2, 4),
        time_emb_dim:        int   = 256,
        cond_type:           str   = "concat",
    ):
        super().__init__()
        assert cond_type in ("concat", "attn"), f"unknown cond_type={cond_type}"
        self.feature_dim = feature_dim

        chs = [base_channels * m for m in channel_multipliers]  # [128,256,512]
        n   = len(chs)                                           # 3

        self.time_emb   = TimeEmbedding(time_emb_dim)
        self.input_proj = nn.Conv1d(feature_dim, chs[0], 1, bias=False)

        # encoder
        self.enc_blocks = nn.ModuleList()
        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            self.enc_blocks.append(
                EncBlock(in_ch, out_ch, time_emb_dim, feature_dim, cond_type,
                         downsample=(i < n - 1))   # last level: no downsample
            )
            in_ch = out_ch

        # bottleneck
        self.mid = ResBlock1D(chs[-1], chs[-1], time_emb_dim)

        # decoder  (reverse order; skip_ch[i] = chs[n-1-i])
        self.dec_blocks = nn.ModuleList()
        for i in range(n):
            skip_ch = chs[n - 1 - i]
            out_ch  = chs[max(n - 2 - i, 0)]
            self.dec_blocks.append(
                DecBlock(in_ch, skip_ch, out_ch, time_emb_dim, feature_dim,
                         cond_type, upsample=(i < n - 1))
            )
            in_ch = out_ch

        self.output_proj = nn.Sequential(
            nn.GroupNorm(_groups(chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv1d(chs[0], feature_dim, 1),
        )

    def forward(
        self,
        x_t:  torch.Tensor,
        t:    torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)             # [B, time_emb_dim]
        h     = self.input_proj(x_t)         # [B, chs[0], L]

        skips: list[torch.Tensor] = []
        for blk in self.enc_blocks:
            h, skip = blk(h, t_emb, cond)
            skips.append(skip)

        h = self.mid(h, t_emb)

        for blk, skip in zip(self.dec_blocks, reversed(skips)):
            h = blk(h, skip, t_emb, cond)

        return self.output_proj(h)           # [B, feature_dim, L]

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── convenience builder ───────────────────────────────────────────────────────

def build_unet(
    feature_dim:   int = 512,
    base_channels: int = 128,
    mults:         tuple = (1, 2, 4),
    time_emb_dim:  int = 256,
    cond_type:     str = "concat",
) -> UNet1D:
    return UNet1D(
        feature_dim=feature_dim,
        base_channels=base_channels,
        channel_multipliers=mults,
        time_emb_dim=time_emb_dim,
        cond_type=cond_type,
    )
