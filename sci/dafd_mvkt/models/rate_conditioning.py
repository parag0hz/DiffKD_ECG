"""
RatePrompt-KD conditioning modules.

Components
----------
ActualTimePositionalEncoding
    PE based on actual time t = i/fs (seconds).
    Output: [B, 1+2K, L] where K = len(frequencies).

SamplingRateEmbedding
    Maps scalar fs → fixed-dim embedding via r=log(fs/50) → MLP.
    Output: [B, embed_dim]

FiLMLayer
    Feature-wise Linear Modulation conditioned on sampling-rate embedding.
    x[B,C,L] × (1+gamma) + beta,  gamma/beta [B,C,1] from MLP(e_fs).
    Initialized near-zero so the model starts close to an unconditioned ResNet.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

_FREQ_DEFAULTS = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]   # 7 frequencies → dim=15


class ActualTimePositionalEncoding(nn.Module):
    """
    Actual-time positional encoding.

    PE(i) = [t/10, sin(2π f_k t), cos(2π f_k t)  for k in 0..K-1]
    where  t = i / fs  (seconds)

    dim = 1 + 2*K  (default 15 for 7 frequencies)
    """

    def __init__(self, frequencies: list[float] | None = None):
        super().__init__()
        if frequencies is None:
            frequencies = _FREQ_DEFAULTS
        self.register_buffer(
            "freqs",
            torch.tensor(frequencies, dtype=torch.float32),  # [K]
        )

    @property
    def dim(self) -> int:
        return 1 + 2 * len(self.freqs)

    def forward(
        self,
        B: int,
        L: int,
        fs: float | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns [B, dim, L] PE tensor.

        fs: scalar float or [B] tensor — uses mean when tensor.
        """
        fs_val = float(fs.float().mean().item()) if isinstance(fs, torch.Tensor) else float(fs)
        t = torch.arange(L, dtype=torch.float32, device=device) / fs_val   # [L]  seconds

        t_norm  = (t / 10.0).unsqueeze(0)                                  # [1, L]
        sin_enc = torch.sin(2.0 * math.pi * self.freqs.unsqueeze(-1) * t)  # [K, L]
        cos_enc = torch.cos(2.0 * math.pi * self.freqs.unsqueeze(-1) * t)  # [K, L]

        pe = torch.cat([t_norm, sin_enc, cos_enc], dim=0)  # [1+2K, L]
        return pe.unsqueeze(0).expand(B, -1, -1).contiguous()              # [B, dim, L]


class SamplingRateEmbedding(nn.Module):
    """
    Embeds sampling rate into a fixed-dim vector.

    r = log(fs / 50)  →  MLP(r)  →  [B, embed_dim]

    fs values: 50 → r=0,  100 → r≈0.693,  250 → r≈1.609,  500 → r≈2.303
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fs: torch.Tensor) -> torch.Tensor:
        """fs: [B] float tensor.  Returns [B, embed_dim]."""
        r = torch.log(fs.float() / 50.0).unsqueeze(-1)  # [B, 1]
        return self.mlp(r)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    x:  [B, C, L]  — feature map from a ResNet stage
    e:  [B, D]     — sampling-rate embedding

    gamma, beta = Linear(e)  →  reshape to [B, C, 1]
    output = x * (1 + gamma) + beta

    Weight/bias initialized to zero so gamma≈0 and beta≈0 at init,
    meaning the model starts identical to the unconditioned ResNet.
    """

    def __init__(self, num_channels: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * num_channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        e: [B, D]
        """
        params = self.fc(e)                    # [B, 2C]
        gamma, beta = params.chunk(2, dim=-1)  # [B, C] each
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
