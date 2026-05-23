"""
RatePrompt ResNet1d — ResNet1d with optional sampling-rate conditioning.

rate_conditioning modes
-----------------------
"none"          Identical to ResNet1d. No new parameters.
"time_pe"       Concatenate ActualTimePositionalEncoding to input channels.
                Backbone in_channels = orig_in_channels + time_pe_dim.
"fs_film"       FiLM conditioning applied after each configured stage.
                Backbone in_channels unchanged; FiLM modules are new.
"time_pe_film"  Both time PE and FiLM.

Backward compat
---------------
rate_conditioning="none" → use ResNet1d directly (see _build_student).
When rate_conditioning != "none", load SimCLR weights via
  load_encoder_init(student.backbone, ckpt_path, strict=False)
so that stem mismatch (time_pe case) is safely skipped.

Forward signature
-----------------
model(x)                      rate_conditioning="none" compatible
model(x, fs=fs_tensor)        when conditioning is active
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.resnet1d import ResNet1d
from models.rate_conditioning import (
    ActualTimePositionalEncoding,
    SamplingRateEmbedding,
    FiLMLayer,
)


class RatePromptResNet1d(nn.Module):
    """
    ResNet1d wrapped with optional rate conditioning (time PE and/or FiLM).

    Parameters
    ----------
    in_channels        : original signal channels (1 for single-lead ECG)
    rate_conditioning  : "none" | "time_pe" | "fs_film" | "time_pe_film"
    time_pe_dim        : PE feature dim (must equal 1+2*K for K freqs; default 15)
    fs_embed_dim       : SamplingRateEmbedding output dim
    film_layers        : stage names to apply FiLM after
                         (default all 4: layer1,layer2,layer3,layer4)
    **resnet_kwargs    : passed to ResNet1d (num_classes, layers, etc.)
    """

    def __init__(
        self,
        in_channels:       int = 1,
        rate_conditioning: str = "none",
        time_pe_dim:       int = 15,
        fs_embed_dim:      int = 64,
        film_layers:       list[str] | None = None,
        **resnet_kwargs,
    ):
        super().__init__()
        self.rate_conditioning = rate_conditioning

        # Stem channel width increases when PE is concatenated
        use_pe   = rate_conditioning in ("time_pe",  "time_pe_film")
        use_film = rate_conditioning in ("fs_film",  "time_pe_film")

        backbone_in_ch = in_channels + time_pe_dim if use_pe else in_channels

        self.backbone = ResNet1d(in_channels=backbone_in_ch, **resnet_kwargs)
        self.feat_dim = self.backbone.feat_dim   # proxy for callers

        # ── time PE ──────────────────────────────────────────────────────────
        self.time_pe_module: ActualTimePositionalEncoding | None = None
        if use_pe:
            self.time_pe_module = ActualTimePositionalEncoding()
            assert self.time_pe_module.dim == time_pe_dim, (
                f"time_pe_dim arg={time_pe_dim} but "
                f"ActualTimePositionalEncoding.dim={self.time_pe_module.dim}. "
                f"Set time_pe_dim=1+2*len(frequencies)."
            )

        # ── fs embedding + FiLM ───────────────────────────────────────────────
        self.fs_embed_module: SamplingRateEmbedding | None = None
        self.film_modules = nn.ModuleDict()
        if use_film:
            self.fs_embed_module = SamplingRateEmbedding(fs_embed_dim)
            film_layer_set = set(
                film_layers if film_layers is not None
                else ["layer1", "layer2", "layer3", "layer4"]
            )
            for i, name in enumerate(self.backbone._STAGE_NAMES):
                if name in film_layer_set:
                    ch = self.backbone._STAGE_CHANNELS[i]
                    self.film_modules[name] = FiLMLayer(ch, fs_embed_dim)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x:                torch.Tensor,
        fs:               torch.Tensor | float | None = None,
        return_features:  bool = True,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        x  : [B, in_channels, L]
        fs : [B] float tensor or scalar (Hz).  Required when conditioning != "none".
        """
        if self.rate_conditioning != "none" and fs is None:
            raise ValueError(
                f"rate_conditioning='{self.rate_conditioning}' requires fs to be provided"
            )

        # ── 1. Time PE concatenation ─────────────────────────────────────────
        if self.time_pe_module is not None:
            B, _, L = x.shape
            pe = self.time_pe_module(B, L, fs, x.device)   # [B, time_pe_dim, L]
            x  = torch.cat([x, pe], dim=1)                 # [B, 1+pe_dim, L]

        # ── 2. FS embedding ──────────────────────────────────────────────────
        e_fs: torch.Tensor | None = None
        if self.fs_embed_module is not None:
            if not isinstance(fs, torch.Tensor):
                fs_t = torch.full((x.shape[0],), float(fs),
                                  dtype=torch.float32, device=x.device)
            else:
                fs_t = fs.to(dtype=torch.float32, device=x.device)
            e_fs = self.fs_embed_module(fs_t)               # [B, fs_embed_dim]

        # ── 3. Backbone forward (with or without FiLM) ───────────────────────
        if self.film_modules:
            return self._forward_with_film(x, e_fs, return_features, return_attention)
        else:
            return self.backbone(x,
                                 return_features=return_features,
                                 return_attention=return_attention)

    def _forward_with_film(
        self,
        x:               torch.Tensor,
        e_fs:            torch.Tensor,
        return_features: bool,
        return_attention: bool,
    ) -> dict[str, torch.Tensor]:
        h = self.backbone.stem(x)

        att_dict: dict[str, torch.Tensor] = {}
        start = 0
        for name, end in zip(self.backbone._STAGE_NAMES, self.backbone._stage_ends):
            for j in range(start, end):
                h = self.backbone.layer_blocks[j](h)
            if name in self.film_modules:
                h = self.film_modules[name](h, e_fs)
            if name in self.backbone.attentions:
                h, att = self.backbone.attentions[name](h)
                att_dict[name] = att
            start = end

        feature_map = h
        pooled = self.backbone.gap(feature_map).squeeze(-1)   # [B, 512]
        logits = self.backbone.classifier(pooled)             # [B, 5]

        out: dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            out["pooled"]      = pooled
            out["feature_map"] = feature_map
            out["proj"]        = self.backbone.proj_head(pooled)
        if return_attention:
            out["attentions"] = att_dict
        return out


# ── convenience builder ───────────────────────────────────────────────────────

def build_student(
    rate_conditioning: str = "none",
    time_pe_dim:       int = 15,
    fs_embed_dim:      int = 64,
    film_layers:       list[str] | None = None,
) -> nn.Module:
    """
    Return the appropriate student model.

    rate_conditioning="none" → plain ResNet1d (identical to F02T01 student).
    Otherwise              → RatePromptResNet1d.
    """
    shared = dict(
        num_classes=5,
        layers=[3, 4, 6, 3],
        base_channels=64,
        proj_dim=128,
        dropout=0.0,
    )
    if rate_conditioning == "none":
        return ResNet1d(in_channels=1, **shared)

    return RatePromptResNet1d(
        in_channels=1,
        rate_conditioning=rate_conditioning,
        time_pe_dim=time_pe_dim,
        fs_embed_dim=fs_embed_dim,
        film_layers=film_layers,
        **shared,
    )
