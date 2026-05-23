"""
DDPM Scheduler with v-parameterization for DiffKD.

Notation
--------
x_0  : clean teacher feature
x_t  : noisy version at timestep t
ε    : noise ~ N(0,I)
α_t  : sqrt(ᾱ_t)   — "signal coefficient"
σ_t  : sqrt(1-ᾱ_t) — "noise coefficient"

Forward process:  x_t = α_t * x_0 + σ_t * ε

v-parameterization (Salimans & Ho 2022):
  v_t = α_t * ε - σ_t * x_0

  From v and x_t we recover:
    x_0_pred = α_t * x_t - σ_t * v_pred
    ε_pred   = σ_t * x_t + α_t * v_pred

Training loss:
  L = MSE(v_pred, v_target)

Inference (DDIM, deterministic, n_steps ≤ num_timesteps):
  x_{t-1} = α_{t-1}/α_t * (x_t - σ_t * pred_ε) + σ_{t-1} * pred_ε
  where pred_ε is recovered from v_pred.
"""
from __future__ import annotations
import torch
import numpy as np


class DDPMScheduler:
    """
    DDPM / DDIM scheduler with v-parameterization.

    Parameters
    ----------
    num_timesteps  : total diffusion steps T (default 100)
    beta_start     : start of linear beta schedule (default 1e-4)
    beta_end       : end of linear beta schedule (default 0.02)
    prediction_type: "v_prediction" | "epsilon"
    """

    def __init__(
        self,
        num_timesteps:   int   = 100,
        beta_start:      float = 1e-4,
        beta_end:        float = 0.02,
        prediction_type: str   = "v_prediction",
    ):
        assert prediction_type in ("v_prediction", "epsilon")
        self.num_timesteps   = num_timesteps
        self.prediction_type = prediction_type

        # ── linear beta schedule ──────────────────────────────────────────────
        betas      = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        alphas     = 1.0 - betas
        alphas_bar = np.cumprod(alphas)                    # ᾱ_t

        # α_t = sqrt(ᾱ_t),  σ_t = sqrt(1-ᾱ_t)
        sqrt_alphas_bar      = np.sqrt(alphas_bar)
        sqrt_one_minus_ab    = np.sqrt(1.0 - alphas_bar)

        # previous ᾱ (for DDIM reverse step)
        alphas_bar_prev      = np.concatenate([[1.0], alphas_bar[:-1]])
        sqrt_alphas_bar_prev = np.sqrt(alphas_bar_prev)
        sqrt_one_minus_prev  = np.sqrt(1.0 - alphas_bar_prev)

        # Register all as float32 tensors (not nn.Parameters — scheduler has no grad)
        def _t(x): return torch.tensor(x, dtype=torch.float32)

        self.betas               = _t(betas)
        self.alphas_bar          = _t(alphas_bar)
        self.sqrt_ab             = _t(sqrt_alphas_bar)      # α_t
        self.sqrt_one_minus_ab   = _t(sqrt_one_minus_ab)   # σ_t
        self.sqrt_ab_prev        = _t(sqrt_alphas_bar_prev)
        self.sqrt_om_prev        = _t(sqrt_one_minus_prev)

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in ("betas", "alphas_bar", "sqrt_ab", "sqrt_one_minus_ab",
                     "sqrt_ab_prev", "sqrt_om_prev"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ── forward (noising) ────────────────────────────────────────────────────

    def q_sample(
        self,
        x_0: torch.Tensor,
        t:   torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0).

        Returns
        -------
        x_t   : noisy sample [B, C, L]
        noise : the noise added (ε) — needed for loss computation
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        a  = self.sqrt_ab[t].view(-1, 1, 1)          # α_t  [B,1,1]
        s  = self.sqrt_one_minus_ab[t].view(-1, 1, 1) # σ_t  [B,1,1]
        x_t = a * x_0 + s * noise
        return x_t, noise

    # ── v-target computation ─────────────────────────────────────────────────

    def get_v_target(
        self,
        x_0:   torch.Tensor,
        noise: torch.Tensor,
        t:     torch.Tensor,
    ) -> torch.Tensor:
        """v_t = α_t * ε - σ_t * x_0"""
        a = self.sqrt_ab[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return a * noise - s * x_0

    # ── recover x_0 / ε from prediction ─────────────────────────────────────

    def predict_x0_from_v(
        self,
        x_t:    torch.Tensor,
        v_pred: torch.Tensor,
        t:      torch.Tensor,
    ) -> torch.Tensor:
        """x_0_pred = α_t * x_t - σ_t * v_pred"""
        a = self.sqrt_ab[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return a * x_t - s * v_pred

    def predict_eps_from_v(
        self,
        x_t:    torch.Tensor,
        v_pred: torch.Tensor,
        t:      torch.Tensor,
    ) -> torch.Tensor:
        """ε_pred = σ_t * x_t + α_t * v_pred"""
        a = self.sqrt_ab[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return s * x_t + a * v_pred

    def predict_x0_from_eps(
        self,
        x_t:    torch.Tensor,
        eps:    torch.Tensor,
        t:      torch.Tensor,
    ) -> torch.Tensor:
        """x_0_pred = (x_t - σ_t * ε) / α_t"""
        a = self.sqrt_ab[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_ab[t].view(-1, 1, 1)
        return (x_t - s * eps) / a.clamp(min=1e-8)

    # ── training loss ─────────────────────────────────────────────────────────

    def training_loss(
        self,
        model_output: torch.Tensor,
        x_0:          torch.Tensor,
        noise:        torch.Tensor,
        t:            torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between model output and target."""
        if self.prediction_type == "v_prediction":
            target = self.get_v_target(x_0, noise, t)
        else:  # epsilon
            target = noise
        return torch.mean((model_output - target) ** 2)

    # ── DDIM deterministic inference ─────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model:      "torch.nn.Module",
        x_T:        torch.Tensor,
        cond:       torch.Tensor,
        n_steps:    int = 50,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """
        Deterministic DDIM reverse process.
        Runs n_steps uniformly sampled from [0, T-1].

        Parameters
        ----------
        model  : UNet1D  (takes x_t, t_int, cond → predicted v or ε)
        x_T    : starting noise [B, C, L]
        cond   : student conditioning [B, C, L]
        n_steps: number of denoising steps

        Returns
        -------
        x_0_pred : denoised feature [B, C, L]
        """
        T   = self.num_timesteps
        # Uniformly spaced timesteps from T-1 down to 0
        step_indices = list(reversed(
            np.linspace(0, T - 1, n_steps, dtype=int).tolist()
        ))

        x = x_T
        for i, t_idx in enumerate(step_indices):
            t_batch = torch.full((x.shape[0],), t_idx,
                                  dtype=torch.long, device=x.device)

            pred = model(x, t_batch, cond)

            if self.prediction_type == "v_prediction":
                x0_pred  = self.predict_x0_from_v(x, pred, t_batch)
                eps_pred = self.predict_eps_from_v(x, pred, t_batch)
            else:
                eps_pred = pred
                x0_pred  = self.predict_x0_from_eps(x, pred, t_batch)

            if clip_denoised:
                x0_pred = x0_pred.clamp(-1, 1)

            # DDIM step: x_{t_prev} = α_{t_prev} * x0_pred + σ_{t_prev} * ε_pred
            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1]
            else:
                t_prev = 0
            t_prev_batch = torch.full_like(t_batch, t_prev)

            a_prev = self.sqrt_ab_prev[t_batch].view(-1, 1, 1)     # α_{t-1}
            s_prev = self.sqrt_om_prev[t_batch].view(-1, 1, 1)     # σ_{t-1}

            # Use actual t_prev alphas (not t_batch)
            a_prev = self.sqrt_ab[t_prev_batch].view(-1, 1, 1)
            s_prev = self.sqrt_one_minus_ab[t_prev_batch].view(-1, 1, 1)

            x = a_prev * x0_pred + s_prev * eps_pred

        return x
