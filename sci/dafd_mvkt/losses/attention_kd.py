"""
Attention Knowledge Distillation utilities.

Functions
---------
make_temporal_saliency_from_feature
    Generate a [B,1,L] saliency map from a teacher feature map [B,C,L].

make_dual_teacher_saliency
    Combine 1L500 and 1L100 teacher saliency maps into one target map,
    resampled to a target length.

attention_kd_loss
    MSE or KL loss between student attention and teacher saliency maps,
    both treated as probability distributions over the time axis.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# saliency from feature
# ─────────────────────────────────────────────────────────────────────────────

def make_temporal_saliency_from_feature(
    feat: torch.Tensor,
    target_len: int | None = None,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """
    Derive a temporal saliency map from a feature map via channel-mean abs.

    Args
    ----
    feat        : [B, C, L]  teacher feature map (will be detached)
    target_len  : if not None, interpolate output to this length
    apply_sigmoid : if True, apply sigmoid before normalisation

    Returns
    -------
    saliency : [B, 1, L'] where L' = target_len (or L if None)
    """
    with torch.no_grad():
        sal = feat.detach().abs().mean(dim=1, keepdim=True)  # [B,1,L]

        if apply_sigmoid:
            sal = torch.sigmoid(sal)

        # sample-wise min-max normalisation along time axis
        mn  = sal.min(dim=-1, keepdim=True)[0]
        mx  = sal.max(dim=-1, keepdim=True)[0]
        sal = (sal - mn) / (mx - mn + _EPS)

        if target_len is not None and sal.shape[-1] != target_len:
            sal = F.interpolate(sal, size=target_len,
                                mode="linear", align_corners=False)
    return sal


def make_dual_teacher_saliency(
    feat_500: torch.Tensor,
    feat_100: torch.Tensor,
    target_len: int,
    w500: float = 0.60,
    w100: float = 0.40,
) -> torch.Tensor:
    """
    Weighted combination of 1L500 and 1L100 teacher saliency maps.

    Both maps are resized to target_len before combining.

    Args
    ----
    feat_500  : [B, C, L500]  1L500 teacher feature map
    feat_100  : [B, C, L100]  1L100 teacher feature map
    target_len: target time-axis length (student attention length)
    w500, w100: mixture weights (will be normalised to sum=1)

    Returns
    -------
    combined : [B, 1, target_len]
    """
    total = w500 + w100 + _EPS
    w500, w100 = w500 / total, w100 / total

    sal_500 = make_temporal_saliency_from_feature(feat_500, target_len)
    sal_100 = make_temporal_saliency_from_feature(feat_100, target_len)
    return w500 * sal_500 + w100 * sal_100


# ─────────────────────────────────────────────────────────────────────────────
# attention KD loss
# ─────────────────────────────────────────────────────────────────────────────

def attention_kd_loss(
    student_att: torch.Tensor,
    teacher_saliency: torch.Tensor,
    mode: str = "mse_prob",
) -> torch.Tensor:
    """
    Attention KD loss between student attention map and teacher saliency.

    Both are normalised to time-axis probability distributions before
    computing the loss, so absolute scale differences are irrelevant.

    Args
    ----
    student_att      : [B, 1, L_s]  student temporal attention (in [0,1])
    teacher_saliency : [B, 1, L_t]  teacher saliency map
    mode             : "mse_prob" (MSE on distributions) or "kl" (KL div)

    Returns
    -------
    Scalar loss.
    """
    # ── align lengths ─────────────────────────────────────────────────────────
    L_s = student_att.shape[-1]
    L_t = teacher_saliency.shape[-1]
    if L_t != L_s:
        teacher_saliency = F.interpolate(
            teacher_saliency.detach(), size=L_s,
            mode="linear", align_corners=False,
        )
    else:
        teacher_saliency = teacher_saliency.detach()

    # ── normalise to probability distribution over time axis ──────────────────
    s = student_att.clamp_min(_EPS)
    s = s / (s.sum(dim=-1, keepdim=True) + _EPS)       # [B,1,L]

    t = teacher_saliency.clamp_min(_EPS)
    t = t / (t.sum(dim=-1, keepdim=True) + _EPS)       # [B,1,L]

    # ── loss ──────────────────────────────────────────────────────────────────
    if mode == "kl":
        # KL(t || s) — teacher as target distribution
        loss = F.kl_div(s.log(), t, reduction="batchmean")
    else:  # mse_prob
        loss = F.mse_loss(s, t)

    return loss
