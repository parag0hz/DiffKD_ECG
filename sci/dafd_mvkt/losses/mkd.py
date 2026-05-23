"""
Multi-Label Knowledge Distillation (MKD) loss.

For each class c, treats distillation as binary:
  - positive logit: z_c / T
  - negative logit: 0 / T (anchor)

KL( log_softmax(student_binary), softmax(teacher_binary) ), × T²

Averaged over batch and classes.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def multi_label_kd_loss_per_class(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Per-sample, per-class binary KD loss (no reduction).

    Args:
        student_logits: [B, C]
        teacher_logits: [B, C]  will be detached
        temperature:    T

    Returns:
        loss: [B, C]  — BCE-with-soft-target per sample per class, scaled by T²
    """
    teacher_logits = teacher_logits.detach()
    T = temperature

    s_T = student_logits / T                          # [B, C]
    t_prob = torch.sigmoid(teacher_logits / T)        # [B, C]  soft target

    # BCE with soft target, numerically stable via built-in
    # loss_ij = -t_ij * log σ(s_ij) - (1-t_ij) * log σ(-s_ij)
    loss = F.binary_cross_entropy_with_logits(
        s_T, t_prob, reduction="none"
    ) * (T ** 2)                                      # [B, C]

    return loss


def multi_label_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Multi-label knowledge distillation via class-wise binary distributions.

    Args:
        student_logits: [B, C]  student raw logits (not sigmoid)
        teacher_logits: [B, C]  teacher raw logits, will be detached internally
        temperature:    temperature T for softening distributions

    Returns:
        Scalar loss.
    """
    teacher_logits = teacher_logits.detach()
    T = temperature

    # binary logits: [B, C, 2]  — dim 0 = positive class, dim 1 = anchor (0)
    zeros = torch.zeros_like(student_logits)

    t_bin = torch.stack([teacher_logits / T, zeros], dim=-1)  # [B, C, 2]
    s_bin = torch.stack([student_logits / T, zeros], dim=-1)  # [B, C, 2]

    # soft targets from teacher
    t_soft = F.softmax(t_bin, dim=-1)                         # [B, C, 2]

    # KL divergence: sum_i p_t * (log p_t - log p_s)
    # F.kl_div expects log-probs as input and probs as target
    s_log_soft = F.log_softmax(s_bin, dim=-1)                 # [B, C, 2]

    # kl_div(input=log_q, target=p): computes p*(log p - log q)
    loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)

    return loss
