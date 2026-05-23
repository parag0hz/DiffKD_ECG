"""
Multi-Teacher Knowledge Distillation loss.

Extends the existing MKD binary formulation (losses/mkd.py) to N teachers
with per-teacher weights.

For each teacher i:
  L_i = KL( log_softmax(student_binary/T), softmax(teacher_i_binary/T) ) * T^2
Total: sum_i(w_i * L_i) / sum(w_i)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _binary_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Per-teacher binary KD loss (same formulation as losses/mkd.py).
    teacher_logits must already be detached.
    Returns scalar.
    """
    T = temperature
    zeros = torch.zeros_like(student_logits)
    t_bin = torch.stack([teacher_logits / T, zeros], dim=-1)   # [B, C, 2]
    s_bin = torch.stack([student_logits / T, zeros], dim=-1)   # [B, C, 2]
    t_soft    = F.softmax(t_bin, dim=-1)
    s_log_soft = F.log_softmax(s_bin, dim=-1)
    return F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)


class MultiTeacherKDLoss(nn.Module):
    """
    Multi-label multi-teacher KD.

    Args:
        temperature: softening temperature (default 2.0)

    Forward:
        student_logits:       [B, C]
        teacher_logits_list:  list of [B, C]   (already from .detach() is not required
                              — detach is applied internally)
        weights:              list of float, same length as teacher_logits_list
                              (will be normalised to sum=1)

    Returns:
        loss:    scalar
        diag:    dict with per-teacher and total KD values
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: list[torch.Tensor],
        weights: list[float],
    ) -> tuple[torch.Tensor, dict]:
        assert len(teacher_logits_list) == len(weights), \
            "teacher_logits_list and weights must have same length"

        w_sum = sum(weights)
        assert w_sum > 0, "weights must sum to a positive value"

        diag: dict = {}
        total = torch.zeros(1, device=student_logits.device)

        for i, (t_logits, w) in enumerate(zip(teacher_logits_list, weights)):
            kd_i = _binary_kd(
                student_logits, t_logits.detach(), self.temperature
            )
            norm_w = w / w_sum
            total = total + norm_w * kd_i
            diag[f"kd_T{i+1}"]     = kd_i.item()
            diag[f"weight_T{i+1}"] = norm_w

        diag["kd_total"] = total.item()
        return total.squeeze(), diag
