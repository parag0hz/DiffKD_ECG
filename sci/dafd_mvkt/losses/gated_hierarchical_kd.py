"""
Confidence-Gated Hierarchical Knowledge Distillation.

Adaptively mixes 12-lead teacher KD and 1-lead TA KD using per-class
confidence and teacher-TA agreement as soft gating weights.

Gate modes:
  ta_only              — only TA KD, ignore teacher
  confidence           — weight by sigmoid confidence of each source
  confidence_agreement — weight by confidence × teacher-TA agreement
  oracle_fixed         — fixed normalized lambda weights (debugging)
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_multilabel_kl_per_class(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Per-class binary KL divergence: target → student.

    For each class c, forms a 2-class distribution over [z_c/T, 0].
    Returns [B, C] per-class loss (not yet averaged).
    """
    T = temperature
    zeros = torch.zeros_like(student_logits)

    t_bin = torch.stack([target_logits / T, zeros], dim=-1)   # [B, C, 2]
    s_bin = torch.stack([student_logits / T, zeros], dim=-1)  # [B, C, 2]

    t_soft = F.softmax(t_bin, dim=-1)                         # [B, C, 2]
    s_log_soft = F.log_softmax(s_bin, dim=-1)                 # [B, C, 2]

    # kl_div(log_q, p) = p*(log p - log q); sum over 2-class dim → [B, C]
    kl = F.kl_div(s_log_soft, t_soft, reduction="none").sum(dim=-1)  # [B, C]
    return kl * (T ** 2)


def _confidence(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-class confidence: 1 - normalised binary entropy of sigmoid(logits).
    High when prediction is near 0 or 1.
    Returns [B, C] in [0, 1].
    """
    p = torch.sigmoid(logits)
    entropy = -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)
    return 1.0 - entropy / math.log(2)


class ConfidenceGatedHierarchicalKD(nn.Module):
    """
    Confidence-Gated Hierarchical KD loss.

    Args:
        lambda_teacher:  base weight for teacher KD contribution
        lambda_ta:       base weight for TA KD contribution
        temperature:     KD temperature T
        gate_mode:       one of ta_only / confidence / confidence_agreement / oracle_fixed
        eps:             numerical stability epsilon
    """

    VALID_MODES = {"ta_only", "confidence", "confidence_agreement", "oracle_fixed"}

    def __init__(
        self,
        lambda_teacher: float = 0.3,
        lambda_ta: float = 1.0,
        temperature: float = 2.0,
        gate_mode: str = "confidence_agreement",
        eps: float = 1e-8,
    ):
        super().__init__()
        if gate_mode not in self.VALID_MODES:
            raise ValueError(f"gate_mode must be one of {self.VALID_MODES}, got {gate_mode!r}")
        self.lambda_teacher = lambda_teacher
        self.lambda_ta = lambda_ta
        self.temperature = temperature
        self.gate_mode = gate_mode
        self.eps = eps

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, C]
        teacher_logits: torch.Tensor,   # [B, C]
        ta_logits: torch.Tensor,        # [B, C]
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss:         scalar
            diagnostics:  dict with mean_w_teacher, mean_w_ta, mean_agreement,
                          mean_teacher_conf, mean_ta_conf
        """
        teacher_logits = teacher_logits.detach()
        ta_logits = ta_logits.detach()

        kd_teacher = binary_multilabel_kl_per_class(
            student_logits, teacher_logits, self.temperature
        )  # [B, C]
        kd_ta = binary_multilabel_kl_per_class(
            student_logits, ta_logits, self.temperature
        )  # [B, C]

        teacher_conf = _confidence(teacher_logits, self.eps)  # [B, C]
        ta_conf = _confidence(ta_logits, self.eps)            # [B, C]
        agreement = 1.0 - torch.abs(
            torch.sigmoid(teacher_logits) - torch.sigmoid(ta_logits)
        )  # [B, C]

        if self.gate_mode == "ta_only":
            w_teacher = torch.zeros_like(kd_teacher)
            w_ta = torch.ones_like(kd_ta)

        elif self.gate_mode == "oracle_fixed":
            total = self.lambda_teacher + self.lambda_ta + self.eps
            w_teacher = torch.full_like(kd_teacher, self.lambda_teacher / total)
            w_ta = torch.full_like(kd_ta, self.lambda_ta / total)

        elif self.gate_mode == "confidence":
            w_teacher_raw = self.lambda_teacher * teacher_conf
            w_ta_raw = self.lambda_ta * ta_conf
            w_sum = w_teacher_raw + w_ta_raw + self.eps
            w_teacher = w_teacher_raw / w_sum
            w_ta = w_ta_raw / w_sum

        else:  # confidence_agreement
            w_teacher_raw = self.lambda_teacher * teacher_conf * agreement
            w_ta_raw = self.lambda_ta * ta_conf
            w_sum = w_teacher_raw + w_ta_raw + self.eps
            w_teacher = w_teacher_raw / w_sum
            w_ta = w_ta_raw / w_sum

        loss = (w_teacher * kd_teacher + w_ta * kd_ta).mean()

        diag = {
            "mean_w_teacher":   float(w_teacher.mean().item()),
            "mean_w_ta":        float(w_ta.mean().item()),
            "mean_agreement":   float(agreement.mean().item()),
            "mean_teacher_conf":float(teacher_conf.mean().item()),
            "mean_ta_conf":     float(ta_conf.mean().item()),
        }
        return loss, diag
