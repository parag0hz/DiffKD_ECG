"""
Cross-Rate Contrastive Feature (CRF) loss.

Symmetric InfoNCE between teacher (500 Hz, 12-lead) and student
(low Hz, 1-lead) projected features within the same batch.

Positive pair: same ECG record (diagonal of similarity matrix).
Negatives: all other records in the batch.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def cross_rate_contrastive_loss(
    student_pooled: torch.Tensor,    # [B, D]  (unused in base impl)
    teacher_pooled: torch.Tensor,    # [B, D]  (unused in base impl)
    student_proj: torch.Tensor,      # [B, proj_dim]  L2-normalized
    teacher_proj: torch.Tensor,      # [B, proj_dim]  L2-normalized
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-rate contrastive loss (symmetric InfoNCE).

    Uses the L2-normalized projection heads from each model.
    student_pooled / teacher_pooled are accepted for API compatibility
    but the contrastive computation operates on proj features.

    Args:
        student_pooled: raw pooled features [B, D]
        teacher_pooled: raw pooled features [B, D]
        student_proj:   L2-normalized proj [B, 128]
        teacher_proj:   L2-normalized proj [B, 128]
        temperature:    InfoNCE temperature τ

    Returns:
        Scalar loss.
    """
    teacher_proj = teacher_proj.detach()

    B = student_proj.size(0)
    labels = torch.arange(B, device=student_proj.device)

    # similarity matrix [B, B]
    logits = student_proj @ teacher_proj.T / temperature  # [B, B]

    # symmetric InfoNCE
    loss_s2t = F.cross_entropy(logits,   labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_s2t + loss_t2s)
