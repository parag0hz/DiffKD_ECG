"""
NT-Xent (SimCLR) contrastive loss for ECG encoder pretraining.

Differences from CLECGLoss (MoCo):
  - No momentum encoder or queue.
  - All negatives come from the current batch.
  - Both views (z1, z2) trained with gradient.
  - Larger effective batch size needed for good negatives.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations", ICML 2020.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Symmetric NT-Xent loss for SimCLR.

    Args:
        temperature:  softmax temperature τ (default 0.1)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            z1: [B, D] L2-normalized embeddings (view 1)
            z2: [B, D] L2-normalized embeddings (view 2)

        Returns:
            loss:  scalar
            info:  dict with 'acc' and 'n_negatives'
        """
        B = z1.size(0)

        # Concatenate: [z1; z2] → (2B, D)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Pairwise cosine similarity matrix
        sim = z @ z.T / self.T  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair labels:
        #   for z1[i] (row i):   positive = z2[i] (row i+B)
        #   for z2[i] (row i+B): positive = z1[i] (row i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            preds = sim.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        return loss, {"acc": acc, "n_negatives": 2 * B - 2}
