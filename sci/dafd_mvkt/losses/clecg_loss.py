"""
MoCo-style InfoNCE loss for CLECG contrastive pretraining.

Queue:
  - Fixed-size FIFO queue of K=2048 key embeddings (L2-normalized).
  - Keys are enqueued after each batch; oldest keys dequeued.
  - Gradients do NOT flow through queue entries.

Loss (NT-Xent / InfoNCE):
  L = -log( exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ_neg exp(q·k_neg / τ)) )

where k+ is the key from the momentum encoder for the same sample, and
k_neg comes from the queue.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoQueue(nn.Module):
    """
    FIFO queue of L2-normalized key embeddings.

    Args:
        feat_dim:  embedding dimension
        queue_size: K (number of negative keys to maintain, default 2048)
    """

    def __init__(self, feat_dim: int, queue_size: int = 2048):
        super().__init__()
        self.K = queue_size
        # Queue is not a parameter — no gradients.
        self.register_buffer("queue", F.normalize(torch.randn(feat_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor) -> None:
        """
        Add keys to the queue, replacing the oldest entries.

        Args:
            keys: [B, D] L2-normalized key embeddings
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Simple wrap-around: if batch doesn't divide queue evenly, truncate
        space = self.K - ptr
        if B <= space:
            self.queue[:, ptr : ptr + B] = keys.T
            ptr = (ptr + B) % self.K
        else:
            # Fill to end, then wrap
            self.queue[:, ptr:] = keys[:space].T
            remainder = B - space
            self.queue[:, :remainder] = keys[space:].T
            ptr = remainder

        self.queue_ptr[0] = ptr

    def get_keys(self) -> torch.Tensor:
        """Returns a clone of the queue as [D, K] (detached, own storage)."""
        return self.queue.clone().detach()


class CLECGLoss(nn.Module):
    """
    MoCo-style InfoNCE loss for CLECG.

    Args:
        feat_dim:   projection head output dimension
        queue_size: K, number of negatives in queue (default 2048)
        temperature: τ (default 0.07)
    """

    def __init__(
        self,
        feat_dim: int,
        queue_size: int = 2048,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.T = temperature
        self.queue = MoCoQueue(feat_dim, queue_size)

    def forward(
        self,
        q: torch.Tensor,        # [B, D]  query embeddings (L2-normalized)
        k: torch.Tensor,        # [B, D]  key embeddings (L2-normalized, momentum enc)
        update_queue: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute InfoNCE loss.

        Args:
            update_queue: if False, skip enqueue (use during validation to avoid
                          contaminating the training queue with val-set keys)
        Returns:
            loss: scalar
            info: dict with {"loss": float, "accuracy": float}
        """
        B = q.shape[0]
        device = q.device

        # Positive logit: (B, 1)
        pos = torch.einsum("bd,bd->b", q, k).unsqueeze(1)  # [B, 1]

        # Negative logits: (B, K) via queue
        neg = torch.einsum("bd,dk->bk", q, self.queue.get_keys())  # [B, K]

        # Concat: column 0 is positive, columns 1..K are negatives
        logits = torch.cat([pos, neg], dim=1) / self.T  # [B, 1+K]

        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean()

        if update_queue:
            self.queue.enqueue_dequeue(k.detach())

        return loss, {"loss": loss.item(), "accuracy": acc.item()}
