"""
Class-adaptive wrapper for per-class KD losses.

Applies learnable or pre-computed per-class weights to a [B, C] per-class
loss tensor, allowing harder classes to receive stronger distillation signal.
"""
from __future__ import annotations
import json

import torch
import torch.nn as nn

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ClassAdaptiveWrapper(nn.Module):
    """
    Scales a per-class loss tensor [B, C] by pre-computed class weights,
    then returns the weighted mean.

    Args:
        class_weights: list of C floats (normalized so mean == 1.0)
    """

    def __init__(self, class_weights: list[float]):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", w)   # [C]

    def forward(self, per_class_loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_class_loss: [B, C] per-class loss values
        Returns:
            scalar weighted mean loss
        """
        # mean over batch → [C], then weighted mean over classes
        class_mean = per_class_loss.mean(dim=0)          # [C]
        return (self.weights * class_mean).mean()


def load_class_weights(json_path: str, classes: list[str] = CLASSES) -> list[float]:
    """Load per-class weights from JSON, returned in class order."""
    with open(json_path) as f:
        data = json.load(f)
    return [float(data.get(c, 1.0)) for c in classes]
