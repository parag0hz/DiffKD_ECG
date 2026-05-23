"""Exponential Moving Average (EMA) teacher utilities."""
from __future__ import annotations
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains an EMA copy of a model's weights.
    EMA parameters have requires_grad=False and the model stays in eval mode.
    Call update() after every optimizer.step().
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        for ema_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            if ema_b.is_floating_point():
                ema_b.data.mul_(self.decay).add_(m_b.data, alpha=1.0 - self.decay)
            else:
                ema_b.copy_(m_b)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for ema_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            m_p.data.copy_(ema_p.data)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


def create_ema_model(model: nn.Module, decay: float = 0.999) -> ModelEMA:
    return ModelEMA(model, decay=decay)
