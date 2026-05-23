from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, model: nn.Module, epoch: int,
                    metrics: dict, cfg: dict,
                    extra: dict | None = None) -> None:
    state = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "cfg": cfg,
    }
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module,
                    device: torch.device,
                    strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return ckpt
