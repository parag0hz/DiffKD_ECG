"""
Feature extraction hooks for DiffKD.

Registers forward hooks on ResNet1d layer4 (and optionally layer3) to capture
intermediate feature maps without modifying the backbone.

Student  layer4 output: [B, 512, 16]   (input 500@50Hz)
Teacher  1L100  layer4: [B, 512, 32]   (input 1000@100Hz)
Teacher  1L500  layer4: [B, 512, 157]  (input 5000@500Hz)

Teacher features are pooled to match the student's temporal dim via
F.adaptive_avg_pool1d before returning.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHook:
    """
    Attaches a forward hook to a named submodule and stores the output.

    Usage
    -----
    hook = FeatureHook(model, "layer_blocks.15")  # layer4 last block
    with torch.no_grad():
        model(x)
    feat = hook.feature   # [B, C, L]
    hook.remove()
    """

    def __init__(self, model: nn.Module, module_name: str):
        self.feature: torch.Tensor | None = None
        self._handle = None
        # Resolve the named module
        module = dict(model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(
                f"Module '{module_name}' not found. "
                f"Available: {list(dict(model.named_modules()).keys())[:20]}"
            )
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature = output  # [B, C, L]

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        self.remove()


def _resnet1d_stage_last_block_name(model: nn.Module, stage: int) -> str:
    """
    Return the name of the last BasicBlock1d in a given stage (0-indexed).
    stage 3 = layer4, stage 2 = layer3, etc.
    Uses model._stage_ends to find boundaries.
    """
    ends = model._stage_ends   # e.g. [3, 7, 13, 16]
    last_block_idx = ends[stage] - 1
    return f"layer_blocks.{last_block_idx}"


class DualTeacherFeatureExtractor:
    """
    Attaches hooks to student + both teachers to extract layer-N features.
    Automatically pools teacher temporal dims to match student.

    Parameters
    ----------
    student   : ResNet1d (Lead-II 50Hz)
    teacher500: ResNet1d (Lead-II 500Hz)
    teacher100: ResNet1d (Lead-II 100Hz)
    layer     : "layer4" | "layer3"
    """

    _STAGE_IDX = {"layer1": 0, "layer2": 1, "layer3": 2, "layer4": 3}

    def __init__(
        self,
        student:    nn.Module,
        teacher500: nn.Module,
        teacher100: nn.Module,
        layer:      str = "layer4",
    ):
        self.layer = layer
        stage = self._STAGE_IDX[layer]

        s_name  = _resnet1d_stage_last_block_name(student,    stage)
        t5_name = _resnet1d_stage_last_block_name(teacher500, stage)
        t1_name = _resnet1d_stage_last_block_name(teacher100, stage)

        self._hook_s  = FeatureHook(student,    s_name)
        self._hook_t5 = FeatureHook(teacher500, t5_name)
        self._hook_t1 = FeatureHook(teacher100, t1_name)

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (feat_student, feat_t500_pooled, feat_t100_pooled).
        All pooled to student's temporal dim.
        Teachers pooled via adaptive_avg_pool1d.
        """
        fs  = self._hook_s.feature
        ft5 = self._hook_t5.feature
        ft1 = self._hook_t1.feature
        assert fs is not None and ft5 is not None and ft1 is not None, \
            "Hooks not triggered yet — run a forward pass first."

        L = fs.shape[-1]
        ft5_pool = F.adaptive_avg_pool1d(ft5, L) if ft5.shape[-1] != L else ft5
        ft1_pool = F.adaptive_avg_pool1d(ft1, L) if ft1.shape[-1] != L else ft1
        return fs, ft5_pool, ft1_pool

    def remove(self):
        self._hook_s.remove()
        self._hook_t5.remove()
        self._hook_t1.remove()

    def __del__(self):
        self.remove()


def verify_shapes(
    student:    nn.Module,
    teacher500: nn.Module,
    teacher100: nn.Module,
    device:     torch.device,
    batch_size: int = 4,
    layer:      str = "layer4",
) -> None:
    """
    Run a single batch through all models, print feature shapes.
    Used in Step 0.4 to confirm actual feature dimensions.
    """
    extractor = DualTeacherFeatureExtractor(student, teacher500, teacher100, layer)

    student.eval()
    teacher500.eval()
    teacher100.eval()

    x_s   = torch.randn(batch_size, 1,  500, device=device)   # 50Hz
    x_t5  = torch.randn(batch_size, 1, 5000, device=device)   # 500Hz
    x_t1  = torch.randn(batch_size, 1, 1000, device=device)   # 100Hz

    with torch.no_grad():
        student(x_s)
        teacher500(x_t5)
        teacher100(x_t1)

    fs, ft5, ft1 = extractor.get()

    print(f"\n[feature shapes @ {layer}]")
    print(f"  student   ({layer}): {tuple(fs.shape)}   (before pooling)")
    print(f"  t500      ({layer}): {tuple(extractor._hook_t5.feature.shape)}  → pooled to {tuple(ft5.shape)}")
    print(f"  t100      ({layer}): {tuple(extractor._hook_t1.feature.shape)}  → pooled to {tuple(ft1.shape)}")
    print(f"  canonical diffusion space: [B, {fs.shape[1]}, {fs.shape[2]}]")

    extractor.remove()
