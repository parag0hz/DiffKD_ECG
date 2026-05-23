"""
Sanity checks for TG-MTA attention modules.

Tests
-----
1. All three attention modules produce correct output shapes.
2. TGMTAModule gamma starts at 0 (identity residual).
3. Backward compatibility: attention_type="none" gives identical output
   to original ResNet1d (no attention) and loads checkpoints with strict=True.
4. attention_type="tgmta" forward shape is correct.
5. attention_kd_loss runs without error and returns scalar.
6. make_dual_teacher_saliency length alignment.

Run from repo root:
    cd /home/kwy00/sci
    /home/kwy00/anaconda3/envs/ecg/bin/python -m pytest dafd_mvkt/tests/test_attention_modules.py -v
or:
    /home/kwy00/anaconda3/envs/ecg/bin/python dafd_mvkt/tests/test_attention_modules.py
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Allow running from repo root or from within dafd_mvkt/
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
try:
    import pytest
except ImportError:
    pytest = None

# ─────────────────────────────────────────────────────────────────────────────

def _rand(B=2, C=64, L=32):
    return torch.randn(B, C, L)


class TestSE1D:
    def test_output_shape(self):
        from models.attention_modules import SE1D
        m = SE1D(channels=64)
        x = _rand(2, 64, 32)
        enhanced, att = m(x)
        assert enhanced.shape == x.shape, f"enhanced shape mismatch: {enhanced.shape}"
        assert att.shape == (2, 1, 32), f"att shape mismatch: {att.shape}"

    def test_temporal_att_all_ones(self):
        from models.attention_modules import SE1D
        m = SE1D(channels=64)
        x = _rand(2, 64, 32)
        _, att = m(x)
        assert att.allclose(torch.ones_like(att)), "SE1D temporal att should be all-ones"


class TestCBAM1D:
    def test_output_shape(self):
        from models.attention_modules import CBAM1D
        m = CBAM1D(channels=128)
        x = _rand(2, 128, 48)
        enhanced, att = m(x)
        assert enhanced.shape == x.shape
        assert att.shape == (2, 1, 48)

    def test_att_range(self):
        from models.attention_modules import CBAM1D
        m = CBAM1D(channels=64)
        x = _rand(2, 64, 32)
        _, att = m(x)
        assert att.min() >= 0.0 and att.max() <= 1.0, "CBAM att should be in [0, 1]"


class TestTGMTAModule:
    def test_output_shape(self):
        from models.attention_modules import TGMTAModule
        m = TGMTAModule(channels=256)
        x = _rand(2, 256, 64)
        enhanced, att = m(x)
        assert enhanced.shape == x.shape
        assert att.shape == (2, 1, 64)

    def test_gamma_zero_init(self):
        from models.attention_modules import TGMTAModule
        m = TGMTAModule(channels=64)
        assert m.gamma.item() == 0.0, "gamma should initialise to 0"

    def test_identity_at_init(self):
        """With gamma=0, output should equal input exactly."""
        from models.attention_modules import TGMTAModule
        m = TGMTAModule(channels=64)
        m.eval()
        x = _rand(2, 64, 32)
        with torch.no_grad():
            enhanced, _ = m(x)
        assert torch.allclose(enhanced, x), "TGMTAModule with gamma=0 should be identity"

    def test_wide_kernel(self):
        from models.attention_modules import TGMTAModule
        m = TGMTAModule(channels=64, use_wide_kernel=True)
        x = _rand(2, 64, 32)
        enhanced, att = m(x)
        assert enhanced.shape == x.shape
        assert att.shape == (2, 1, 32)
        assert len(m.branches) == 4, "wide kernel should add 4th branch"

    def test_gradient_flow(self):
        from models.attention_modules import TGMTAModule
        m = TGMTAModule(channels=64)
        x = _rand(2, 64, 32)
        enhanced, att = m(x)
        loss = enhanced.mean() + att.mean()
        loss.backward()
        assert m.gamma.grad is not None, "gamma should receive gradient"


class TestBuildAttention:
    def test_se(self):
        from models.attention_modules import build_attention, SE1D
        m = build_attention("se", 64)
        assert isinstance(m, SE1D)

    def test_cbam(self):
        from models.attention_modules import build_attention, CBAM1D
        m = build_attention("cbam", 128)
        assert isinstance(m, CBAM1D)

    def test_tgmta(self):
        from models.attention_modules import build_attention, TGMTAModule
        m = build_attention("tgmta", 256)
        assert isinstance(m, TGMTAModule)

    def test_tgmta_wide(self):
        from models.attention_modules import build_attention, TGMTAModule
        m = build_attention("tgmta", 256, use_wide_kernel=True)
        assert isinstance(m, TGMTAModule)
        assert len(m.branches) == 4

    def test_unknown_raises(self):
        from models.attention_modules import build_attention
        try:
            build_attention("unknown", 64)
            assert False, "should have raised"
        except ValueError:
            pass


class TestResNet1dBackwardCompat:
    """Verify attention_type='none' behaviour is unchanged."""

    def _make_model_no_att(self):
        from models.resnet1d import ResNet1d
        return ResNet1d(in_channels=1, num_classes=5,
                        layers=[3, 4, 6, 3], base_channels=64,
                        attention_type="none")

    def test_forward_shape(self):
        model = self._make_model_no_att()
        model.eval()
        x = torch.randn(2, 1, 500)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape == (2, 5)
        assert out["feature_map"].shape[0] == 2
        assert out["feature_map"].shape[1] == 512

    def test_no_attention_dict(self):
        model = self._make_model_no_att()
        model.eval()
        x = torch.randn(2, 1, 500)
        with torch.no_grad():
            out = model(x, return_attention=True)
        assert out.get("attentions") == {}, "no-att model should return empty dict"

    def test_checkpoint_strict_load(self, tmp_path):
        """Save and reload with strict=True — should work for no-att model."""
        model = self._make_model_no_att()
        ckpt = tmp_path / "test_no_att.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt)
        model2 = self._make_model_no_att()
        sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
        model2.load_state_dict(sd, strict=True)


class TestResNet1dWithAttention:
    def test_tgmta_layer4_shape(self):
        from models.resnet1d import ResNet1d
        model = ResNet1d(in_channels=1, num_classes=5,
                         layers=[3, 4, 6, 3], base_channels=64,
                         attention_type="tgmta",
                         attention_layers=["layer4"])
        model.eval()
        x = torch.randn(2, 1, 500)
        with torch.no_grad():
            out = model(x, return_features=True, return_attention=True)
        assert out["logits"].shape == (2, 5)
        assert "layer4" in out["attentions"]
        att = out["attentions"]["layer4"]
        assert att.shape[0] == 2 and att.shape[1] == 1

    def test_tgmta_layer3_layer4(self):
        from models.resnet1d import ResNet1d
        model = ResNet1d(in_channels=1, num_classes=5,
                         layers=[3, 4, 6, 3], base_channels=64,
                         attention_type="tgmta",
                         attention_layers=["layer3", "layer4"])
        model.eval()
        x = torch.randn(2, 1, 500)
        with torch.no_grad():
            out = model(x, return_attention=True)
        atts = out["attentions"]
        assert "layer3" in atts and "layer4" in atts

    def test_strict_false_loads_base_weights(self, tmp_path):
        """Base ResNet1d checkpoint loads into TG-MTA model with strict=False."""
        from models.resnet1d import ResNet1d
        base = ResNet1d(in_channels=1, num_classes=5,
                        layers=[3, 4, 6, 3], base_channels=64,
                        attention_type="none")
        ckpt = tmp_path / "base.pt"
        torch.save({"model_state_dict": base.state_dict()}, ckpt)

        att_model = ResNet1d(in_channels=1, num_classes=5,
                             layers=[3, 4, 6, 3], base_channels=64,
                             attention_type="tgmta",
                             attention_layers=["layer4"])
        sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
        missing, unexpected = att_model.load_state_dict(sd, strict=False)
        # only attention keys should be missing (from att_model)
        for k in missing:
            assert "attentions" in k, f"unexpected missing key: {k}"
        assert len(unexpected) == 0, f"unexpected keys: {unexpected}"


class TestAttentionKDLoss:
    def test_mse_prob_scalar(self):
        from losses.attention_kd import attention_kd_loss
        s = torch.rand(4, 1, 16)
        t = torch.rand(4, 1, 16)
        loss = attention_kd_loss(s, t, mode="mse_prob")
        assert loss.shape == (), f"expected scalar, got {loss.shape}"
        assert loss.item() >= 0.0

    def test_kl_scalar(self):
        from losses.attention_kd import attention_kd_loss
        s = torch.rand(4, 1, 16)
        t = torch.rand(4, 1, 16)
        loss = attention_kd_loss(s, t, mode="kl")
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_length_mismatch_handled(self):
        from losses.attention_kd import attention_kd_loss
        s = torch.rand(4, 1, 16)
        t = torch.rand(4, 1, 32)
        loss = attention_kd_loss(s, t, mode="mse_prob")
        assert loss.shape == ()


class TestMakeTeacherSaliency:
    def test_saliency_shape_and_range(self):
        from losses.attention_kd import make_temporal_saliency_from_feature
        feat = torch.randn(4, 512, 32)
        sal = make_temporal_saliency_from_feature(feat)
        assert sal.shape == (4, 1, 32)
        assert sal.min() >= 0.0 and sal.max() <= 1.0 + 1e-5

    def test_target_len_interpolation(self):
        from losses.attention_kd import make_temporal_saliency_from_feature
        feat = torch.randn(4, 512, 32)
        sal = make_temporal_saliency_from_feature(feat, target_len=16)
        assert sal.shape == (4, 1, 16)

    def test_dual_teacher_saliency(self):
        from losses.attention_kd import make_dual_teacher_saliency
        f500 = torch.randn(4, 512, 157)
        f100 = torch.randn(4, 512, 32)
        sal = make_dual_teacher_saliency(f500, f100, target_len=16)
        assert sal.shape == (4, 1, 16)
        assert sal.min() >= 0.0 and sal.max() <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = [
        TestSE1D, TestCBAM1D, TestTGMTAModule, TestBuildAttention,
        TestResNet1dBackwardCompat, TestResNet1dWithAttention,
        TestAttentionKDLoss, TestMakeTeacherSaliency,
    ]
    passed = failed = 0
    for cls in tests:
        obj = cls()
        for name in [n for n in dir(cls) if n.startswith("test")]:
            method = getattr(obj, name)
            # handle methods with tmp_path fixture manually
            import inspect
            sig = inspect.signature(method)
            kwargs = {}
            if "tmp_path" in sig.parameters:
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    kwargs["tmp_path"] = Path(d)
                    try:
                        method(**kwargs)
                        print(f"  PASS  {cls.__name__}.{name}")
                        passed += 1
                    except Exception:
                        print(f"  FAIL  {cls.__name__}.{name}")
                        traceback.print_exc()
                        failed += 1
            else:
                try:
                    method()
                    print(f"  PASS  {cls.__name__}.{name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL  {cls.__name__}.{name}")
                    traceback.print_exc()
                    failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed  {failed} failed")
    sys.exit(0 if failed == 0 else 1)
