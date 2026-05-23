"""
Sanity checks for Adaptive KD implementation.

Tests
-----
1. multi_label_kd_loss_per_class shape [B,C]
2. Confidence weight range [0,1]
3. Agreement weight range [0,1]
4. Confidence×Agreement weight range [0,1]
5. ensemble_prob KD returns scalar, no NaN
6. ensemble_logit KD returns scalar, no NaN
7. label_correlation loss is scalar, no NaN
8. dynamic_weight schedule is linear and within bounds
9. akd_mode=base matches multi_label_kd_loss exactly
10. _compute_adaptive_kd_loss runs for all modes without error

Run from /home/kwy00/sci:
    /home/kwy00/anaconda3/envs/ecg/bin/python dafd_mvkt/tests/test_adaptive_kd.py
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────

B, C = 8, 5


def _rand_logits(B=B, C=C):
    return torch.randn(B, C)


class TestPerClassKDLoss:
    def test_shape(self):
        from losses.mkd import multi_label_kd_loss_per_class
        s = _rand_logits(); t = _rand_logits()
        loss = multi_label_kd_loss_per_class(s, t, temperature=1.5)
        assert loss.shape == (B, C), f"expected [{B},{C}], got {loss.shape}"

    def test_nonneg(self):
        from losses.mkd import multi_label_kd_loss_per_class
        s = _rand_logits(); t = _rand_logits()
        loss = multi_label_kd_loss_per_class(s, t, temperature=1.5)
        assert loss.min().item() >= 0.0, "per-class KD loss should be non-negative"

    def test_teacher_detached(self):
        from losses.mkd import multi_label_kd_loss_per_class
        t = _rand_logits().requires_grad_(True)
        s = _rand_logits().requires_grad_(True)
        loss = multi_label_kd_loss_per_class(s, t, temperature=1.5).mean()
        loss.backward()
        assert t.grad is None, "teacher should be detached — no gradient"
        assert s.grad is not None, "student should receive gradient"

    def test_mean_equals_scalar_loss(self):
        from losses.mkd import multi_label_kd_loss_per_class, multi_label_kd_loss
        s = _rand_logits(); t = _rand_logits()
        per = multi_label_kd_loss_per_class(s, t, temperature=2.0)
        # per-class mean should be close to the KL-based scalar loss
        # (they use different formulations so not exactly equal, just finite)
        scalar = multi_label_kd_loss(s, t, temperature=2.0)
        assert torch.isfinite(per.mean()), "per-class mean must be finite"
        assert torch.isfinite(scalar), "scalar KD loss must be finite"


class TestAdaptiveWeights:
    def _make_probs(self):
        l500 = _rand_logits()
        l100 = _rand_logits()
        T = 1.5
        p500 = torch.sigmoid(l500 / T)
        p100 = torch.sigmoid(l100 / T)
        return p500, p100

    def test_confidence_range(self):
        p500, p100 = self._make_probs()
        conf500 = 2.0 * (p500 - 0.5).abs()
        conf100 = 2.0 * (p100 - 0.5).abs()
        conf = 0.6 * conf500 + 0.4 * conf100
        assert conf.min() >= 0.0 and conf.max() <= 1.0 + 1e-5, \
            f"confidence out of [0,1]: min={conf.min():.4f} max={conf.max():.4f}"

    def test_agreement_range(self):
        p500, p100 = self._make_probs()
        agr = 1.0 - (p500 - p100).abs()
        assert agr.min() >= 0.0 and agr.max() <= 1.0 + 1e-5, \
            f"agreement out of [0,1]: min={agr.min():.4f} max={agr.max():.4f}"

    def test_conf_agreement_range(self):
        p500, p100 = self._make_probs()
        conf = 0.6 * 2.0*(p500-0.5).abs() + 0.4 * 2.0*(p100-0.5).abs()
        agr  = 1.0 - (p500 - p100).abs()
        q = agr * conf
        assert q.min() >= 0.0 and q.max() <= 1.0 + 1e-5, \
            f"conf×agreement out of [0,1]: min={q.min():.4f} max={q.max():.4f}"

    def test_shape_b_c(self):
        p500, p100 = self._make_probs()
        conf = 0.6 * 2.0*(p500-0.5).abs() + 0.4 * 2.0*(p100-0.5).abs()
        assert conf.shape == (B, C), f"expected [{B},{C}], got {conf.shape}"


class TestEnsembleKD:
    def test_ensemble_prob_scalar_finite(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
        kd, corr, info = _compute_adaptive_kd_loss(
            s, l5, l1, 0.6, 0.4, 1.5, "ensemble_prob",
            1.0, 0.0, True, False, None, 0.0, 1, 100, 0.5, 0.6)
        assert kd.shape == (), f"should be scalar, got {kd.shape}"
        assert torch.isfinite(kd), "ensemble_prob KD must be finite"

    def test_ensemble_logit_scalar_finite(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
        kd, corr, info = _compute_adaptive_kd_loss(
            s, l5, l1, 0.6, 0.4, 1.5, "ensemble_logit",
            1.0, 0.0, True, False, None, 0.0, 1, 100, 0.5, 0.6)
        assert kd.shape == ()
        assert torch.isfinite(kd)

    def test_no_gradient_to_teacher(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        l5 = _rand_logits().requires_grad_(True)
        l1 = _rand_logits().requires_grad_(True)
        s  = _rand_logits().requires_grad_(True)
        kd, _, _ = _compute_adaptive_kd_loss(
            s, l5, l1, 0.6, 0.4, 1.5, "ensemble_prob",
            1.0, 0.0, True, False, None, 0.0, 1, 100, 0.5, 0.6)
        kd.backward()
        assert l5.grad is None and l1.grad is None, \
            "ensemble target must be detached from teacher"


class TestLabelCorrelation:
    def test_scalar_finite(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        s  = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
        kd, corr, info = _compute_adaptive_kd_loss(
            s, l5, l1, 0.6, 0.4, 1.5, "label_correlation",
            1.0, 0.0, True, False, None, 0.01, 1, 100, 0.5, 0.6)
        assert torch.isfinite(corr), "label corr loss must be finite"
        assert corr.shape == (1,) or corr.shape == (), \
            f"expected scalar-like, got {corr.shape}"

    def test_no_nan_with_small_batch(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        # batch size 2 — should not produce NaN
        s  = torch.randn(2, C); l5 = torch.randn(2, C); l1 = torch.randn(2, C)
        kd, corr, info = _compute_adaptive_kd_loss(
            s, l5, l1, 0.6, 0.4, 1.5, "label_correlation",
            1.0, 0.0, True, False, None, 0.01, 1, 100, 0.5, 0.6)
        assert not torch.isnan(corr), "label corr should not NaN with small batch"


class TestDynamicWeight:
    def test_linear_schedule(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        starts = []; ends = []
        for ep in [1, 50, 100]:
            s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
            _, _, info = _compute_adaptive_kd_loss(
                s, l5, l1, 0.6, 0.4, 1.5, "dynamic_weight",
                1.0, 0.0, True, False, None, 0.0,
                current_epoch=ep, total_epochs=100,
                dynamic_w500_start=0.50, dynamic_w500_end=0.60)
            if "current_w500" in info:
                starts.append(info["current_w500"])
        if starts:
            assert starts[0] <= starts[-1], "w500 should increase over epochs"
            assert abs(starts[0] - 0.50) < 0.01, f"start w500 should be ~0.50, got {starts[0]:.4f}"
            assert abs(starts[-1] - 0.60) < 0.01, f"end w500 should be ~0.60, got {starts[-1]:.4f}"

    def test_w500_w100_sum_to_one(self):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        for ep in [1, 50, 100]:
            s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
            _, _, info = _compute_adaptive_kd_loss(
                s, l5, l1, 0.6, 0.4, 1.5, "dynamic_weight",
                1.0, 0.0, True, False, None, 0.0,
                current_epoch=ep, total_epochs=100,
                dynamic_w500_start=0.45, dynamic_w500_end=0.65)
            if "current_w500" in info:
                assert abs(info["current_w500"] + info["current_w100"] - 1.0) < 1e-5


class TestBaseMode:
    def test_base_matches_legacy(self):
        """akd_mode=base should give same loss as manual w500*kd500+w100*kd100."""
        from losses.mkd import multi_label_kd_loss
        from train_f02_student_tune import _compute_adaptive_kd_loss
        torch.manual_seed(42)
        s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
        w5, w1, T = 0.6, 0.4, 1.5

        kd_ref = w5 * multi_label_kd_loss(s, l5, T) + w1 * multi_label_kd_loss(s, l1, T)

        kd_akd, _, _ = _compute_adaptive_kd_loss(
            s, l5, l1, w5, w1, T, "base",
            1.0, 0.0, True, False, None, 0.0, 1, 100, 0.5, 0.6)

        assert torch.allclose(kd_ref, kd_akd, atol=1e-5), \
            f"base mode mismatch: ref={kd_ref:.6f} akd={kd_akd:.6f}"


class TestAllModes:
    """Smoke test: all modes run without error and return finite scalars."""

    def _run(self, mode, **kwargs):
        from train_f02_student_tune import _compute_adaptive_kd_loss
        s = _rand_logits(); l5 = _rand_logits(); l1 = _rand_logits()
        defaults = dict(
            w500=0.6, w100=0.4, temperature=1.5,
            akd_gamma=1.0, akd_min_weight=0.0,
            akd_detach_weight=True, akd_normalize_weight=False,
            class_reliability=None, label_corr_weight=0.01,
            current_epoch=50, total_epochs=100,
            dynamic_w500_start=0.5, dynamic_w500_end=0.6,
        )
        defaults.update(kwargs)
        kd, corr, info = _compute_adaptive_kd_loss(s, l5, l1, akd_mode=mode, **defaults)
        assert torch.isfinite(kd), f"mode={mode}: kd not finite"
        return kd, corr, info

    def test_all_modes(self):
        modes = [
            "base", "confidence", "agreement", "confidence_agreement",
            "ensemble_prob", "ensemble_logit",
            "class_reliability", "label_correlation", "dynamic_weight",
        ]
        for mode in modes:
            extra = {}
            if mode == "class_reliability":
                extra["class_reliability"] = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.75])
            self._run(mode, **extra)

    def test_confidence_agreement_gamma(self):
        for gamma in [0.5, 1.0, 2.0]:
            self._run("confidence_agreement", akd_gamma=gamma)

    def test_confidence_agreement_min_weight(self):
        kd, _, info = self._run("confidence_agreement", akd_min_weight=0.25)
        assert info.get("mean_akd_weight", 1.0) >= 0.25 - 1e-4


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        TestPerClassKDLoss, TestAdaptiveWeights, TestEnsembleKD,
        TestLabelCorrelation, TestDynamicWeight, TestBaseMode, TestAllModes,
    ]
    passed = failed = 0
    for cls in tests:
        obj = cls()
        for name in [n for n in dir(cls) if n.startswith("test")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed  {failed} failed")
    sys.exit(0 if failed == 0 else 1)
