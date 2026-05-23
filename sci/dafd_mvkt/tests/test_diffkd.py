"""
Unit tests for DiffKD infrastructure (Step 0.3).

Tests
-----
1. Forward process is deterministic (same input+t → same noisy output)
2. Denoising step is numerically stable (no NaN/Inf)
3. v-prediction is mathematically consistent with epsilon-prediction
4. Conditioning affects output (different cond → different output)
5. UNet param count is printed
6. Feature hook captures correct shapes

Run:
    python dafd_mvkt/tests/test_diffkd.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import traceback

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results: list[tuple[str, bool, str]] = []

def test(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()


# ── fixtures ──────────────────────────────────────────────────────────────────

B, C, L = 4, 512, 16
T_STEPS  = 100
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dafd_mvkt.diffkd.scheduler import DDPMScheduler
from dafd_mvkt.diffkd.unet1d    import UNet1D, build_unet


def make_scheduler(pred="v_prediction"):
    return DDPMScheduler(num_timesteps=T_STEPS, prediction_type=pred).to(DEVICE)


def make_unet(cond_type="concat"):
    return build_unet(feature_dim=C, base_channels=128,
                      mults=(1, 2, 4), cond_type=cond_type).to(DEVICE).eval()


# ── Test 1: forward process determinism ───────────────────────────────────────

def _test_forward_determinism():
    sched = make_scheduler()
    x0    = torch.randn(B, C, L, device=DEVICE)
    t     = torch.randint(0, T_STEPS, (B,), device=DEVICE)
    noise = torch.randn_like(x0)

    xt_a, _ = sched.q_sample(x0, t, noise)
    xt_b, _ = sched.q_sample(x0, t, noise)

    assert torch.allclose(xt_a, xt_b, atol=1e-6), \
        "q_sample is not deterministic given the same noise"
    # Different noise → different xt
    noise2 = torch.randn_like(x0)
    xt_c, _ = sched.q_sample(x0, t, noise2)
    assert not torch.allclose(xt_a, xt_c, atol=1e-4), \
        "q_sample ignores noise (always same output)"


# ── Test 2: denoising stability ───────────────────────────────────────────────

def _test_denoising_stability():
    sched = make_scheduler()
    net   = make_unet()
    x0    = torch.randn(B, C, L, device=DEVICE)
    cond  = torch.randn(B, C, L, device=DEVICE)
    t     = torch.randint(0, T_STEPS, (B,), device=DEVICE)

    xt, noise = sched.q_sample(x0, t)
    with torch.no_grad():
        v_pred = net(xt, t, cond)

    assert not v_pred.isnan().any(),  "NaN in v_pred"
    assert not v_pred.isinf().any(),  "Inf in v_pred"
    assert v_pred.shape == (B, C, L), f"wrong shape: {v_pred.shape}"

    # Recover x0 and check it's finite
    x0_rec = sched.predict_x0_from_v(xt, v_pred, t)
    assert not x0_rec.isnan().any(), "NaN in recovered x0"


# ── Test 3: v-prediction ↔ epsilon-prediction consistency ─────────────────────

def _test_vpred_epsilon_consistency():
    sched = make_scheduler()
    x0    = torch.randn(B, C, L, device=DEVICE)
    t     = torch.randint(1, T_STEPS, (B,), device=DEVICE)
    noise = torch.randn_like(x0)

    xt, _ = sched.q_sample(x0, t, noise)

    # v_target  from true x0, noise
    v_tgt = sched.get_v_target(x0, noise, t)

    # Recover ε and x0 from v_target — must match originals
    eps_rec = sched.predict_eps_from_v(xt, v_tgt, t)
    x0_rec  = sched.predict_x0_from_v(xt, v_tgt, t)

    assert torch.allclose(eps_rec, noise, atol=1e-4), \
        f"ε recovery failed: max_err={((eps_rec - noise).abs().max()):.4e}"
    assert torch.allclose(x0_rec, x0, atol=1e-4), \
        f"x0 recovery failed: max_err={((x0_rec - x0).abs().max()):.4e}"

    # ε-pred scheduler recovers x0 consistently
    sched_eps = make_scheduler(pred="epsilon")
    x0_from_eps = sched_eps.predict_x0_from_eps(xt, noise, t)
    assert torch.allclose(x0_from_eps, x0, atol=1e-4), \
        "epsilon→x0 recovery inconsistent"


# ── Test 4: conditioning affects output ───────────────────────────────────────

def _test_conditioning_effect():
    for ctype in ("concat", "attn"):
        net  = make_unet(cond_type=ctype)
        xt   = torch.randn(B, C, L, device=DEVICE)
        t    = torch.zeros(B, dtype=torch.long, device=DEVICE)
        c1   = torch.randn(B, C, L, device=DEVICE)
        c2   = torch.randn(B, C, L, device=DEVICE)  # different conditioning

        with torch.no_grad():
            out1 = net(xt, t, c1)
            out2 = net(xt, t, c2)

        diff = (out1 - out2).abs().mean().item()
        assert diff > 1e-4, \
            f"cond_type='{ctype}': conditioning has no effect (mean diff={diff:.2e})"


# ── Test 5: param count ───────────────────────────────────────────────────────

def _test_param_count():
    for ctype in ("concat", "attn"):
        net = make_unet(cond_type=ctype)
        n   = net.count_params()
        print(f"\n    UNet1D(cond_type='{ctype}'): {n/1e6:.3f}M parameters")
        assert n > 1_000_000, f"suspiciously few params: {n}"
        assert n < 20_000_000, f"suspiciously many params: {n} — check architecture"


# ── Test 6: DDIM sampling completes ──────────────────────────────────────────

def _test_ddim_sampling():
    sched = make_scheduler()
    net   = make_unet()
    x_T   = torch.randn(B, C, L, device=DEVICE)
    cond  = torch.randn(B, C, L, device=DEVICE)

    x0_pred = sched.ddim_sample(net, x_T, cond, n_steps=5)   # fast: 5 steps
    assert x0_pred.shape == (B, C, L), f"wrong shape: {x0_pred.shape}"
    assert not x0_pred.isnan().any(), "NaN in DDIM output"
    assert not x0_pred.isinf().any(), "Inf in DDIM output"


# ── Test 7: feature hook ──────────────────────────────────────────────────────

def _test_feature_hook():
    # ResNet1d uses relative imports ("from models.heads"), must run from dafd_mvkt/
    _dafd = str(Path(__file__).parents[1])
    if _dafd not in sys.path:
        sys.path.insert(0, _dafd)
    from models.resnet1d import ResNet1d
    from diffkd.feature_hook import verify_shapes

    student    = ResNet1d(in_channels=1, num_classes=5, layers=[3,4,6,3]).to(DEVICE)
    teacher500 = ResNet1d(in_channels=1, num_classes=5, layers=[3,4,6,3]).to(DEVICE)
    teacher100 = ResNet1d(in_channels=1, num_classes=5, layers=[3,4,6,3]).to(DEVICE)

    verify_shapes(student, teacher500, teacher100, DEVICE, batch_size=2, layer="layer4")
    verify_shapes(student, teacher500, teacher100, DEVICE, batch_size=2, layer="layer3")


# ── run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"DiffKD Unit Tests  (device={DEVICE})")
    print(f"{'='*55}")

    test("forward process determinism",          _test_forward_determinism)
    test("denoising step numerical stability",    _test_denoising_stability)
    test("v-pred ↔ ε-pred mathematical consistency", _test_vpred_epsilon_consistency)
    test("conditioning affects output",           _test_conditioning_effect)
    test("param count",                           _test_param_count)
    test("DDIM sampling completes",               _test_ddim_sampling)
    test("feature hook shape verification",       _test_feature_hook)

    passed = sum(r[1] for r in _results)
    total  = len(_results)
    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("FAILED:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  ✗ {name}: {msg}")
    else:
        print("All tests passed ✓")
    print(f"{'='*55}\n")
    sys.exit(0 if passed == total else 1)
