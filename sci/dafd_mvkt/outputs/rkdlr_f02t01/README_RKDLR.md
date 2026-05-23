# RKDLR — Recoverability-aware KD for Low-Rate ECG

**Status: FAILED — All variants below F02T01. Direction closed.**

---

## Baseline Audit (2026-05-23)

| metric | value | source |
|--------|-------|--------|
| F02T01 **test** AUC | **0.84644** | `macro_auc` in `metrics_test_student_II_50hz_f02_F02T01_seed0.json` |
| F02T01 **val** AUC | 0.85037 | `macro_auc` in `metrics_val_…json` (= `val_auc_best` in test JSON) |
| F02T01 test F1_tuned | 0.62544 | `macro_f1_tuned` in metrics_test JSON |
| best epoch (val-AUC-based) | — | checkpoint saved at highest val AUC epoch |

**Checkpoint selection criterion**: highest val AUC per epoch.
**Primary evaluation**: `macro_auc` from `metrics_test_*.json` = **test AUC**.

The "0.8464" figure consistently used as baseline throughout this project is the **test AUC**, not val AUC.
The companion computer reports F02T01 test AUC = 0.8404 (same code, different seed/run, expected variance ~±0.005).

---

## Results (vs F02T01 test AUC = 0.84644)

| variant | mode | test AUC | ΔAUC (vs 0.8464) | r̄ | val_best |
|---------|------|----------|-------------------|---|---------|
| F02T01 (baseline) | — | **0.8464** | — | — | 0.8504 |
| RKDLR01 | gate_full γ=1.0 | 0.8412 | -0.0052 | 0.947 | 0.8429 |
| RKDLR04 | gate_full min_w=0.25 | 0.8404 | -0.0061 | 0.960 | 0.8443 |
| RKDLR03 | gate_full γ=2.0 | 0.8403 | -0.0061 | 0.899 | 0.8439 |
| RKDLR06 | mixed_target | 0.8402 | -0.0062 | 0.947 | 0.8446 |
| RKDLR02 | gate_full γ=0.5 | 0.8402 | -0.0063 | 0.973 | 0.8439 |
| RKDLR05 | degraded_target | 0.8327 | -0.0137 | 0.947 | 0.8405 |

## Root Cause

**Recoverability r ≈ 0.95 across all samples and classes.**

50Hz downsampling of a 500Hz or 100Hz signal via linear interpolation barely changes the teacher's predictions. The 1-lead ECG at 50Hz already captures almost all the discriminative information for these 5 superclasses — the teacher's predictions are nearly identical before and after degradation.

Consequence: the recoverability gate provides virtually no per-sample or per-class discrimination. Multiplying KD loss by r ≈ 0.95 is essentially equivalent to multiplying by a constant slightly less than 1, which hurts rather than helps.

## Do Not Continue
- No further RKDLR variants
- degraded_target is worst (-0.0137) — never combine with other methods
- Rate-based signal analysis is not the bottleneck; **lead deficiency is**

## Next Direction
CLRD — 12-lead teacher lead occlusion importance distillation
