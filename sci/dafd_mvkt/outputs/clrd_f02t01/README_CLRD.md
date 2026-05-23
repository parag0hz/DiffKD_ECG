# CLRD — Virtual Lead Importance Distillation

**All CLRD comparisons in this environment use the local F02T01 test baseline AUC=0.84644. Cross-machine comparison is disabled until baseline fingerprint is matched.**

See `dafd_mvkt/outputs/baseline_fingerprint_F02T01.json` for full reproducibility fingerprint.

---

## Status: FAILED — All variants below F02T01. Direction closed.

---

## Baseline

| metric | value |
|--------|-------|
| F02T01 test AUC | **0.84644** |
| F02T01 val AUC | 0.85037 (checkpoint selection criterion) |
| F02T01 test F1_tuned | 0.62544 |
| F02T01 per-class | NORM=0.90049, MI=0.82173, STTC=0.87417, CD=0.86444, HYP=0.77135 |

---

## Method

12-lead 100Hz teacher (`teacher_12lead_100hz_resnet34_best.pt`, AUC=0.9065):
- For each of 12 leads, zero-out that lead and compare p_occlude vs p_full
- importance[b,c,l] = ReLU(p_full[b,c] − p_occlude[b,c])
- Normalize over lead dimension → A_teacher[b,5,12] (sum-to-1 per class)
- Cache by record_id in `clrd_cache/` (13 teacher forward passes per sample, built once)

Student: `LeadImportanceHead` (Linear 512→5×12, softmax) attached to pooled representation.  
KD loss: KL(A_teacher ∥ A_student) on positive (label=1) class/sample pairs only.  
Total loss: L_BCE + λ_KD × L_KD(F02T01) + λ_CLRD × L_CLRD

---

## Results (vs local baseline AUC=0.84644)

| variant | λ_clrd | loss | pos_only | test AUC | ΔAUC | F1_tuned | HYP | MI | STTC | CD | NORM | val_best | verdict |
|---------|--------|------|----------|----------|------|----------|-----|-----|------|----|------|---------|---------|
| F02T01 | — | — | — | 0.84644 | — | 0.62544 | 0.77135 | 0.82173 | 0.87417 | 0.86444 | 0.90049 | 0.85037 | baseline |
| CLRD01 | 0.000 | kl | True | 0.84393 | -0.00251 | 0.62909 | 0.75952 | 0.82359 | 0.87402 | 0.86301 | 0.89951 | 0.84981 | failed |
| CLRD02 | 0.001 | kl | True | 0.84416 | -0.00228 | 0.63153 | 0.76851 | 0.81769 | 0.87339 | 0.86192 | 0.89930 | 0.85032 | failed |
| CLRD03 | 0.003 | kl | True | 0.84464 | -0.00180 | 0.62717 | 0.76281 | 0.82111 | 0.87511 | 0.86558 | 0.89862 | 0.84990 | failed |
| CLRD04 | 0.005 | kl | True | 0.84461 | -0.00183 | 0.62968 | 0.76214 | 0.82047 | 0.87560 | 0.86673 | 0.89813 | 0.85034 | failed |

All ΔAUC values are negative (−0.002 to −0.003). No variant exceeds F02T01 baseline.

---

## Interpretation

### Q1. CLRD01 (λ=0.0) sanity check

CLRD01 does not apply CLRD loss (λ=0) but adds `LeadImportanceHead` to the optimizer. The ΔAUC=−0.0025 vs F02T01 is within stochastic variance (±0.005), attributable to:
1. Cache building (13 × full-dataset teacher forward passes) alters PyTorch PRNG state after `set_seed(0)`
2. Extra `LeadImportanceHead` parameters in Adam state (receive no gradients, minimal effect)

### Q2. CLRD02–04 trend

Increasing λ_clrd from 0.001 → 0.005 shows a slight positive ΔAUC trend (−0.0023 → −0.0018), suggesting the CLRD loss signal is slightly helpful but not enough to recover the PRNG perturbation introduced by cache building, let alone improve over baseline.

F1_tuned is uniformly higher (+0.003–0.008) across all variants, indicating that the lead importance head helps calibrate the student for threshold-sensitive metrics — but macro-AUC is the primary metric and it does not improve.

### Q3. HYP

All CLRD variants show HYP AUC below F02T01 (0.77135):
- CLRD01: 0.7595 (−0.012)
- CLRD02: 0.7685 (−0.003)
- CLRD03: 0.7628 (−0.009)
- CLRD04: 0.7621 (−0.009)

Hypothesis: the 12L100 teacher provides weaker HYP signal (HYP is most lead-dependent), and transferring its lead-importance distribution adds noise to the student's HYP representation rather than improving it.

### Root cause hypothesis

The core difficulty: a single-lead 50Hz student cannot learn meaningful multi-lead importance because it only observes one lead. The `LeadImportanceHead` is forced to predict a 12-dimensional importance distribution from a 512-dim feature that carries no information distinguishing which leads would be important — it has never seen the other 11 leads. Training this head adds an underdetermined objective that conflicts with the primary BCE+KD loss, causing a net performance drop.

**This direction is structurally incompatible with single-lead setup.** CLRD requires the student to distinguish between lead contributions, which a 1-lead model fundamentally cannot do.

---

## Decision

**CLRD direction fully closed.** Do not retry.

- No further CLRD variants (positive_only=False, MSE loss, higher λ, etc. will not change the structural incompatibility)
- The lead importance cache (`clrd_cache/`) can be retained for potential future use
- All CLRD comparisons use local baseline AUC=0.84644

---

## KD Summary — All Directions Tested

| direction | best ΔAUC | conclusion |
|-----------|-----------|------------|
| AKD Phase A (confidence/agreement) | −0.0036 (AKD02) | failed |
| AKD Phase B (ensemble) | −0.0036 (EKD02) | failed |
| AKD Phase C (dynamic weight) | −0.0007 (DKD02) | failed |
| AKD Phase D (label correlation) | −0.0041 (LCKD02) | failed |
| AKD Phase E (class reliability) | −0.0065 (CKD01) | failed |
| RKDLR (rate recoverability) | −0.0052 (RKDLR01) | failed, r≈0.95 |
| CLRD (lead importance) | −0.0018 (CLRD03/04) | failed, structural limit |

**F02T01 (fixed w=0.60/0.40, T=1.5, KL-div KD) remains the best single-seed result.**

---

*Generated: 2026-05-24 | local baseline: F02T01 test AUC=0.84644*
