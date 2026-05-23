# Adaptive Knowledge Distillation — F02T01 Baseline

**Baseline: F02T01** — Lead II, 50Hz, ResNet1d-34, T=1.5, w500=0.60/w100=0.40  
**Baseline metrics**: AUC=0.8464, F1_tuned=0.6254  
**Per-class AUC**: NORM=0.9005, MI=0.8217, STTC=0.8742, CD=0.8644, HYP=0.7713  
**Primary metric**: macro-AUC (F1 is secondary)  
**Fixed conditions**: attention_type=none, no CRF, no FeatureKD, seed=0  
**Overall verdict**: **AKD 전 방향 중단** — 어떤 variant도 F02T01을 넘지 못함

---

## Section 1: Failed Adaptive KD (Phase A+B) — STOPPED

**Status: STOPPED** — All variants below F02T01 by > 0.001 AUC

### Phase A — Sample-wise adaptive weighting

| variant | mode | AUC | ΔAUC | ΔHYP | ΔMI | notes |
|---------|------|-----|------|------|-----|-------|
| AKD01 | confidence γ=1.0 | 0.8330 | -0.0134 | -0.0308 | -0.0154 | failed |
| AKD02 | agreement γ=1.0 | 0.8406 | -0.0058 | -0.0112 | -0.0059 | failed |
| AKD03 | conf×agree γ=1.0 | 0.8310 | -0.0154 | -0.0340 | -0.0190 | failed |
| AKD04 | conf×agree γ=0.5 | 0.8363 | -0.0101 | -0.0121 | -0.0247 | failed |
| AKD05 | conf×agree γ=2.0 | 0.8319 | -0.0145 | -0.0321 | -0.0216 | failed |
| AKD06 | conf×agree min_w=0.25 | 0.8354 | -0.0110 | -0.0282 | -0.0120 | failed |

### Phase B — Ensemble teacher targets

| variant | mode | AUC | ΔAUC | ΔHYP | ΔMI | notes |
|---------|------|-----|------|------|-----|-------|
| EKD01 | ensemble_prob | 0.8413 | -0.0051 | -0.0114 | -0.0052 | failed |
| EKD02 | ensemble_logit | 0.8428 | -0.0036 | -0.0054 | -0.0053 | failed |

### Interpretation

두 teacher(1L500, 1L100)의 weight 0.60/0.40은 이미 최적화된 상태(TW15 tuning으로 확인됨).
여기에 추가적인 sample-wise 또는 class-wise adaptive gate를 얹으면:
- confidence/agreement가 낮은 샘플(hard cases)의 KD 신호를 줄여 → student의 어려운 case 학습이 저하됨
- agreement 기반이 confidence 기반보다 덜 나쁨 (AKD02 > AKD01) — teacher 합의가 그나마 meaningful하지만 여전히 부족
- ensemble 방식(EKD01/02)은 dual-teacher 구조를 단일 target으로 합쳐 → 구조적 신호 손실

**이 방향은 재시도하지 않는다.**

---

## Section 2: Remaining KD Variants (Phase C–E) — ALL FAILED

**Status: STOPPED** — All variants below F02T01 (threshold: AUC < 0.8460 → stop)

### Phase C — Dynamic Weight KD (epoch-wise w500 schedule)

| variant | mode | w500 schedule | AUC | ΔAUC | ΔF1 | ΔHYP | ΔMI | verdict |
|---------|------|--------------|-----|------|-----|------|-----|---------|
| DKD01 | dynamic_weight | 0.50→0.60 | 0.8443 | -0.0021 | +0.0007 | -0.0110 | +0.0044 | failed |
| DKD02 | dynamic_weight | 0.45→0.65 | 0.8457 | -0.0007 | +0.0004 | -0.0007 | -0.0025 | failed |
| DKD03 | dynamic_weight | 0.55→0.65 | 0.8443 | -0.0021 | +0.0015 | -0.0120 | +0.0045 | failed |

DKD02가 가장 근접 (ΔAUC=-0.0007, F1≈동등)하지만 threshold(0.8460) 미달 및 AUC 기준 negative.  
DKD01/03: MI는 +0.004~0.005 개선되나 HYP가 -0.011~0.012으로 크게 하락 — macro-AUC 개선으로 이어지지 않음.

### Phase D — Label Correlation KD

| variant | mode | label_corr_weight | AUC | ΔAUC | ΔF1 | ΔHYP | notes |
|---------|------|------------------|-----|------|-----|------|-------|
| LCKD01 | label_correlation | 0.001 | 0.8411 | -0.0053 | -0.0032 | -0.0112 | failed |
| LCKD02 | label_correlation | 0.003 | 0.8423 | -0.0041 | -0.0015 | -0.0093 | failed |
| LCKD03 | label_correlation | 0.005 | 0.8414 | -0.0050 | -0.0083 | -0.0098 | failed |

LCKD02(0.003)이 가장 낫지만 여전히 -0.0041. covariance MSE loss가 주 KD 신호를 방해하는 것으로 보임.

### Phase E — Class Reliability KD

| variant | mode | reliability (NORM,MI,STTC,CD,HYP) | AUC | ΔAUC | ΔHYP | ΔMI | notes |
|---------|------|----------------------------------|-----|------|------|-----|-------|
| CKD01 | class_reliability | 1.0,1.0,1.0,1.0,0.75 | 0.8399 | -0.0065 | -0.0112 | -0.0070 | failed |
| CKD02 | class_reliability | 1.0,1.0,1.0,1.0,0.50 | 0.8356 | -0.0108 | -0.0362 | -0.0070 | failed |
| CKD03 | class_reliability | 1.0,0.95,1.0,1.0,0.75 | 0.8370 | -0.0094 | -0.0284 | -0.0083 | failed |

HYP weight를 줄일수록 HYP AUC가 오히려 더 크게 하락 (CKD02: -0.0362). teacher의 HYP 신호가 약해도 그게 있어야 student가 HYP를 배울 수 있음 — down-weighting은 역효과.

---

## Section 3: Final Judgment

### Q1. F02T01을 넘은 variant가 있는가?
**없음.** 17개 AKD variant 전부 F02T01(0.8464) 미달.

### Q2. AUC 0.8470 이상 후보가 있는가?
**없음.** 최고 DKD02=0.8457.

### Q3. F1 또는 HYP/MI만 개선된 variant가 있는가?
- DKD01/DKD03: MI +0.004 개선, 그러나 HYP -0.011~0.012로 상쇄 → macro-AUC 기준 채택 불가
- DKD02: F1 +0.0004 (무시 가능), AUC -0.0007 → macro-AUC 기준 불합격
- macro-AUC를 primary metric으로 유지하는 한 어떤 variant도 채택 기준 미달

### Q4. Multi-seed 후보가 있는가?
**없음.** AUC threshold(>0.8464+0.001=0.8474) 기준에 미달하는 variant만 존재.

### 최종 권고
**Adaptive KD 계열 전체 중단.** F02T01(고정 weight, KL-div KD)이 현재 codebase 내 최적.

---

## Section 4: 다음 방향 (AKD 대체 후보)

AKD가 전부 실패했으므로 아래 두 방향 중 하나로 전환:

### A. RKDLR — Recoverability-aware KD
1L500 teacher의 full-rate 예측과 50Hz-degraded 예측의 차이를 recoverability signal로 사용.  
rate-drop에 민감한 class에 더 강한 KD 신호 → lead/rate 결핍을 직접 다룸.  
AKD Phase A+B와 달리 teacher confidence/agreement가 아닌 **teacher 자신의 rate sensitivity**를 사용.

### B. CLRD — Virtual Lead Importance Distillation
12L100 teacher의 lead-occlusion response를 이용해 per-sample KD 가중치 설정.  
missing lead가 중요한 샘플에 더 강한 KD → lead 결핍을 직접 다룸.  
논문 contribution 후보 — AKD와 명확히 구분되는 방향.

---

*Generated: 2026-05-23 | summary.csv: 18 rows (F02T01 + 17 AKD variants)*
