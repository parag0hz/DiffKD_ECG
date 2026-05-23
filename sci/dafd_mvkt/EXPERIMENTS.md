# DAFD-MVKT 실험 현황 및 인수인계 문서

> 작성일: 2026-05-22  
> 작성자: Claude (Sonnet 4.6)  
> 목적: 다른 컴퓨터/다른 Claude 세션에서 실험을 이어받기 위한 완전한 컨텍스트 문서

---

## 1. 프로젝트 개요

**목표**: Single-lead (Lead II, 50Hz) ECG로 5-class 심장질환 분류  
**데이터**: PTB-XL (17,084 train / 2,146 val / 2,158 test)  
**5 classes**: NORM, MI, STTC, CD, HYP (superclass, multi-label)  
**Primary metric**: macro-AUC (secondary: macro-F1 with tuned threshold)  
**작업 디렉토리**: `/home/kwy00/sci` (모든 명령은 여기서 실행)

---

## 2. 데이터셋 설정

PTB-XL 데이터는 `comper_repo/ptb_xl/`에 위치해야 함:
```
comper_repo/ptb_xl/ptbxl_database.csv
comper_repo/ptb_xl/scp_statements.csv
comper_repo/ptb_xl/records100/...
comper_repo/ptb_xl/records500/...
```

Split policy: strat_fold 1-8=train, 9=val, 10=test (PTB-XL 공식 split)  
Normalization: per-sample z-score (DataLoader에서 적용, 전처리 시 미적용)

---

## 3. 핵심 아키텍처: F02 Fork-Join Dual Distillation

```
12L100 Teacher (ResNet1d-34, 12-lead, 100Hz)
    ↓ (branch 2)                ↓ (branch 3)
1L500 Teacher               1L100 Teacher
(Lead II, 500Hz)            (Lead II, 100Hz)
    ↓                           ↓
    └──────── Student ──────────┘
          (Lead II, 50Hz)
          ResNet1d-34, in_ch=1
```

**Student 모델**: `ResNet1d(in_channels=1, num_classes=5, layers=[3,4,6,3], base_channels=64)`  
- feat_dim=512, proj_dim=128  
- 파라미터: ~7.55M  
- 입력 shape: [B, 1, 500] (50Hz × 10s)

**Teacher 모델들**:
- 12L100 Teacher: `dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt`
- 1L500 Branch: `dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt`
- 1L100 Branch: `dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt`

**Teacher 입력 shape**:
- 1L500: [B, 1, 5000] (500Hz × 10s)
- 1L100: [B, 1, 1000] (100Hz × 10s)

---

## 4. 학습 설정 (고정값 — 절대 변경 금지)

```yaml
# dafd_mvkt/configs/adaptive_50hz.yaml
epochs: 100
batch_size: 512
lr: 1.0e-3
weight_decay: 1.0e-4
scheduler: cosine (CosineAnnealingLR, T_max=epochs)
grad_clip: 1.0
num_workers: 4
optimizer: AdamW
```

**Loss function**:
```
L = BCE(s_logits, y) + lambda_kd * (w500 * MKD(1L500→S) + w100 * MKD(1L100→S))
```

**MKD (Multi-Label KD)**: per-class binary KL divergence with temperature scaling  
- `multi_label_kd_loss(s_logits, t_logits, T)` in `dafd_mvkt/losses/mkd.py`
- T² scaling on KD loss 포함

**Threshold tuning**: val set에서 grid search (0.05~0.95, step 0.01), per-class 최적 threshold 적용

---

## 5. 학습 진입점

```bash
# 작업 디렉토리: /home/kwy00/sci
conda activate ecg  # Python: /home/kwy00/anaconda3/envs/ecg/bin/python

python dafd_mvkt/train_f02_student_tune.py \
    --config dafd_mvkt/configs/adaptive_50hz.yaml \
    --data_dir comper_repo/ptb_xl \
    --lead II \
    --teacher_1l500_ckpt dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt \
    --teacher_1l100_ckpt dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt \
    --student_init_ckpt  dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \
    --weights "<w500>,<w100>" \
    --temperature <T> \
    --attention_type none \
    --lambda_att 0.0 \
    --variant_id <ID> \
    --output_dir <OUT_DIR> \
    --seed 0
```

**주요 CLI 옵션**:
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--weights` | w500,w100 (자동 normalize) | `0.5,0.5` |
| `--temperature` | KD temperature | `2.0` |
| `--lambda_kd` | KD loss weight | `1.0` |
| `--attention_type` | `none`/`se`/`cbam`/`tgmta` | `none` |
| `--lambda_att` | Attention KD weight | `0.0` |
| `--use_crf` | Cross-Rate contrastive loss | off |
| `--use_feature_kd` | Feature KD loss | off |

---

## 6. 비교 기준선 (Reference Baselines)

| 모델 | 조건 | AUC | F1_tuned |
|------|------|-----|----------|
| BCE only | Lead II, 50Hz, no KD | 0.8061 | 0.5740 |
| PROG-SimCLR | progressive KD | 0.8396 | 0.6176 |
| C14 (parallel) | 14-class parallel KD | 0.8425 | 0.6243 |
| MVKT | multi-view KD (기존 최고) | 0.8430 | 0.6260 |
| **F02 (fork-join)** | **F02 원래 결과** | **0.8447** | **0.6275** |

> **목표**: macro-AUC > 0.8447 (F02 original) 이상 달성 및 최적화

---

## 7. 완료된 실험 결과

### 7.1 Stage 1: Teacher Weight Tuning at T=2.0 (F02W01–07)

출력 디렉토리: `dafd_mvkt/outputs/f02_opt/`

| variant | w500 | w100 | T | AUC | F1_tuned | 비고 |
|---------|------|------|---|-----|----------|------|
| F02W01 | 0.500 | 0.500 | 2.0 | 0.8445 | 0.6309 | |
| F02W02 | 0.400 | 0.600 | 2.0 | 0.8443 | 0.6260 | |
| F02W03 | 0.300 | 0.700 | 2.0 | 0.8437 | 0.6242 | |
| **F02W04** | **0.600** | **0.400** | **2.0** | **0.8448** | **0.6289** | **Stage 1 best** |
| F02W05 | 0.700 | 0.300 | 2.0 | 0.8433 | 0.6288 | |
| F02W06 | 0.250 | 0.750 | 2.0 | 0.8440 | 0.6271 | |
| F02W07 | 0.750 | 0.250 | 2.0 | 0.8422 | 0.6257 | |

**결론**: w500=0.60이 유리, 1L500 teacher가 약간 더 중요함.

### 7.2 Stage 2: CRF / Feature KD at T=2.0 (F02F01–06)

기준: F02W04 (w=0.60,0.40, T=2.0)

| variant | CRF | β_crf | FeatureKD | γ | AUC | F1_tuned |
|---------|-----|-------|-----------|---|-----|----------|
| F02F01 | Y | 0.05 | N | - | 0.8443 | 0.6291 |
| F02F02 | Y | 0.10 | N | - | 0.8441 | 0.6270 |
| F02F03 | N | - | Y | 0.10 | 0.8445 | 0.6272 |
| F02F04 | N | - | Y | 0.20 | 0.8446 | 0.6224 |
| F02F05 | Y | 0.05 | Y | 0.10 | 0.8438 | 0.6254 |
| F02F06 | Y | 0.10 | Y | 0.20 | 0.8444 | 0.6296 |

**결론**: CRF/FeatureKD 모두 유의미한 개선 없음. F02W04 유지.

### 7.3 Stage 3: Temperature Tuning (F02T01–04)

기준: F02W04 (w=0.60,0.40), best CRF/Feat = none

| variant | T | AUC | F1_tuned | 비고 |
|---------|---|-----|----------|------|
| **F02T01** | **1.5** | **0.8464** | **0.6254** | **★ 현재 전체 최고** |
| F02T02 | 2.0 | 0.8448 | 0.6289 | = F02W04 |
| F02T03 | 3.0 | 0.8439 | 0.6210 | |
| F02T04 | 4.0 | 0.8441 | 0.6253 | |

**결론**: T=1.5가 최적. F02T01이 현재 전체 최고 (AUC=0.8464, F02 대비 +0.0017).

**F02T01 per-class AUC**: NORM=0.9005, MI=0.8217, STTC=0.8742, CD=0.8644, HYP=0.7713  
**F02T01 per-class F1_tuned**: NORM=0.8113, MI=0.5918, STTC=0.6637, CD=0.6777, HYP=0.3827

### 7.4 Stage 4: TG-MTA Attention (ATTN03–07) ← 중단됨

기준: F02T01 (w=0.60,0.40, T=1.5)  
출력 디렉토리: `dafd_mvkt/outputs/tgmta_f02/`

| variant | att_type | layers | λ_att | params | AUC | F1_tuned | vs F02T01 |
|---------|----------|--------|-------|--------|-----|----------|-----------|
| F02T01 (base) | none | - | 0 | 7.55M | 0.8464 | 0.6254 | 기준 |
| ATTN03 | tgmta | layer4 | 0.000 | 8.65M | 0.8443 | 0.6309 | -0.0021 |
| ATTN04 | tgmta | layer3,layer4 | 0.000 | 8.65M | 0.8439 | 0.6270 | -0.0025 |
| ATTN05 | tgmta | layer4 | 0.010 | 8.65M | 0.8444 | 0.6267 | -0.0020 |
| ATTN06 | tgmta | layer4 | 0.050 | 8.65M | 0.8439 | 0.6204 | -0.0025 |
| ATTN07 | tgmta | layer4 | 0.100 | 8.65M | 0.8440 | 0.6197 | -0.0024 |

**결론**: TG-MTA가 AUC 기준 F02T01보다 모두 낮음 → **TG-MTA branch 중단**.  
(F1_tuned는 일부 높지만 primary metric AUC가 하락하므로 개선 불인정)

---

## 8. 현재 진행 중인 실험

### Stage 5: T=1.5 Teacher Weight Tuning (TW15)

**배경**: Stage 1 weight tuning은 T=2.0에서 수행됨. F02T01이 T=1.5에서 w=0.60,0.40을 사용하지만, T=1.5 조건에서 최적 weight가 다를 수 있음.

**실험 설정**:
- temperature=1.5 (고정)
- attention_type=none, lambda_att=0.0 (고정)
- w500+w100은 auto-normalize됨

**출력 디렉토리**: `dafd_mvkt/outputs/tw15_weight_tune/`  
**실행 스크립트**: `dafd_mvkt/scripts/run_tw15_weight_tune.sh`

```bash
# 실행 방법
DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
  bash dafd_mvkt/scripts/run_tw15_weight_tune.sh
```

| variant | w500 | w100 | T | 상태 |
|---------|------|------|---|------|
| F02T01 | 0.600 | 0.400 | 1.5 | ✓ 완료 (기준, AUC=0.8464) |
| TW15_01 | 0.500 | 0.500 | 1.5 | 완료 (AUC=0.8446) |
| TW15_02 | 0.550 | 0.450 | 1.5 | 진행 중 |
| TW15_03 | skip | - | - | 건너뜀 |
| TW15_04 | 0.650 | 0.350 | 1.5 | 대기 |
| TW15_05 | 0.700 | 0.300 | 1.5 | 대기 |
| TW15_06 | 0.400 | 0.600 | 1.5 | 대기 |
| TW15_07 | 0.300 | 0.700 | 1.5 | 대기 |
| TW15_08 | 0.750 | 0.250 | 1.5 | 대기 |

**판정 기준**:
- Primary metric: macro-AUC
- AUC > F02T01(0.8464) + 0.001 → 다음 단계 multi-seed 대상
- AUC 차이 ±0.0005 이내 → 동률, F1_tuned 및 class-wise 안정성 비교
- F1_tuned가 높아도 AUC가 F02T01보다 낮으면 최종 best 불인정

---

## 9. 다음 실험 계획 (TW15 이후)

### 가능한 다음 단계들

**Option A: Multi-seed 검증** (TW15에서 best가 나온 경우)
```bash
# seed=1,2로 best variant 재실행
python dafd_mvkt/train_f02_student_tune.py \
    ... --weights <best_w> --temperature 1.5 \
    --variant_id <best_id>_seed1 --seed 1
```

**Option B: Fine-grained weight sweep** (TW15 결과에서 peak 탐색)
- best 구간 주변 ±0.05 단위로 추가 탐색

**Option C: T=1.5 + 다른 T 조합 재탐색** (만약 TW15에서 best weight가 0.60,0.40이면)
- 이미 최적점이므로 다른 개선 방향 모색

**Option D: ATTN08-10 실행** (현재 중단된 TG-MTA 나머지)
```bash
RUN_ALL=1 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
  bash dafd_mvkt/scripts/run_tgmta_f02.sh
```
- ATTN08: tgmta layer4, λ=0.05, KL loss
- ATTN09: tgmta layer3+4, λ=0.05, mse_prob
- ATTN10: tgmta layer4, wide_kernel, λ=0.05

---

## 10. 코드 구조

```
dafd_mvkt/
├── configs/
│   └── adaptive_50hz.yaml          # 학습 설정 (고정)
├── data/
│   └── ptbxl_multiteacher_dataset.py  # Dataset class
├── losses/
│   ├── mkd.py                      # multi_label_kd_loss
│   ├── contrastive.py              # cross_rate_contrastive_loss
│   ├── feature_kd.py               # FeatureKDLoss
│   └── attention_kd.py             # make_dual_teacher_saliency, attention_kd_loss
├── models/
│   ├── resnet1d.py                 # ResNet1d backbone (attention 지원)
│   ├── attention_modules.py        # SE1D, CBAM1D, TGMTAModule
│   └── heads.py                    # ProjectionHead
├── utils/
│   ├── metrics.py                  # compute_metrics, find_best_thresholds
│   ├── checkpoint.py               # load_checkpoint, save_checkpoint
│   ├── seed.py                     # set_seed
│   └── simclr_checkpoint.py        # load_encoder_init
├── scripts/
│   ├── run_f02_optimization.sh     # Stage 1-3 실험 (완료)
│   ├── run_tgmta_f02.sh            # TG-MTA 실험 (중단)
│   └── run_tw15_weight_tune.sh     # ★ 현재 실험
├── tests/
│   └── test_attention_modules.py   # Sanity check (26/26 pass)
├── outputs/
│   ├── forkjoin/                   # Branch teacher 체크포인트
│   ├── simclr/                     # SimCLR pretrained encoder
│   ├── f02_opt/                    # Stage 1-3 결과
│   ├── tgmta_f02/                  # TG-MTA 결과 (중단)
│   └── tw15_weight_tune/           # ★ 현재 결과 저장 위치
├── train_f02_student_tune.py       # ★ 메인 학습 스크립트
└── EXPERIMENTS.md                  # 이 파일
```

---

## 11. 중요 구현 노트

### 11.1 sigmoid bug fix (중요)
`_run_eval()` 함수가 raw logits 반환 → probs로 수정됨 (현재 코드는 수정된 상태):
```python
logits = torch.cat(all_logits).numpy()
probs  = 1.0 / (1.0 + np.exp(-logits))   # sigmoid 반드시 적용
return probs, torch.cat(all_labels).numpy(), all_ids
```
이전 실험(F02W~)은 AUC는 정확하지만 F1_tuned 값이 잘못됨. 현재 코드로 재현 시 AUC는 동일, F1_tuned는 약간 다를 수 있음.

### 11.2 ResNet1d backward compatibility
`attention_type="none"` (기본값)이면 기존 checkpoint를 `strict=True`로 로드 가능.  
`attention_type="tgmta"` 등을 사용하면 `strict=False`로 로드 (attention 가중치만 새로 초기화).

### 11.3 학습 재현성
- `set_seed(0)` 적용: random, numpy, torch, cuda 모두 고정
- `DataLoader(shuffle=True)`의 worker seed도 고정됨
- 동일 seed/config → 동일 결과 재현 가능

### 11.4 MKD temperature 적용 방식
```python
# multi_label_kd_loss 내부
# T^2 scaling: KL(σ(t/T), σ(s/T)) * T^2
```
T를 낮추면(1.5) softer target 효과가 약해지고 peak가 더 sharp해짐. T=1.5가 T=2.0보다 좋았음.

---

## 12. 빠른 실험 재현 명령어

```bash
# F02T01 재현 (현재 best)
python dafd_mvkt/train_f02_student_tune.py \
    --config dafd_mvkt/configs/adaptive_50hz.yaml \
    --data_dir comper_repo/ptb_xl --lead II \
    --teacher_1l500_ckpt dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt \
    --teacher_1l100_ckpt dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt \
    --student_init_ckpt  dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \
    --weights 0.60,0.40 --temperature 1.5 \
    --variant_id F02T01_repro --output_dir /tmp/test_repro --seed 0
# 기대 AUC: ~0.8464

# sanity check
python dafd_mvkt/tests/test_attention_modules.py
# 기대: 26/26 passed

# TW15 실험 현황 확인
python -c "
import json, glob
for f in sorted(glob.glob('dafd_mvkt/outputs/tw15_weight_tune/metrics_test_*.json')):
    d = json.load(open(f))
    print(f\"{d['variant_id']}: w={d['w500']:.2f}/{d['w100']:.2f} AUC={d['macro_auc']:.4f} F1t={d['macro_f1_tuned']:.4f}\")
"
```

---

## 13. 실험 관리 규칙

1. **기존 결과 덮어쓰지 말 것**: `run_or_skip()` 패턴 사용 — metrics_test JSON이 있으면 skip
2. **metric 계산 방식 변경 금지**: compute_metrics, find_best_thresholds 함수 그대로 사용
3. **데이터 split 변경 금지**: strat_fold 1-8/9/10 고정
4. **모든 결과 summary.csv에 기록**: 좋지 않아도 숨기지 말 것
5. **primary metric = macro-AUC**: F1_tuned는 참고용
6. **threshold tuning 방식 고정**: val set grid search (0.05~0.95, step 0.01)

---

## 14. 환경 설정

```bash
conda activate ecg
# 또는
/home/kwy00/anaconda3/envs/ecg/bin/python

# 주요 패키지: torch, numpy, pandas, scikit-learn, scipy, matplotlib, tqdm, wfdb, yaml
```

GPU 사용 여부는 자동 감지 (`torch.cuda.is_available()`).
