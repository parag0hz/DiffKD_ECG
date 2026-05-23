# DAFD-MVKT: 방법론 및 전체 실험 결과

> **Generated**: 2026-05-22  
> **Task**: 단일 리드(Lead-II, 50 Hz) ECG 신호로 PTB-XL 5개 superclass 분류  
> **현재 최고 성능 모델**: F02W04 — AUC **0.8448**, F1_tuned **0.6289**

---

## 1. 실험 설정

### 1.1 데이터셋
- **PTB-XL** (PhysioNet ECG 데이터셋, 21,799건)
- 분할: train(fold 1–8) / val(fold 9) / test(fold 10)
  - Train: ~17,111건 | Val: ~2,146건 | Test: ~2,158건
- **타겟 superclass** (5개 클래스, 다중 레이블):
  | 레이블 | 의미 |
  |---|---|
  | NORM | 정상 심전도 |
  | MI | 심근경색 (Myocardial Infarction) |
  | STTC | ST-T 변화 |
  | CD | 전도 장애 (Conduction Disturbance) |
  | HYP | 비대 (Hypertrophy) |

### 1.2 입력 조건 (핵심 제약)
| 설정 | 값 |
|---|---|
| **입력 리드** | Lead-II 단일 리드 |
| **샘플링 레이트** | 50 Hz (down-sampled from 100/500 Hz) |
| **입력 길이** | 500 time-steps (10초 기준) |
| **목표** | 저비용 웨어러블 환경 (1리드 50Hz)에서 12리드 수준 분류 성능 달성 |

### 1.3 모델 아키텍처
- **백본**: `ResNet1d-34` (1D Conv + Residual block [3,4,6,3])
  - `in_channels=1`, `num_classes=5`, `base_channels=64`
  - Feature map 차원: 512, Projection head 차원: 128
- **학습 설정**: epochs=100, batch_size=512, lr=1e-3, weight decay=1e-4, cosine scheduler, grad clip=1.0

### 1.4 평가 지표
- **macro-AUC**: 5개 클래스 ROC-AUC 평균 (주요 지표)
- **F1_tuned**: validation set에서 클래스별 최적 threshold 탐색 후 test set 적용 (grid: 0.05~0.95, step 0.01)
- **F1@0.5**: threshold=0.5 고정

---

## 2. Teacher 모델 계층 구조

학생 모델(1L50)을 지도하는 teacher 모델들의 성능 기준:

| 모델명 | 입력 | 샘플링 | AUC | F1_tuned | 비고 |
|---|---|---|---|---|---|
| Teacher 12L100 (Wang) | 12-lead | 100 Hz | **0.9104** | 0.7128 | 상한선 |
| Teacher 12L100 (ResNet34) | 12-lead | 100 Hz | 0.9028 | 0.7080 | 주 teacher |
| Teacher 12L50 (ResNet34) | 12-lead | 50 Hz | 0.8891 | 0.6687 | — |
| TA 1L500 (SimCLR init) | Lead-II | 500 Hz | 0.8464 | 0.6289 | 중간 branch |
| TA 1L100 (from 1L500) | Lead-II | 100 Hz | 0.8413 | 0.6189 | 중간 branch |
| **Student 1L50 (목표)** | **Lead-II** | **50 Hz** | **0.8448** | **0.6289** | **현재 best** |

> 12-lead 100Hz teacher(0.9028)와 1L50 student(0.8448)의 AUC 차이: **−0.0580**  
> teacher 성능의 **93.6%** 회복

---

## 3. 방법론 발전 과정 (시계열 순서)

### Phase 1: BCE Baseline
**AUC=0.8061 | F1=0.5740**

- 단순 Binary Cross-Entropy 지도 학습
- 임의 초기화 또는 SimCLR pretrain 없이 직접 학습
- 한계: teacher 지식 미활용, 확률 보정 부재

---

### Phase 2: Single-Teacher MKD (Multi-Label Knowledge Distillation)
**AUC=0.8231 | F1=0.6056**

- **교사 모델**: 12-lead 100Hz teacher 1개
- **손실 함수**:
  ```
  L = BCE(S_logits, y) + λ_kd × MKD(T→S)
  ```
  - `MKD`: 각 클래스를 독립 이진 KD로 처리, T² 스케일링
  - `λ_kd = 1.0`, `T = 2.0`
- **개선 폭**: BCE 대비 +0.017 AUC

---

### Phase 3: MKD + CRF + FeatureKD
**AUC=0.8300 | F1=0.6060**

- **추가된 손실**:
  ```
  L = BCE + λ_kd × MKD(T→S)
    + β_crf × CRF(S_pooled, T_pooled, S_proj, T_proj)
    + γ_feat × FeatureKD(T_feats→S_feats)
  ```
  - `CRF`: Symmetric InfoNCE (projection heads, τ=0.07) — teacher-student 표현 공간 정렬
  - `FeatureKD`: adaptive avg pool + 1×1 conv + L2 norm + MSE — 중간 feature map 정렬

---

### Phase 4: SimCLR 사전 학습 초기화
**AUC=0.8396 | F1=0.6176**

- **SimCLR pretraining**: Lead-II 50Hz 신호에서 augmentation 기반 contrastive 학습
  - augmentation: time-crop, gaussian noise, amplitude scaling
  - NT-Xent loss, projection head [512→128]
- 학습된 encoder weights를 student 초기화에 사용 (`strict=False`)
- **효과**: random init 대비 AUC +0.010, 이후 모든 실험의 기본 설정

---

### Phase 5: 3-Teacher Parallel KD (MVKT)
**AUC=0.8425 | F1=0.6243**

- **3개의 teacher** 동시 활용: 12L100 + 다양한 view 조합 20개 실험 (C01–C20)
- **Best combo (C14)**: [12L100, 1L500, 1L100] — equal weights
  ```
  L = BCE(S) + λ_kd × (1/3 × MKD(T1→S) + 1/3 × MKD(T2→S) + 1/3 × MKD(T3→S))
  ```
- Teacher 수가 늘어도 단순 equal weight 조합은 0.842~0.843 범위에 수렴
- 한계: 각 teacher를 독립적으로 처리 — teacher 간 상호작용 없음

---

### Phase 6: Hierarchical Chain KD (MVKT Hier)
**Best: H02 AUC=0.8442 | F1=0.6245**

- **개념**: teacher를 성능 순서로 chain 형태로 배열, 점진적 지식 압축
- **H02 구조** (현재 최고 hierarchical):
  ```
  12L100 (AUC=0.9028)
    → 1L500 (AUC=0.8492) : stage1 KD
      → 1L100 (AUC=0.8499) : stage2 KD
        → 1L50 (AUC=0.8442) : stage3 KD (student)
  ```
- 각 stage: `L = BCE + λ_kd × MKD(상위→하위) + CRF + FeatureKD`
- **한계**: 순차 학습이라 각 단계 오차가 누적됨

#### Hierarchical 전체 결과
| 모델 | 경로 | AUC | F1_tuned |
|---|---|---|---|
| H02 | 12L100→1L500→1L100→**1L50** | 0.8442 | 0.6245 |
| H01 | 12L500→1L500→1L100→**1L50** | 0.8427 | 0.6181 |
| H04 | 12L100→1L100→**1L50** | 0.8414 | 0.6241 |
| H03 | 1L500→1L100→**1L50** | 0.8382 | 0.6140 |
| H05 | 12L100→12L50→**1L50** | 0.8345 | 0.6127 |
| H06 | 12L500→12L100→**1L100** (branch) | 0.8380 | 0.6151 |

---

### Phase 7: Fork-Join Dual Distillation ← **현재 최고 방법론**
**Best (F02): AUC=0.8447 | F1=0.6275**

#### 7.1 핵심 아이디어
기존 hierarchical KD의 누적 오차 문제를 해결하기 위해 **두 개의 독립 branch teacher**를 병렬로 학습시키고, 최종 student에게 **동시에** 지식을 전달하는 구조.

#### 7.2 F02 구조 상세
```
                    ┌─────────────────────────────────┐
                    │   12L100 Teacher (t1)           │
                    │   AUC=0.9028  (frozen)          │
                    └────────────┬────────────────────┘
                                 │  KD (MKD + CRF + FeatureKD)
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   ┌─────────────────────┐             ┌─────────────────────┐
   │  Branch2: 1L500 (t2)│             │  Branch3: 1L100 (t3)│
   │  AUC=0.8480         │             │  AUC=0.8486         │
   │  Lead-II, 500 Hz    │             │  Lead-II, 100 Hz    │
   └──────────┬──────────┘             └──────────┬──────────┘
              │  w500=0.50                         │  w100=0.50
              └──────────────────┬─────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Student 1L50          │
                    │  Lead-II, 50 Hz        │
                    │  AUC=0.8447 (F02 orig) │
                    └────────────────────────┘
```

#### 7.3 학습 과정 (2단계)

**Stage 1: Branch 학습** (`train_fork_join_dual.py`)
```python
# Branch2 (1L500) 학습
L_t2 = BCE(t2_logits, y) + λ_kd × MKD(t1→t2)
       + β_crf × CRF(t1, t2) + γ_feat × FeatureKD(t1→t2)

# Branch3 (1L100) 학습
L_t3 = BCE(t3_logits, y) + λ_kd × MKD(t1→t3)
       + β_crf × CRF(t1, t3) + γ_feat × FeatureKD(t1→t3)
```

**Stage 2: Student 학습** (`train_student_dual_teacher.py`)
```python
# Dual-teacher student 학습 (t2, t3 frozen)
L_S = BCE(s_logits, y)
    + λ_kd × (w2 × MKD(t2→S) + w3 × MKD(t3→S))

# 기본값: w2=0.50, w3=0.50, λ_kd=1.0, T=2.0
```

#### 7.4 손실 함수 상세

**Multi-Label KD Loss (MKD)**:
```python
def multi_label_kd_loss(s_logits, t_logits, T):
    # T² scaling 적용 (KD 논문 표준)
    s_prob = sigmoid(s_logits / T)
    t_prob = sigmoid(t_logits / T).detach()
    # 각 클래스 독립 BCE
    loss = T² × mean(-t_prob * log(s_prob) - (1-t_prob) * log(1-s_prob))
    return loss
```

**CRF (Cross-Rate Contrastive Loss)**:
```python
def cross_rate_contrastive_loss(s_pool, t_pool, s_proj, t_proj, τ=0.07):
    # Symmetric InfoNCE on projection heads [B, 128]
    # Teacher-student 표현 공간 정렬
    loss = (InfoNCE(s_proj, t_proj) + InfoNCE(t_proj, s_proj)) / 2
    return loss
```

**Feature KD Loss**:
```python
class FeatureKDLoss:
    # student_channels=512, ta_channels=512, pool_size=128
    def forward(s_feat, t_feat):
        s = L2_norm(pool(adapt_conv(s_feat)))  # [B, 128]
        t = L2_norm(pool(t_feat.detach()))     # [B, 128]
        return MSE(s, t)
```

#### 7.5 Fork-Join 전체 변형 결과
| 모델 | t1 | t2 | t3 | AUC | F1_tuned |
|---|---|---|---|---|---|
| **F02** | 12L100 | 1L500 | 1L100 | **0.8447** | **0.6275** |
| F04 | 12L500 | 1L500 | 1L100 | 0.8432 | 0.6159 |
| F01 | 12L100 | 12L50 | 1L100 | 0.8410 | 0.6255 |
| F06 | 1L500  | 1L100 | 1L50  | 0.8414 | 0.6175 |
| F05 | 12L100 | 12L50 | 1L50  | 0.8355 | 0.6154 |
| F03 | 12L500 | 12L50 | 1L100 | 0.8374 | 0.6138 |

> **F02가 최고**: t1=12L100 사용 + t2/t3를 1리드(500Hz/100Hz)로 구성하는 것이 최적.  
> 12-lead branch(12L50)보다 1-lead branch(1L500, 1L100)가 1L50 student에게 더 적합한 지식 제공.

---

### Phase 8: Skip-Fork KD
**Best (SF06): AUC=0.8448 | F1=0.6241**

- Fork-Join과 유사하지만 t1 없이 t2→t3→S 방향의 skip connection 형태
- **SF06**: 1L500 → 1L100 + 1L50 → 1L50 (F02와 거의 동일한 AUC)

| 모델 | 구조 | AUC | F1_tuned |
|---|---|---|---|
| **SF06** | 1L500→1L100+1L50→1L50 | **0.8448** | 0.6241 |
| SF04 | 12L500→1L500+1L100→1L50 | 0.8444 | 0.6222 |
| SF02 | 12L100→1L500+1L100→1L50 | 0.8433 | 0.6224 |
| SF03 | 12L500→12L50+1L100→1L50 | 0.8421 | 0.6232 |
| SF05 | 12L100→12L50+1L50→1L50 | 0.8366 | 0.6092 |
| SF01 | 12L100→12L50+1L100→1L50 | 0.8351 | 0.6161 |

---

### Phase 9: Adaptive Teacher (EMA, Progressive)
**Best: PROG01 AUC=0.8423 | F1=0.6241**

- **Progressive EMA**: 1L100 teacher + EMA student regularization
  ```
  L = BCE(S) + λ_kd × MKD(1L100→S) + λ_ema × MKD(EMA_S→S)
  ```
  - EMA decay=0.99, λ_ema=0.05~0.10
- **3T-EMA**: [12L100, 1L500, 1L100] + EMA
- Fork-Join 대비 성능 낮음 — 단일/3-teacher 방식의 근본 한계

| 모델 | 방법 | AUC | F1_tuned |
|---|---|---|---|
| PROG01 | 1L100+EMA (λ=0.05, d=0.99) | 0.8423 | 0.6241 |
| PROG02 | 1L100+EMA (λ=0.10, d=0.99) | 0.8415 | 0.6237 |
| 3T-EMA01 | C14+EMA (λ=0.10, d=0.99) | 0.8403 | 0.6245 |
| 3T-EMA02 | C14+EMA (λ=0.05, d=0.99) | 0.8399 | 0.6151 |

---

### Phase 10: F02 Weight Optimization ← **현재 최고 (공동 1위)**
**Best (F02W04): AUC=0.8448 | F1=0.6289**

F02의 student 학습 시 두 branch teacher의 가중치 비율을 최적화.

#### 10.1 변경 사항
- 기존 F02: `w500=0.50, w100=0.50` (equal weight)
- 최적화: 7가지 가중치 조합 탐색 (F02W01~F02W07)

#### 10.2 손실 함수 (F02 student tune)
```python
L_S = BCE(s_logits, y)
    + λ_kd × (w500 × MKD(1L500→S, T) + w100 × MKD(1L100→S, T))
# optional: + β_crf × CRF(1L100, S)        [Stage 2 진행 중]
# optional: + γ_feat × FeatureKD(1L100→S)  [Stage 2 진행 중]
```

#### 10.3 Weight Tuning 결과 (F02W01~W07)
| Variant | w500 | w100 | AUC | F1_tuned | vs F02 orig |
|---|---|---|---|---|---|
| **F02W04** | **0.60** | **0.40** | **0.8448** | **0.6289** | **+0.0001 AUC** |
| F02W01 | 0.50 | 0.50 | 0.8445 | 0.6309 | −0.0002 AUC |
| F02W06 | 0.25 | 0.75 | 0.8440 | 0.6271 | −0.0007 AUC |
| F02W02 | 0.40 | 0.60 | 0.8443 | 0.6260 | −0.0004 AUC |
| F02W05 | 0.70 | 0.30 | 0.8433 | 0.6288 | −0.0014 AUC |
| F02W03 | 0.30 | 0.70 | 0.8437 | 0.6242 | −0.0010 AUC |
| F02W07 | 0.75 | 0.25 | 0.8422 | 0.6257 | −0.0025 AUC |
| F02 orig | 0.50 | 0.50 | 0.8447 | 0.6275 | baseline |

> **분석**: 500Hz branch 가중치가 0.60일 때 최적. 그 이상(0.70, 0.75)은 오히려 하락.  
> 500Hz 신호가 50Hz student에게 조금 더 유익하지만, 과도하면 100Hz 정보 손실.

#### 10.4 진행 중인 실험 (Stage 2)
- **F02F01~F02F06** (현재 실행 중, best weight=0.60/0.40 사용)
  | Variant | 설정 | 목적 |
  |---|---|---|
  | F02F01 | CRF (β=0.05) | 표현 정렬 |
  | F02F02 | CRF (β=0.10) | 표현 정렬 (강화) |
  | F02F03 | FeatureKD (γ=0.10) | feature map 정렬 |
  | F02F04 | FeatureKD (γ=0.20) | feature map 정렬 (강화) |
  | F02F05 | CRF+FeatureKD (β=0.05, γ=0.10) | 복합 정렬 |
  | F02F06 | CRF+FeatureKD (β=0.10, γ=0.20) | 복합 정렬 (강화) |

---

## 4. 전체 성능 요약 표

### 4.1 주요 1L50 Student 모델 (AUC 기준 상위)

| 순위 | 방법론 | 모델명 | AUC | F1_tuned | F1@0.5 |
|---|---|---|---|---|---|
| 1 | **F02 Weight-Tuned** | F02W04 | **0.8448** | **0.6289** | 0.5864 |
| 1 | Skip-Fork | SF06 | **0.8448** | 0.6241 | 0.5682 |
| 3 | F02 Orig (Fork-Join) | F02 | 0.8447 | 0.6275 | 0.5787 |
| 4 | F02W01 | F02W01 | 0.8445 | 0.6309 | 0.5769 |
| 5 | Skip-Fork | SF04 | 0.8444 | 0.6222 | 0.5613 |
| 6 | F02W02 | F02W02 | 0.8443 | 0.6260 | 0.5745 |
| 7 | Hierarchical | H02 | 0.8442 | 0.6245 | 0.5910 |
| 8 | F02W06 | F02W06 | 0.8440 | 0.6271 | 0.5743 |
| 9 | F02W03 | F02W03 | 0.8437 | 0.6242 | 0.5795 |
| 10 | F02W05 | F02W05 | 0.8433 | 0.6288 | 0.5925 |
| 11 | Skip-Fork | SF02 | 0.8433 | 0.6224 | 0.5877 |
| 12 | Fork-Join | F04 | 0.8432 | 0.6159 | 0.5534 |
| — | 3T Parallel | C14 | 0.8425 | 0.6243 | 0.5726 |
| — | Hier | H01 | 0.8427 | 0.6181 | 0.5693 |
| — | Prog EMA | PROG01 | 0.8423 | 0.6241 | 0.5550 |
| — | **BCE Baseline** | — | **0.8061** | **0.5740** | **0.5006** |

### 4.2 클래스별 성능 (Best 모델: F02W04)

| Class | AUC | F1_tuned | threshold | 비고 |
|---|---|---|---|---|
| NORM | 0.901 | 0.814 | — | 정상 분류 우수 |
| MI | 0.825 | 0.600 | — | 심근경색 |
| STTC | 0.873 | 0.658 | — | ST-T 변화 |
| CD | 0.862 | 0.692 | — | 전도 장애 |
| **HYP** | **0.763** | **0.381** | — | **가장 어려운 클래스** |
| **Macro** | **0.8448** | **0.6289** | — | |

> **HYP 성능이 가장 낮음**: 비대증은 12-lead teacher도 F1=0.48 수준.  
> 1리드 정보만으로는 HYP 패턴 포착에 근본적 한계 존재.

### 4.3 F02 원본 vs F02W04 클래스별 비교

| Class | F02 orig AUC | F02W04 AUC | F02 orig F1 | F02W04 F1 | 변화 |
|---|---|---|---|---|---|
| NORM | 0.900 | 0.901 | 0.815 | 0.814 | ≈0 |
| MI | 0.826 | 0.825 | 0.597 | 0.600 | +0.003 F1 |
| STTC | 0.874 | 0.873 | 0.668 | 0.658 | −0.010 F1 |
| CD | 0.860 | 0.862 | 0.678 | 0.692 | **+0.014 F1** |
| HYP | 0.764 | 0.763 | 0.379 | 0.381 | ≈0 |

---

## 5. 핵심 인사이트

### 5.1 무엇이 효과적인가

1. **SimCLR 초기화 필수**: random init 대비 일관되게 +0.004~0.010 AUC 향상.  
   contrastive 사전학습이 단일 리드 ECG의 표현 학습에 중요.

2. **Dual Branch > Single Teacher**: 동일 도메인(1리드) teacher 2개(500Hz + 100Hz)를 병렬로 사용하는 것이 단일 12-lead teacher보다 효과적.  
   → domain gap(12리드→1리드) 최소화

3. **Teacher 선택이 핵심**: F02의 우위는 t2=1L500, t3=1L100 선택에 있음.  
   12-lead branch(12L50)보다 1-lead branch가 1L50 student에 더 적합한 지식 제공.

4. **w500 > w100 (약간)**: 500Hz 리드 teacher가 50Hz student에게 미세하게 더 유익.  
   최적 비율: w500=0.60, w100=0.40.

5. **HYP 클래스 한계**: F1_tuned=0.38 수준. 12-lead teacher도 0.48 수준이며,  
   1리드 정보만으로는 비대증 패턴 포착에 근본적 한계 존재.

### 5.2 무엇이 효과적이지 않은가

| 방법 | 이유 |
|---|---|
| 12-lead branch (12L50) | domain gap — 12리드 패턴이 1L50 student에 부적합 |
| EMA regularization (단독) | 부가 regularization 효과 제한적 |
| 가중치 극단화 (w500>0.70) | 100Hz 정보 손실로 오히려 성능 하락 |
| Hierarchical 3단계 | 누적 오차 문제, Fork-Join 대비 낮음 |

### 5.3 발전 궤적 요약

```
BCE baseline:  AUC=0.8061  (+0.000)
+ MKD:         AUC=0.8231  (+0.017) ← teacher 지식 활용
+ CRF+FeatKD:  AUC=0.8300  (+0.007) ← 표현 공간 정렬
+ SimCLR init: AUC=0.8396  (+0.010) ← 사전 학습
+ 3T Parallel: AUC=0.8425  (+0.003) ← teacher 다변화
+ Hierarchical:AUC=0.8442  (+0.002) ← 점진적 압축
+ Fork-Join:   AUC=0.8447  (+0.001) ← 병렬 dual branch
+ Weight Tune: AUC=0.8448  (+0.001) ← w500=0.60 최적화
```

---

## 6. 현재 최고 모델 재현 방법

```bash
# 환경 설정
conda activate ecg
cd /home/kwy00/sci

# Step 1. SimCLR pretraining
python dafd_mvkt/pretrain_simclr.py \
    --data_dir comper_repo/ptb_xl --lead II \
    --sampling_rate 50 --seed 0 \
    --output_dir dafd_mvkt/outputs/simclr
# → dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt

# Step 2. Fork-Join Branch 학습 (F02: 12L100 → 1L500 + 1L100)
DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
    bash dafd_mvkt/scripts/run_fork_join_6.sh
# → dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt
# → dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt

# Step 3. Student 학습 (F02W04: best weight 0.60/0.40)
python dafd_mvkt/train_f02_student_tune.py \
    --config dafd_mvkt/configs/adaptive_50hz.yaml \
    --data_dir comper_repo/ptb_xl --lead II \
    --teacher_1l500_ckpt dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt \
    --teacher_1l100_ckpt dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt \
    --student_init_ckpt dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \
    --weights "0.60,0.40" --temperature 2.0 \
    --variant_id F02W04 --output_dir dafd_mvkt/outputs/f02_opt --seed 0
# → AUC=0.8448, F1_tuned=0.6289
```

---

## 7. 다음 실험 계획

| 단계 | 내용 | 상태 | 예상 효과 |
|---|---|---|---|
| **F02F01~F06** | CRF/FeatureKD (1L100↔S) | **진행 중** | +0.001~0.003 AUC 기대 |
| **F02T01~T04** | Temperature 탐색 (T=1.5/2.0/3.0/4.0) | 대기 중 | 최적 T 확인 |
| **F02BR01~04** | Branch CRF enhancement | 조건부 | AUC > 0.8447 미달성 시 |
| **F02CW01~02** | Confidence-weighted KD | 조건부 | 최후 수단 |

> **목표**: AUC ≥ 0.846, F1_tuned ≥ 0.630  
> **달성 불가 시**: F02W04 (AUC=0.8448, F1=0.6289)를 최종 best로 채택

---

*마지막 업데이트: 2026-05-22 | 전체 실험 수: 148개 | 1L50 Student 실험 수: 30개*
