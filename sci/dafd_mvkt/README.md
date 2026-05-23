# DAFD-MVKT

**Diagnosis-Aware Frequency Distillation for Low-Sampling-Rate Single-Lead ECG Classification**

Extends multi-view ECG knowledge transfer (MVKT) to a realistic wearable setting:
a frozen 12-lead 500 Hz **teacher** transfers diagnostic knowledge to a 1-lead
100/50 Hz **student** via three distillation objectives.

---

## Goal

| Setting | Input | Hz |
|---------|-------|----|
| Teacher | 12-lead | 500 |
| Student | 1-lead (default: Lead II) | 100 or 50 |
| Task | PTB-XL 5-superclass multi-label classification | — |

Classes: `NORM`, `MI`, `STTC`, `CD`, `HYP`

---

## Dataset Preparation

Download PTB-XL from PhysioNet and organise as:

```
/path/to/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records500/        ← 500 Hz WFDB records
└── records100/        ← 100 Hz WFDB records (optional)
```

No pre-processing needed — the dataset loader handles anti-aliased
downsampling on the fly.

---

## Training

### 1. Teacher (12-lead 500 Hz)

```bash
python train_teacher.py \
    --config configs/teacher_500hz.yaml \
    --data_dir /path/to/ptbxl
```

Best checkpoint → `outputs/teacher_best.pt`

### 2. Student baseline (BCE only)

```bash
python train_student.py \
    --config configs/student_100hz.yaml \
    --data_dir /path/to/ptbxl \
    --teacher_ckpt outputs/teacher_best.pt \
    --losses bce
```

### 3. Student + full distillation

```bash
python train_student.py \
    --config configs/student_100hz.yaml \
    --data_dir /path/to/ptbxl \
    --teacher_ckpt outputs/teacher_best.pt \
    --losses bce,mkd,crf,daf
```

Same commands with `configs/student_50hz.yaml` for 50 Hz experiments.

---

## Evaluation

```bash
python evaluate.py \
    --config configs/student_100hz.yaml \
    --data_dir /path/to/ptbxl \
    --ckpt outputs/student_100hz_best.pt \
    --split test \
    --model_name student_100hz_full \
    [--tune_threshold]
```

`--tune_threshold` finds per-class thresholds maximising F1 on the validation
set and applies them to the test set.

Outputs:
- `outputs/predictions_test_student_100hz_full.csv`
- `outputs/metrics_test_student_100hz_full.json`

---

## Loss Functions

### L_BCE — Supervised multi-label BCE
Standard `BCEWithLogitsLoss` on student predictions.

### L_MKD — Multi-Label Knowledge Distillation
Class-wise binary distillation (Hinton KD extended to multi-label).
For each class `c`, forms a 2-class distribution over `[z_c/T, 0]`
and minimises KL-divergence between teacher and student soft distributions.
Temperature `T=2.0`.

### L_CRF — Cross-Rate Contrastive Feature Loss
Symmetric InfoNCE between teacher and student L2-normalised projection
embeddings within a batch. Same ECG record forms positive pairs across
sampling rates. Temperature `τ=0.07`.

### L_DAF — Diagnosis-Aware Frequency Distillation *(main novelty)*
Aligns frequency-domain content of teacher and student temporal feature maps,
gated by a learnable **diagnosis-frequency gate** `W ∈ R^{C×F}`.

```
student_spec  = |FFT(AdaptivePool(student_fm))|   # [B, F]
teacher_spec  = |FFT(AdaptivePool(teacher_fm))|   # [B, F]
gate          = sigmoid(gate_logits)              # [C, F]  learnable
weights       = sigmoid(teacher_logits) @ gate    # [B, F]
L_DAF         = mean( weights * (student_spec - teacher_spec)^2 )
```

The gate learns which frequency bands carry diagnostic information for each
superclass (e.g. low-frequency ST changes for MI vs. high-frequency QRS
features for CD).

### Total Loss
```
L = L_BCE + α·L_MKD + β·L_CRF + γ·L_DAF
```
Defaults: `α=1.0`, `β=0.1`, `γ=0.1` (configurable in YAML).

---

## Experiment Table

| Method | Lead | Hz | Macro AUC | Macro F1 |
|--------|------|----|-----------|----------|
| Teacher | 12 | 500 | 0.9121 | 0.7190 |
| Baseline | II | 100 | 0.8387 | 0.6202 |
| + MKD | II | 100 | 0.8378 | 0.6123 |
| + MKD+CRF | II | 100 | 0.8396 | 0.6137 |
| **+ MKD+CRF+DAF** | **II** | **100** | **0.8391** | **0.6061** |
| Baseline | II | 50 | 0.8309 | 0.6098 |
| + MKD+CRF+DAF | II | 50 | 0.8299 | 0.6086 |

---

## Experimental Protocol After Initial Results

### Observed Results (single seed, threshold-tuned F1)

| Method | Lead | Hz | Macro AUC | Macro F1 |
|---|---|---:|---:|---:|
| Teacher | 12 | 500 | 0.9121 | 0.7190 |
| TA | II | 500 | 0.8457 | 0.6234 |
| Baseline | II | 100 | 0.8387 | 0.6202 |
| Direct MKD+CRF | II | 100 | 0.8396 | 0.6137 |
| DAF full | II | 100 | 0.8391 | 0.6061 |
| **TA-hier** | **II** | **100** | **0.8429** | **0.6287** |
| Baseline | II | 50 | 0.8309 | 0.6098 |
| DAF full | II | 50 | 0.8299 | 0.6086 |

### Key observations

- **DAF** is disabled by default — frequency distillation did not improve over the
  BCE baseline in any configuration.
- **Direct KD** (teacher → student) gives marginal gain. The 12-lead 500Hz →
  1-lead 100/50Hz gap is too large for effective response or contrastive KD.
- **TA-hier** is the current best direction: decomposing the transfer into a
  *spatial* stage (12-lead → 1-lead TA at 500Hz) and a *temporal* stage
  (1-lead 500Hz TA → 1-lead 100/50Hz student) improves both AUC and F1.

### Running experiments

```bash
# 100Hz ablation (sequential, max VRAM per run)
DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_ta_hier_ablation_100hz.sh

# 50Hz ablation
DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_ta_hier_ablation_50hz.sh

# Multi-seed (5 seeds × 5 ablations × 2 Hz)
DATA_DIR=comper_repo/ptb_xl SEEDS="0 1 2 3 4" bash dafd_mvkt/scripts/run_multiseed_ta_hier.sh

# Aggregate results
python dafd_mvkt/experiments/aggregate_results.py \
    --results_dir dafd_mvkt/outputs \
    --out_dir dafd_mvkt/outputs
```

---

## Hierarchical Teacher Assistant Distillation

### Motivation

Direct 12-lead 500 Hz → 1-lead 100/50 Hz distillation involves two simultaneous gaps:

1. **Spatial gap** — 12 leads → 1 lead (loss of cross-lead correlations)
2. **Temporal-resolution gap** — 500 Hz → 100/50 Hz (loss of high-frequency features)

Results show this gap is too large for effective KD: MKD, CRF, and DAF all
give marginal or negative improvements over the BCE baseline.

### Solution: Teacher Assistant

Introduce a 1-lead 500 Hz **Teacher Assistant (TA)** to decompose the gap:

```
12-lead 500 Hz teacher
        │  (Step 1: close spatial gap)
        ▼
 1-lead 500 Hz TA
        │  (Step 2: close temporal-resolution gap)
        ▼
 1-lead 100/50 Hz student
```

### Step 1 — Train Teacher Assistant

```bash
python train_ta.py \
    --config configs/ta_ii_500hz.yaml \
    --data_dir /path/to/ptbxl \
    --teacher_ckpt outputs/teacher_best.pt \
    --lead II
```

Best checkpoint → `outputs/ta_ii_500hz_best.pt`

TA loss: `L = L_BCE + α·L_MKD(teacher→TA) + β·L_CRF(teacher↔TA)`
Defaults: `α=1.0`, `β=0.1`

### Step 2 — Train Hierarchical Student

```bash
# 100 Hz student — TA supervision only (recommended default)
python train_student_hier.py \
    --config configs/student_100hz_ta.yaml \
    --data_dir /path/to/ptbxl \
    --teacher_ckpt outputs/teacher_best.pt \
    --ta_ckpt outputs/ta_ii_500hz_best.pt \
    --lead II \
    --losses bce,ta_mkd,ta_crf,feature

# 50 Hz student
python train_student_hier.py \
    --config configs/student_50hz_ta.yaml \
    --data_dir /path/to/ptbxl \
    --teacher_ckpt outputs/teacher_best.pt \
    --ta_ckpt outputs/ta_ii_500hz_best.pt \
    --lead II \
    --losses bce,ta_mkd,ta_crf,feature
```

#### Supported `--losses` combinations

| Flag | Loss components |
|------|----------------|
| `bce` | BCE only |
| `bce,ta_mkd` | + TA soft-label KD |
| `bce,ta_mkd,ta_crf` | + TA contrastive |
| `bce,ta_mkd,ta_crf,feature` | + TA feature-map KD **(recommended)** |
| `bce,teacher_mkd,ta_mkd,ta_crf,feature` | + direct teacher KD |

#### Hierarchical student loss

```
L = L_BCE
    + alpha_teacher * L_MKD(teacher→student)   # direct teacher KD (optional)
    + alpha_ta      * L_MKD(TA→student)
    + beta_ta_crf   * L_CRF(TA↔student)
    + gamma_feature * L_FeatureKD(TA→student)
```
Defaults: `alpha_teacher=0.3`, `alpha_ta=1.0`, `beta_ta_crf=0.1`, `gamma_feature=0.2`

### Feature KD Loss

Aligns TA and student temporal feature maps:
1. Temporal alignment — AdaptiveAvgPool1d to K=128
2. Channel alignment  — 1×1 Conv1d if channel dims differ (identity if equal)
3. L2 normalisation along channel dim
4. MSE loss

### Evaluation

```bash
# Evaluate TA
python evaluate.py \
    --config configs/ta_ii_500hz.yaml \
    --data_dir /path/to/ptbxl \
    --ckpt outputs/ta_ii_500hz_best.pt \
    --split test --model_name ta_ii_500hz

# Evaluate hierarchical student
python evaluate.py \
    --config configs/student_100hz_ta.yaml \
    --data_dir /path/to/ptbxl \
    --ckpt outputs/student_ii_100hz_hier_best.pt \
    --split test --model_name student_ii_100hz_hier \
    --tune_threshold
```

### Expected Result Table

| Method | Lead | Hz | Macro AUC | Macro F1@0.5 | Macro F1 tuned |
|--------|------|----|-----------|--------------|----------------|
| Teacher | 12 | 500 | 0.9121 | — | 0.7190 |
| TA | II | 500 | — | — | — |
| Baseline | II | 100 | 0.8387 | — | 0.6202 |
| Direct MKD+CRF | II | 100 | 0.8396 | — | 0.6137 |
| TA-MKD | II | 100 | — | — | — |
| TA-MKD+CRF | II | 100 | — | — | — |
| **TA-MKD+CRF+Feature** | **II** | **100** | **—** | **—** | **—** |
| Teacher+TA Full | II | 100 | — | — | — |
| Baseline | II | 50 | 0.8309 | — | 0.6098 |
| **TA-MKD+CRF+Feature** | **II** | **50** | **—** | **—** | **—** |

---

## Project Structure

```
dafd_mvkt/
├── configs/              YAML hyperparameter configs
├── data/ptbxl_dataset.py PTB-XL loader with anti-aliased downsampling
├── models/
│   ├── resnet1d.py       Shared 1D ResNet-34 backbone
│   └── heads.py          Projection and channel-projection heads
├── losses/
│   ├── mkd.py                      Multi-label knowledge distillation
│   ├── contrastive.py              Cross-rate InfoNCE
│   ├── feature_kd.py               Feature-map MSE KD (TA→student)
│   └── frequency_distillation.py   DAF loss (learnable gate, optional)
├── utils/
│   ├── metrics.py        AUC, F1, precision, recall, specificity
│   ├── signal.py         Anti-aliased downsampling, z-score normalisation
│   ├── seed.py           Reproducibility
│   └── checkpoint.py     Save / load helpers
├── train_teacher.py
├── train_student.py        Direct teacher→student distillation
├── train_ta.py             1-lead 500 Hz Teacher Assistant training
├── train_student_hier.py   Hierarchical TA→student distillation
├── evaluate.py             Evaluation for teacher / TA / student
└── requirements.txt
```
