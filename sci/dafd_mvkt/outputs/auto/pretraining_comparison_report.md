# Pretraining Comparison Report — 2026-05-20 11:16:14

Split: `test` | MVKT target: AUC 0.843, F1 0.626

## 1. SimCLR Source

| Item | Value |
|---|---|
| comper_repo encoder | ConvNeXt1d (12-lead, `downsample_layers.*`) |
| dafd_mvkt encoder | ResNet1d (1-lead, `stem.*`, `layer_blocks.*`) |
| Architecture compatible | **NO** — different backbone, weight transfer impossible |
| SimCLR implementation | `dafd_mvkt/pretrain_simclr.py` (native ResNet1d) |
| Checkpoint format | `{"state_dict": ..., "hz": ..., "lead": ...}` (CLECG-compatible) |

## 2. Lead II 100Hz — Pretraining Comparison

| Method | AUC | F1_tuned | ΔAUC vs random | ΔF1 vs random | Beats MVKT? |
|---|---:|---:|---:|---:|:---:|
| **MVKT-ECG (ref)** | **0.8430** | **0.6260** | — | — | — |
| Random init | 0.8397 | 0.6195 | +0.0000 | +0.0000 |  |
| CLECG student | 0.8412 | 0.6187 | +0.0015 | -0.0008 |  |
| CLECG TA+Student | 0.8418 | 0.6234 | +0.0021 | +0.0039 |  |
| SimCLR student | 0.8428 | 0.6204 | +0.0031 | +0.0009 |  |
| **SimCLR TA+Student** | **0.8480** | **0.6328** | +0.0083 | +0.0133 | ✓ |

## 3. Lead I 100Hz — Pretraining Comparison

| Method | AUC | F1_tuned | ΔAUC vs random | Beats MVKT? |
|---|---:|---:|---:|:---:|
| **MVKT-ECG (ref)** | **0.8430** | **0.6260** | — | — |
| Random init | 0.8292 | 0.6064 | +0.0000 |  |
| CLECG student | 0.8349 | 0.6072 | +0.0057 |  |
| CLECG TA+Student | 0.8297 | 0.6137 | +0.0005 |  |
| SimCLR student | 0.8342 | 0.6092 | +0.0050 |  |
| SimCLR TA+Student | 0.8384 | 0.6131 | +0.0093 |  |

## 4. Lead II 50Hz — Progressive HST-KD Comparison

| Method | AUC | F1_tuned | ΔAUC vs BCE 50Hz |
|---|---:|---:|---:|
| BCE 50Hz (baseline) | 0.8061 | 0.5740 | +0.0000 |
| Progressive HST-KD | 0.8320 | 0.6106 | +0.0259 |
| Progressive HST-KD + CLECG | — | — | — |
| Progressive HST-KD + SimCLR | 0.8396 | 0.6176 | +0.0335 |

## 5. Recommendation

- **Best Lead II 100Hz method**: `II_100hz_bce,ta_mkd,ta_crf,feature_simclrta_simclr`
  AUC 0.8480  F1 0.6328  (beats MVKT ✓)
- **Recommendation**: SimCLR full pipeline (TA+Student init) is best. Recommend replacing CLECG.

## 6. Warnings

- ⚠ comper_repo SimCLR uses ConvNeXt (not ResNet1d). dafd_mvkt SimCLR is independently pretrained with ResNet1d backbone. Architectures are NOT interchangeable.
- ⚠ SimCLR improves over CLECG by +0.0062 AUC on Lead II 100Hz.
