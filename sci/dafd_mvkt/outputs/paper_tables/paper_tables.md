## Table 1: Lead I 100Hz — Comparison with MVKT-ECG

| Method | AUC | F1_tuned | ΔAUC | ΔF1 | n |
|---|---:|---:|---:|---:|---:|
| **MVKT-ECG (reported)** | **0.8430** | **0.6260** | — | — | 1 |
| BCE only | 0.8138 | 0.5825 | -0.0292 | -0.0435 | 1 |
| BCE + MKD | 0.8282 | 0.5977 | -0.0148 | -0.0283 | 1 |
| HST-KD | 0.8292 | 0.6064 | -0.0138 | -0.0196 | 1 |
| HST-KD + CLECG | 0.8349 | 0.6072 | -0.0081 | -0.0188 | 1 |
| HST-KD + CLECG-TA + CLECG | 0.8297 | 0.6137 | -0.0133 | -0.0123 | 1 |

## Table 2: Sampling Rate Trade-off (Lead II)

| Model | Hz | AUC | F1_tuned | ΔAUC vs 100Hz | n |
|---|---:|---:|---:|---:|---:|
| BCE 100Hz | 100 | 0.8168 | 0.5865 | — | 1 |
| HST-KD 100Hz | 100 | 0.8397 | 0.6195 | — | 1 |
| HST-KD+CLECG 100Hz | 100 | 0.8418±0.0032 | 0.6234±0.0052 | — | 4 |
| BCE 50Hz | 50 | 0.8061 | 0.5740 | -0.0357 | 1 |
| HST-KD 50Hz | 50 | 0.8300 | 0.6060 | -0.0118 | 1 |
| HST-KD+CLECG 50Hz | 50 | 0.8318 | 0.6052 | -0.0100 | 1 |

## Table 3: Ablation — Loss Components (Lead II 100Hz)

| Losses / Method | AUC | F1@0.5 | F1_tuned | ΔAUC | n |
|---|---:|---:|---:|---:|---:|
| BCE only | 0.8168 | 0.5392 | 0.5865 | +0.0000 | 1 |
| + ta_mkd | 0.8379 | 0.5587 | 0.6181 | +0.0211 | 1 |
| + ta_crf + feature (HST-KD) | 0.8397 | 0.5534 | 0.6195 | +0.0229 | 1 |
| HST-KD + CLECG student | 0.8412 | 0.5545 | 0.6187 | +0.0244 | 1 |
| HST-KD + CLECG-TA + CLECG (full) | 0.8418±0.0032 | 0.5742±0.0097 | 0.6234±0.0052 | +0.0250 | 4 |

## Table 4: Class-wise AUC — Best Method vs MVKT-ECG

| Method | NORM | MI | STTC | CD | HYP | Macro |
|---|---:|---:|---:|---:|---:|---:|
| MVKT-ECG | — | — | — | — | — | 0.8430 |
| **Best Lead I** (I_100hz_bce,ta_mkd,ta_crf,feature_clecg) | 0.8877 | 0.7872 | 0.8775 | 0.8179 | 0.8041 | **0.8349** |
| **Best Lead II** (II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg) | 0.8972 | 0.8210 | 0.8740 | 0.8588 | 0.7582 | **0.8418±0.0032** |