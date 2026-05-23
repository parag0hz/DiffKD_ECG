# Pipeline Status Report — 2026-05-20 06:50:32

## 1. MVKT Target
- Target: AUC **0.843**, F1_tuned **0.626**
- **Beaten: YES ✓**

## 2. Best 100Hz Result
| | Value | Gap to MVKT |
|---|---|---|
| Method | II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg | — |
| Lead | II | — |
| AUC | **0.8460** | -0.0030 |
| F1_tuned | **0.6303** | -0.0043 |

## 3. Lead Comparison (100Hz)
- **Lead I**: AUC=0.8349  F1=0.6072  [I_100hz_bce,ta_mkd,ta_crf,feature_clecg]
- **Lead II**: AUC=0.8460  F1=0.6303  [II_100hz_bce,ta_mkd,ta_crf,feature_clecgta_clecg]

## 4. Best 50Hz Result (trade-off)
- Method: II_50hz_bce,ta_mkd,ta_crf,feature_prog
- AUC: 0.8320 (+0.0259)
- F1_tuned: 0.6106 (+0.0366)
- Gap to MVKT AUC: 0.0110
- Gain vs BCE 50Hz AUC: 0.0259

## 5. Pipeline Status
- Status: `mvkt_beaten`
- Next recommended: `50hz_tradeoff`

## 6. Warnings
- ⚠ Lead I is weak (AUC=0.8349). Prioritize Lead II.
