# Skip-Fork-Join Dual Distillation Analysis

Loaded 6/6 results.

## 1. Best Overall

- Best AUC: **SF06** (1L500→{1L100,1L50}→S+skip) AUC=0.8448  F1=0.6241
- Best F1:  **SF06** (1L500→{1L100,1L50}→S+skip) AUC=0.8448  F1=0.6241

## 2. Reference Comparison

| SF | Structure | AUC | F1_tuned | >Prog | >C14 | >MVKT AUC | >MVKT F1 |
|----|-----------|-----|----------|-------|------|-----------|----------|
| SF06 | 1L500→{1L100,1L50}→S+skip | 0.8448 | 0.6241 | ✅ | ✅ | ✅ | ❌ |
| SF04 | 12L500→{1L500,1L100}→S+skip | 0.8444 | 0.6222 | ✅ | ✅ | ✅ | ❌ |
| SF02 | 12L100→{1L500,1L100}→S+skip | 0.8433 | 0.6224 | ✅ | ✅ | ✅ | ❌ |
| SF03 | 12L500→{12L50,1L100}→S+skip | 0.8421 | 0.6232 | ✅ | ❌ | ❌ | ❌ |
| SF05 | 12L100→{12L50,1L50}→S+skip | 0.8366 | 0.6092 | ❌ | ❌ | ❌ | ❌ |
| SF01 | 12L100→{12L50,1L100}→S+skip | 0.8351 | 0.6161 | ❌ | ❌ | ❌ | ❌ |

## 3. SF02 vs Parallel C14 (same teacher set)

SF02 = 12L100 → {1L500, 1L100} → S + skip 12L100 → S
C14  = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)

| Method | AUC | F1_tuned | ΔAUC |
|--------|-----|----------|------|
| C14 (parallel) | 0.8425 | 0.6243 | ref |
| SF02 (skip-fork) | 0.8433 | 0.6224 | +0.0008 |

→ Skip-fork-join **improves** over parallel C14 with same teachers.

## 4. SF02 vs Ordinary Fork-Join F02

SF02 = 12L100 → {1L500, 1L100} → S  + skip 12L100 → S
F02  = 12L100 → {1L500, 1L100} → S  (no skip)

| Method | AUC | F1_tuned | ΔAUC |
|--------|-----|----------|------|
| F02  (fork, no skip) | 0.8447 | 0.6275 | ref |
| SF02 (skip-fork)     | 0.8433 | 0.6224 | -0.0013 |

→ Skip connection from t1 **does not help** F02: -0.0013 AUC.

## 5. SF01 vs Ordinary Fork-Join F01

SF01 = 12L100 → {12L50, 1L100} → S  + skip 12L100 → S
F01  = 12L100 → {12L50, 1L100} → S  (no skip)

| Method | AUC | F1_tuned | ΔAUC |
|--------|-----|----------|------|
| F01  (fork, no skip) | 0.8410 | 0.6255 | ref |
| SF01 (skip-fork)     | 0.8351 | 0.6161 | -0.0059 |

→ Skip connection from t1 **does not help** F01: -0.0059 AUC.

## 6. Skip from 12L100 (SF01, SF02, SF05)

- n=3: mean AUC=0.8383 (SF01, SF02, SF05)
  - SF01: AUC=0.8351  F1=0.6161  weights=[0.2, 0.4, 0.4]
  - SF02: AUC=0.8433  F1=0.6224  weights=[0.2, 0.4, 0.4]
  - SF05: AUC=0.8366  F1=0.6092  weights=[0.2, 0.4, 0.4]

## 7. Skip from 12L500 (SF03, SF04)

- n=2: mean AUC=0.8433 (SF03, SF04)
  - SF03: AUC=0.8421  F1=0.6232  weights=[0.1, 0.45, 0.45]
  - SF04: AUC=0.8444  F1=0.6222  weights=[0.1, 0.45, 0.45]

## 8. Branch with 1L50 vs without

- With 1L50 branch (n=2): mean AUC=0.8407 (SF05, SF06)
- Without 1L50 branch (n=4): mean AUC=0.8412 (SF01, SF02, SF03, SF04)
→ 1L50 branch hurts.

## 9. Recommendation

Best skip-fork by AUC: **SF06** = 1L500→{1L100,1L50}→S+skip
  AUC=0.8448  F1=0.6241
  weights (skip/b2/b3)=[0.2, 0.4, 0.4]

✅ **Strong success**: beats MVKT target AUC.

Next step: run scripts/tune_best_skip_fork_weights.sh with SF_ID=SF06
