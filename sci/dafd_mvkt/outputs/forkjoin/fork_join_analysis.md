# Fork-Join Dual Distillation Analysis

Loaded 6/6 results.

## 1. Best Overall

- Best AUC: **F02** (12L100→{1L500,1L100}→S) AUC=0.8447  F1=0.6275
- Best F1:  **F02** (12L100→{1L500,1L100}→S) AUC=0.8447  F1=0.6275

## 2. Reference Comparison

| Fork | Structure | AUC | F1_tuned | >Prog | >C14 | >MVKT AUC | >MVKT F1 |
|------|-----------|-----|----------|-------|------|-----------|----------|
| F02 | 12L100→{1L500,1L100}→S | 0.8447 | 0.6275 | ✅ | ✅ | ✅ | ✅ |
| F04 | 12L500→{1L500,1L100}→S | 0.8432 | 0.6159 | ✅ | ✅ | ✅ | ❌ |
| F06 | 1L500→{1L100,1L50}→S | 0.8414 | 0.6175 | ✅ | ❌ | ❌ | ❌ |
| F01 | 12L100→{12L50,1L100}→S | 0.8410 | 0.6255 | ✅ | ❌ | ❌ | ❌ |
| F03 | 12L500→{12L50,1L100}→S | 0.8374 | 0.6138 | ❌ | ❌ | ❌ | ❌ |
| F05 | 12L100→{12L50,1L50}→S | 0.8355 | 0.6154 | ❌ | ❌ | ❌ | ❌ |

## 3. F02 vs Parallel C14 (same teacher set)

F02 = 12L100 → {1L500, 1L100} → S  (fork-join)
C14 = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)

| Method | AUC | F1_tuned | ΔAUC |
|--------|-----|----------|------|
| C14 (parallel) | 0.8425 | 0.6243 | ref |
| F02 (fork-join) | 0.8447 | 0.6275 | +0.0022 |

→ Fork-join **improves** over parallel with same teachers.

## 4. F01: Spatial (12L) + Rate (1L) Factorization

F01 = 12L100 → {12L50, 1L100} → S
Spatial branch: 12L50 (same leads, lower rate)
Rate branch:    1L100 (fewer leads, same rate)

F01 AUC=0.8410  F1=0.6255  ΔvsC14=-0.0015

## 5. Upstream t1: 12L500 vs 12L100

- t1=12L500 (n=2): mean AUC=0.8403 (F03, F04)
- t1=12L100 (n=3): mean AUC=0.8404 (F01, F02, F05)
→ 12L100 is sufficient.

## 6. Branch with 1L50 vs without

- With 1L50 branch (n=2): mean AUC=0.8384 (F05, F06)
- Without 1L50 branch (n=4): mean AUC=0.8416 (F01, F02, F03, F04)
→ 1L50 branch hurts.

## 7. Recommendation

Best fork by AUC: **F02** = 12L100→{1L500,1L100}→S
  AUC=0.8447  F1=0.6275

✅ **Strong success**: beats MVKT target AUC.

Next step: run scripts/run_best_fork_join_full.sh with fork_id=F02
