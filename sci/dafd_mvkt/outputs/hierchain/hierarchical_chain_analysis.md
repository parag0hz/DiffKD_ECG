# Hierarchical Chain KD Analysis

Loaded 6/6 results.

## 1. Best Overall

- Best AUC:    **H02** (12L100→1L500→1L100→S) AUC=0.8442  F1=0.6245
- Best F1:     **H02** (12L100→1L500→1L100→S) AUC=0.8442  F1=0.6245

## 2. Reference Comparison

| Chain | AUC | F1_tuned | >Prog+SimCLR | >Parallel C14 | >MVKT AUC | >MVKT F1 |
|-------|-----|----------|-------------|---------------|-----------|----------|
| H02 (12L100→1L500→1L100→S) | 0.8442 | 0.6245 | ✅ | ✅ | ✅ | ❌ |
| H01 (12L500→1L500→1L100→S) | 0.8427 | 0.6181 | ✅ | ✅ | ❌ | ❌ |
| H04 (12L100→1L100→1L50→S) | 0.8414 | 0.6241 | ✅ | ❌ | ❌ | ❌ |
| H03 (1L500→1L100→1L50→S) | 0.8382 | 0.6140 | ❌ | ❌ | ❌ | ❌ |
| H06 (12L500→12L100→1L100→S) | 0.8380 | 0.6151 | ❌ | ❌ | ❌ | ❌ |
| H05 (12L100→12L50→1L50→S) | 0.8345 | 0.6127 | ❌ | ❌ | ❌ | ❌ |

## 3. H02 vs Parallel C14 (same teacher set)

H02 = 12L100 → 1L500 → 1L100 → S  (hierarchical)
C14 = 12L100 + 1L500 + 1L100 → S  (parallel, AUC=0.8425 F1=0.6243)

| | AUC | F1_tuned | ΔAUC |
|--|-----|----------|------|
| C14 (parallel) | 0.8425 | 0.6243 | ref |
| H02 (hier) | 0.8442 | 0.6245 | +0.0017 |

→ Hierarchy **improves** over parallel KD.

## 4. Chains Starting with 12L500

- With 12L500 t1 (n=2): mean AUC=0.8403
- Without 12L500 t1 (n=4): mean AUC=0.8396
→ 12L500 as t1 helps.

## 5. t3 View Before Student: 1L100 vs 1L50

- t3=1L100 chains (n=3): mean AUC=0.8416 (H01, H02, H06)
- t3=1L50 chains  (n=3):  mean AUC=0.8380 (H03, H04, H05)
→ 1L100 as t3 is better.

## 6. Recommendation

Best chain by AUC: **H02** = 12L100→1L500→1L100→S
  AUC=0.8442  F1=0.6245

✅ **Strong success**: beats MVKT target AUC.

Next step: run scripts/run_best_hierarchical_chain_full.sh with chain_id=H02
