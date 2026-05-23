# Fork-Join Dual Distillation Results

Results: 6/6 forks complete

**References:**
- BCE 50Hz:           AUC=0.8061  F1=0.574
- Progressive+SimCLR: AUC=0.8396 F1=0.6176
- Parallel C14:       AUC=0.8425 F1=0.6243
- MVKT target:        AUC=0.843  F1=0.626

| Rank | Fork | Structure | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |
|------|------|-----------|-----|----------|-------------|-------------|-------|------|-------|
| 1 | F02 | 12L100→{1L500,1L100}→S | 0.8447 | 0.6275 | +0.0386 | +0.0022 | YES | YES | YES |
| 2 | F04 | 12L500→{1L500,1L100}→S | 0.8432 | 0.6159 | +0.0371 | +0.0007 | YES | YES | YES |
| 3 | F06 | 1L500→{1L100,1L50}→S | 0.8414 | 0.6175 | +0.0353 | -0.0011 | YES | no | no |
| 4 | F01 | 12L100→{12L50,1L100}→S | 0.841 | 0.6255 | +0.0349 | -0.0015 | YES | no | no |
| 5 | F03 | 12L500→{12L50,1L100}→S | 0.8374 | 0.6138 | +0.0313 | -0.0051 | no | no | no |
| 6 | F05 | 12L100→{12L50,1L50}→S | 0.8355 | 0.6154 | +0.0293 | -0.0071 | no | no | no |
