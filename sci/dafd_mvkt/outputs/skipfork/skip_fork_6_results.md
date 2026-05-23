# Skip-Fork-Join Dual Distillation Results

Results: 6/6 skip-forks complete

**References:**
- BCE 50Hz:           AUC=0.8061  F1=0.574
- Progressive+SimCLR: AUC=0.8396 F1=0.6176
- Parallel C14:       AUC=0.8425  F1=0.6243
- MVKT target:        AUC=0.843 F1=0.626

| Rank | SF | Structure | Weights | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |
|------|----|-----------|---------|----|----------|-------------|-------------|-------|------|-------|
| 1 | SF06 | 1L500→{1L100,1L50}→S+skip | 0.2/0.4/0.4 | 0.8448 | 0.6241 | +0.0387 | +0.0023 | YES | YES | YES |
| 2 | SF04 | 12L500→{1L500,1L100}→S+skip | 0.1/0.45/0.45 | 0.8444 | 0.6222 | +0.0383 | +0.0019 | YES | YES | YES |
| 3 | SF02 | 12L100→{1L500,1L100}→S+skip | 0.2/0.4/0.4 | 0.8433 | 0.6224 | +0.0372 | +0.0008 | YES | YES | YES |
| 4 | SF03 | 12L500→{12L50,1L100}→S+skip | 0.1/0.45/0.45 | 0.8421 | 0.6232 | +0.0360 | -0.0004 | YES | no | no |
| 5 | SF05 | 12L100→{12L50,1L50}→S+skip | 0.2/0.4/0.4 | 0.8366 | 0.6092 | +0.0305 | -0.0059 | no | no | no |
| 6 | SF01 | 12L100→{12L50,1L100}→S+skip | 0.2/0.4/0.4 | 0.8351 | 0.6161 | +0.0290 | -0.0074 | no | no | no |
