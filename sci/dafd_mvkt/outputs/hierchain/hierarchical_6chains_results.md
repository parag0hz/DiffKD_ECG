# Hierarchical 6-Chain KD Results

Results: 6/6 chains complete

**References:**
- BCE 50Hz:              AUC=0.8061  F1=0.574
- Progressive+SimCLR:    AUC=0.8396 F1=0.6176
- Parallel 3T C14:       AUC=0.8425 F1=0.6243
- MVKT target:           AUC=0.843  F1=0.626

| Rank | Chain | Path | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs C14 | >Prog | >C14 | >MVKT |
|------|-------|------|-----|----------|-------------|-------------|-------|------|-------|
| 1 | H02 | 12L100→1L500→1L100→S | 0.8442 | 0.6245 | +0.0381 | +0.0017 | YES | YES | YES |
| 2 | H01 | 12L500→1L500→1L100→S | 0.8427 | 0.6181 | +0.0366 | +0.0002 | YES | YES | no |
| 3 | H04 | 12L100→1L100→1L50→S | 0.8414 | 0.6241 | +0.0353 | -0.0011 | YES | no | no |
| 4 | H03 | 1L500→1L100→1L50→S | 0.8382 | 0.614 | +0.0321 | -0.0043 | no | no | no |
| 5 | H06 | 12L500→12L100→1L100→S | 0.838 | 0.6151 | +0.0319 | -0.0045 | no | no | no |
| 6 | H05 | 12L100→12L50→1L50→S | 0.8345 | 0.6127 | +0.0284 | -0.0080 | no | no | no |
