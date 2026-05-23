| Method | Lead | Hz | Losses | AUC mean±std | F1@0.5 mean±std | F1_tuned mean±std | n |
|---|---|---:|---|---:|---:|---:|---:|
| teacher_12lead_500hz | 12 | 500 | bce | 0.9121 | 0.7190 | 0.7190 | 1 |
| II_100hz_bce,teacher_mkd,ta_mkd,ta_crf,feature | II | 100 | bce,teacher_mkd,ta_mkd,ta_crf,feature | 0.8389 | 0.5599 | 0.6234 | 1 |
| II_50hz_bce,teacher_mkd,ta_mkd,ta_crf,feature | II | 50 | bce,teacher_mkd,ta_mkd,ta_crf,feature | 0.8300 | 0.5361 | 0.6137 | 1 |
| ta_ii_500hz | ii | 500 | bce,mkd,crf | 0.8457 | 0.6234 | 0.6234 | 1 |
| II_100hz_bce | II | 100 | bce | 0.8168 | 0.5392 | 0.5865 | 1 |
| II_100hz_bce,ta_mkd | II | 100 | bce,ta_mkd | 0.8379 | 0.5587 | 0.6181 | 1 |
| II_100hz_bce,ta_mkd,feature | II | 100 | bce,ta_mkd,feature | 0.8374 | 0.5550 | 0.6225 | 1 |
| II_100hz_bce,ta_mkd,ta_crf | II | 100 | bce,ta_mkd,ta_crf | 0.8392 | 0.5554 | 0.6154 | 1 |
| II_100hz_bce,ta_mkd,ta_crf,feature | II | 100 | bce,ta_mkd,ta_crf,feature | 0.8397 | 0.5534 | 0.6195 | 1 |
| II_50hz_bce | II | 50 | bce | 0.8061 | 0.5006 | 0.5740 | 1 |
| II_50hz_bce,ta_mkd | II | 50 | bce,ta_mkd | 0.8231 | 0.5329 | 0.6056 | 1 |
| II_50hz_bce,ta_mkd,feature | II | 50 | bce,ta_mkd,feature | 0.8269 | 0.5468 | 0.6067 | 1 |
| II_50hz_bce,ta_mkd,ta_crf | II | 50 | bce,ta_mkd,ta_crf | 0.8283 | 0.5475 | 0.6027 | 1 |
| II_50hz_bce,ta_mkd,ta_crf,feature | II | 50 | bce,ta_mkd,ta_crf,feature | 0.8300 | 0.5455 | 0.6060 | 1 |