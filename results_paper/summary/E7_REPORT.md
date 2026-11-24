# E7 timing and FLOPs

CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz  
Platform: Windows-11-10.0.26200-SP0  
Fixed threads: 4; warm-ups: 50; timed iterations: 500.

| Model | Batch | Median ms | IQR ms | MACs | Parameters | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|
| MTN (baseline) | 1 | 50.685050 | 5.720850 | 3329504 | 133700 | 1909507 |
| MTN (baseline) | 8 | 83.909500 | 6.048600 | 26636032 | 133700 | 1909507 |
| PACE-Net without SATIE (SAGE + MASC) | 1 | 53.849100 | 6.481625 | 3334592 | 139142 | 1939667 |
| PACE-Net without SATIE (SAGE + MASC) | 8 | 88.005150 | 7.241475 | 26676736 | 139142 | 1939667 |
| PACE-Net (SAGE + MASC + SATIE) | 1 | 57.561550 | 5.432150 | 4809152 | 130694 | 1902595 |
| PACE-Net (SAGE + MASC + SATIE) | 8 | 95.038950 | 12.719875 | 38473216 | 130694 | 1902595 |
| PACE-Net with ROUTE | 1 | 56.744000 | 5.748075 | 3510848 | 154007 | 2006119 |
| PACE-Net with ROUTE | 8 | 88.499000 | 7.765900 | 28086784 | 154007 | 2006119 |
| PACE-Net with ROUTE and SATIE | 1 | 53.153500 | 5.586750 | 4985408 | 145559 | 1969111 |
| PACE-Net with ROUTE and SATIE | 8 | 88.128750 | 6.396700 | 39883264 | 145559 | 1969111 |
