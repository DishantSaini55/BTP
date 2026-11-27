# E1 from-scratch summary

All values below are generated from the 50 run directories. Confirmatory inference is seed-level for H1 and H2 only, with Holm correction over those two tests. The remaining eight pairs are exploratory and report unadjusted p values. Track-cluster and moving-block bootstraps are secondary because all test tracks come from one video.

| Model | MSE mean ± sample SD | C-MSE mean ± sample SD | CF-MSE mean ± sample SD | Epochs mean ± sample SD |
|---|---:|---:|---:|---:|
| MTN (baseline) | 3265.529449 ± 1159.103851 | 3033.020175 ± 1150.258903 | 6043.768731 ± 2520.717969 | 40.10 ± 8.12 |
| PACE-Net without SATIE (SAGE + MASC) | 4200.780387 ± 2124.625058 | 3969.733803 ± 2068.106094 | 7560.103792 ± 3110.172872 | 35.20 ± 11.02 |
| PACE-Net (SAGE + MASC + SATIE) | 4221.712756 ± 2406.945424 | 3954.003750 ± 2384.442833 | 7330.911971 ± 3897.228204 | 34.30 ± 11.87 |
| PACE-Net with ROUTE | 5249.691266 ± 3392.133608 | 4951.514335 ± 3226.021836 | 8328.732830 ± 4278.945772 | 31.80 ± 9.11 |
| PACE-Net with ROUTE and SATIE | 4191.840883 ± 3167.570534 | 3921.489050 ± 2964.277347 | 7147.624957 ± 4536.314936 | 36.10 ± 11.02 |
