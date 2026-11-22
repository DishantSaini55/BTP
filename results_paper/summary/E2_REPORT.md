# E2 SAGE gate ablation report

Post-hoc evaluations of the matching E1 checkpoints; no E2 run retrained any model.

| Model | Ablation | Runs | MSE mean ± SD | C-MSE mean | CF-MSE mean |
|---|---|---:|---:|---:|---:|
| PACE-Net (SAGE + MASC + SATIE) | always_open | 10 | 4063.461038 ± 2500.375722 | 3819.136927 | 7288.680537 |
| PACE-Net (SAGE + MASC + SATIE) | always_closed | 10 | 5597.617218 ± 3117.639909 | 5176.828464 | 7725.045204 |
| PACE-Net (SAGE + MASC + SATIE) | soft_sigmoid | 10 | 4160.576696 ± 2526.229767 | 3920.171872 | 7351.209369 |
| PACE-Net (SAGE + MASC + SATIE) | context_only | 10 | 4151.919357 ± 2443.368153 | 3891.631597 | 7328.329222 |
| PACE-Net without SATIE (SAGE + MASC) | always_open | 10 | 4131.879288 ± 2155.014497 | 3911.696138 | 7540.457202 |
| PACE-Net without SATIE (SAGE + MASC) | always_closed | 10 | 5008.712172 ± 2319.472110 | 4689.927887 | 7892.420707 |
| PACE-Net without SATIE (SAGE + MASC) | soft_sigmoid | 10 | 4196.839402 ± 2216.367997 | 3974.132855 | 7535.979882 |
| PACE-Net without SATIE (SAGE + MASC) | context_only | 10 | 4156.161686 ± 2145.285592 | 3932.730472 | 7509.512988 |

## H3 learned gate vs always-open

- **PACE-Net (SAGE + MASC + SATIE)**: mean difference 158.251718; 95% t CI [-84.770167, 401.273602]; d_z=0.465828; paired t p=0.174821; learned gate lower-error seeds=0/10.
- **PACE-Net without SATIE (SAGE + MASC)**: mean difference 68.901099; 95% t CI [-29.334676, 167.136874]; d_z=0.501741; paired t p=0.147055; learned gate lower-error seeds=0/10.
