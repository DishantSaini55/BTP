# Phase 6B Official-Test Runtime Report

## Official-test inference

- Eligible windows: 2,615
- Eligible videos: 13
- Track clusters: 861
- Models/records: 15 neural seed evaluations plus AO-0, AO-CP and AO-RIDGE.
- Official-test wall-clock execution time was not instrumented in this evaluation invocation and is therefore not reported.

## Phase 6A matched CPU latency

| Model | Batch | Median ms | IQR ms | Peak sampled process bytes |
|---|---:|---:|---:|---:|
| AO-2 | 1 | 99.049 | 181.088 | 240713728 |
| AO-2 | 8 | 114.574 | 186.512 | 264704000 |
| AO-3 | 1 | 49.395 | 39.370 | 272629760 |
| AO-3 | 8 | 102.235 | 117.540 | 297631744 |

Measured on Intel64 Family 6 Model 142 Stepping 12, four threads, 50 warm-ups and 500 timed iterations.

## Checkpoint sizes

- AO-2/AO-3 seed-20260928 best checkpoints: 1,624,085 bytes each.
- Neural latest checkpoints remain approximately 3.89–3.97 MB.
- Ridge coefficient archive: `classical/AO-RIDGE/seed_20260928/ridge_coefficients.npz`.
