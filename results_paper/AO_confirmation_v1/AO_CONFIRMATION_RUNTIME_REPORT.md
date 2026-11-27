# Phase 6A AO confirmation runtime report

Training and memory measurements are CPU-only run observations. They are not GPU comparisons or performance claims beyond this machine.

| Model | Seeds | Mean training/fit seconds | Mean alpha-selection seconds | Mean wall seconds | Max sampled process bytes | Mean best-checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|
| AO-0 | 5 | 0.00 | n/a | n/a | 221495296 | 0 |
| AO-1 | 5 | 2226.90 | n/a | 2245.18 | 331931648 | 1624085 |
| AO-2 | 5 | 1191.79 | n/a | 1220.87 | 332161024 | 1624085 |
| AO-3 | 5 | 1056.24 | n/a | 1068.77 | 332378112 | 1624085 |
| AO-CP | 1 | 0.00 | n/a | n/a | n/a | 0 |
| AO-RIDGE | 1 | 0.24 | 4.49 | n/a | n/a | 0 |

## Final-checkpoint inference latency

| Model | Batch | CPU | Threads | Warm-ups | Iterations | Median ms | IQR ms | MACs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AO-2 | 1 | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 4 | 50 | 500 | 99.049 | 181.088 | 227520 |
| AO-2 | 8 | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 4 | 50 | 500 | 114.574 | 186.512 | 1820160 |
| AO-3 | 1 | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 4 | 50 | 500 | 49.395 | 39.370 | 227520 |
| AO-3 | 8 | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 4 | 50 | 500 | 102.235 | 117.540 | 1820160 |

