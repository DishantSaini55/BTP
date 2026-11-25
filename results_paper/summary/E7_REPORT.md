# E7 timing and encoder attention MACs

CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz  
Platform: Windows-11-10.0.26200-SP0  
Fixed threads: 4; warm-ups: 50; timed iterations: 500.

| Model | Batch | Median ms | IQR ms | Encoder self-attention MACs (analytic) | Parameters | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|
| MTN (baseline) | 1 | 50.685050 | 5.720850 | 227520 | 133700 | 1909507 |
| MTN (baseline) | 8 | 83.909500 | 6.048600 | 1820160 | 133700 | 1909507 |
| PACE-Net without SATIE (SAGE + MASC) | 1 | 53.849100 | 6.481625 | 227520 | 139142 | 1939667 |
| PACE-Net without SATIE (SAGE + MASC) | 8 | 88.005150 | 7.241475 | 1820160 | 139142 | 1939667 |
| PACE-Net (SAGE + MASC + SATIE) | 1 | 57.561550 | 5.432150 | 227520 | 130694 | 1902595 |
| PACE-Net (SAGE + MASC + SATIE) | 8 | 95.038950 | 12.719875 | 1820160 | 130694 | 1902595 |
| PACE-Net with ROUTE | 1 | 56.744000 | 5.748075 | 227520 | 154007 | 2006119 |
| PACE-Net with ROUTE | 8 | 88.499000 | 7.765900 | 1820160 | 154007 | 2006119 |
| PACE-Net with ROUTE and SATIE | 1 | 53.153500 | 5.586750 | 227520 | 145559 | 1969111 |
| PACE-Net with ROUTE and SATIE | 8 | 88.128750 | 6.396700 | 1820160 | 145559 | 1969111 |

## MAC audit

The former whole-model THOP MAC values are withdrawn. THOP's recursive hook registration visits a shared encoder-attention module through each of its three layer references, registering each Q/K/V/O linear hook three times. SATIE still executes that shared module three times, so those projections were counted nine times rather than three. This creates an artificial per-sample excess of
`8 × 3 × 4 × 15 × 32 × 32 = 1,474,560` MACs, exactly the observed difference between SATIE and its unshared counterpart. It is profiler hook duplication, not additional encoder computation. The raw E7 record retains the former THOP outputs under `*_unverified` names for audit; reported parameter counts come directly from the models.

The replacement column reports only analytically verified encoder self-attention MACs, not whole-model MACs. For sequence length `L=15`, model width `d=32`, each layer performs `4Ld² + 2L²d = 75,840` MACs: four Q/K/V/O projections and the QKᵀ and attention-value products. Bias additions, softmax, and elementwise operations are excluded. Three encoder layers therefore perform 227,520 MACs per sample, or 1,820,160 for batch 8. SATIE ties parameters but does not reduce the three layer executions; these counts are consequently identical with and without sharing. The calculation was checked against runtime hooks confirming three encoder self-attention calls with `[batch, 15, 32]` Q/K/V inputs in both configurations.
