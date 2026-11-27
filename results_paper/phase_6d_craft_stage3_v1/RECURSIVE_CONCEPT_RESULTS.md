# Recursive concept results

For each parent, exactly 1482 of 14820 training tokens (top decile by saved coefficient) were selected. Earlier activations were captured for 466 unique windows at encoder layer 1 post-ReLU FFN. Recursive NMF rank was fixed at 2; seeds were (20260928, 20260929).

| Parent | Seed | Fit relative error | Coefficient zero fraction | Iterations | Warning |
|---|---:|---:|---:|---:|---|
| C1 | 20260928 | 0.078481 | 0.0233 | 200 | True |
| C1 | 20260929 | 0.078481 | 0.0233 | 200 | True |
| C1 | stability | — | — | — | matched_mean_cosine=1.000000000;minimum=1.000000000;supported=False |
| C2 | 20260928 | 0.165391 | 0.0766 | 146 | False |
| C2 | 20260929 | 0.165391 | 0.0766 | 146 | False |
| C2 | stability | — | — | — | matched_mean_cosine=1.000000000;minimum=1.000000000;supported=True |
| C3 | 20260928 | 0.361628 | 0.1296 | 29 | False |
| C3 | 20260929 | 0.361628 | 0.1296 | 29 | False |
| C3 | stability | — | — | — | matched_mean_cosine=1.000000000;minimum=1.000000000;supported=True |
| C4 | 20260928 | 0.202348 | 0.0597 | 92 | False |
| C4 | 20260929 | 0.202348 | 0.0597 | 92 | False |
| C4 | stability | — | — | — | matched_mean_cosine=1.000000000;minimum=1.000000000;supported=True |

Numerical support flags (both fits converged without warning, error ≤0.50, matched cosine ≥0.90): `[False, True, True, True]`. No semantic labels were assigned.
