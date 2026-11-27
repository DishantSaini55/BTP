# SAGE V2 gradient audit

## What this audit does and does not establish

Phase 1 measured the historical failure pathway: all-closed batches bypassed the context branch, leaving gate and context-encoder gradients absent; a rejected sample in a mixed batch contributed zero to those tested pathways. The measured all-open Phase 1 gradient L2 norms were 0.009474 (gate) and 0.863635 (context encoder). This was evidence of a pathway, not proof that it was the sole cause of gate collapse.

V2 retains a context token in the decoder memory and makes its admission affect the context attention logit through `log(p + 1e-8)`. This creates a continuous loss pathway even at low but nonzero probabilities; exact hard rejection at inference still uses the motion-only path.

## Deterministic measured results

Measurements below are from CPU, one thread, seed 20260928, fixed synthetic inputs, normalized-coordinate MSE. The complete JSON records finite checks and absent-versus-zero gradient counts.

| Case | Gate implementation | Admission probability | Gate gradient L2 | Context-encoder gradient L2 | Gate params absent / zero | Context params absent / zero |
|---|---|---:|---:|---:|---:|---:|
| Low admission | Actual learned gate MLP | 0.00245011 | 0.00000755293 | 0.00107058 | 0 / 0 of 4 | 0 / 0 of 12 |
| Mixed | Fixed-logit pathway probe: 0.0179862 and 0.982014 | both values | 0.000735635 | 0.120916 | 0 / 0 of 1 probe parameter | 0 / 0 of 12 |
| Saturated boundary | Fixed-logit probe: `3.78e-44` and `1.0` | near 0 and 1 | 0.0 | 0.129963 | 0 / 1 of 1 probe parameter | 0 / 0 of 12 |

All predictions and reported gradients were finite. The near-zero/one probe demonstrates that finite arithmetic does not eliminate softmax saturation: the gate gradient was present but exactly zero at that extreme. It is not representative evidence of useful learning there. At a low but nonzero probability of 0.00245, the actual gate MLP had nonzero gradients in all four parameter tensors and the context encoder had nonzero gradients in all twelve.

The unit tests also compare the attention-prior derivative to a centered finite difference, require exact zero attention on masked positions, verify exact hard-closed motion fallback, and check bitwise equality between P2-B and P2-C forced-open predictions from their shared initialization.

## Interpretation limits

- Synthetic gradient probes prove the tested implementation pathway, not optimization stability across the full dataset.
- A nonzero gradient does not imply a useful update or improved validation accuracy.
- The tests do not establish that the gradient bypass was the sole cause of the historical gate behavior.
- The capped smoke losses are infrastructure checks only and are not full-split performance evidence.
- The actual full-split one-seed screen is exploratory and has no multi-seed statistical inference.

Reproduction:

```powershell
python -m unittest tests.test_sage_v2 -v
python MTN\sage_v2_gradient_diagnostics.py
```
