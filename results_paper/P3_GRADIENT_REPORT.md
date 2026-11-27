# Phase 3 prediction-head gradient report

## Diagnostic setup

The explicit zero-correction-head option was tested on a deterministic
synthetic CPU batch (seed `20260928`, one CPU thread). This is an implementation
and gradient-flow check only; it is not a training result or accuracy claim.
The output-head weight and bias were initialized to zero. The initial target
was a fixed random tensor, and one SGD update was applied before measuring
upstream gradients again.

Machine-readable values are in
`results_paper/P3_diagnostics/gradient_diagnostics.json`.

| Measurement | Value |
|---|---:|
| Initial maximum absolute prediction difference from CV | 0.0 |
| Initial normalized MSE | 277.8792114 |
| Initial prediction-head gradient L2 | 63.0663759 |
| Prediction-head parameter tensors with gradients | 2 / 2 |
| Upstream gradient L2 before head update | 0.0 |
| Upstream tensors with gradients before update | 222 present, 16 absent |
| Post-update normalized MSE | 240.7705994 |
| Upstream gradient L2 after head update | 12.5955388 |
| Upstream tensors with gradients after update | 222 present, 16 absent |
| Post-update prediction and loss finite | yes |

The 16 absent gradients belong to predictor parameters not participating in
the exercised prediction path; they are distinct from present-but-zero
gradients. The aggregate upstream norm is zero on the initial backward pass
because a zero output projection transmits zero derivatives to its input.
After the output head changes, upstream gradients are nonzero. This is a
verified property of the optional initialization, not evidence that the
screening configurations require it.

## Screening initialization

The controlled P3 screening leaves the existing nonzero Xavier prediction
head initialization unchanged. P3-B/P3-C and P3-D/P3-E begin with identical
parameter states within each matched pair. The zero-head option is not applied
to screening or historical checkpoints.
