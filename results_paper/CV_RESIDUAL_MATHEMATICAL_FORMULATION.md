# Constant-velocity residual formulation

## Historical baseline

Let `b_t = [x1, y1, x2, y2]` be the final observed bounding box at observed
frame `t` (the 15th observed frame). The B1 baseline estimates coordinatewise
velocity over the last two source-frame intervals:

```text
v_t = (b_t - b_(t-2)) / 2
```

For future horizon `h = 1, ..., 45`, the forecast is:

```text
CV_(t+h) = b_t + h * v_t
```

This is the exact historical `constant_velocity_last_3_frames` estimator. It
uses only the 15 observed boxes; no future box or label enters the forecast.
The four coordinates are extrapolated independently, without clipping.

## Learned residual prediction

The original model path is:

```text
Y_hat = Y_constant_position + Generator(decoder_output)
```

The Phase 3 version changes only the residual anchor:

```text
Y_hat = Y_constant_velocity + Generator(decoder_output)
```

`Generator(decoder_output)` is the learned normalized-coordinate correction.
`ResidualAnchorModel.forward_batch()` computes one selected anchor, passes it
to the existing MTN/SAGE forward path, and adds it exactly once at the original
output site (`generator(decoded) + residual`). The MTN decoder target/query
construction remains the original zero tensor with shape `[B, 45, 4]`; cached
flow inputs, scene inputs, MASC, SATIE, optimizer, loss, and training target are
unchanged.

P3-B/P3-C share a byte-identical initial state for their MTN parameters. P3-D/
P3-E share a byte-identical initial state for their SAGE + MASC + SATIE
parameters. Both matched pairs use the same nonzero historical prediction-head
initialization in screening, preventing head initialization from being a
confound. The optional `--zero-correction-head` mode is implemented separately
and is not enabled for screening.

## Zero-correction-head option

When explicitly requested, the existing `Linear(32, 4)` output head's weight
and bias are zeroed. Then the initial learned correction is zero and
`Y_hat == Y_constant_velocity` exactly. This choice changes gradient flow at
initialization: the head receives loss gradients, but the zero weight
mathematically blocks the first backward pass into upstream decoder/encoder
parameters. After one optimizer step changes the head, upstream gradients can
flow. Measured values are recorded in
`results_paper/P3_diagnostics/gradient_diagnostics.json` and summarized in
`P3_GRADIENT_REPORT.md`.

The zero-head option is never applied to a historical model or checkpoint.
All P3 artifacts are written beneath a new, versioned P3 results directory.
