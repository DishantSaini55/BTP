# Constant-velocity baseline verification

## Source implementation

The historical B1 implementation is in `MTN/run_b1_from_scratch.py`, in
`evaluate()` under `"Constant velocity (last 3 frames)"`. Its exact calculation is:

```python
velocity = (boxes[:, -1] - boxes[:, -3]).div(3 - 1)
steps = torch.arange(1, 46, dtype=velocity.dtype).view(1, 45, 1)
prediction = boxes[:, -1:, :] + steps * velocity.unsqueeze(1)
```

The Phase 3 utility in `MTN/cv_residual_v1.py` reproduces this formula, including
the 45 future steps. The deterministic P3-A baseline directly applies that
utility to the 15 normalized observed boxes; no neural model is trained.

## Inputs and units

- The input sequence has 15 boxes; `boxes[:, -3]` is observed index 12 (the
  13th observed source frame using one-based numbering), and `boxes[:, -1]` is
  observed index 14 (the 15th observed source frame).
- The difference therefore spans two consecutive source-frame intervals.
  Dividing by two gives a per-source-frame displacement. The dataset split
  materializer verifies that all 60 frames in an eligible window are consecutive.
- Forecast step 1 is the source frame immediately after observed frame 15;
  step 45 is 45 frame intervals after it. The baseline uses no seconds-based
  conversion or separate frame-rate parameter.
- Coordinates are `[x1, y1, x2, y2]` in the annotation's pixel coordinate
  system. Each coordinate is extrapolated independently, so changing box size
  is extrapolated too.
- The formula contains no image-boundary clipping or other validity constraint.

## Normalization equivalence

The B1 dataset wrapper normalizes every box coordinate with the same
train-derived affine transform used by E1:
`z_j = (b_j - mean_j) / std_j`. The Phase 3 formula operates on these
normalized observed boxes to remain compatible with the existing model inputs
and targets. Because velocity is a difference, the mean cancels and the
coordinate standard deviation divides the pixel-space velocity. In exact
arithmetic, forecasting normalized boxes and then denormalizing is equivalent
to forecasting in pixels.

The deterministic CPU test measured a maximum absolute discrepancy of
`1.52587890625e-05` normalized units for a float32 affine round trip with
pixel-scale means and standard deviations; the test tolerance is `2e-05`.
The historical formula and Phase 3 utility agreed exactly (`0.0` maximum
difference) on the deterministic float32 comparison input.

## Data-use boundary and split metrics

The Phase 3 runner calls the existing validation-only split loader and creates
only `train` and `validation` datasets. It does not construct a test dataset,
select a checkpoint from test data, or evaluate test predictions. The cache is
a single pickle containing role-keyed data; the validation-only loader
deserializes that container and exposes only its train and validation roles to
the runner. No test-role samples are passed to model or metric code.

P3-A's full-split train and validation measurements are saved in:

- `results_paper/P3_initial_screen_v1/p3_a/20260928/train_metrics.json`
- `results_paper/P3_initial_screen_v1/p3_a/20260928/validation_metrics.json`

See `P3_VALIDATION_SUMMARY.csv` for the validation row and
`P3_PER_WINDOW_COMPARISON.csv` for validation-window errors. Capped smoke
metrics in `P3_smoke_v1` are infrastructure checks only and are not full-split
evidence.
