# Phase 3: Constant-Velocity Residual Anchor Screen

## Status and scope

The bounded Phase 3 screen is complete. It used seed `20260928`, the full
training and validation splits, batch size 8, Adam at `1e-3`, MSE, four CPU
threads, and at most 10 epochs per trainable configuration. All six trainable
runs completed 10 epochs; validation-selected checkpoints and validation-only
metrics are recorded under `P3_initial_screen_v1/`. P3-A is a deterministic
constant-velocity baseline and required no training. No test predictions or
test metrics were generated. No files under `paper/` were edited, and no
commit or push was made.

This is a one-seed exploratory screen, not confirmatory evidence. The 43
validation windows all come from `gp_set_0006_vid_0001`, so windows are
correlated and the validation results have a single-video limitation. No
confidence intervals or significance claims are warranted.

## Protocol and configurations

The historical B1 baseline extrapolates each box coordinate from the last
three observed boxes, exactly as implemented in
`MTN/run_b1_from_scratch.py`. The new wrapper changes only the learned
prediction anchor: either the final observed box (constant position, CP) or
the historical constant-velocity forecast (CV). The residual head adds its
learned correction exactly once. Historical model implementations and E1/E2/
E3/E7 outputs were not changed.

| ID | Configuration | Anchor | Selected validation inference |
|---|---|---|---|
| P3-A | Historical constant-velocity baseline | CV | Exact B1 formula |
| P3-B | MTN | CP | Base |
| P3-C | MTN | CV | Base |
| P3-D | Original SAGE + MASC + SATIE | CP | Hard gate |
| P3-E | Original SAGE + MASC + SATIE | CV | Hard gate |
| P3-F | Always-open context + MASC + SATIE | CV | Always open |
| P3-G | Differentiable SAGE V2, context-only gate + MASC + SATIE | CV | Soft gate for checkpoint selection; soft and hard evaluated |

P3-B/C and P3-D/E used identical initial states within their matched pairs.
Checkpoint selection used normalized validation MSE only. P3-G's checkpoint
was selected with soft admission; its hard result is a post-hoc inference
comparison, not a separately selected checkpoint. Forced-open and
forced-closed evaluations are diagnostic counterfactuals.

## Validation results

All metric values below are from the selected checkpoint and the same 43
validation windows. Raw MSE, C-MSE, and CF-MSE are in pixel coordinates;
normalized MSE is the training-statistics-normalized box MSE. Lower is better.
The normalized and raw rankings differ substantially for P3-D, so both are
reported rather than choosing whichever favors a model.

| ID / policy | Train normalized MSE | Validation normalized MSE | Validation raw MSE | C-MSE | CF-MSE | Epoch | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3-A exact CV | 0.3426 | 1.2188 | 3,630.32 | 3,074.49 | 15,769.10 | n/a | 0 |
| P3-B base | 0.1338 | 1.0061 | 7,106.32 | 6,554.38 | 20,871.90 | 9 | 133,700 |
| P3-C base | 0.2570 | 1.0097 | 3,341.44 | 2,596.37 | 11,865.54 | 8 | 133,700 |
| P3-D hard | 0.1128 | 0.9993 | 8,010.10 | 7,344.73 | 17,571.08 | 10 | 130,694 |
| P3-E hard | 0.2512 | 1.0118 | 3,275.40 | 2,414.86 | 11,814.23 | 8 | 130,694 |
| P3-F always open | 0.2491 | 1.0357 | 3,379.50 | 2,585.67 | 11,596.67 | 8 | 130,068 |
| P3-G soft | 0.2562 | 0.9962 | 3,106.42 | 2,421.37 | 11,851.80 | 8 | 130,630 |
| P3-G hard | 0.2562 | 0.9927 | 3,082.83 | 2,394.19 | 11,889.57 | 8 | 130,630 |

Per-window comparisons against P3-A CV:

| Configuration / selected policy | Lower raw MSE than CV | Higher raw MSE than CV | Mean raw MSE change vs CV |
|---|---:|---:|---:|
| P3-B | 5 / 43 | 38 / 43 | +95.75% |
| P3-C | 21 / 43 | 22 / 43 | -7.96% |
| P3-D hard | 4 / 43 | 39 / 43 | +120.65% |
| P3-E hard | 27 / 43 | 16 / 43 | -9.78% |
| P3-F always open | 25 / 43 | 18 / 43 | -6.91% |
| P3-G soft | 26 / 43 | 17 / 43 | -14.43% |
| P3-G hard | 26 / 43 | 17 / 43 | -15.08% |

The CV residual anchor substantially lowered P3-B's raw error relative to
the matched constant-position MTN (7,106 to 3,341), while normalized MSE was
slightly higher (1.0061 to 1.0097). The CV anchor in P3-E lowered raw error
relative to P3-D's constant-position counterpart (8,010 to 3,275), but P3-D
had the slightly lower normalized validation MSE. This illustrates why
the metric discrepancy and checkpoint-selection metric must remain explicit.
The deterministic CV baseline itself remained a strong reference: P3-C, P3-E,
P3-F, and P3-G had lower mean raw validation MSE, but each still worsened
individual windows and these single-seed comparisons are not conclusive.

P3-A's train normalized MSE was 0.3426 (212 windows) versus 1.2188 on
validation. This descriptive gap is consistent with a train/validation
distribution difference but does not establish its cause.

## Gate and context diagnostics

- P3-D's mean admission probability was 0.9164 (SD 0.0768), and it hard-opened
  all 43 validation windows. Its hard and forced-open predictions therefore
  matched. This validation run did not demonstrate sample-selective hard
  admission.
- P3-E's mean probability was 0.5525 (SD 0.0924); hard admission was 32/43
  windows (74.4%). Hard inference had raw MSE 3,275.40 versus 3,250.78 when
  forced open; forced open was better by 24.62 raw-MSE units. Hard inference
  had slightly lower normalized MSE (1.0118 versus 1.0161). This metric
  disagreement is not evidence that the learned gate improves prediction.
- P3-G's mean probability was 0.5524 (SD 0.0058), with a narrow observed
  range [0.5422, 0.5583]. All 43 windows hard-opened at the configured
  threshold, making hard predictions identical to forced-open predictions.
  Soft and hard mean raw MSEs were 3,106.42 and 3,082.83, respectively.
- P3-F is a more relevant always-open comparison: its raw validation MSE was
  3,379.50. P3-G's lower one-seed validation error is exploratory; because
  all P3-G hard decisions were open, it does not show a benefit from
  sample-dependent hard admission.

These gate distributions show that P3-E varied its probabilities and hard
decisions on this validation video; they do not establish that the gate
learned a generally meaningful context-benefit rule. P3-G's probabilities
were narrowly distributed and all crossed the hard-open threshold. No gate
rate was targeted or tuned. Context attention weights are retained in the
CSV and per-window outputs as diagnostics, not treated as an accuracy
measure.

## Interpretation and limitations

1. **Residual-anchor effect:** The matched pairs show lower raw validation
   errors with a CV anchor than with a CP anchor, but normalized loss and raw
   metrics do not always agree. This is a promising, validation-only
   observation, not an accuracy guarantee.
2. **Always-open context:** P3-F's mean raw error was 6.91% below the CV
   baseline, but P3-C was slightly better than P3-F. The screen does not show
   a clear benefit from adding always-open context to MTN.
3. **Differentiable admission:** P3-G had the lowest mean raw validation MSE
   in this screen, but its hard gate admitted every validation window and its
   probabilities varied very little. Its result therefore does not validate
   useful sample-dependent hard admission.
4. **Hard versus soft inference:** For P3-G, hard was slightly better on
   normalized and raw MSE than soft at the same checkpoint; the difference
   is descriptive and one-seed only. For P3-E, hard and forced-open rankings
   differed by metric. No robust inference penalty or gain is established.
5. **Need for multi-seed work:** These results alone are insufficient to
   justify a confirmatory claim. If the owner approves follow-up training,
   the next bounded step should be a predeclared multi-seed comparison of
   P3-C, P3-E, P3-F, and P3-G, with the primary metric and paired analysis
   fixed in advance. Do not select among them from this one seed.
6. **Zero-head diagnostic:** The optional zero-correction-head test
   reproduced the CV forecast exactly. Its measured upstream gradient norm
   was zero before the head update and 12.5955 after one update; this mode was
   not used in screening and is not evidence of an accuracy improvement.

### Test-data isolation caveat

No test dataset was constructed, iterated, passed to a model, used for
normalization, or evaluated. However, the existing
`flow_splits.pkl` is a monolithic pickle, and `load_split(validation_only=True)`
calls `pickle.load` before selecting train and validation roles. Thus the
serialized object containing the test role was deserialized into the process,
even though the runner only created train/validation datasets and did not
access test samples. The strict literal requirement to avoid deserializing
the test role was therefore not met at the storage layer. This limitation is
recorded rather than hidden; future strict isolation requires role-separated
cache files or an equivalent loader that does not deserialize the test role.
There is no evidence that test values influenced training, checkpoint
selection, metrics, or figures.

## Deliverables and verification

- [Validation summary CSV](P3_initial_screen_v1/P3_VALIDATION_SUMMARY.csv)
- [Per-window comparison CSV](P3_initial_screen_v1/P3_PER_WINDOW_COMPARISON.csv)
- [Per-window aggregates JSON](P3_initial_screen_v1/P3_PER_WINDOW_AGGREGATES.json)
- [Frozen qualitative selection](P3_initial_screen_v1/P3_QUALITATIVE_SELECTION.md)
- [Validation trajectory SVGs](P3_initial_screen_v1/figures/)
- [Runtime report](P3_RUNTIME_REPORT.md)
- [CV anchor formulation](CV_RESIDUAL_MATHEMATICAL_FORMULATION.md)
- [Gradient diagnostic report](P3_GRADIENT_REPORT.md)
- [Machine-readable gradient diagnostic](P3_diagnostics/gradient_diagnostics.json)

The initial run command's training completed, but its optional plotting step
returned an error because `matplotlib` was unavailable. The plotting code was
changed to emit SVG using the standard library, after which result
finalization and both plots completed. All seven run directories report
`completed`; each trainable configuration has 10 history rows. The final
regression command `python -m unittest discover -s tests -v` passed all 40
tests, and `py_compile` passed for the new Phase 3 files.

No Phase 4 work, additional seeds, test evaluation, commit, push, or manuscript
edit was performed.
