# Phase 6B Frozen Official-Test Evaluation Protocol

## Status

Preparation-only. Phase 6A is complete and no official-test annotation,
cache, image, video, or record is authorized for access at this stage.

## Scope and model identities

The official-test candidate identities are frozen in
`FROZEN_MODEL_MANIFEST.json`:

| Identity | Role | Parameters |
|---|---|---:|
| AO-2 | Primary neural accuracy candidate | 76,804 |
| AO-3 | Compact neural candidate | 68,356 |
| AO-RIDGE | Strong classical candidate | 11,700 fitted coefficients |
| AO-0 | Deterministic constant-velocity reference | 0 |
| AO-1 | Constant-position residual ablation | 76,804 |
| AO-CP | Constant-position baseline | 0 |

All five AO-1/AO-2/AO-3 selected checkpoints are preserved. No seed is
chosen after test evaluation. No ensemble is allowed.

## Frozen ridge model

The ridge coefficient file is:
`AO_confirmation_v1/classical/AO-RIDGE/seed_20260928/ridge_coefficients.npz`.

The alpha is frozen at 100. The file contains train-fitted feature means,
feature scales, residual means and coefficients. Validation and test data
must not alter coefficients or preprocessing.

## Window protocol

- Fifteen observed frames.
- Immediately following 45 target frames.
- Absolute pixel `xyxy` coordinates.
- 30 FPS convention.
- 60-consecutive-frame eligible segment, stride 30.
- Outside, malformed, non-positive-area and out-of-frame boxes split a
  segment and make the window ineligible.
- No interpolation.
- No scene, ego-speed, flow or future-derived input.
- Frozen train-only coordinate normalization.
- Exact sample IDs and identical ordering across every model.

## Metrics

Primary metric: pooled raw-pixel MSE across 45 future frames and four
coordinates.

Secondary metrics:

- Normalized MSE.
- C-MSE.
- CF-MSE.
- Horizon MSE at steps 1, 15, 30 and 45.
- Per-video MSE and macro-video MSE.
- Per-track MSE and macro-track MSE.
- Fraction of windows improved over AO-0.
- Parameter counts.
- Recorded CPU latency and memory under matched conditions.

## Interpretation rules

The winner is determined only after the complete eligible test set is
evaluated, using the frozen pooled primary metric and no post hoc metric
switching. Ridge must be reported if it wins. CV must be reported if it
wins. AO-2/AO-3 advantages or trade-offs must be reported as measured.
Annotation-only results are not directly comparable to historical full-input
published IDD-PeD scores.

## Exact preparation commands

```powershell
python -m unittest tests.test_official_test_evaluator_v1 tests.test_ao_confirmation_finalize_v1
python MTN\official_test_evaluator_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1 --checkpoint-root results_paper\AO_confirmation_v1 --output-json results_paper\AO_confirmation_v1\PHASE6B_DEVELOPMENT_EVALUATOR_SMOKE.json
```

Official-test execution is prohibited during preparation. Do not add
`--allow-official-test` until explicit written authorization and a separately
frozen official-test approval manifest exist.
