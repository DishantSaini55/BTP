# Phase 5B: Annotation-only multi-video trajectory prediction

## Outcome

The Phase 5B pilot is complete. A separate annotation-only dataset protocol and AO-0–AO-3 models were evaluated on the frozen development-only split. AO-4 was skipped because scene-label availability is incomplete across the selected validation videos and prediction-time availability is not verified. All three learned runs completed the ten-epoch cap; there was no test-set evaluation, raw-video download, optical-flow generation, paper edit, commit, or push.

This is a one-seed exploratory screen over four training videos and two validation videos. Overlapping windows are not independent replicates. The results do not establish statistical significance or generalization to IDD-PeD test sets.

## Data and isolation

The nested reference repository already contained the needed per-video development annotations; no acquisition was required. The split was frozen before model comparison:

- Train: GP1-0001, GP2-0001, GP4-0001, GP6-0002 (988 windows, 261 tracks, 4 videos).
- Validation: GP7-0004 and GP7-0005 (432 windows, 120 tracks, 2 videos).
- Window: 60 consecutive annotated frames, 15 observed and 45 future, stride 30.
- Coordinates: absolute pixel `xyxy`; FPS convention is 30 FPS, verified from the dataset description.
- Normalization: coordinate-wise statistics from unique train-role frames only.
- Observation inputs: observed boxes only. Ego speed was excluded because units are unverified; occlusion fields conflict; GP7-0005 has no POI labels.

The train and validation caches are separate NPZ files with source hashes, sample-ID digests, and explicit role membership. The historical GP6 test video was excluded; official test annotations and test caches were not opened. Details and provenance are in `ANNOTATION_AVAILABILITY_REPORT.md`, `ANNOTATION_ONLY_PROTOCOL.md`, and `OBSERVATION_SAFETY_AUDIT.md`.

## Architectures and correctness

| ID | Configuration | Unique trainable parameters | Result |
|---|---|---:|---|
| AO-0 | Exact deterministic constant-velocity formula | 0 | Evaluated |
| AO-1 | Bbox-only transformer; constant-position residual; unshared encoder attention | 76,804 | Trained from scratch |
| AO-2 | Matched bbox-only transformer; constant-velocity residual; unshared encoder attention | 76,804 | Trained from scratch |
| AO-3 | AO-2 with encoder Q/K/V/O self-attention shared across layers | 68,356 | Trained from scratch; parameter identity verified |
| AO-4 | Observation-safe scene-context model | — | Skipped: labels/availability not verified consistently |

These are versioned annotation-only architectures, not exact reproductions of historical full-input MTN/PACE-Net. No synthetic optical-flow inputs were supplied. Architecture changes are documented in `AO_ARCHITECTURE_COMPARISON.md`.

The repository test suite passed **70 tests** (`python -m unittest discover -s tests -v`). This includes the annotation loader/split checks, observation-only and target-shape checks, historical CV baseline regression, model shape/gradient/residual/SATIE checks, runner smoke/resume/validation-only behavior, and existing regressions. `py_compile` passed for the new source and test files. Pylance reported no diagnostics in the annotation loader, models, or runner tests; it noted that the Windows-excluded branch of the runner's platform guard is not analyzed.

## Validation results

Raw-pixel MSE averages squared error across the 45 future steps and four box coordinates. C-MSE is center-coordinate error pooled over the 45 steps; CF-MSE is center-coordinate error at the final step. Each learned model's checkpoint was chosen by validation normalized MSE.

| Model | Best epoch / epochs | Train normalized MSE at selected epoch | Validation normalized MSE | Raw-pixel MSE | C-MSE | CF-MSE | Windows improved vs CV | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AO-0 | — / 0 | — | 0.108174 | 3,531.505 | 2,889.519 | 14,435.050 | 0.0% | 0 |
| AO-1 | 4 / 10 | 0.111616 | 0.133124 | 7,822.443 | 7,265.028 | 25,943.253 | 11.1% | 76,804 |
| AO-2 | 4 / 10 | 0.109355 | 0.106061 | 3,432.318 | 2,644.576 | 12,768.007 | 47.2% | 76,804 |
| AO-3 | 4 / 10 | 0.109793 | 0.106441 | 3,454.714 | 2,667.826 | 12,935.696 | 48.6% | 68,356 |

AO-2's pooled raw-pixel MSE is 99.19 (2.8%) below AO-0; AO-3 is 76.79 (2.2%) below AO-0. This difference is not consistent across the two validation videos: AO-2/AO-3 are worse than AO-0 on GP7-0004 and better on GP7-0005. AO-1 is worse than AO-0 on both videos. AO-2 and AO-3 are very close; the one-seed result does not establish a predictive benefit from SATIE sharing. Sharing reduced parameters by 8,448 (11.0%) and recorded training-loop time by about 4.9%, with similar pooled validation error.

All learned models selected epoch 4, while their training normalized MSE was lower at epoch 10. This is consistent with the checkpoint-selection criterion and a widening train/validation gap after the selected epoch; it is not, by itself, evidence about the cause. The models do not uniformly improve on constant velocity, and the modest pooled AO-2/AO-3 differences are not enough to claim a robust benefit.

Horizon MSE at steps 1/15/30/45 (raw pixel coordinates) was:

| Model | Step 1 | Step 15 | Step 30 | Step 45 |
|---|---:|---:|---:|---:|
| AO-0 | 0.970 | 355.256 | 3,299.344 | 16,633.734 |
| AO-1 | 1,918.902 | 2,330.774 | 8,053.943 | 27,574.482 |
| AO-2 | 287.522 | 580.745 | 3,183.950 | 15,106.443 |
| AO-3 | 251.756 | 573.336 | 3,247.510 | 15,245.815 |

The full validation, per-video, and per-track tables are in `AO_initial_screen_v1/AO_VALIDATION_SUMMARY.csv`, `AO_PER_VIDEO_RESULTS.csv`, and `AO_PER_TRACK_RESULTS.csv`. They include 4 model rows, 8 video rows, and 480 model-by-track rows. Per-video and per-track averages do not turn overlapping windows into independent samples.

## Interpretation and limitations

1. **Implementation and isolation:** the tested annotation-only data path is role-separated, observation-only for model inputs, and train-only for normalization. Unit/regression tests passed. This does not certify anything about historical full-input runs or test-set performance.
2. **Constant-velocity residual:** AO-2 materially outperformed AO-1, but the variants differ in residual anchor; with one seed this is exploratory.
3. **SATIE sharing:** AO-3 had slightly higher pooled error than AO-2, with fewer parameters and less recorded train time. The validation evidence does not justify a claim that sharing improves prediction.
4. **Context:** AO-4 was not run. No ego-speed, occlusion, scene labels, or optical-flow features were introduced.
5. **Generalization and uncertainty:** a single seed and two validation videos do not support significance testing or broad generalization claims. No confidence intervals or p-values are reported.
6. **Protocol comparability:** these annotation-only results must not be directly presented as comparable to full-input published IDD-PeD scores.

The bounded single-seed initial screen is complete. Do not start multi-seed training or Phase 6 without a separate decision and a prediction-independent validation strategy.

## Runtime and reproduction

All learned runs completed under the configured 1,800-second per-run wall-clock budget. Their recorded process times were 1,509.431, 826.591, and 788.784 seconds for AO-1, AO-2, and AO-3 respectively. Total training-loop time was 3,079.067 seconds; peak sampled process RSS was 310,083,584 bytes. Timing details and exact commands are in `AO_RUNTIME_REPORT.md`.

The fresh-output reproduction commands are:

```powershell
python MTN\ao_dataset_v1.py prepare --output-root results_paper\splits\ao_annotation_only_v1_reproduction
python MTN\run_ao_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1_reproduction --output-root results_paper\AO_initial_screen_v1_reproduction --models AO-0,AO-1,AO-2,AO-3 --seed 20260928 --max-epochs 10 --patience 10 --batch-size 8 --learning-rate 0.001 --cpu-threads 4 --wall-clock-budget-seconds 1800
python MTN\run_ao_v1.py --finalize-only --output-root results_paper\AO_initial_screen_v1_reproduction
python -m unittest discover -s tests -v
```

## Deliverables

- Inventory and availability: `ANNOTATION_INVENTORY.csv`, `ANNOTATION_AVAILABILITY_REPORT.md`.
- Frozen protocol and safety: `ANNOTATION_ONLY_PROTOCOL.md`, `OBSERVATION_SAFETY_AUDIT.md`.
- Models and budget: `AO_ARCHITECTURE_COMPARISON.md`, `AO_COMPUTATIONAL_BUDGET.md`, `AO_RUNTIME_REPORT.md`.
- Results: `AO_initial_screen_v1/AO_VALIDATION_SUMMARY.csv`, `AO_PER_VIDEO_RESULTS.csv`, `AO_PER_TRACK_RESULTS.csv`.
- Source and tests: `MTN/ao_dataset_v1.py`, `MTN/ao_models_v1.py`, `MTN/run_ao_v1.py`, and `tests/test_ao_*.py`.
