# PACE-Net paper work log

## 2026-11-21 — orientation only

- Extracted the author-supplied `overleaf_project (2).zip` to `paper/`; no TeX source was edited.
- Audited the local code, the existing artifacts under `D:\IDD-PeD-Evaluation`, the supplied manuscript source, and the official IDD-PeD MTN evaluation source.
- Instantiated all five prescribed models. Parameter counts are exactly 133,700; 139,142; 130,694; 154,007; and 145,559 in the requested model order. SATIE passed module identity, tensor `data_ptr()`, and unique-optimizer-parameter checks.
- Wrote `ORIENTATION.md`. Per the task boundary, no smoke test, training, evaluation, result migration, qualitative analysis, baseline download, or manuscript update was started.

Current repository commit observed before this documentation change: `2572a86567f9d7ef6b6d690ca83f06579288c0b9`.

## 2026-11-21 — leakage audit (blocking)

- Read the official default IDD-PeD split implementation and inspected the local official annotation cache.
- Found that the official MTN training split includes `gp_set_0006`, whose videos are `gp_set_0006_vid_0001`, `_0002`, and `_0003`.
- The local 32/8/24 protocol assigns those same videos to validation, local train, and local test respectively. Therefore the official MTN warm-start checkpoint contaminates the local test video.
- Wrote `ORIENTATION_ADDENDUM.md` and stopped before runner implementation, smoke tests, E1, B1, E2/E3 changes, or baseline work, awaiting the author's decision.
- E8 remains skipped because the SAGE x_T input is normalised; the requested E2/E3 changes have not yet been implemented because the leakage stop condition took precedence.

## 2026-11-21 — leakage-free test-asset inventory (blocking)

- Following the author's Option 1 decision, checked every official default-test set (`gp_set_0003`, `_0005`, `_0008`, `_0009`) for local pedestrian annotations, extracted images and source videos before selecting windows or computing flow.
- Only GP8 has local annotation XMLs (videos 0001, 0004 and 0005); none of those videos has local images or a source video. GP3, GP5 and GP9 have no local test annotations or video assets.
- There are zero locally flow-usable, leakage-free official-test videos. This is below the pre-registered minimum of three from at least two sets.
- Updated `ORIENTATION_ADDENDUM.md` and stopped. `results_paper/config/SPLIT.md` was intentionally not created because no valid test set can yet be frozen. No flow, legacy migration, runner work, smoke test, B1 or E1 was started.

## 2026-11-21 — from-scratch Option 1b protocol frozen

- The author selected a leakage-free from-scratch study within GP6 rather than the unavailable official-test-video option.
- Wrote `config/SPLIT.md` before computing flow or running any model. It forbids all official-checkpoint use, fixes video-level train/validation/test roles, a 60-frame stride-30 window rule, train-only normalization, 50 maximum epochs, validation-only early stopping with patience 10, ten seeds, and a two-window moving-block bootstrap caveat.
- Pending before launch: count/materialise eligible windows, verify the local Farneback environment, estimate CPU time for flow and training, and report that estimate to the author. No flow, model execution, smoke test, baseline, legacy migration, or E1 run has begun.

## 2026-11-21 — pre-launch workload estimate

- Read-only materialisation found 212 train windows from GP6 video 2, 43 validation windows from video 1 before frame 4000, and 217 test windows from video 3, all 60 consecutive source frames with stride 30.
- Recorded the exact counts and the missing OpenCV dependency in `config/SPLIT.md`.
- Wrote `config/CPU_TIME_ESTIMATE.md`: estimated 1-3 CPU hours for cache-aware Farneback flow (3-8 without caching), 10-18 worst-case CPU hours for E1, and 4-7 hours if early stopping averages 20 epochs; B1 adds an estimated 1-3 hours.
- Confirmed that the active Python environment lacks `cv2`; no flow, runner, smoke test, baseline, legacy migration, or E1 run has been launched. Awaiting pre-launch approval and dependency-install direction.

## 2026-11-21 — from-scratch flow cache and runner smoke tests

- Installed CPU OpenCV (`opencv-python-headless` 5.1.0) after the fixed split and workload estimate were written. A single cache-aware sequential Farneback job wrote `splits/gp6_from_scratch_v1/flow_splits.pkl` and `flow_metadata.json`; it did not overwrite any artifact.
- The cache contains 212 train, 43 validation and 217 test windows. Every `ego_op_flow` array is `[60,64,2]`, every `ped_op_flow` array is `[60,9,2]`, all values are finite, and the intentionally unused final flow slot is zero. The frozen test-ID SHA-256 is `6e19d53af0552d271ec1ec7dfd63c0a9aaa803d52250b018fdcbdbf655a71dbe`.
- OpenCV/FFmpeg emitted recoverable H.264 decode warnings while reading source videos. The job completed normally and feature validation passed; the warning log is retained beside the cache as `flow_stderr.log`.
- Archived copies of the earlier official-checkpoint GP6 result records under `legacy_warmstart/` with a README marking them contaminated/adaptation artifacts. They will not enter any result summary or test.
- Implemented the fresh E1 runner. Each run records exact `state_dict` initialization-key names/shapes/SHA-256 digests, `loaded_keys: []`, the source revision and local source hashes, full validation history, checkpoint, per-window predictions, gate decisions/logits where applicable, and decoder source-attention weights for every layer/head. It never loads an official checkpoint.
- Five-model smoke test passed: parameter counts are 133,700; 139,142; 130,694; 154,007; and 145,559; all outputs are finite; SATIE object/parameter/data-pointer sharing and unique optimizer registration passed. The forced SAGE g=0 path now bypasses the appended context token and exactly equals the motion-only output. The initial implementation appended a masked token and differed by at most `7.15e-7` due to attention reduction order; it was corrected before any training run.
- Implemented and smoke-tested B1. Constant-position and three-observed-frame constant-velocity predictors are deterministic; fresh 32-hidden-unit autoregressive LSTM baselines use boxes+ego speed, with a separately specified static-scene-label variant. Both LSTM outputs are `[batch,45,4]` and finite; their counts are 9,988 and 10,760. No B1 or E1 metric run has yet begun.

## 2026-11-22 — E1 attention serialization recovery

- E1 completed the ten MTN seeds and eight PACE-Net-without-SATIE seeds before `PACE-Net without SATIE`, seed 20260936, failed after validation-selected checkpoint creation and before any test metric, per-sample CSV, or prediction file was written.
- Cause: an exact all-closed SAGE batch takes the 15-memory-token motion-only path, while a batch with one or more admitted samples has 16 tokens. The evaluator attempted a direct concatenation of those variable-width attention arrays. This is an output-serialization defect, not a split, training, validation-selection, or test-data issue.
- Preserved the failed checkpoint and its original run configuration under `failed_runs/E1_attention_serialization_20260936/`; nothing was overwritten or deleted. Repaired the evaluator to pad only the absent context-token attention column with `NaN` and save a per-sample `attention_memory_tokens` field.
- The repair passed a read-only 217-test-window evaluation check. Restarted only the failed seed and the not-yet-started E1 model/seed combinations under the same fixed seed and protocol. Completed E1 metrics are not rerun. The E1 gate remains closed until 50 metrics and `summary/` exist.

## 2026-11-22 — E1 gate opened and report generated

- The from-scratch E1 pipeline completed all 50 model-seed run directories and 50 `metrics.json` files across MTN, PACE-Net without SATIE, PACE-Net with SATIE, PACE-Net with ROUTE, and PACE-Net with ROUTE and SATIE.
- Rebuilt `results_paper/summary/` from the 50 immutable run directories after detecting that the earlier summary CSV contained only 10 MTN rows. The stale/partial summaries were preserved under `summary_stale_partial_20261122/`, `summary_failed_partial_20261122/`, and `summary_regeneration_20261122/`; no run directory or checkpoint was overwritten.
- Repaired `MTN/summarize_e1.py` to union variable metric keys when writing `all_runs.csv`; SAGE gate models add `admission_rate`, while ROUTE models add route-share fields.
- Wrote `results_paper/E1_REPORT.md` and summary artifacts with 50 per-run rows, aggregate MSE/C-MSE/CF-MSE and epoch summaries, H1/H2 Holm-adjusted results, track-cluster bootstraps, per-seed values, B1 baselines, and anomaly reporting.
- No non-finite E1 metric was found. Eight runs reached 50 epochs; seed 20260936 is a visible high-MSE MTN outlier and was retained.

## 2026-11-22 — E2 SAGE gate ablations

- Completed 80 post-hoc E2 checkpoint evaluations: four ablations (`always_open`, `always_closed`, `soft_sigmoid`, `context_only`) across PACE-Net with and without SATIE for seeds 20260928–20260937.
- The learned-gate configurations were reused from matching E1 checkpoints; no E2 model was retrained.
- Wrote `results_paper/summary/E2_all_runs.csv`, `E2_summary.csv`, `E2_summary.tex`, `E2_statistics.json`, and `E2_REPORT.md`.
- E2 resume logic skips existing immutable run directories. One empty always-closed seed 20260935 directory was identified and removed specifically, then that variant was rerun; no completed E2 result was overwritten.

## 2026-11-22 — E3 gate behaviour and E8 trigger

- Ran post-hoc gate/route analysis over all matching E1 saved test outputs for seeds 20260928–20260937; no training or test-set tuning was used.
- Wrote E3 CSV/JSON/Markdown/LaTeX summaries, including admitted/rejected MSE, gate logits, scene and occlusion slices, and normalized `x_T` ranges.
- The normalized test `x_T` range reached `[-38.396156, 4.739763]`, exceeding the protocol's `±5` trigger in train, validation, and test. E8 will therefore be run.

## 2026-11-22 — E8 launch stopped after stalled first run

- E8 was launched because E3 triggered the `±5` normalized `x_T` condition.
- The first E8 clipped-`x_T` seed 20260928 run remained active for more than one hour without writing a metric or training log and was stopped to avoid an uncontrolled parallel experiment.
- The empty run directory was removed; no partial checkpoint or fabricated result is retained.
- E8 remains explicitly incomplete and is not included in any summary or test-set conclusion.

## 2026-11-23 — B1 LSTM diagnosis and corrected rerun

- Diagnosed the original B1 LSTM instability: no gradient clipping, unmasked 45-step autoregressive rollout, and incorrect scene vocabulary cardinalities. Input/target normalization and raw-denormalization were consistent; the reported million-scale values remain a training instability/result rather than a unit-conversion defect.
- Patched the B1 runner and added `run_b1_fixed.py`, using teacher forcing during training, own predictions during test rollout, Adam 1e-3, batch 8, patience 10, and gradient clipping at norm 1.0.
- Completed 20 corrected reruns in `results_paper/runs/B1_FIXED_CORRECTED/` plus 20 separate intermediate diagnostic runs in `B1_FIXED/`; neither directory overwrote an existing B1 run.
- Wrote `results_paper/B1_FIXED_REPORT.md`. Both corrected variants remain far worse than constant velocity (1496.018040 test MSE), with mean MSE 446028.863458 for boxes+speed and 430463.202011 with scene labels.

## 2026-11-24 — E1 diagnostics

- Added 30 independent epoch-level diagnostic reruns for MTN, PACE-Net without SATIE, and PACE-Net with SATIE. They save train MSE, validation MSE, normalized test MSE, and raw test MSE for every epoch. Test curves were never used for checkpoint selection.
- Wrote `results_paper/summary/E1_DIAGNOSTICS_summary.csv`, `E1_DIAGNOSTICS_aggregate.csv`, and `E1_DIAGNOSTICS_REPORT.md`.
- Aggregate absolute validation-versus-test-optimal epoch gaps were 7.10 for MTN, 10.50 for PACE-Net without SATIE, and 7.40 for PACE-Net with SATIE.
- MTN seed 20260932 selected epoch 34 but its test-optimal diagnostic epoch was 22. It stopped at 44 epochs in the diagnostic rerun, versus 50 in the original run. This shows video-shift/selection concern but does not prove early stopping alone caused seed 20260932's failure.

## 2026-11-24 — E7 timing and FLOPs

- Benchmarked all five E1 model families at batch sizes 1 and 8 with 4 threads, 50 warm-ups, and 500 timed iterations.
- CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz.
- Used `thop` for MACs and recorded parameter counts and seed-20260928 checkpoint sizes.
