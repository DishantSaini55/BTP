# PACE-Net paper work log

## 2026-11-27 — Phase 6D CRAFT Stage 1 minimal AO-3 pilot

- Captured the genuine AO-3 seed-20260928 final encoder ReLU FFN activations
  for all 988 training and 432 validation windows from development caches only.
- Verified the frozen checkpoint hash and development cache manifest; no
  official-test role was read. Identity activation intervention matched the
  original model output exactly (maximum absolute difference 0.0).
- Six focused regression tests passed. The bounded rank-4/rank-8 NMF pilot
  used a fixed lexicographic 96-window train subset and feature RMS scaling.
  Rank 4 train relative reconstruction error was 0.39036; rank 8 reached its
  120-iteration cap with a logged convergence warning and error 0.29474.
- Held-out NNLS relative reconstruction error was approximately 1.0, so the
  projection is degenerate and no concept interpretation is supported.
- Saved independently hashed activations, NMF pilot factors, audit, metrics and
  a stage report under `phase_6d_craft_stage1_v1/`. No AO-2, recursive
  decomposition, Sobol, insertion/deletion, training, official-test access,
  paper edit, commit or push occurred.

## 2026-11-27 — Phase 6D CRAFT robust recovery attempt

- Inspected and preserved all four invalid/incomplete Phase 6D CRAFT partial
  directories; verified no experiment stage was complete and no Python process
  was active at recovery start.
- Reverified AO-2/AO-3 seed-20260928 checkpoint hashes and the development-only
  cache manifest (988 train, 432 validation; official test sets not read).
- Diagnosed prior NMF warnings as iteration-cap convergence warnings on an
  unscaled 14,820-by-128 activation matrix. Drafted `MTN/run_craft_trajectory_v2.py`
  with deterministic feature scaling, bounded video-balanced fitting rows and
  an exclusive lock; syntax compilation passed.
- Recovery stopped before smoke testing or experiment execution because the
  required regression suite, stage-level resumability and intervention-scale
  correctness are not yet validated. `phase_6d_craft_execution_v2/RECOVERY_STATUS.md`
  records the exact remaining work. No new scientific results are claimed.
- No official-test access, training, model modification, raw-video download,
  `paper/` edit, commit or push occurred.

## 2026-11-27 — Phase 6C literature, comparison and novelty review

- Completed a documentation-only review in `phase_6c_literature_v1/` using the
  corrected Phase 6B v2 bundle and v3 derivatives as canonical.
- Created a verified-reference database, protocol-compatibility/SOTA taxonomy,
  manuscript claim audit, novelty audit, research-direction recommendation and
  CRAFT/HIVE feasibility materials.
- The review identifies AO-0/AO-CP/AO-1/AO-2/AO-3/AO-RIDGE as the only direct
  numerical comparison set. Published full-input, stochastic, world-coordinate
  and best-of-K results are reported as unmatched context only.
- Recommended a comparative empirical paper framing: ridge is the strongest
  frozen official-test predictor; neural improvements over CV are marginal and
  not conclusive under the seed-level analysis. SATIE is retained only as a
  compactness result; SAGE/MASC/ROUTE are not supported headline claims.
- No training, official-test inference, architecture modification,
  hyperparameter tuning, raw-video download, `paper/` edit, Phase 6D execution,
  commit or push occurred.

## 2026-11-27 — Phase 6B official-test preparation

- Froze the Phase 6A model identities: AO-2 as primary neural accuracy
  candidate, AO-3 as compact candidate, AO-RIDGE as strong classical candidate,
  AO-0 as CV reference, AO-1 as constant-position residual ablation and AO-CP
  as constant-position baseline.
- Froze ridge alpha 100 and preserved the train-fitted coefficient archive.
- Prepared the official-test membership plan for GP3/GP5/GP8/GP9 using the
  authoritative dataset documentation, while preserving GP6 as a separate
  historical local test protocol.
- Created the evaluation-only implementation and refused official-test access
  through the preparation state.
- Added 10,000-repetition Phase 6A finalization outputs, checkpoint/seed
  provenance, runtime and latency records.
- Ran 21 AO-focused regression tests successfully.
- No training, official-test access, raw archive download, paper edit,
  commit or push was performed.

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

## 2026-11-24 — Protocol assessment

- Wrote `results_paper/PROTOCOL_ASSESSMENT.md`.
- Documented cross-video shift, small GP6 training volume, constant-position residual-head context, and constant-velocity superiority.
- Inventoried local GP6/GP8 pedestrian XML, GP6/GP8 OBD XML, GP6 source videos, and local cache files.
- Explicitly recorded that no training protocol change was made and that approval is required before any future change.

## 2026-11-24 — E7 MAC audit correction

- Confirmed THOP's shared-module hook duplication: the SATIE encoder self-attention object is called three times, but THOP registers its Q/K/V/O linear hooks three times, yielding nine hook counts for three executions. This accounts exactly for the false 1,474,560-MAC-per-sample increase.
- With sequence length 15 and model width 32, analytically counted 75,840 MACs per encoder self-attention layer and 227,520 across all three layers (1,820,160 for batch 8), identical with and without SATIE sharing.
- Replaced the whole-model THOP MAC column in E7 summaries with the verified encoder self-attention component count. Preserved previous raw THOP outputs only under explicitly unverified audit field names. Removed THOP profiling from the E7 runner.

## 2026-11-24 — B1 LSTM rollout root-cause rerun

- Compared teacher-forced and autoregressive predictions from the prior `B1_FIXED_CORRECTED` checkpoints. On seed 20260928 boxes+speed, validation normalized MSE was 0.007217 teacher-forced versus 5.599401 autoregressive; test was 0.025593 versus 20.478539. At test steps 1/15/30/45, autoregressive normalized MSE on the first eight windows rose from 0.0114 to 65.2463, while teacher-forced MSE remained 0.0262 at step 45.
- Verified train-only z-score input/target normalization, normalized residual addition and inverse scaling are consistent. A one-window autoregressive overfit reached 0.000581 normalized rollout MSE after 500 updates.
- Corrected training and validation to use autoregressive rollout, retaining Adam 1e-3, batch 8, max 50 epochs, patience 10, gradient clipping norm 1.0, frozen split and seed list. Completed both LSTM variants for all ten seeds in new `runs/B1_AUTOREGRESSIVE/` directories.
- Mean test MSE was 281,652.402782 (boxes+speed) and 338,899.533790 (with scene labels), versus 446,028.863458 and 430,463.202010 for the prior teacher-forced reruns. This improves 8/10 and 7/10 paired seeds, respectively, but remains far worse than constant velocity (1,496.018040).
- Wrote `B1_ROOT_CAUSE_REPORT.md` and `summary/B1_AUTOREGRESSIVE_summary.csv`. No other experiment was started.

## 2026-11-25 — Qualitative selection and overlay figures

- Before opening any image, froze and pushed `qualitative/SELECTION.md` for seed 20260928, including six RNG-selected distinct pedestrians, two largest PACE-Net improvements, two largest degradations, and available gate examples.
- Both saved SAGE gates were closed on all 217 test windows, leaving no `g=1` window; no alternate seed or forced gate was substituted. The frozen selection includes 12 unique windows (6 random, 2 improvement, 2 degradation, 2 closed-gate) and records the unavailable open-gate case.
- Rendered a 12-page vector-overlay PDF using the source GP6 video at each selected window's last observed frame; included observed boxes, ground truth, MTN, both PACE-Net variants, constant velocity, per-window MSE, gates, and scene labels. Wrote the page-to-window metrics CSV.

## 2026-11-25 — Saved decoder attention and gate-mask analysis

- Aggregated all three decoder layers, four heads, and 45 query positions from the saved seed-20260928 test outputs. Wrote per-head/per-token summaries for all selected windows and all 217 test windows.
- Both learned SAGE gates were `g=0` for all 217 cases, so the E1 attention arrays contain 15 trajectory keys; there is no natural 16th context-token attention value or `g=1` counterfactual window. Scene-stratified gate counts and the not-estimable masking effect are recorded.
- Loaded the saved PACE-Net checkpoint for a post-hoc mixed `[g=1,g=0]` mask-mechanics check only (no weights updated). In all 3 layers and 4 heads, the forced g=0 row's context key had exactly zero attention; forced g=1 values are marked diagnostic, not learned-gate results. Wrote the page-to-window metrics CSV.

## 2026-11-25 — Exploratory CRAFT-inspired trajectory concepts

- Read the CRAFT GitHub README/PyTorch API and Fel et al., CVPR 2023 / arXiv:2211.10154. Recorded why this is a CRAFT-inspired regression probe rather than an unchanged application of the classifier/image pipeline.
- Audited encoder box/speed and decoder ReLU FFN activations, plus context post-GELU activations. All FFN activations were finite and non-negative. Context post-GELU negatives accounted for 13.81% of train and 13.58% of validation absolute activation mass before clamping.
- Fit NMF bases on train activations and chose ranks by the frozen validation-only 1%-from-best rule over candidate ranks 2–16: final encoder box FFN rank 15 (validation relative reconstruction error 0.3397), clamped context rank 14 (0.1334). The preliminary 2–8 rank sweep is retained separately because it reached its upper bound.
- Captured per-step factors for all 15 observed steps, top-activation training frame/window examples, and descriptive scene/gate associations. The learned gate admitted 49/212 train and 17/43 validation windows.
- Ran a fixed-reference training-window functional probe: per-step ablation and 1,024-point Sobol indices for final predicted center displacement. Effects are local to the gate-closed reference sample and are explicitly not global importance estimates.
- No model weights changed, no test dataset was iterated, and no files under `paper/` were edited. Wrote `qualitative/CRAFT_DECISION.md` with limitations; concepts were not clearly human-interpretable.

## 2026-11-25 — HIVE-style study design

- Read the HIVE repository and Kim et al., ECCV 2022 / arXiv:2112.03184; wrote `qualitative/HIVE_DESIGN.md` adapting distinction, agreement, and comprehension tasks to the saved trajectory predictions and attention displays.
- Specified participant-safe stimulus masking, randomization, the six random primary cases and six exploratory cases, response schema, scoring, and a future analysis-script contract. The design records that all selected examples have `g=0`, natural `g=1` examples are unavailable, and the 12-window single-video bank is suitable only for a feasibility pilot.
- Design only: no participant materials were rendered or deployed, no participant responses were generated or analyzed, and `paper/` was not edited.

## 2026-11-25 — Zoomed frozen qualitative overlays

- Added `MTN/render_qualitative_zoom.py`, reading the unchanged IDs from `qualitative/SELECTION.md`. The six-panel page uses the first two listed random windows, both largest improvements, and both largest degradations; page two contains the two frozen closed-gate windows.
- For each of the eight windows, computed the crop from the union of the 15 observed boxes, 45 ground-truth boxes, and all 45 boxes from each of the four saved predictors. Added a 30% margin on each side, expanded to the source-frame aspect ratio, and drew all paths/boxes with pixel-coordinate axes and a full-frame inset.
- Wrote `qualitative/ZOOMED_OVERLAYS_seed20260928.pdf` as a two-page PDF with vector paths/boxes and raster source frames. Verified page count, panel IDs, crop containment/aspect ratio, and visually inspected both pages. No selection changed and no files under `paper/` were edited.

## 2026-11-25 — Fixed-window architecture asset pack

- Added `figures_paper/assets/generate_asset_pack.py` and `README.md` for the single frozen window `gp_set_0006/gp_set_0006_vid_0003/gp_6933/01415/01474`. Exported observed/predicted source-frame crops and strips, cached Farneback flow maps, speed bars, label metadata, saved-model memory/attention strips, ROUTE operator outputs, and encoder self-attention parameter bars.
- Loaded E1 seed-20260928 PACE-Net and ROUTE checkpoints for forward-only token/operator capture. The saved SAGE gate for this sample is `g=0`; ROUTE selects additive `[0,1,0]`. The saved 16-token strip is explicitly documented as a pre-gate context candidate, not the actual E1 decoder memory.
- The raw cached observed ego speed is zero at all 15 steps; this is shown as zero-height bars/markers, not altered. Flow maps use cached Farneback 8×8/3×3 grids, nearest-neighbor enlarged for display. Scene/occlusion labels and numeric color ranges are in `scene_labels.json`; no annotation text is rasterized.
- Ran the asset exporter and verified the image dimensions/DPI, saved sample/gates, attention shape and normalization, and that ROUTE operators and model token hooks matched the saved checkpoints. No training was run and no files under `paper/` were edited.

## 2026-11-25 — Second fixed-window architecture asset pack

- Before viewing any candidate image, filtered the frozen test split to windows with nonzero mean raw OBD speed over the first 15 frames and maximum observed occlusion >= 1, then selected the lexicographically first ID. Five strict matches existed; no relaxation was used. The frozen selection is `gp_set_0006/gp_set_0006_vid_0003/gp_6933/01205/01264`, with mean observed speed 1.5726103782653809 and observed max occlusion 2. The rule and ID are in `figures_paper/assets2/README.md`.
- Exported the same 14-image/JSON asset set to `figures_paper/assets2/` using real source-video frames, cached flow, and saved E1 seed-20260928 outputs. The observed speeds are recorded unchanged in `scene_labels.json`; cached Farneback grids are enlarged only for display. Saved PACE-Net gate is `g=0`, and ROUTE selects additive.
- During the first attempt, image drawing exposed saved PACE-Net boxes with reversed horizontal endpoints at 16 of 45 future steps. Updated the shared crop/drawing helper to use the min/max endpoint envelope for visualization only; did not alter saved predictions. Exact affected zero-based steps are recorded in the JSON and the limitation is documented in README.
- Regenerated successfully; verified the frozen selection is the lexicographically first strict match, all PNGs decode and are 300 DPI, speed/occlusion values satisfy the rule, and gate metadata matches saved outputs. No training was run and no files under `paper/` were edited.

## 2026-11-26 — Phase 3 CV residual-anchor screen

- Added the separate `MTN/cv_residual_v1.py` model wrapper and `MTN/run_p3_screen.py` resumable runner. Historical model implementations and prior experiment outputs were left unchanged.
- Passed deterministic unit tests and capped smoke checks, then ran P3-B through P3-G sequentially for seed 20260928, 10 epochs maximum, full train/validation splits, batch size 8, and four CPU threads. P3-A was evaluated as a deterministic CV baseline. No test forward pass, prediction, or metric was performed.
- All seven run statuses are `completed`; the six trainable runs have 10 epoch-history entries. The validation split has 43 windows, all from `gp_set_0006_vid_0001`. Results and per-window comparisons are in `P3_initial_screen_v1/`.
- Best raw validation MSE among the selected checkpoints was P3-G hard (3,082.83) versus 3,630.32 for exact CV; P3-G hard-opened all 43 validation windows, so this is not evidence for sample-dependent hard admission. These are one-seed exploratory results, not statistically conclusive.
- Recorded train-loop elapsed time summed to 1,789.535 seconds across six trainable runs; maximum sampled RSS was 349,159,424 bytes. The exact runtime table and commands are in `P3_RUNTIME_REPORT.md`.
- The initial post-training plot step failed due to missing `matplotlib`; it was replaced with standard-library SVG generation. Finalization and both validation plots then succeeded without rerunning training.
- Integrity caveat: `load_split(validation_only=True)` only creates train and validation datasets and the runner never accesses test values, but the monolithic flow pickle is deserialized in full, including its test role. This does not satisfy literal no-test-deserialization; it is disclosed in `PHASE_3_RESEARCH_REPORT.md` and `P3_RUNTIME_REPORT.md`. Train-only normalization was retained.
- Wrote `PHASE_3_RESEARCH_REPORT.md`, `P3_RUNTIME_REPORT.md`, and this work-log entry. Final validation: `python -m unittest discover -s tests -v` passed all 40 tests; Python compilation checks passed. No `paper/` files, additional seeds, Phase 4 work, commit, or push.

## 2026-11-27 — Phase 4 cache isolation and runner validation

- Migrated the trusted monolithic GP6 flow cache once to role-separated files under `splits/gp6_from_scratch_v1/roles_v1/`. The source SHA-256 is `3c888c7c089de88bffd44349c8336b4ca9b76177402d141e990ea79a1c306001`; the migration verified role counts, IDs, video membership, per-sample hashes, and copied metadata without recomputing flow. The one-time migration did deserialize all three source roles and is explicitly documented.
- Added `dataset_roles_v1.py`, which only accepts and constructs train and validation paths, and `migrate_flow_cache_v1.py`, which refuses overwrite and requires explicit trusted-cache acknowledgement. Phase 4 normalization is computed from train samples only.
- Added the P4-A–F model configurations with unique trainable counts 133,700; 125,252; 138,516; 130,068; 130,694; and 130,630. Parameter identity and SATIE audit tests passed.
- Implemented the resumable validation-only runner and corrected its incremental summary-field mapping. Added explicit no-test role checks, full-split metrics, per-epoch checkpoint/budget handling, summary updates after each run, and the `IndividualTF` type check for decoder attention.
- Ran all six capped smoke configurations against a temporary cache containing only train/validation pickles and manifests; all completed. Then measured one full train/validation epoch for every configuration at four CPU threads. Measured epoch times ranged 24.97–44.90 seconds; these are budget inputs, not accuracy claims. A one-thread P4-F epoch was 18.77 seconds and is only a single paired runtime observation.
- Wrote `DATASET_ISOLATION_REPORT.md` and `P4_COMPUTATIONAL_BUDGET.md`. Temporary testless cache directories were removed. The independent test cache was not opened during smoke or benchmark screening.
- Added a validation-only finalizer for seed-paired metrics, paired t intervals, pedestrian-track bootstrap intervals, gate interventions, parameter-efficiency reporting, runtime summaries, and provenance. It refuses to summarize capped, incomplete, non-validation-only, or differently configured runs.
- Completed the sequential three-seed P4-A–F screen: all 18 runs finished with explicit `completed` states using 325 total epochs. Per-run and per-configuration metrics, timings, sampled RSS, checkpoint sizes, and run metadata are in `P4_initial_screen_v1/`.
- Finalized validation-only comparisons, 10,000-repetition pedestrian-track bootstrap intervals, gate utility interventions, parameter-efficiency summaries, and provenance. The finalizer verified all 18 runs, 212 train and 43 validation samples, the frozen seeds/configurations/settings, and matching role-cache hashes; metadata records `test_split_loaded: false` and train-only normalization.
- Total recorded training-loop time was 11,223.612 seconds (3.118 hours), excluding process startup and final report generation; longest run was 1,114.306 seconds and maximum sampled RSS was 336,777,216 bytes. Updated `P4_COMPUTATIONAL_BUDGET.md` with measured totals alongside the original projection.
- No files under `paper/` were changed; no commit or push was made.

## 2026-11-26 — Phase 5 dataset feasibility stop

- Performed a metadata-only inventory of `D:\IDD-PeD-Evaluation`; no experiment test cache was opened or deserialized, and the local annotation database pickle was not inspected.
- Found three local GP6 videos, GP6/GP8 pedestrian and OBD annotations, one GP6 archive, and GP8 annotations without their matching raw videos. The GP6 archive is only a likely duplicate of the extracted videos; byte-for-byte identity remains unverified.
- Confirmed the historical GP6 split uses video 0003 as its test video, so it cannot be presented as a newly acquired untouched multi-video test.
- Recorded that prior DNS failures prevented verification of remote archive sizes. Did not download, extract, build a split/cache, compute flow, or train.
- Created `DATASET_INVENTORY.csv`, `DATASET_AVAILABILITY_REPORT.md`, and the explicitly unfrozen `DATASET_EXPANSION_PLAN.md`. The proposed development split uses GP1/2/4 for training and GP7 for validation while leaving official test sets GP3/5/8/9 untouched.
- Stopped at the feasibility gate pending user approval for data acquisition and the draft protocol. No files under `paper/` were changed; no commit or push was made.

## 2026-11-26 — Phase 5A selective acquisition feasibility

- Re-inventoried D: storage and existing IDD-PeD files by metadata only. D: had 75,205,443,584 bytes free; the dataset directory had 125 files totalling 8,495,455,289 bytes. The local GP6 archive's three member names/sizes match the extracted videos, but byte identity was not checked. Prior GP8 partial files are absent; the retained log ends at byte 201,326,591.
- Did not open or deserialize `test.pkl` or `test_full.pkl`. Existing annotation and flow material was inventoried without generating flow or modifying caches.
- HEAD and one-byte Range probes for GP1/2/4/7 failed DNS resolution at `cvit.iiit.ac.in`; sizes, redirects, range support, resume validators, and remote checksums remain unknown. Repeated the metadata-only probe after implementation; all four hosts remained unresolvable.
- Streamed only official-train annotations for GP1/2/4/7 at Hugging Face revision `9be5757b550bc2e60581c1af2b6ce4b3a78bfd93`. Annotation estimates show 506 legacy-filtered (520 unfiltered) candidate training windows from local GP6 video 0002 plus GP1 videos 0001/0003, and 228 legacy-filtered (432 unfiltered) GP7 validation windows across two videos. The split remains unfrozen because of annotation-version and outcome-related eligibility concerns.
- Added `MTN/download_iddped_archives.py`, its focused test file, `REMOTE_ARCHIVE_INVENTORY.csv`, `STORAGE_FEASIBILITY_REPORT.md`, `MINIMUM_DATASET_EXPANSION_PLAN.md`, and this acquisition report. The utility requires explicit `--execute`, destination on D:, and an explicit byte cap; no actual video archive download or extraction was started.
- Focused downloader tests passed (6/6); `py_compile` passed; Pylance diagnostics for both modified Python files returned no issues. The final metadata-only probes reported DNS resolution errors, with no HTTP negotiation.
- D: has about 70 GiB free, but required peak storage is unknown because GP1/GP7 archive and extracted-video sizes remain unknown. No informed download approval was requested; stop until fresh metadata and a storage plan are available. No training, flow generation, test-cache access, `paper/` edit, commit, or push occurred.

## 2026-11-27 — Phase 5B annotation-only pilot

- Audited the separate development annotations and froze four training videos (988 windows, 261 tracks) and two GP7 validation videos (432 windows, 120 tracks); excluded historical GP6 test video 0003. Used separate role caches and train-only normalization. No official-test annotations/cache, raw video, or optical flow was accessed or created.
- Added versioned bbox-only AO-0–AO-3 models, independent loaders, resumable runner, and focused regression tests. AO-4 was skipped because scene labels and prediction-time availability are not verified consistently across GP7-0004/0005. Ego speed and occlusion were also excluded from model inputs for their documented availability/consistency limitations.
- Completed the sequential one-seed screen (20260928): AO-0 evaluated; AO-1, AO-2, and AO-3 each completed ten epochs. All learned runs selected epoch 4 by validation normalized MSE and remained below the 1,800-second per-run cap.
- Validation raw-pixel MSE was 3,531.505 (AO-0), 7,822.443 (AO-1), 3,432.318 (AO-2), and 3,454.714 (AO-3). AO-2/AO-3 were worse than CV on one validation video and better on the other; the one-seed evidence is exploratory and does not establish significance or broad generalization.
- Finalized 4-row model summary, 8-row per-video table, and 480-row per-track table. All 70 repository tests passed; Python compilation checks passed. Pylance reported no diagnostics on the annotation loader, models, and runner tests; it noted that the Windows-excluded branch of the runner's platform guard is not analyzed.
- Recorded 3,079.067 seconds of training-loop time across learned runs (51.32 minutes), 3,124.806 seconds of process time (52.08 minutes), and maximum sampled RSS 310,083,584 bytes. Updated `AO_COMPUTATIONAL_BUDGET.md`; added `AO_RUNTIME_REPORT.md` and `PHASE_5B_RESEARCH_REPORT.md`.
- No `paper/` edits, official-test evaluation, multi-seed training, Phase 6 work, commit, or push.

## 2026-11-27 — Phase 6A confirmation preflight

- Reviewed the frozen AO split, protocol, model configurations, continuation checkpoints, and CPU budget. Confirmation output was absent; no multi-seed training had started.
- Corrected ridge metric bookkeeping discovered during preflight: the closed-form coefficient fit had been recorded as zero training time. Preserved the original preflight freeze and outputs; created immutable `AO_CONFIRMATION_MANIFEST_v2.json` for the corrected source, with unchanged split/cache, model, and training configuration hashes.
- Reran the ridge preflight into a new output directory. With the predefined validation-selected alpha 100, AO-RIDGE achieved validation raw-pixel MSE 3,129.163, C-MSE 2,767.625, CF-MSE 13,508.873, and improved 55.8% of validation windows relative to CV. The selected fit took 0.481 seconds; selection across the five predefined alphas took 3.489 seconds. The model has 11,700 fitted coefficients/parameters.
- AO-RIDGE selection uses the fixed validation split and is exploratory, not an independent test; the result does not establish generalization. AO-CP validation MSE was 10,233.668. No official-test data, raw video, or optical-flow data were accessed.
- Added the validation-only finalizer for paired seed-level t intervals, track-cluster bootstrap intervals, per-video/per-track tables, correction magnitudes, runtime/checkpoint summaries, explicit incomplete-run handling, and final-test readiness. It verifies frozen manifest/cache provenance and reads only the separate train/validation caches.
- Focused confirmation tests passed (3/3 for ridge; 3/3 for finalization helpers); Python compilation and Pylance syntax/diagnostic checks passed. The five-seed AO confirmation matrix is running sequentially under its frozen wall-clock/reserve gates.

## 2026-11-27 — CRAFT Stage 3 bounded development analysis

- Completed the bounded frozen AO-3 Stage 3 experiment in 105.65s under the 30-minute budget and four-thread limit. Reused saved Stage 1 activations and the canonical Stage 2B rank-4 bank; no training, final-layer activation recapture, or official-test access.
- Saved protocol, regression results, recursive factors, motion associations, Sobol estimates/convergence, intervention/fidelity tables, figures, report, and hashes in `results_paper/phase_6d_craft_stage3_v1/`.
- See `CRAFT_FINAL_EXPERIMENT_REPORT.md` for actual findings and limitations. No manuscript edits, commit, or push.


## 2026-11-27 -- Phase 7 supervisor manuscript package

- Created the versioned manuscript and delivery package in `results_paper/final_manuscript_v1/`; the legacy `paper/` directory was not modified.
- Integrated canonical Phase 6B corrected v2/v3 evidence, Phase 6C compatibility guidance, Phase 6D CRAFT Stage 3 results, and source-verified AO architecture.
- Added a compiled 15-page paper PDF, Overleaf source ZIP, editable architecture diagrams, metrics figures, canonical data copies, provenance documentation, and prospective HIVE protocol templates.
- Direct pdfLaTeX and BibTeX compilation passed; final logs had no unresolved references/citations or layout overfull warnings. `latexmk` is unavailable because Perl is not installed.
- Official-test trajectory overlays and blinded AO HIVE stimuli remain blocked because canonical official-test records do not store coordinate predictions. No inference rerun was performed. No participants or responses exist.
- No training, inference, CRAFT computation, raw-video download, hyperparameter selection, manuscript edit under `paper/`, commit, push, or later research phase.
