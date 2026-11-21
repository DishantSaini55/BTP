# Frozen from-scratch split and training protocol

**Frozen before flow generation or any model execution: 2026-11-21.**

## Leakage status and initialization

This replaces the contaminated official-checkpoint warm-start protocol. Every model, including MTN, will be initialised **from scratch** using the repository constructors and the run seed. No tensor, optimiser state, scheduler state, normalization statistic, prediction, checkpoint, or hyperparameter selected using the official MTN checkpoint may be loaded.

The three GP6 videos are in the official MTN checkpoint's training split. That fact does not contaminate this experiment because no official checkpoint is used. They are assigned only within this new from-scratch, video-level split:

| Role | Video | Selection rule |
|---|---|---|
| train | `gp_set_0006_vid_0002` | Every eligible 60-frame trajectory window beginning at a deterministic stride-30 start. |
| validation | `gp_set_0006_vid_0001` | Every eligible 60-frame trajectory window beginning at a deterministic stride-30 start, with the window's final source frame strictly below 4000. Used only for sanity checks and validation-MSE early stopping. |
| test | `gp_set_0006_vid_0003` | Every eligible 60-frame trajectory window beginning at a deterministic stride-30 start. Used only once per completed run for final metrics and saved outputs. |

No source video appears in more than one role. Window eligibility uses the existing trajectory sequence definition: valid pedestrian annotation, all required ego-speed/flow inputs, and 60 consecutive source frames after the repository's standard trajectory filters. A candidate's sample ID is `set/video/pid/start_frame/end_frame`; IDs are sorted lexicographically by `(video, pid, start_frame)` before splitting, and the SHA-256 of the newline-separated test IDs will be saved with every run.

The fixed stride is **30 frames**. Adjacent selected 60-frame windows therefore overlap by exactly 30 frames (50%), never more. There is no outcome/error/gate/image-dependent selection, no resampling, and no test-set tuning. The fixed selection seed is **20260928**; it is recorded for deterministic runner behaviour but does not alter the all-eligible-window rule.

Read-only materialisation against the cached official GP6 annotations, before flow generation, found these counts. Each selected 60-frame window has verified consecutive source-frame numbers.

| Role | Video | Eligible tracks | Windows (stride 30) | Unique source frames referenced |
|---|---|---:|---:|---:|
| train | `gp_set_0006_vid_0002` | 43 | 212 | 4,812 |
| validation | `gp_set_0006_vid_0001` before frame 4000 | 81 | 43 | 1,568 |
| test | `gp_set_0006_vid_0003` | 24 | 217 | 1,685 |

## Shared training protocol

All five paper models and both trained B1 LSTM baselines use the same cached Farneback flow, split, normalisation derived from train windows only, Adam at learning rate 1e-3, batch size 8, unweighted box MSE, and seeds 20260928 through 20260937 inclusive.

- Maximum epochs: **50**.
- Validation criterion: mean normalised box MSE over the fixed validation split, computed after every epoch in `model.eval()` mode.
- Early stopping: retain the checkpoint at the lowest validation MSE; stop after **10 consecutive epochs** without a strictly lower validation MSE (`min_delta = 0`). No test metric, gate statistic, qualitative output, or training-loss value influences selection.
- The selected validation checkpoint alone is evaluated on the test set once. Its epoch, full validation history and stopping reason are saved.

Fresh initialization is made after setting every run seed. The runner must record each model's full `state_dict` key list and initial per-tensor SHA-256 digest; no loaded-key list is expected because the official checkpoint is forbidden.

## Evaluation and statistics caveat

MSE, C-MSE and CF-MSE are evaluated in pixels after denormalisation. Per-video metrics are saved even though this split contains one test video only. Seed-level paired tests with Holm correction are primary.

The secondary test-window bootstrap is explicitly exploratory: all test windows come from one video and cannot establish cross-video generalisation. It uses a moving block bootstrap with **two consecutive selected windows per block** (60 source frames at the fixed 30-frame stride), 10,000 resamples, and reports this one-video limitation alongside every confidence interval.

## Flow

Farneback settings must exactly equal `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\farneback_config.json`. Flow is generated only after the materialised split manifest is reviewed and hashed. The resulting manifests must record each video ID, sample ID, source window bounds, selection rule/version, flow settings digest, and test-ID SHA-256. OpenCV `5.1.0` was installed in the active CPU environment after this protocol was frozen; one cache-aware flow job was started on 2026-11-21 after the counts and hash above were materialised.

## Superseded warm-start evidence

All pre-existing 32/8/10 and 32/8/24 GP6 results initialised from the official MTN checkpoint are legacy contaminated/adaptation artifacts, not evidence for this from-scratch study. They will be segregated under `results_paper/legacy_warmstart/` with provenance only and will not populate any summary table, figure, hypothesis test, or manuscript result.
