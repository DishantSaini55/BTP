# Phase 6A AO confirmation protocol — frozen before training

## Freeze and scope

The initial preflight freeze, `AO_CONFIRMATION_MANIFEST.json`, is retained
unchanged. Before confirmation training began, review found that the ridge
preflight summary incorrectly recorded zero training time despite fitting
coefficients. The corrected ridge bookkeeping is frozen in
`AO_CONFIRMATION_MANIFEST_v2.json`, created at
`2026-11-26T21:50:04+05:30`; use v2 for this experiment. It contains
exact per-sample IDs, video IDs, track IDs, frame starts, annotation
provenance, cache hashes, model hashes, training settings, and the Phase 5B
seed-20260928 continuation checkpoint hashes. Both manifests are immutable;
neither may be rewritten to fit observed results.

The frozen development split is unchanged:

| Role | Videos | Windows | Tracks |
|---|---|---:|---:|
| Train | GP1-0001, GP2-0001, GP4-0001, GP6-0002 | 988 | 261 |
| Validation | GP7-0004, GP7-0005 | 432 | 120 |

Coordinates are absolute pixel `xyxy`; each sample has 15 observed frames and
the immediately following 45 target frames at the verified 30-FPS convention.
Windows are contiguous and stride 30. Inputs contain only the 15 observed
bounding boxes. Future boxes are supervision and evaluation targets only.
Ego speed, scene labels, occlusion, optical flow, images, and raw video are
not inputs. Normalization uses unique training-role frames only.

The frozen SHA-256 values are:

| File | SHA-256 |
|---|---|
| `train.npz` | `f089004844578c98c2b43aabe77e962f02ff94aae9b5ba0ed65514ce5d352df7` |
| `validation.npz` | `01141782b19d21bcc14ef77529cb6a856ff919ffefd246c013eca4717e7d5c63` |
| `split_manifest.json` | `e0fdf042eb4f9f30e5ea066a4b23490cb6582d7ad9356df77b8c5bd72b503c01` |

The ordered train and validation sample-ID digests are
`9b3c69920c67f33836fd4811da045b8905fbcdb793950ffb69aea1a877fc82cf` and
`eb78190af81de2588ae222bd81472bcf36fd25979f19dae0eb105e56d4d783ee`.
The manifest records source annotation XML hashes and exact video/track
membership. The historical GP6 test video and official test sets are excluded;
`official_test_sets_read` is empty.

## Model and training freeze

| Model | Residual anchor | Shared encoder self-attention | Unique parameters | Configuration SHA-256 |
|---|---|---|---:|---|
| AO-1 | Constant position | No | 76,804 | `e4d41c70197a19f36a60b0805c5f21856c9c07f61ed01f691341bbf4989429f9` |
| AO-2 | Constant velocity | No | 76,804 | `9dbbc7d5410b220fd30cb8eff9c17f58a4e4fd174062feadfcc6d9fd94b5647d` |
| AO-3 | Constant velocity | Q/K/V/O module shared across encoder layers 1–3 | 68,356 | `1e617d10ce80692c36903b7168145e1a5b23ffb46ebf3602e8a9066e5cc1cefb` |

AO-2 and AO-3 use the same source, anchor, dimensions, decoder, feed-forward
blocks, normalization, dropout, and outputs; the sole model-configuration
difference is the encoder attention-sharing flag. Identity is verified at
runtime. AO-3 has 8,448 fewer unique parameters (11.0%). It still executes
three encoder attention calls per forward pass.

The fixed model settings are three encoder and decoder layers, `d_model=32`,
`d_ff=128`, four heads, dropout 0.1, Adam (`lr=0.001`, betas `(0.9, 0.999)`,
epsilon `1e-8`, zero weight decay), MSE loss, batch size 8, four CPU threads,
and validation-normalized-MSE checkpoint selection. The five predefined seeds
are **20260928, 20260929, 20260930, 20260931, 20260932**. Training shuffles
windows by a seed-and-epoch-specific permutation, matched across model IDs
for a seed. Maximum epochs are 30 and early-stopping patience is 10.

The training configuration SHA-256 is
`7325810eddd182f211dbb146b0cbd92dfaf63e16aff4e90513a681a23ee8a306`.
Model and source hashes are in the manifest. No split, model dimension, data
feature, optimizer, loss, seed, or selection rule may be changed after this
freeze.

### Seed-20260928 continuation

The Phase 5B 10-epoch screen remains preserved in
`AO_initial_screen_v1/`. For confirmation only, each seed-20260928 run is
forked into a new output directory from its verified latest training
checkpoint, including optimizer and RNG states, and continued through the
30-epoch maximum/early-stopping rule. Only the maximum epoch cap and
wall-clock allowance differ in the old run configuration; the training
operations through epoch 10 are unchanged. This is a continuation, not a
relabeling of the old 10-epoch result. Other seeds start from scratch. Source
checkpoint and configuration SHA-256 values are recorded in the manifest.

## Classical comparison freeze

AO-CP repeats the last observed `xyxy` box for all 45 steps. AO-0 is the exact
coordinate-wise CV forecast
`b[T+k] = b[T] + k * (b[T] - b[T-2]) / 2`.

AO-RIDGE predicts the direct, 45-by-4 correction to that CV forecast. Its
64 input features are the 15 observed boxes normalized with train-only
coordinate statistics, expressed relative to the last observed box, plus the
normalized last box. Feature means/scales and coefficients are fitted on train
only. The target is the normalized future trajectory minus the normalized CV
forecast. A centered, direct multi-output ridge solve (no autoregressive
rollout) uses the frozen grid `[0.01, 0.1, 1, 10, 100]`; select alpha by pooled
raw-pixel MSE on the fixed validation split. This alpha selection is
exploratory validation tuning and will be disclosed. Neural and classical
models use the same observed-box inputs; no unavailable ego speed is fabricated.

## Metrics and decision rules

Primary predictive metric: pooled raw-pixel MSE averaged over the 45 future
steps and four `xyxy` coordinates. Report validation normalized MSE, C-MSE,
CF-MSE, raw horizon MSE at steps 1/15/30/45, per-video results, macro-video
averages, per-track results, and fraction of windows improved over CV.
Report neural correction magnitude as per-window RMS Euclidean `xyxy`
correction relative to each model's declared anchor, with mean, median, and
90th percentile.

For each fixed-seed model, report selected epoch and actual compute. Across
matched seeds, report individual values and mean, sample SD, and median of
paired differences. Five seeds vary initialization/training stochasticity on
the **same two validation videos**; they are not five independent datasets.
Track bootstrap intervals, if reported, resample whole pedestrian tracks
within each of the two fixed videos and are conditional on those videos. They
do not establish population-level video generalization. No significance,
equivalence, or broad generalization claim follows from these analyses.

Choose the best accuracy candidate by the lowest mean pooled raw-pixel
validation MSE across completed matched seeds. Report macro-video averages
separately. AO-3 may be recommended as a compact Pareto candidate only when
its measured accuracy/parameter trade-off supports that choice; do not claim
equivalence based on a nonsignificant test. Keep AO-2 as the accuracy candidate
if it is materially better. Report deterministic CV and ridge honestly even
if either performs best.

## CPU timing and budget gate

The preflight benchmark uses the Phase 5B seed-20260928 checkpoints to
characterize these architectures before confirmation. It matches E7's
protocol: four threads, 50 warm-ups, 500 timed iterations at batch sizes 1
and 8. The initial medians were noisy, so no speed claim is made; final
latency is measured again on confirmation checkpoints. The preflight CSV and
JSON record all quantiles and machine details.

Analytical encoder self-attention MACs use E7's verified formula
`3 * (4*B*L*d^2 + 2*B*L^2*d)` with `L=15`, `d=32`: 227,520 per sample and
1,820,160 for batch 8, both with and without SATIE. One hook per *unique*
attention module verifies exactly three executions for each architecture;
shared-module hooks are not registered once per layer reference.

The observed Phase 5B mean epoch times (10-epoch histories) were 149.651 s
(AO-1), 81.121 s (AO-2), and 77.135 s (AO-3), with substantial per-epoch
variation. A conservative all-30-epoch, five-seed projection is 46,186
seconds (12.83 hours). Continuing the three seed-20260928 checkpoints after
epoch 10 reduces the projected remaining work to about 43,107 seconds
(11.97 hours). Set a 14-hour total CPU wall-clock budget, a two-hour per-run
budget with resumable checkpoints, and a six-hour reserve gate before seeds
20260931–20260932. After the first three seeds, proceed with the final two only
if at least six hours remain; otherwise preserve resumable states and report
the incomplete matrix. Disk space at freeze was approximately 75.17 GB free;
no raw videos, flow, or test data are required.

## Test isolation and reproduction

No official test annotation, cache, image, video, or record may be downloaded,
opened, deserialized, summarized, or used in this phase. Do not edit `paper/`.
Do not commit or push.

```powershell
python MTN\freeze_ao_confirmation_v1.py --output results_paper\AO_CONFIRMATION_MANIFEST_v2.json
python MTN\benchmark_ao_latency_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1 --checkpoint-root results_paper\AO_initial_screen_v1 --output results_paper\AO_confirmation_preflight\latency.csv --warmups 50 --timed-iterations 500
python MTN\run_ao_confirmation_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1 --manifest results_paper\AO_CONFIRMATION_MANIFEST_v2.json --output-root results_paper\AO_confirmation_v1
```

The first command creates the v2 manifest and refuses overwrite. Resume an
interrupted matrix with the same arguments plus `--resume`. Ridge evaluation
follows the AO-0 validation run:

```powershell
python MTN\ao_confirmation_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1 --confirmation-root results_paper\AO_confirmation_v1 --output-root results_paper\AO_confirmation_v1\classical
python MTN\benchmark_ao_latency_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1 --checkpoint-root results_paper\AO_confirmation_v1 --output results_paper\AO_confirmation_v1\AO_FINAL_LATENCY.csv --warmups 50 --timed-iterations 500
python MTN\finalize_ao_confirmation_v1.py --output-root results_paper\AO_confirmation_v1 --cache-dir results_paper\splits\ao_annotation_only_v1 --manifest results_paper\AO_CONFIRMATION_MANIFEST_v2.json --classical-root results_paper\AO_confirmation_v1\classical --bootstrap-repetitions 10000 --bootstrap-seed 20260928
```

Run the final latency benchmark only after the seed-20260928 AO-2 and AO-3
best checkpoints are present. Run finalization after the orchestration matrix
has reached a terminal state; incomplete runs are listed as incomplete and do
not produce five-seed comparisons. Finalization writes seed/model summaries,
paired comparisons, per-video/per-track tables, correction magnitudes,
runtime/checkpoint reporting, final-test readiness notes, and provenance
metadata under `AO_confirmation_v1/`. It refuses to overwrite these reports.
