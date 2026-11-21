# Pre-launch CPU-time estimate

Estimate date: 2026-11-21. This is an operational estimate, not an experiment result.

## Fixed workload

The frozen split (`SPLIT.md`) has 212 train, 43 validation and 217 test windows. At 59 optical-flow transitions per 60-frame window, generating unshared per-window features would require 27,848 transition evaluations:

| Split | Windows | Transitions |
|---|---:|---:|
| train | 212 | 12,508 |
| validation | 43 | 2,537 |
| test | 217 | 12,803 |
| total | 472 | 27,848 |

The source-frame workload can be reduced by caching decoded frames and dense Farneback fields for repeated adjacent source-frame pairs. There are 4,812, 1,568 and 1,685 distinct referenced frames in train, validation and test respectively. Cached dense-flow computation is expected to be bounded by roughly 8,062 distinct adjacent frame pairs, plus per-box pooling; the cached and uncached variants must produce identical feature tensors for a validated sample before use.

The fixed Farneback configuration is 320x180, scale 0.5, 3 pyramid levels, window 15, 3 iterations, poly_n 5 and poly_sigma 1.2.

## Flow estimate

The prior 40-window GP6 POC generated about 2,360 window-transition features; the larger flow workload is 11.8 times that size. Allow **1-3 CPU hours** for a cache-aware, sequential-decode implementation, or **3-8 CPU hours** if it must compute every window transition separately. The range covers the very long GP6 video-2 decode span and disk/video-decoder variability.

## Training and evaluation estimate

The saved previous CPU run reports approximately 2.6 seconds for one 32-train/8-validation epoch of a context model with four training batches. Scaling by the train-batch count (27 batches for 212 windows) and validation batches (6 batches for 43 windows) gives an estimated **15-25 seconds per epoch per PACE-Net model** on four CPU threads. ROUTE is expected near the upper end; MTN is expected near the lower end.

For five models x ten seeds with a maximum of 50 epochs:

- E1 worst case: **10-18 CPU hours** training, plus approximately 10-20 minutes final test evaluation/checkpoint serialization.
- If early stopping occurs around 20 epochs on average: **4-7 CPU hours** training.
- B1's two LSTM variants, ten seeds each, are expected to add roughly **1-3 CPU hours**, pending their implementation and smoke benchmark.

These estimates exclude flow time, runner development, smoke tests, plotting, FLOP measurement and B2/B3. Execution should be serial to keep the CPU timing and determinism controlled.

## Blocking environment fact

The current Python interpreter has `torch` and `numpy` but does **not** have `cv2`/OpenCV. Therefore no Farneback flow can be generated or validated until a compatible OpenCV package is installed. No flow or model job has been launched.
