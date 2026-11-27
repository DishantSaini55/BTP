# Annotation-only computational feasibility

## Dataset and preprocessing

| Quantity | Measured |
|---|---:|
| Train videos / tracks / windows | 4 / 261 / 988 |
| Validation videos / tracks / windows | 2 / 120 / 432 |
| Train cache | 564,867 bytes |
| Validation cache | 247,262 bytes |
| Split manifest | 4,222 bytes |
| Selected split parse + cache construction | 25.475 s |
| Historical GP6 test video used | No |
| Optical-flow preprocessing | None |

The preprocessing measurement reran only the six frozen development-video annotations into a temporary directory on D:; the temporary cache was removed afterward. The broader inventory parsed 19 development annotation files and completed after the command's 180-second initial wait, but its exact elapsed duration was not captured.

## Model sizes

| Model | Unique trainable parameters |
|---|---:|
| AO-0 | 0 |
| AO-1 | 76,804 |
| AO-2 | 76,804 |
| AO-3 | 68,356 |
| AO-4 | Not implemented / skipped |

## Full-split epoch benchmark

The benchmark used AO-2, seed 20260928, batch size 8, four CPU threads, all 988 training windows, and all 432 validation windows. It completed one epoch and is marked `benchmark_complete`, not as a ten-epoch pilot result.

| Measure | Observed |
|---|---:|
| Train + validation epoch time | 107.006 s |
| Final full-validation forward pass | 36.227 s |
| Validation forward latency | 83.859 ms / window |
| Peak sampled process RSS | 301,355,008 bytes |
| Process elapsed for benchmark and final artifacts | 158.592 s |

The one-epoch loss values saved with this timing run are not used as final predictive results. See `AO_epoch_benchmark_v1/AO-2/seed_20260928/metrics.json`.

## Pre-screen projection and feasibility decision

Using the observed 107.006-second epoch as a conservative common estimate for each learned model, three models × ten epochs project to about 3,210 seconds (53.5 minutes), before startup and final artifact writing. Adding one final validation pass per model gives approximately 55.3 minutes. Each model's expected training time is about 18 minutes, below the configured 30-minute per-run wall-clock cap.

The actual CPU was verified as an Intel Core i5-8365U on Windows 11 Pro. At the time of the feasibility check, Windows reported about 2.0 GiB free physical memory; the benchmark process itself peaked near 288 MiB. The small model and cache fit the observed memory headroom, but unrelated system load can vary. The pilot runs sequentially and retain atomic resumable checkpoints. If any run reaches its time cap, its status remains `incomplete` and it is not included in the completed comparison.

**Decision:** proceed with the bounded one-seed pilot AO-0–AO-3; skip AO-4 for the documented label-availability and prediction-time-safety reasons. No additional seeds or epochs beyond the ten-epoch cap are authorized by this protocol.

## Measured initial-screen cost

The sequential initial screen completed AO-0 and all ten epochs of AO-1, AO-2, and AO-3. The three trainable runs used 3,079.067 seconds of recorded training-loop time (51.32 minutes) and 3,124.806 seconds of recorded process wall time (52.08 minutes) in total. Individual recorded process times were 1,509.431 seconds (AO-1), 826.591 seconds (AO-2), and 788.784 seconds (AO-3). They all finished within the 1,800-second per-run cap; none is marked incomplete.

Maximum sampled RSS across the learned-model runs was 310,083,584 bytes (AO-1, approximately 295.6 MiB). Peak RSS measurements are process samples, not a continuous guarantee. The deterministic AO-0 validation computation took 0.037 seconds; artifact writing and Python startup are not included in that figure. Measured run-by-run latency, checkpoint sizes, and the variable per-epoch timings are in `AO_RUNTIME_REPORT.md`.

The initial 107.006-second benchmark projection was close to the observed 102.636-second mean training-loop time per learned-model epoch, but individual models and epochs varied substantially. These timings are one sequential run on one machine and are not a general CPU performance claim.
