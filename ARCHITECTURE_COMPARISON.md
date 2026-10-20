# Architecture Comparison Report

## 1. Executive Summary

The previous MTN and current context-aware MTN both completed every identical synthetic test sample with finite `(4, 45, 4)` predictions. No measured trajectory accuracy comparison is possible: the IDD-PeD dataset, optical-flow inputs, labels, and checkpoint directory are absent. Therefore, no architecture is recommended for the actual use case yet.

## 2. Testing Methodology

- Device: CPU; PyTorch threads: 1; seed: `20260927`.
- Four fixed samples, identical input tensors and ordering for both models.
- The previous model's parameters exactly match the current model's shared MTN base branch (`same_base_weights=True`).
- Both ran in evaluation mode with 10 warm-up passes and 100 timed passes. Timings are per batch of four, using `perf_counter_ns`.
- Current model used its deployment-mode hard context gate. No targets were fabricated; accuracy fields remain unavailable.

## 3. Test Dataset/Samples

Synthetic contract samples: 15 observed boxes, 60 speed steps, 60 ego/pedestrian flow steps, six categorical scene fields, and 15 observed-only occlusion labels. Saved inputs and predictions: `outputs/architecture_comparison_predictions.npz`.

IDD-PeD availability: `IDDPedestrian=False`, `flow=False`, `IDDP_model=False`.

## 4–7. Architecture and Metric Results

| Metric | Previous MTN | Current context MTN | Difference |
| --- | ---: | ---: | ---: |
| Output shape | (4, 45, 4) | (4, 45, 4) | same |
| Finite outputs | True | True | same |
| Parameters | 40,132 | 46,166 | +15.04% |
| Parameter memory | 156.77 KiB | 180.34 KiB | +15.04% |
| Mean latency | 19.7988 ms | 22.9948 ms | +16.14% |
| Median latency | 19.1066 ms | 22.4181 ms | — |
| P95 latency | 23.5507 ms | 28.1170 ms | — |
| Accuracy / MSE | unavailable | unavailable | no labels installed |

## 8. Sample-wise Differences

| Sample | Context gate | Prediction RMSE vs. previous | Maximum absolute difference |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.01823795 | 0.05716145 |
| 1 | 1 | 0.03001770 | 0.06645036 |
| 2 | 1 | 0.01567047 | 0.03604913 |
| 3 | 1 | 0.01354708 | 0.02993071 |


## 9. Performance Improvements / Regressions

Only structural and runtime measurements are supported. The current model adds 6,034 parameters and changed measured mean latency by +16.14%. This is not an accuracy improvement or regression.

## 10. Errors and Failure Cases

No forward-pass failures, non-finite values, or shape mismatches occurred. The blocked evaluation case is real-data accuracy: all required IDD-PeD assets are missing.

## 11. Resource / Latency Comparison

Parameter storage is reported exactly. CPU process RSS and accelerator memory were not captured because this run used CPU and no process-memory profiler is installed. Timing variability is represented by median and P95 values above.

## 12–13. Key Findings and Trade-offs

The current model's context path and hard gate execute correctly on the same inputs. The added branch has a measurable parameter/latency cost. Whether its day/night, scene, and occlusion information improves trajectory prediction cannot be determined from unlabelled synthetic data.

## 14. Final Recommendation

**Do not finalize either architecture from this benchmark.** Install the identical IDD-PeD split, flow arrays, time-of-day map, and held-out labels; train both architectures with matched seed, schedule, and checkpoints; then compare held-out MSE/CMSE overall and by night/day, occlusion, signalization, and interaction.
