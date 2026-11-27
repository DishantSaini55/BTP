# Phase 4 Validation Results

All runs use the role-separated cache and only train/validation data.
The 43 validation windows come from one video and overlap in time.
Three-seed comparisons are exploratory; no test metrics, test-based
selection, or confirmatory significance claims are reported.

| Configuration | Policy | Parameters | Epochs (mean) | Validation normalized MSE | Raw pixel MSE | C-MSE | CF-MSE | Correction RMS (px) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P4_A | base | 133700 | 16.33 | 1.111469 | 4566.179 | 3678.264 | 12365.193 | 47.096 |
| P4_B | base | 125252 | 20.67 | 1.014245 | 3256.154 | 2484.312 | 11500.039 | 34.086 |
| P4_C | always_open | 138516 | 18.33 | 1.047057 | 3552.724 | 2739.629 | 11038.571 | 40.686 |
| P4_D | always_open | 130068 | 19.33 | 1.038590 | 3819.907 | 2934.702 | 11608.962 | 42.268 |
| P4_E | hard | 130694 | 17.33 | 1.077824 | 5095.549 | 4190.278 | 12765.901 | 49.333 |
| P4_F | soft | 130630 | 16.33 | 1.078929 | 5774.419 | 4939.521 | 12896.134 | 52.847 |

Deterministic CV validation raw MSE: 3630.315.

Read `P4_PAIRED_ABLATIONS.md`, `P4_GATE_UTILITY.md`, and
`P4_PARAMETER_EFFICIENCY.md` for paired and diagnostic analyses. No
model-selection decision was based on test data. These limited-video
results cannot establish robustness to video or dataset shift.
