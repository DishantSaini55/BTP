# B1 LSTM diagnostic rerun

## Diagnosis

The original B1 LSTM run was unstable:

- It decoded for 45 autoregressive steps by treating the network output as an increment, but the training loop was not teacher-forced.
- It had no global gradient-norm clipping.
- The learning rate was the shared 1e-3 rate, and the original scene-embedding cardinalities did not match the frozen context schema.
- Input and target data were normalized consistently, and raw-space MSE was computed only after denormalisation. The ~371k and ~408k reported MSE values are therefore not solely a denormalisation-unit bug.

The fixed rerun uses:

- normalized absolute-box targets,
- teacher forcing from the exact prior target during training,
- autoregressive own-prediction rollout during test inference,
- Adam 1e-3,
- batch 8,
- validation-only early stopping with patience 10,
- global gradient clipping at norm 1.0,
- scene cardinalities `(7, 5, 5, 5, 3)` plus the native occlusion context.

The fixed rerun is stored only in `results_paper/runs/B1_FIXED_CORRECTED/`; the original and intermediate directories were not overwritten.

## Seed-level comparison

| Seed | Original no scene MSE | Corrected no scene MSE | Original scene MSE | Corrected scene MSE |
|---:|---:|---:|---:|---:|
| 20260928 | 51920.336995 | 124662.270269 | 43706.145638 | 134114.734411 |
| 20260929 | 632575.979119 | 505728.866791 | 816339.830069 | 511434.849510 |
| 20260930 | 348965.021817 | 485711.399626 | 333887.020593 | 477876.687140 |
| 20260931 | 987629.689336 | 804902.810142 | 834196.436348 | 753514.402659 |
| 20260932 | 88250.946645 | 483080.966590 | 120858.972044 | 488248.882488 |
| 20260933 | 145577.547235 | 204110.698445 | 148372.640355 | 210556.147825 |
| 20260934 | 789531.214286 | 1247135.010081 | 1093840.376152 | 1222093.907834 |
| 20260935 | 333679.534850 | 273828.758641 | 300394.834533 | 177372.948031 |
| 20260936 | 152070.673423 | 298671.085685 | 186699.084317 | 304293.183684 |
| 20260937 | 181422.321429 | 32456.768312 | 207156.132993 | 25126.276518 |

## Finding

Both corrected-LSTM variants remain unstable and far worse than the 1496.018040 constant-velocity test MSE. The corrected no-scene mean MSE is 446028.863458; the corrected scene mean is 430463.202011. Corrected LSTMs beat the original only in 4 of 10 seeds for each variant. This is a real instability/result, not evidence that the MSE values are caused by a denormalisation mistake.
