# B1 LSTM rollout diagnosis

## Finding

The large error in `B1_FIXED_CORRECTED` is primarily a training/inference mismatch. That rerun passed the ground-truth prior box into the decoder at every training and validation step, while test inference fed back its own predicted box. Its validation checkpoint was therefore selected on teacher-forced loss rather than the deployed 45-step rollout. Small one-step residual errors compound during free-running inference. The train-video-only split and modest 212-window training set remain generalization constraints; the rollout correction is not expected to guarantee constant-velocity-level test error.

This identifies the cause of the approximately 446k teacher-forced rerun, not a single cause for every earlier B1 failure. The original B1 code already trained autoregressively but did not clip gradients and used the old scene-label cardinalities. Its poor results are consistent with instability and generalization failure as well; the new autoregressive rerun below retains clipping and the corrected label vocabulary.

## Scaling and rollout audit

The target and observed boxes use train-split z-scores, `z=(box-mean)/std`. The linear head predicts a normalized-coordinate displacement, and each next prediction is `z_previous + delta_z`. In raw coordinates this is `raw_previous + std * delta_z`; the implementation denormalizes both predicted and target boxes with the same train mean and standard deviation. No residual mean is added, and the output head is not accidentally left in raw units. This path is consistent.

For the seed-20260928 boxes+speed checkpoint in `B1_FIXED_CORRECTED`, split-wide normalized losses were:

| Split | Teacher-forced MSE | Autoregressive MSE |
|---|---:|---:|
| Train (212 windows) | 0.000646 | 0.140456 |
| Validation (43 windows) | 0.007217 | 5.599401 |
| Test (217 windows) | 0.025593 | 20.478539 |

The validation objective understated the inference-mode loss by about 776×; on test it understated it by about 800×. For the first eight test windows, normalized per-step MSE at steps 1, 15, 30 and 45 was `0.0114, 4.8612, 25.1459, 65.2463` autoregressively, versus `0.0114, 0.0226, 0.0258, 0.0262` with teacher forcing. Raw per-step MSE for that batch rose from `75.9` at step 1 to `372,225.0` at step 45. On that batch, the largest absolute predicted raw coordinate was 2,335.7, versus 1,306.7 for targets, consistent with rollout drift rather than a fixed denormalization multiplier error.

## One-window overfit

A fresh boxes+speed model was trained on one training window using its own autoregressive predictions at every step (Adam `1e-3`, gradient clipping at norm 1.0). Normalized rollout MSE fell from `16.9054` at update 1 to `0.0008051` at update 100 and `0.0005810` at update 500. The first- and last-step MSEs at update 500 were `0.0000668` and `0.0011446`. Thus the architecture and residual head can fit a single window with the same free-running path used in the rerun. This test is diagnostic only and was not used for model selection.

## Corrective rerun

The new rerun uses autoregressive predictions during both training and validation, so checkpoint selection matches test-time rollout. It preserves the split, train-only normalization, seeds, Adam `1e-3`, batch size 8, max 50 epochs, patience 10, and gradient clipping at norm 1.0; it uses the corrected scene vocabulary sizes. New artifacts are isolated in `results_paper/runs/B1_AUTOREGRESSIVE/`.

All 20 runs (both variants, seeds 20260928–20260937) completed:

| Variant | Mean test MSE | Median test MSE | Mean C-MSE | Mean CF-MSE | Mean epochs |
|---|---:|---:|---:|---:|---:|
| Boxes + ego speed | 281,652.402782 | 249,433.305426 | 147,355.326989 | 415,046.004665 | 35.9 |
| Boxes + ego speed + scene labels | 338,899.533790 | 239,398.803733 | 203,587.406782 | 570,593.492633 | 36.6 |

Compared with the prior teacher-forced reruns, the autoregressive rerun had lower test MSE in 8/10 boxes+speed seeds and 7/10 scene-label seeds. Mean test MSE fell by 36.85% and 21.27%, respectively. Nevertheless, the new means remain about 188× and 227× the constant-velocity test MSE of 1,496.018040. The rollout mismatch was a real training/evaluation defect, but correcting it did not make these LSTMs competitive; limited training data, video-level shift and high seed variance remain material. Per-seed metrics and epochs are in `results_paper/summary/B1_AUTOREGRESSIVE_summary.csv`.
