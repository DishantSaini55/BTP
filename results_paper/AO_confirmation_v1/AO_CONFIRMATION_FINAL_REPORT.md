# Phase 6A AO confirmation report

Run matrix status: **complete**.
Completed confirmation runs: 20 / 20.
Frozen split: 988 train windows across 4 videos; 432 validation windows across 2 GP7 videos.

All predictive results are validation-only on two fixed GP7 videos. The five seeds measure training stochasticity on the same split; they are not five independent datasets. Track-cluster bootstrap intervals are conditional on those two videos and overlapping windows.

The official test partition was not read or evaluated. No raw video or optical-flow feature was used. No statistical significance or broad video-generalization claim is made.

## Completed model means

| Model | Seeds | Parameters | Epochs | Norm. MSE | Pooled MSE | Macro-video MSE | Macro-track MSE | C-MSE | CF-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AO-0 | 5 | 0 | n/a | 0.108174 | 3531.505 | 3524.452 | 4631.888 | 2889.519 | 14435.050 |
| AO-1 | 5 | 76804 | 26.60 | 0.115715 | 6254.644 | 6128.876 | 6441.238 | 5766.406 | 20672.416 |
| AO-2 | 5 | 76804 | 17.00 | 0.105507 | 3349.963 | 3318.745 | 3904.627 | 2651.030 | 12981.059 |
| AO-3 | 5 | 68356 | 17.00 | 0.105656 | 3353.704 | 3323.057 | 3909.406 | 2650.193 | 13009.314 |
| AO-CP | 1 | 0 | n/a | 0.163478 | 10233.668 | 10251.269 | 13847.624 | 9638.782 | 35470.747 |
| AO-RIDGE | 1 | 11700 | n/a | 0.074342 | 3129.163 | 3086.894 | 3503.891 | 2767.625 | 13508.873 |
Macro-video and macro-track MSE give equal weight to each video or track, respectively; the accompanying per-video and per-track CSVs retain the underlying subgroup values.

## Raw-pixel MSE by forecast horizon

| Model | Step 1 | Step 15 | Step 30 | Step 45 |
|---|---:|---:|---:|---:|
| AO-0 | 0.970 | 355.256 | 3299.344 | 16633.734 |
| AO-1 | 933.345 | 1810.228 | 6860.229 | 22198.181 |
| AO-2 | 202.082 | 508.512 | 3088.287 | 15223.820 |
| AO-3 | 204.710 | 493.510 | 3112.067 | 15245.517 |
| AO-CP | 10.896 | 2416.589 | 11567.997 | 37472.106 |
| AO-RIDGE | 1.433 | 276.805 | 2883.710 | 14790.223 |

## Architecture selection

By the predeclared mean pooled raw-pixel validation MSE rule, AO-2 is the best neural candidate (3349.963), better than AO-0 by 181.543 (3531.505). This is a validation-only ranking, not a significance or generalization claim.
AO-3 uses 11.0% fewer unique parameters than AO-2; its mean raw-pixel MSE is 3.741 higher. The paired uncertainty is reported below; parameter savings do not establish predictive equivalence.
AO-RIDGE is reported separately (one deterministic fit; validation-selected alpha 100; MSE 3129.163) and is not a five-seed architectural comparison.

## Fresh-seed sensitivity (excluding resumed seed 20260928)

The remaining four seeds start from fresh seed-specific initialization. Their
mean pooled validation MSE is AO-1 5862.694 +/- 125.726 SD, AO-2
3329.374 +/- 162.318 SD, and AO-3 3328.451 +/- 142.024 SD. Individual
fresh-seed MSE values are:

- AO-1: 5800.651, 5947.961, 5716.859, 5985.306.
- AO-2: 3434.255, 3267.364, 3485.811, 3130.066.
- AO-3: 3381.508, 3264.657, 3497.481, 3170.158.

Relative to the constant-velocity mean of 3531.505, AO-1 deteriorates by
66.011%, AO-2 improves by 5.724%, and AO-3 improves by 5.750%. Excluding
the resumed seed leaves AO-3 0.923 MSE lower than AO-2 on average
(AO-3 wins 2 of 4 fresh seeds), but this is exploratory and not statistically
conclusive. The seed-20260928 continuation remains a documented protocol
exception rather than a fresh-start run.

## Accuracy-versus-parameter view

Among neural models, AO-2 (76,804 parameters; MSE 3349.963) and AO-3
(68,356 parameters; MSE 3353.704) occupy the practical accuracy/parameter
Pareto trade-off. AO-1 (76,804 parameters; MSE 6254.644) is dominated by the
CV-residual models. AO-3 is 11.0% smaller and has 3.741 higher mean MSE than
AO-2, with an AO-3-minus-AO-2 paired MSE 95% t interval of [-40.004, 47.486].
AO-RIDGE is outside the neural Pareto comparison: it uses 11,700 fitted
parameters and had exploratory validation-selected MSE 3129.163, but alpha
was tuned on the same fixed validation split and it has no five-seed result.

## Paired seed comparisons

Negative first-minus-second deltas favor the first-named model. t-intervals use df = n-1; intervals are descriptive. Track bootstrap intervals are secondary and conditional on the fixed validation videos.

| Comparison | Metric | Mean delta | t 95% CI | d_z | First wins | Track bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Constant-position transformer vs CV | mse | 2723.1 | [1627, 3820] | 3.0833898036325134 | 0/5 | [1313, 4508] |
| Constant-position transformer vs CV | normalized_mse | 0.0075403 | [-0.004639, 0.01972] | 0.7687395143724787 | 0/5 | n/a |
| Constant-position transformer vs CV | cmse | 2876.9 | [1827, 3926] | 3.4037393395935007 | 0/5 | [1551, 4615] |
| Constant-position transformer vs CV | cfmse | 6237.4 | [2420, 1.005e+04] | 2.0286736729706107 | 0/5 | [2568, 1.069e+04] |
| Unshared CV-residual transformer vs CV | mse | -181.54 | [-365.2, 2.123] | -1.2273140660126574 | 5/5 | [-365.8, -44.43] |
| Unshared CV-residual transformer vs CV | normalized_mse | -0.0026674 | [-0.003637, -0.001698] | -3.4156515572832866 | 5/5 | n/a |
| Unshared CV-residual transformer vs CV | cmse | -238.49 | [-392.4, -84.55] | -1.9236869954761013 | 5/5 | [-408.5, -120.4] |
| Unshared CV-residual transformer vs CV | cfmse | -1454 | [-2457, -450.8] | -1.7996824204327695 | 5/5 | [-2175, -944.3] |
| SATIE CV-residual transformer vs CV | mse | -177.8 | [-345.8, -9.756] | -1.3137521729152537 | 5/5 | [-364.1, -36.38] |
| SATIE CV-residual transformer vs CV | normalized_mse | -0.0025187 | [-0.003873, -0.001164] | -2.3090275563628997 | 5/5 | n/a |
| SATIE CV-residual transformer vs CV | cmse | -239.33 | [-366.5, -112.2] | -2.336853637165922 | 5/5 | [-408.5, -120] |
| SATIE CV-residual transformer vs CV | cfmse | -1425.7 | [-2376, -475.2] | -1.8623103730014163 | 5/5 | [-2148, -916] |
| CV-residual vs constant-position transformer | mse | -2904.7 | [-3980, -1829] | -3.3525086726055044 | 5/5 | [-4570, -1572] |
| CV-residual vs constant-position transformer | normalized_mse | -0.010208 | [-0.02213, 0.001718] | -1.0628235558794832 | 5/5 | n/a |
| CV-residual vs constant-position transformer | cmse | -3115.4 | [-4199, -2031] | -3.568166838351297 | 5/5 | [-4808, -1773] |
| CV-residual vs constant-position transformer | cfmse | -7691.4 | [-1.185e+04, -3528] | -2.2937152236709775 | 5/5 | [-1.248e+04, -4003] |
| SATIE vs unshared encoder attention | mse | 3.7409 | [-40, 47.49] | 0.106182144652881 | 2/5 | [-10.14, 21.9] |
| SATIE vs unshared encoder attention | normalized_mse | 0.00014875 | [-0.0008333, 0.001131] | 0.18807730729510538 | 1/5 | n/a |
| SATIE vs unshared encoder attention | cmse | -0.83672 | [-33.91, 32.23] | -0.03141503746820938 | 2/5 | [-11.2, 11.14] |
| SATIE vs unshared encoder attention | cfmse | 28.255 | [-94.29, 150.8] | 0.28628438461110245 | 2/5 | [3.552, 60.54] |
| AO-CP vs AO-0; single-seed validation-only | mse | 6702.2 | n/a | n/a | 0/1 | [4644, 9488] |
| AO-CP vs AO-0; single-seed validation-only | normalized_mse | 0.055303 | n/a | n/a | 0/1 | n/a |
| AO-CP vs AO-0; single-seed validation-only | cmse | 6749.3 | n/a | n/a | 0/1 | [4692, 9518] |
| AO-CP vs AO-0; single-seed validation-only | cfmse | 21036 | n/a | n/a | 0/1 | [1.535e+04, 2.852e+04] |
| AO-RIDGE vs AO-0; single-seed validation-only | mse | -402.34 | n/a | n/a | 1/1 | [-938.1, 132.6] |
| AO-RIDGE vs AO-0; single-seed validation-only | normalized_mse | -0.033833 | n/a | n/a | 1/1 | n/a |
| AO-RIDGE vs AO-0; single-seed validation-only | cmse | -121.89 | n/a | n/a | 1/1 | [-547.4, 321.8] |
| AO-RIDGE vs AO-0; single-seed validation-only | cfmse | -926.18 | n/a | n/a | 1/1 | [-3172, 1481] |

## Incomplete runs

- None.

A model recommendation must be based on the completed validation matrix, parameter counts, measured runtime, and declared criteria. AO-RIDGE's predefined alpha was selected on this same validation split and is exploratory; it is not an independent holdout result.

## Final-test readiness

The confirmation data and predictions are limited to development training and GP7 validation. Before any official-test evaluation, freeze the model-selection decision, checkpoint hashes, metric code, and evaluation protocol; obtain explicit approval to access the official test partition. Model and coefficient SHA-256 values are recorded in the final metadata. This phase did not perform that evaluation.
