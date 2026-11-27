# Corrected E3 observed gate-input ranges

Versioned descriptive diagnostic. The SAGE gate receives the last observed normalized bbox,
frame index 14 (zero-based; frame 15 of the observation).
Normalization mean/std are calculated from train boxes only.

## Per-coordinate statistics

Quantiles are q00, q01, q05, q25, q50, q75, q95, q99, q100; SD is population SD (`ddof=0`).
Outside fraction counts values strictly outside `[-5, 5]`.

| Split | Coordinate | Raw min..max px | Normalized min..max | Mean | SD | q01 | q05 | q50 | q95 | q99 | Outside ±5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | bbox_coord_0 | 439.459991..1430.079956 | -2.077980..2.561285 | 0.119409 | 0.752327 | -1.555024 | -1.026308 | 0.072823 | 1.516167 | 2.511497 | 0.000000 |
| train | bbox_coord_1 | 872.729980..955.140015 | -2.406800..1.672371 | 0.224321 | 0.664952 | -1.962688 | -0.996315 | 0.343089 | 1.144939 | 1.323178 | 0.000000 |
| train | bbox_coord_2 | 499.519989..1464.000000 | -1.968096..2.723383 | 0.104420 | 0.762203 | -1.434440 | -0.994350 | 0.021774 | 1.549309 | 2.629903 | 0.000000 |
| train | bbox_coord_3 | 925.250000..1075.060059 | -1.801909..3.370805 | -0.195049 | 0.703861 | -1.601650 | -1.101879 | -0.313904 | 1.063783 | 2.205880 | 0.000000 |
| validation | bbox_coord_0 | 100.220001..1493.500000 | -3.666706..2.858293 | -0.710692 | 1.554014 | -3.476090 | -2.972062 | -1.085517 | 1.893396 | 2.555659 | 0.000000 |
| validation | bbox_coord_1 | 752.440002..849.289978 | -8.360971..-3.567045 | -5.717073 | 1.242893 | -8.298187 | -8.014333 | -5.533618 | -3.824633 | -3.569747 | 0.744186 |
| validation | bbox_coord_2 | 171.779999..1550.800049 | -3.562308..3.145600 | -0.554185 | 1.605095 | -3.372024 | -2.665942 | -0.992083 | 2.173673 | 2.930310 | 0.000000 |
| validation | bbox_coord_3 | 887.799988..1050.609985 | -3.095001..2.526581 | -0.596471 | 1.303346 | -3.050914 | -2.926881 | -0.509508 | 1.435965 | 2.321959 | 0.000000 |
| test | bbox_coord_0 | 302.070007..1863.810059 | -2.721403..4.592526 | 1.329954 | 1.502140 | -2.286953 | -1.177093 | 1.027446 | 4.019285 | 4.523241 | 0.000000 |
| test | bbox_coord_1 | 298.869995..540.650024 | -30.811997..-18.844253 | -20.716732 | 1.735323 | -27.120757 | -24.211375 | -20.196552 | -19.107586 | -18.976059 | 1.000000 |
| test | bbox_coord_2 | 481.779999..1920.000000 | -2.054388..4.941484 | 1.585565 | 1.529230 | -1.652916 | -0.349875 | 1.254426 | 4.333209 | 4.923428 | 0.000000 |
| test | bbox_coord_3 | 585.690002..848.380005 | -13.526399..-4.456113 | -11.074026 | 1.797610 | -13.448945 | -12.742397 | -11.651158 | -7.178959 | -5.886890 | 0.990783 |

## Aggregate normalized ranges

| Split | n windows | Min | Max | Mean | SD | q01 | q05 | q50 | q95 | q99 | Outside ±5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 212 | -2.406800 | 3.370805 | 0.063275 | 0.738587 | -1.707013 | -1.046035 | 0.017123 | 1.218730 | 2.360668 | 0.000000 |
| validation | 43 | -8.360971 | 3.145600 | -1.894605 | 2.632944 | -8.075830 | -6.584867 | -1.383353 | 1.939969 | 2.698338 | 0.186047 |
| test | 217 | -30.811997 | 4.941484 | -7.218810 | 9.466952 | -24.319885 | -21.884896 | -3.588758 | 3.736711 | 4.612914 | 0.497696 |

## Historical diagnostic correction

The historical `MTN/run_e3.py:x_t_ranges` indexed `sample['bbox'][-1]`, the final
frame of the 60-frame sample. The gate input is `inp[:, -1]` after the 15-frame
observation slice, corresponding to `sample['bbox'][observe_length - 1]`.
Thus the old range included future-target coordinates and was not a measurement of
the gate input. This is a diagnostic frame-selection error; by itself it does not
establish training leakage or invalidate the saved model metrics.

The test split is included only to report descriptive observed-input ranges. Its
statistics were not used to choose clipping thresholds, gate inputs, or
hyperparameters. The historical E3 report and output files are preserved.

Machine-readable detail: `results_paper/diagnostics/E3_v2_observed_input_ranges.json`.
