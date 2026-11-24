# E1 epoch and split diagnostics

These reruns are diagnostics only. Test curves were saved but were never used to select a checkpoint.

## Aggregate checkpoint gap

| Model | Runs | Mean selected epoch | Mean test-optimal epoch | Mean absolute gap | Max absolute gap | Seeds with |gap| > 5 |
|---|---:|---:|---:|---:|---:|---:|
| MTN (baseline) | 10 | 35.50 | 30.00 | 7.10 | 12 | 7 |
| PACE-Net (SAGE + MASC + SATIE) | 10 | 23.00 | 21.80 | 7.40 | 18 | 5 |
| PACE-Net without SATIE (SAGE + MASC) | 10 | 31.30 | 26.60 | 10.50 | 36 | 7 |

## MTN seed 20260932

- Validation-selected epoch: 34.
- Test-optimal epoch: 22.
- Epoch gap: -12.
- Selected test normalized MSE: 0.452402; test raw MSE: 3341.237506.
- Test-optimal normalized MSE: 0.245321; test raw MSE: 3602.225557.
- This diagnostic run stopped early at 44 epochs, not at 50. The original seed 20260932 run stopped at 50 epochs. The shift from GP6 validation video 1 to test video 3 makes the validation-selected checkpoint diagnostic evidence, but it is not by itself proof that seed 20260932 failed solely because of early stopping; the two reruns also differ in completed-epoch behavior.

## Normalized x_T ranges

- Train: {'min': -8.497092247009277, 'max': 8.950606346130371, 'abs_max': 8.950606346130371, 'exceeds_plus_minus_5': True}
- Validation: {'min': -23.226850509643555, 'max': 12.111340522766113, 'abs_max': 23.226850509643555, 'exceeds_plus_minus_5': True}
- Test: {'min': -38.396156311035156, 'max': 4.739763259887695, 'abs_max': 38.396156311035156, 'exceeds_plus_minus_5': True}

## Baseline errors by split

| Baseline | Train MSE / C-MSE / CF-MSE | Validation MSE / C-MSE / CF-MSE | Test MSE / C-MSE / CF-MSE |
|---|---|---|---|
| Constant position | 4923.146615 / 4801.150281 / 19888.327424 | 16953.994406 / 15852.451038 / 60252.517385 | 4696.976911 / 4597.240027 / 13682.455598 |
| Constant velocity (last 3) | 1513.182154 / 1390.725872 / 7250.118652 | 3630.315279 / 3074.487717 / 15769.099758 | 1496.018040 / 655.570860 / 1948.649039 |
