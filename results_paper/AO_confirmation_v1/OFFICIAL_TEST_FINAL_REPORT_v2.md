# Phase 6B Official-Test Evaluation Report

**Status:** complete.

- Eligible windows: 2,615
- Eligible videos: 13
- Track clusters: 861
- Official test sets: GP3, GP5, GP8, GP9
- Protocol: annotation-only 15 observed / 45 predicted frames
- No training, test-set tuning or architecture update was performed.

## Model means

| Model | Runs | Parameters | MSE | SD | Normalized MSE | C-MSE | CF-MSE | Improved vs CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AO-0 | 1 | 0 | 3997.237 | 0.000 | 0.101649 | 6031.335 | 27599.699 | 0.000% |
| AO-1 | 5 | 76804 | 8516.523 | 1145.123 | 0.131472 | 16072.016 | 57260.427 | 30.822% |
| AO-2 | 5 | 76804 | 3975.058 | 82.442 | 0.101344 | 5907.203 | 25938.171 | 46.554% |
| AO-3 | 5 | 68356 | 3962.073 | 82.551 | 0.101307 | 5877.412 | 25924.384 | 46.447% |
| AO-CP | 1 | 0 | 11558.828 | 0.000 | 0.185196 | 21746.627 | 80514.547 | 18.815% |
| AO-RIDGE | 1 | 11700 | 2889.718 | 0.000 | 0.057947 | 5244.799 | 24276.293 | 64.704% |

## Headline findings

- AO-RIDGE: MSE 2889.718; 27.707% better than CV.
- AO-2: MSE 3975.058; 0.555% better than CV.
- AO-3: MSE 3962.073; 0.880% better than CV.
- AO-1: MSE 8516.523; 113.060% worse than CV.
- AO-CP: MSE 11558.828.

## Predeclared paired comparisons

- AO-2 vs AO-1: mean first-minus-second -4541.465; 95% t interval [-5986.323, -3096.606].
- AO-2 vs AO-0: mean first-minus-second -22.179; 95% t interval [-124.544, 80.186].
- AO-3 vs AO-0: mean first-minus-second -35.165; 95% t interval [-137.666, 67.336].
- AO-3 vs AO-2: mean first-minus-second -12.986; 95% t interval [-48.337, 22.366].
- AO-RIDGE vs AO-0: mean first-minus-second -1107.519; 95% t interval [-1107.519, -1107.519].
- AO-RIDGE vs AO-2: mean first-minus-second -1085.340; 95% t interval [-1187.705, -982.975].
- AO-RIDGE vs AO-3: mean first-minus-second -1072.354; 95% t interval [-1174.855, -969.853].

## Track-cluster bootstrap

The resampling unit is `(video_id, track_id)`; overlapping windows belonging to the same track are aggregated before resampling.

- AO-2 vs AO-1: 95% [-5820.981, -4287.765] from 10,000 repetitions.
- AO-2 vs AO-0: 95% [-515.327, -282.596] from 10,000 repetitions.
- AO-3 vs AO-0: 95% [-506.376, -279.113] from 10,000 repetitions.
- AO-3 vs AO-2: 95% [-16.097, -2.739] from 10,000 repetitions.
- AO-RIDGE vs AO-0: 95% [-1530.668, -1082.711] from 10,000 repetitions.
- AO-RIDGE vs AO-2: 95% [-1210.421, -632.147] from 10,000 repetitions.
- AO-RIDGE vs AO-3: 95% [-1206.444, -636.446] from 10,000 repetitions.

## Fresh-seed sensitivity

- AO-1: mean 8049.347 +/- 541.653; individual MSE 8501.334961;7555.997559;7605.474609;8534.582031; change vs CV 101.373%.
- AO-2: mean 3970.820 +/- 94.565; individual MSE 3989.174072;4005.925781;4053.243652;3834.936035; change vs CV -0.661%.
- AO-3: mean 3959.348 +/- 95.062; individual MSE 3954.971924;3962.000244;4076.577393;3843.842285; change vs CV -0.948%.

## Runtime

The existing Phase 6A benchmark measured 99.049/114.574 ms for AO-2 and 49.395/102.235 ms for AO-3 at batches 1/8 on the Intel i5-8365U with four threads, 50 warm-ups and 500 timed iterations.

The official-test wall-clock inference time was not recorded separately and is not reported as a measured value.

## Limitations

- All 13 official-test videos with locally available pedestrian XML were included; GP3/GP5/GP8/GP9 membership is complete for the downloaded annotation set.
- Official raw videos were not needed and not opened.
- The two fixed GP7 validation videos influenced ridge alpha selection.
- Five seeds are stochastic realizations on the same test sample set.
- Track bootstrap intervals are conditional on these 861 test clusters.
- No equivalence claim is made.
