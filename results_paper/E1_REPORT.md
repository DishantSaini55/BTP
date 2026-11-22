# E1 from-scratch report

Generated from 50 immutable E1 run directories. The test split is GP6 video 3 (`217` windows); H1 and H2 are the only confirmatory hypotheses and use a two-test Holm correction. Every other comparison is exploratory.

## Aggregate summary

| Model | MSE mean ± SD | C-MSE mean ± SD | CF-MSE mean ± SD | Epochs mean ± SD |
|---|---:|---:|---:|---:|
| MTN (baseline) | 3265.529449 ± 1159.103851 | 3033.020175 ± 1150.258903 | 6043.768731 ± 2520.717969 | 40.10 ± 8.12 |
| PACE-Net without SATIE (SAGE + MASC) | 4200.780387 ± 2124.625058 | 3969.733803 ± 2068.106094 | 7560.103792 ± 3110.172872 | 35.20 ± 11.02 |
| PACE-Net (SAGE + MASC + SATIE) | 4221.712756 ± 2406.945424 | 3954.003750 ± 2384.442833 | 7330.911971 ± 3897.228204 | 34.30 ± 11.87 |
| PACE-Net with ROUTE | 5249.691266 ± 3392.133608 | 4951.514335 ± 3226.021836 | 8328.732830 ± 4278.945772 | 31.80 ± 9.11 |
| PACE-Net with ROUTE and SATIE | 4191.840883 ± 3167.570534 | 3921.489050 ± 2964.277347 | 7147.624957 ± 4536.314936 | 36.10 ± 11.02 |

## Confirmatory H1/H2

- **H1** (PACE-Net without SATIE (SAGE + MASC) vs MTN (baseline)): mean paired difference 935.250938; 95% t CI [-736.378284, 2606.880161]; d_z=0.400231; paired t p=0.237422; Holm-adjusted p=0.474845; 2 of 10 seeds had lower left-model error.
- **H2** (PACE-Net (SAGE + MASC + SATIE) vs PACE-Net without SATIE (SAGE + MASC)): mean paired difference 20.932369; 95% t CI [-534.732553, 576.597290]; d_z=0.026948; paired t p=0.933954; Holm-adjusted p=0.933954; 5 of 10 seeds had lower left-model error.

## Secondary track-cluster bootstrap

- PACE-Net without SATIE (SAGE + MASC) minus MTN (baseline): observed difference 935.250938; 95% CI [559.763134, 1241.417100], 24 pedestrian-track clusters, 10000 resamples. All test tracks are from one video; this is exploratory and is not cross-video inference.
- PACE-Net (SAGE + MASC + SATIE) minus PACE-Net without SATIE (SAGE + MASC): observed difference 20.932369; 95% CI [-57.517230, 115.768491], 24 pedestrian-track clusters, 10000 resamples. All test tracks are from one video; this is exploratory and is not cross-video inference.

## Per-seed E1 values

| Seed | Model | MSE | C-MSE | CF-MSE | Epochs completed |
|---:|---|---:|---:|---:|---:|
| 20260928 | MTN (baseline) | 2952.866494 | 2702.244876 | 5403.905335 | 35 |
| 20260928 | PACE-Net without SATIE (SAGE + MASC) | 3397.664155 | 3134.091865 | 6688.347790 | 40 |
| 20260928 | PACE-Net (SAGE + MASC + SATIE) | 2513.738554 | 2334.611850 | 5423.553947 | 41 |
| 20260928 | PACE-Net with ROUTE | 2529.745238 | 2380.503475 | 6264.977954 | 32 |
| 20260928 | PACE-Net with ROUTE and SATIE | 2175.394426 | 1995.547041 | 5023.226580 | 32 |
| 20260929 | MTN (baseline) | 3659.327432 | 3274.169808 | 4739.749865 | 23 |
| 20260929 | PACE-Net without SATIE (SAGE + MASC) | 2899.541537 | 2722.255254 | 2940.391057 | 33 |
| 20260929 | PACE-Net (SAGE + MASC + SATIE) | 2395.611777 | 2079.009412 | 3831.331590 | 21 |
| 20260929 | PACE-Net with ROUTE | 2908.172891 | 2694.193170 | 2842.048410 | 39 |
| 20260929 | PACE-Net with ROUTE and SATIE | 2165.010121 | 2031.616975 | 4671.219567 | 31 |
| 20260930 | MTN (baseline) | 2693.573960 | 2603.096858 | 6487.051688 | 44 |
| 20260930 | PACE-Net without SATIE (SAGE + MASC) | 3590.978615 | 3487.210249 | 7863.171469 | 25 |
| 20260930 | PACE-Net (SAGE + MASC + SATIE) | 2701.974261 | 2605.028206 | 6646.806538 | 32 |
| 20260930 | PACE-Net with ROUTE | 2783.336191 | 2700.461532 | 6559.383111 | 42 |
| 20260930 | PACE-Net with ROUTE and SATIE | 2574.180698 | 2486.910942 | 5886.928771 | 41 |
| 20260931 | MTN (baseline) | 2802.911348 | 2408.023866 | 5403.922238 | 45 |
| 20260931 | PACE-Net without SATIE (SAGE + MASC) | 3141.287225 | 2917.750585 | 6133.047363 | 50 |
| 20260931 | PACE-Net (SAGE + MASC + SATIE) | 3553.783077 | 3288.673016 | 6869.887264 | 44 |
| 20260931 | PACE-Net with ROUTE | 3598.221797 | 3258.553327 | 3910.927129 | 45 |
| 20260931 | PACE-Net with ROUTE and SATIE | 3135.211992 | 2723.096283 | 4432.619954 | 44 |
| 20260932 | MTN (baseline) | 3045.557120 | 2838.070963 | 3943.548639 | 50 |
| 20260932 | PACE-Net without SATIE (SAGE + MASC) | 9908.223029 | 9545.195466 | 15377.714295 | 18 |
| 20260932 | PACE-Net (SAGE + MASC + SATIE) | 10538.144391 | 10232.147336 | 16299.802492 | 18 |
| 20260932 | PACE-Net with ROUTE | 10923.352108 | 10055.808818 | 15721.520262 | 19 |
| 20260932 | PACE-Net with ROUTE and SATIE | 12139.545803 | 11246.991563 | 17300.144421 | 19 |
| 20260933 | MTN (baseline) | 2654.917517 | 2516.023833 | 5480.932788 | 41 |
| 20260933 | PACE-Net without SATIE (SAGE + MASC) | 2789.517915 | 2625.965511 | 7778.100426 | 44 |
| 20260933 | PACE-Net (SAGE + MASC + SATIE) | 4494.621685 | 4357.435920 | 11226.555655 | 37 |
| 20260933 | PACE-Net with ROUTE | 5955.276334 | 5834.985167 | 12217.153422 | 28 |
| 20260933 | PACE-Net with ROUTE and SATIE | 6790.887189 | 6634.127466 | 13329.609922 | 28 |
| 20260934 | MTN (baseline) | 3763.713146 | 3489.315701 | 7304.209582 | 39 |
| 20260934 | PACE-Net without SATIE (SAGE + MASC) | 5243.781869 | 4926.452491 | 6876.312154 | 23 |
| 20260934 | PACE-Net (SAGE + MASC + SATIE) | 5309.100283 | 5020.919701 | 8302.259162 | 23 |
| 20260934 | PACE-Net with ROUTE | 5877.062444 | 5452.137695 | 7931.810261 | 23 |
| 20260934 | PACE-Net with ROUTE and SATIE | 4724.925020 | 4392.429237 | 7355.387831 | 23 |
| 20260935 | MTN (baseline) | 2791.639024 | 2563.302607 | 5304.215925 | 50 |
| 20260935 | PACE-Net without SATIE (SAGE + MASC) | 3422.194321 | 3323.464236 | 8138.139701 | 50 |
| 20260935 | PACE-Net (SAGE + MASC + SATIE) | 3079.520839 | 2861.300784 | 6806.779164 | 50 |
| 20260935 | PACE-Net with ROUTE | 4366.911686 | 4147.617995 | 9589.671403 | 34 |
| 20260935 | PACE-Net with ROUTE and SATIE | 1825.049201 | 1732.260024 | 3937.239692 | 50 |
| 20260936 | MTN (baseline) | 6249.603594 | 6051.533946 | 12568.945456 | 34 |
| 20260936 | PACE-Net without SATIE (SAGE + MASC) | 4141.544987 | 3859.429534 | 6918.147881 | 35 |
| 20260936 | PACE-Net (SAGE + MASC + SATIE) | 4229.926204 | 3697.950522 | 4424.338767 | 27 |
| 20260936 | PACE-Net with ROUTE | 11432.467818 | 11052.159476 | 13307.354065 | 20 |
| 20260936 | PACE-Net with ROUTE and SATIE | 3740.104813 | 3496.388795 | 5921.019704 | 50 |
| 20260937 | MTN (baseline) | 2041.184853 | 1884.419288 | 3801.205797 | 40 |
| 20260937 | PACE-Net without SATIE (SAGE + MASC) | 3473.070217 | 3155.522842 | 6887.665781 | 34 |
| 20260937 | PACE-Net (SAGE + MASC + SATIE) | 3400.706485 | 3062.960754 | 3477.805128 | 50 |
| 20260937 | PACE-Net with ROUTE | 2122.366150 | 1938.722692 | 4942.482284 | 36 |
| 20260937 | PACE-Net with ROUTE and SATIE | 2648.099572 | 2475.522173 | 3618.853130 | 43 |

## B1 matched-protocol baselines

| B1 model | Runs | MSE mean | C-MSE mean | CF-MSE mean | Epochs |
|---|---:|---:|---:|---:|---|
| constant_position | 1 | 4696.976911 | 4597.240027 | 13682.455598 | n/a |
| constant_velocity_last_3_frames | 1 | 1496.018040 | 655.570860 | 1948.649039 | n/a |
| lstm_boxes_plus_ego_speed | 10 | 371162.326513 | 210857.647737 | 583259.608595 | 15, 32, 38, 39, 42, 50, 50, 50, 50, 50 |
| lstm_boxes_plus_ego_speed_plus_scene_labels | 10 | 408545.147304 | 245176.129909 | 675010.659539 | 15, 27, 32, 33, 38, 40, 42, 44, 50, 50 |

## Anomalies and validity

- No non-finite MSE, C-MSE, or CF-MSE values were found.
- Eight E1 runs completed all 50 epochs: MTN seeds 20260932 and 20260935; PACE-Net SATIE seeds 20260935 and 20260937; PACE-Net without SATIE seeds 20260931 and 20260935; PACE-Net with ROUTE and SATIE seeds 20260935 and 20260936.
- Seed 20260936 has large-seed concern for MTN (6249.603594 MSE) and is therefore the largest visible source of seed variance, not a protocol violation.
- A one-seed or one-run anomaly must not be silently removed; it is retained and reported.
- Two-window moving-block bootstrap sensitivity was generated but is not confirmatory; all test tracks are from one video.
- B2/B3 published baselines remain excluded as full-IDD-PeD published results and are not comparable.
