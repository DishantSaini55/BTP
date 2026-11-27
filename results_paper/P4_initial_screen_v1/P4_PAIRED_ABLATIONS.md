# Phase 4 Paired Ablation Report

Primary paired contrasts use seed-level validation raw-pixel MSE and
normalized MSE. Negative deltas favor the first-named model. The
t-intervals and standardized paired effect sizes are descriptive with
three seeds; no confirmatory significance claim is made. Track-cluster
bootstrap intervals are secondary and based on one validation video.

| Comparison | Metric | Mean paired delta | t 95% CI | d_z | First wins | Track bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| MTN vs deterministic CV | normalized_mse | -0.1073 | [-0.5754, 0.3608] | -0.5693574670140288 | 2/3 | n/a |
| MTN vs deterministic CV | raw_pixel_mse | 935.86 | [-4779, 6651] | 0.4068005472478293 | 2/3 | [-85.88, 1966] |
| MTN vs deterministic CV | cmse | 603.78 | [-4841, 6048] | 0.2754744531256289 | 2/3 | [-318, 1511] |
| MTN vs deterministic CV | cfmse | -3403.9 | [-5798, -1010] | -3.5323114185365516 | 3/3 | [-6635, -943.3] |
| SATIE MTN vs deterministic CV | normalized_mse | -0.20452 | [-0.3472, -0.06188] | -3.561690898205148 | 3/3 | n/a |
| SATIE MTN vs deterministic CV | raw_pixel_mse | -374.16 | [-807.6, 59.32] | -2.1441881211703966 | 3/3 | [-1066, 229.7] |
| SATIE MTN vs deterministic CV | cmse | -590.18 | [-950.4, -229.9] | -4.069402649548492 | 3/3 | [-1243, -60.59] |
| SATIE MTN vs deterministic CV | cfmse | -4269.1 | [-5408, -3131] | -9.315078647093639 | 3/3 | [-6825, -2271] |
| Always-open context without SATIE vs deterministic CV | normalized_mse | -0.17171 | [-0.3879, 0.04449] | -1.9729855713313544 | 3/3 | n/a |
| Always-open context without SATIE vs deterministic CV | raw_pixel_mse | -77.591 | [-1273, 1118] | -0.16126890552163126 | 2/3 | [-947.9, 680] |
| Always-open context without SATIE vs deterministic CV | cmse | -334.86 | [-1762, 1092] | -0.5829284380741523 | 2/3 | [-1084, 343.7] |
| Always-open context without SATIE vs deterministic CV | cfmse | -4730.5 | [-5942, -3519] | -9.701500848013202 | 3/3 | [-7696, -2454] |
| Always-open context with SATIE vs deterministic CV | normalized_mse | -0.18017 | [-0.3798, 0.01945] | -2.2421190450504014 | 3/3 | n/a |
| Always-open context with SATIE vs deterministic CV | raw_pixel_mse | 189.59 | [-1823, 2202] | 0.2340070696362983 | 2/3 | [-682.4, 977.1] |
| Always-open context with SATIE vs deterministic CV | cmse | -139.79 | [-2064, 1785] | -0.1804229113696542 | 2/3 | [-906, 532.9] |
| Always-open context with SATIE vs deterministic CV | cfmse | -4160.1 | [-5258, -3063] | -9.415095258436807 | 3/3 | [-7215, -1840] |
| Original hard SAGE vs deterministic CV | normalized_mse | -0.14094 | [-0.4717, 0.1898] | -1.058588289144363 | 2/3 | n/a |
| Original hard SAGE vs deterministic CV | raw_pixel_mse | 1465.2 | [-5829, 8760] | 0.49899034806426323 | 2/3 | [519.1, 2410] |
| Original hard SAGE vs deterministic CV | cmse | 1115.8 | [-6025, 8257] | 0.38813551892609677 | 2/3 | [261.4, 1978] |
| Original hard SAGE vs deterministic CV | cfmse | -3003.2 | [-7390, 1383] | -1.7006999326990067 | 3/3 | [-5889, -730.5] |
| Differentiable context-only SAGE soft vs deterministic CV | normalized_mse | -0.13984 | [-0.6145, 0.3348] | -0.7318839213544236 | 2/3 | n/a |
| Differentiable context-only SAGE soft vs deterministic CV | raw_pixel_mse | 2144.1 | [-8541, 1.283e+04] | 0.4984871401019764 | 2/3 | [969.1, 3425] |
| Differentiable context-only SAGE soft vs deterministic CV | cmse | 1865 | [-8576, 1.231e+04] | 0.44374187393879594 | 2/3 | [723.8, 3073] |
| Differentiable context-only SAGE soft vs deterministic CV | cfmse | -2873 | [-9471, 3725] | -1.0816010530981472 | 2/3 | [-6221, -192.1] |
| SATIE effect on MTN | normalized_mse | -0.097224 | [-0.4318, 0.2373] | -0.7219267992901712 | 3/3 | n/a |
| SATIE effect on MTN | raw_pixel_mse | -1310 | [-7454, 4834] | -0.5296907602919723 | 1/3 | [-1939, -793.6] |
| SATIE effect on MTN | cmse | -1194 | [-6789, 4401] | -0.5300809974998073 | 1/3 | [-1775, -712.7] |
| SATIE effect on MTN | cfmse | -865.15 | [-4284, 2554] | -0.6286392116774683 | 2/3 | [-2475, 673.6] |
| SATIE effect on always-open context | normalized_mse | -0.0084679 | [-0.04887, 0.03193] | -0.5206558009725332 | 2/3 | n/a |
| SATIE effect on always-open context | raw_pixel_mse | 267.18 | [-557.1, 1091] | 0.8051929785222478 | 0/3 | [124.6, 458.3] |
| SATIE effect on always-open context | cmse | 195.07 | [-302.8, 693] | 0.9732987038492479 | 0/3 | [73.52, 351.4] |
| SATIE effect on always-open context | cfmse | 570.39 | [445.2, 695.6] | 11.319798540748321 | 0/3 | [197.7, 1057] |
| Always-open context without SATIE vs MTN | normalized_mse | -0.064411 | [-0.3848, 0.256] | -0.49941137040142386 | 2/3 | n/a |
| Always-open context without SATIE vs MTN | raw_pixel_mse | -1013.5 | [-5536, 3509] | -0.556715444789699 | 2/3 | [-1451, -640] |
| Always-open context without SATIE vs MTN | cmse | -938.63 | [-4959, 3081] | -0.5800081699373679 | 2/3 | [-1342, -579.5] |
| Always-open context without SATIE vs MTN | cfmse | -1326.6 | [-2686, 32.57] | -2.424607570131703 | 3/3 | [-2729, -208.9] |
| Always-open context with SATIE vs MTN with SATIE | normalized_mse | 0.024345 | [-0.03832, 0.08701] | 0.9650281516771371 | 1/3 | n/a |
| Always-open context with SATIE vs MTN with SATIE | raw_pixel_mse | 563.75 | [-1879, 3006] | 0.5733389202120233 | 1/3 | [307.1, 855.4] |
| Always-open context with SATIE vs MTN with SATIE | cmse | 450.39 | [-1629, 2530] | 0.5381003580280956 | 1/3 | [230.8, 692.6] |
| Always-open context with SATIE vs MTN with SATIE | cfmse | 108.92 | [-1861, 2079] | 0.1373726769449602 | 2/3 | [-666, 793.2] |
| Original hard SAGE vs independently trained always-open | normalized_mse | 0.039234 | [-0.1289, 0.2074] | 0.579667100628911 | 1/3 | n/a |
| Original hard SAGE vs independently trained always-open | raw_pixel_mse | 1275.6 | [-4014, 6566] | 0.5990417279050128 | 1/3 | [952.3, 1652] |
| Original hard SAGE vs independently trained always-open | cmse | 1255.6 | [-3996, 6507] | 0.5939106197557675 | 1/3 | [939.6, 1619] |
| Original hard SAGE vs independently trained always-open | cfmse | 1156.9 | [-2282, 4596] | 0.8357716028656789 | 0/3 | [670.8, 1694] |
| Differentiable context-only SAGE soft vs always-open | normalized_mse | 0.040339 | [-0.2552, 0.3358] | 0.33911171811666 | 2/3 | n/a |
| Differentiable context-only SAGE soft vs always-open | raw_pixel_mse | 1954.5 | [-6725, 1.063e+04] | 0.5593945185442482 | 1/3 | [1327, 2698] |
| Differentiable context-only SAGE soft vs always-open | cmse | 2004.8 | [-6536, 1.055e+04] | 0.5831149300371905 | 1/3 | [1364, 2803] |
| Differentiable context-only SAGE soft vs always-open | cfmse | 1287.2 | [-4281, 6856] | 0.5742062381931036 | 1/3 | [462.2, 2119] |
| Differentiable context-only SAGE soft vs original hard SAGE | normalized_mse | 0.001105 | [-0.1447, 0.1469] | 0.018827376845397384 | 2/3 | n/a |
| Differentiable context-only SAGE soft vs original hard SAGE | raw_pixel_mse | 678.87 | [-2712, 4069] | 0.4974041680581442 | 2/3 | [317.8, 1108] |
| Differentiable context-only SAGE soft vs original hard SAGE | cmse | 749.24 | [-2554, 4053] | 0.5634116607296845 | 1/3 | [394.2, 1173] |
| Differentiable context-only SAGE soft vs original hard SAGE | cfmse | 130.23 | [-2203, 2463] | 0.13866353016158353 | 1/3 | [-393.5, 609.8] |
| P4-F hard inference minus soft inference | raw_pixel_mse | -69.389 | [-198.2, 59.4] | -1.3383947405312175 | 3/3 | [-93.38, -49.54] |
| P4-F hard inference minus soft inference | cmse | -73.889 | [-214.7, 66.96] | -1.3031380915404134 | 3/3 | [-96.51, -55.12] |
| P4-F hard inference minus soft inference | cfmse | -57.643 | [-266.9, 151.7] | -0.6841620736230927 | 2/3 | [-86.75, -32.16] |

The three-seed t intervals are highly uncertain and are not a substitute
for more independent videos. The 43 validation windows overlap in time;
pedestrian-track resampling does not resolve the single-video limitation.
All results are validation-only and exploratory.
