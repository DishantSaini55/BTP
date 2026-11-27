# NNLS diagnostic report — Stage 2 v1

## Finding

The Stage 1 held-out NNLS failure was a scaling pathology, not an NNLS orientation or coefficient-solver bug. Stage 1's basis is stored component-first with shape `(4, 128)`, so the correct SciPy call is `nnls(W.T, x)` and row coefficients reconstruct as `u @ W`. The old transform applied a feature scale as small as `1e-8` to 13 dimensions. This made the validation activation norm in scaled coordinates `2,591,539,144.35`; NNLS could fit only 128 rows at a time in the recorded diagnostic, producing relative error `0.9999999998` and 75% zero coefficients. On the original Stage 1 fit rows, the same NNLS orientation gave relative error `0.3216`, confirming the basis was projectable in its own scaled space.

The correction estimates per-feature RMS using only the deterministic, video-balanced training fit subset, defines a documented floor as `max(0.001 × maximum training RMS, 1e-6)`, and excludes features with RMS at or below that floor from the concept basis. It then applies the same training-derived scales to training and validation and reverses the scale only on retained features when reporting original-coordinate errors. The resulting floor was `0.0014841299`; 110/128 features were retained and 18 excluded. Validation scaled values and NNLS coefficients were finite. The largest absolute scaled validation activation was `890.2951`, so the floor prevents the previous `1e-8` explosion but the held-out scaling remains numerically high for some features.

## Interpretation

The two tested concept bases improve on the zero-reconstruction baseline, but validation gains remain modest: rank 4 reduces relative error from 1.0 to 0.9816 scaled / 0.9456 original; rank 8 reaches 0.9561 scaled / 0.9193 original. This correction fixes the pathological denominator; it does not establish that the concepts represent the validation activation distribution adequately. No recursion or semantic interpretation is justified from this result.
