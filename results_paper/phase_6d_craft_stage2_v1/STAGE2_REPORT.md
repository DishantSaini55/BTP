# CRAFT trajectory concept Stage 2 — executed results

## Execution and inputs

This run reused `results_paper/phase_6d_craft_stage1_v1/AO3_FFN_activations.npz` and the saved annotation-only train/validation split. All six Stage 1 input hashes were verified before fitting. No model forward pass, model training, official-test access, recursive decomposition, or Sobol analysis occurred. The single process completed in 12.34 seconds. PyTorch and numerical libraries were limited to four CPU threads.

Training RMS/scaling was estimated from 128 deterministic, video-balanced training windows: sort video identifiers, then take the first 32 sample identifiers per video. All 15 temporal tokens per selected window contributed to the fit. The floor was `max(0.001 × max(feature RMS), 1e-6)` = `0.0014841298587`. Features whose training RMS was at or below the floor were omitted; the identical fit-derived scales were used for train and GP7 validation projection. Full feature-by-feature retained scale information is in `FEATURE_SCALING_STATISTICS.csv`; rank-specific banks include feature indices, RMS, and scale.

NMF used ranks 4 and 8, `nndsvda`, seeds 20260928 and 20260929, at most 100 iterations, tolerance 0.001. Each configuration converged without a warning (41 iterations at rank 4; 69 at rank 8). Coefficients for the full saved train and GP7 validation activation tensors were obtained by fixed-basis nonnegative least squares. The small NNLS training-fit figures below are a projection of the same fitting subset; NMF's own fitting objective is separately listed in `NMF_CONFIGURATIONS.csv`.

## Reconstruction errors

Errors are relative Frobenius norms. “Scaled” means training-RMS-scaled coordinates over the 110 retained features. “Original” means inverse-scaled activation coordinates over those same retained features. The zero-reconstruction baseline is 1.0 in both coordinate systems.

| Rank | Set | Token rows | Scaled relative error | Original-coordinate relative error | Zero baseline | NNLS zero coefficient fraction |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Balanced training fit subset | 1,920 | 0.4993 | 0.1956 | 1.0000 | 0.3849 |
| 4 | All training activations | 14,820 | 0.9888 | 0.8967 | 1.0000 | 0.3789 |
| 4 | GP7 validation activations | 6,480 | 0.9816 | 0.9456 | 1.0000 | 0.3164 |
| 8 | Balanced training fit subset | 1,920 | 0.3557 | 0.1155 | 1.0000 | 0.5721 |
| 8 | All training activations | 14,820 | 0.9623 | 0.8314 | 1.0000 | 0.5513 |
| 8 | GP7 validation activations | 6,480 | 0.9561 | 0.9193 | 1.0000 | 0.5263 |

All configurations converged without warnings. The rank-4 and rank-8 mean matched absolute factor correlations across the two seeds were respectively 1.0000 and 1.0000 to reported precision. These strong stability scores are for the fitted component vectors under the documented absolute-correlation/Hungarian matching calculation; they do not prove semantic validity.

The predeclared training-only parsimony rule selects the smallest tested rank whose matched stability is at least 0.50, which is rank 4. Rank 8's lower training and validation reconstruction errors are reported as a comparison, not used to relabel the selection rule. Even rank 8's GP7 held-out error remains 0.9193 in original retained-feature coordinates (versus 1.0 for zero reconstruction); rank 4 is 0.9456. This is weak held-out representation. The observed validation maximum absolute scaled activation is 890.2951 despite finite values, another reason not to proceed to recursive factorization or concept attribution.

## Stage 1 error correction

Stage 1 did not fail because of component orientation: component-first `W` requires `nnls(W.T, x)`, and the resulting coefficients reconstruct with `u @ W`. The cause was applying a `1e-8` floor to 13 extremely low-RMS dimensions, which magnified validation coordinates and dominated the norm. The training-derived RMS floor and exclusion policy above correct that scale pathology while leaving the frozen AO-3 model, weights, split, and evaluation policy unchanged. This is an activation-representation correction, not test-driven model tuning: it uses development training activations only, and neither parameters nor rank selection use official-test results. The GP7 validation figures are held-out diagnostics, not tuning criteria.

## Regression tests and stop decision

The prior focused numerical regression run completed with 9/9 tests passing (6 Stage 1 checks and 3 Stage 2 checks, including NNLS orientation, scale/inverse-scale, and near-zero feature handling). The Stage 2 execution itself also asserted the Stage 1 artifact hashes and wrote finite, nonnegative fixed-basis coefficients. The first attempted launch stopped before fitting because the hash strings differed only in letter case; the guard was normalized to compare SHA-256 hex case-insensitively, then all hashes verified. No fitting output was produced by that failed launch.

**Stage 2 numerical execution is complete. Concept quality is inadequate for Stage 3. Stop here:** no recursion, Sobol, insertion/deletion, semantic labels, or further rank search is warranted from this evidence.
