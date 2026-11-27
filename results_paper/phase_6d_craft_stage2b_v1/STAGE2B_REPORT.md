# CRAFT Stage 2B — full-training unscaled NMF diagnostic

## Scope and protocol

This was an exploratory follow-up motivated by weak Stage 2 reconstruction outside its 128-window fitting subset; it was not preregistered as a confirmatory experiment. The pre-execution choices and thresholds are recorded in `SCREENING_PROTOCOL.md`.

The run reused the saved AO-3, seed 20260928 Stage 1 nonnegative ReLU activations (988 train × 15 × 128; 432 GP7 validation × 15 × 128). It kept all 128 features and fit directly in original activation coordinates, without RMS scaling or feature exclusion. The saved Stage 1 activation and split hashes were verified. The frozen AO-3 checkpoint used for the conditional intervention matched Stage 1's recorded SHA-256: `5f96001dd3b9b9e5f87561f0444c845c6f441c6dd45a2129ef0bcaa9d893114f`.

The grouped training-only holdout was the lexicographically last training video, `gp_set_0006_vid_0002` (216 windows); the other three videos (772 windows) were used for candidate fits. Ranks 4, 8, and 16 were each fit with seeds 20260928 and 20260929, `nndsvda`, max 150 iterations, and tolerance 0.001. NNLS projected held-out video coefficients against each fixed basis. Stability is Hungarian-matched mean cosine similarity of unit-L2-normalized component vectors across the two seeds.

Before fitting, the engineering screen was set to mean grouped holdout relative Frobenius error ≤0.50 and matched cosine stability ≥0.90. These are pragmatic project screening thresholds, not universal CRAFT standards. The selection rule was the smallest rank passing both thresholds; GP7 was not used for rank selection. If none passed, the rank with the lowest grouped holdout error would be used only for diagnosis and no output intervention would run.

## Fixed-grid candidate results

Zero-reconstruction baseline is relative Frobenius error 1.0. Errors below are in original activation coordinates; coefficient sparsity is the exact-zero fraction.

| Rank | Seed | Training fit error | Grouped holdout error | Fit zero fraction | Holdout zero fraction | Iterations | Converged within cap? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 20260928 | 0.181899 | 0.198829 | 0.2403 | 0.3294 | 150 | No; cap warning |
| 4 | 20260929 | 0.181899 | 0.198829 | 0.2403 | 0.3294 | 150 | No; cap warning |
| 8 | 20260928 | 0.105454 | 0.163130 | 0.4392 | 0.6031 | 150 | No; cap warning |
| 8 | 20260929 | 0.105454 | 0.163130 | 0.4392 | 0.6031 | 150 | No; cap warning |
| 16 | 20260928 | 0.069057 | 0.126337 | 0.5001 | 0.5948 | 150 | No; cap warning |
| 16 | 20260929 | 0.069094 | 0.126180 | 0.4988 | 0.5943 | 150 | No; cap warning |

All six candidate fits reached the 150-iteration cap and emitted the recorded convergence warning. Matched factor stability was 1.000000 at rank 4, 1.000000 at rank 8, and 0.999450 at rank 16 (minimum matched cosine at rank 16: 0.995840). Despite cap warnings, the two-seed matches are highly similar; the warnings remain a limitation on claiming numerical convergence.

All ranks pass the predefined error and stability thresholds. The training-only rule therefore selects the smallest passing rank, **rank 4**. Its grouped holdout error is 0.198829 and its cross-seed matched mean cosine is 1.000000.

## Selected rank refit on all training windows

Rank 4 was refit from scratch on all 988 training windows (14,820 temporal-token rows) using both fixed seeds. Both all-training fits converged without warnings after 62 iterations. Each seed had training relative error 0.175750 and exact-zero coefficient fraction 0.2425. Projection of all 432 GP7 windows (6,480 token rows) by NNLS yielded validation relative error 0.371239 and exact-zero fraction 0.3802 for each seed. Zero reconstruction remains 1.0. Full-refit factor stability was 1.000000.

## Conditional trajectory-output intervention

Because the training-only quality/stability gates passed, a deterministic subset of 32 GP7 windows (first 32 by sorted sample ID) was evaluated with the validated FFN hook and the frozen checkpoint. The original no-hook output and identity-hook output matched exactly (maximum absolute error 0). Relative L2 deviation and coordinate MAE from the original model output were:

| Intervention | Relative L2 deviation | Coordinate MAE |
|---|---:|---:|
| Original activation (reference) | 0 | 0 |
| Identity-hook activation | 0 | 0 |
| Rank-4 NMF-reconstructed activation | 0.0027753 | 0.0045858 |
| Zero activation | 0.0085229 | 0.0151949 |

The reconstructed activation's output deviation was below the prespecified 0.05 relative-L2 engineering threshold, so it preserves this model's trajectory output on this 32-window development check. Zero activation also caused a small output change; therefore, this intervention alone does not establish that individual factors are functionally important or semantically meaningful. It only establishes low output discrepancy for this intervention and sample.

## Runtime, integrity, and conclusion

The single-process run completed in **42.15 seconds**, with numerical libraries and PyTorch limited to four CPU threads. No activation extraction was repeated, no model was trained or modified, and no official-test data were accessed. Rank-specific candidate coefficients, final full-training banks and coefficients, grouped holdout results, stability results, gate decision, and output comparisons are saved in this directory. `ARTIFACT_HASHES.csv` records the Stage 2B outputs and the Stage 1 inputs.

This result reverses the previous negative conclusion about feasibility of low-rank activation reconstruction: in original coordinates, the fixed rank-4 representation is stable across seeds, reconstructs the grouped training holdout and GP7 below the declared 0.50 threshold, and preserves trajectory outputs in the bounded intervention check. **Recursive decomposition and a carefully bounded development-only Sobol analysis are technically justified as a next investigation, but were not run here.** The findings do not establish faithful reproduction of the CRAFT method, semantic interpretation of concepts, human utility, generalization beyond these development splits, or causal importance of any factor. Candidate-grid convergence warnings and the small 32-window intervention limit the strength of the conclusion.
