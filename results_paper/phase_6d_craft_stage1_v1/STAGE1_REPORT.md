# CRAFT recovery Stage 1: AO-3 activation and NMF pilot

Status: completed bounded minimal pilot; NMF held-out projection is degenerate
and does not support concept interpretation.

## Scope and provenance

AO-3, seed 20260928 only. Frozen checkpoint SHA-256:
`5f96001dd3b9b9e5f87561f0444c845c6f441c6dd45a2129ef0bcaa9d893114f`.
Development caches only: 988 training windows and 432 GP7 validation windows;
the cache manifest lists `official_test_sets_read: []`. Train and validation
cache SHA-256 values are recorded in `artifact_hashes.csv`.

## Validation

Six focused regression tests passed. They cover feature scaling/inverse scaling,
temporal tensor dimensions, identity activation intervention, exclusive lock
semantics, atomic marker replacement, and NNLS recovery on a known synthetic
factorization. Model identity intervention had maximum absolute output error
0.0 on the 432 validation windows.

## Activation capture

Hook: `encoder_layers[2].feedforward[1]`, the actual ReLU output. Captured shape
is `(988,15,128)` for train and `(432,15,128)` for validation. Temporal token `t`
remains aligned with observed input frame `t`. Train activation range was
`[0, 2.1875253]`, negative fraction 0, zero fraction 0.5271693.

## Bounded factorization pilot

Fixed sample rule: stable lexicographic sample-ID order, first 96 train windows,
all 15 temporal tokens. Fit ranks 4 and 8 using deterministic NNDSVDa, seed
20260928, feature-wise RMS scaling, max 120 iterations, tolerance 1e-3. Rank 4
relative train reconstruction error was 0.39036 after 27 iterations with no
warning. Rank 8 error was 0.29474 at the 120-iteration cap and raised
`ConvergenceWarning`.

Held-out NNLS ran on 6,480 validation temporal tokens. Relative reconstruction
error in scaled feature space was 1.0 (0.99999999975), indicating a degenerate
projection for this pilot bank. This result is recorded as a failed
generalization check; no concepts are interpreted or named. The pilot used rank
4 as the fixed bounded fit solely to save a reproducible pilot artifact; rank
selection/stability is not established.

## Runtime and artifacts

Runner-reported analysis runtime: 19.63 seconds. The six regression tests took
0.23 seconds. No training, weight changes, official-test access, raw video,
AO-2, recursive decomposition, Sobol analysis, insertion/deletion, or figures
were used in this stage.

Activation and factorization arrays are in `AO3_FFN_activations.npz` and
`AO3_NMF_pilot.npz`; their hashes and the input hashes are in
`artifact_hashes.csv`. These are raw numerical artifacts for audit/recovery,
not publication evidence. Next action should diagnose why the held-out NNLS
coefficients collapse before any broader concept experiment.
