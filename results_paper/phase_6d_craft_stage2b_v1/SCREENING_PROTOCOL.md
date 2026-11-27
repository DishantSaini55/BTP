# CRAFT Stage 2B — exploratory screening protocol

**Status:** Written before Stage 2B factorization; exploratory follow-up motivated by poor Stage 2 out-of-subset reconstruction. It is not a preregistered confirmatory experiment.

## Frozen inputs and scope

Reuse the saved AO-3 seed 20260928 Stage 1 ReLU FFN activations only (`train`: 988 × 15 × 128; `validation`: 432 × 15 × 128). Keep all 128 activation features in original nonnegative coordinates. No feature scaling, feature exclusion, additional activation capture, official-test data, training, or parameter/model changes.

## Candidate fits

NMF ranks are fixed at 4, 8, and 16; initialization is `nndsvda`; fixed seeds are 20260928 and 20260929; iteration cap 150; tolerance 0.001; CPU threads ≤4. Numerical warnings will be recorded. NNLS projection uses the fitted component basis without refitting on the held-out group or GP7.

## Grouped training-only assessment

Sort the four training video IDs lexicographically and hold out the last video as the grouped training-only holdout. Fit each candidate only on all windows from the other three videos. Report fit and NNLS projected holdout relative Frobenius errors in original activation coordinates, with zero reconstruction defined as 1.0. This group is used only for rank screening; GP7 metrics are descriptive and cannot select rank.

For each rank, compare the two fitted component banks by normalizing each component to unit L2 norm, calculating pairwise cosine similarity, and using the Hungarian assignment to maximize mean matched similarity. Stability is the mean matched cosine.

## Engineering screening thresholds and rank rule

A candidate passes the pragmatic engineering screen only if the mean grouped holdout relative error across the two seeds is ≤0.50 **and** matched factor stability is ≥0.90. These are project-specific engineering screening thresholds, not universal CRAFT standards. Select the smallest passing rank. If none passes, select the rank with the lowest mean grouped holdout error (tie: smaller rank) solely for a full-training diagnostic, mark the screen failed, and do not perform the conditional model-output intervention.

After selection, refit the selected rank with both fixed seeds on all 988 training windows and project all 432 GP7 validation windows by NNLS. Report all-training fit error and held-out GP7 error. No changes to ranks, scaling, thresholds, seeds, or iteration cap are allowed after viewing GP7 results.

## Conditional intervention gate

Only if the selected full-training bank retains the predeclared quality/stability screen (training-only grouped holdout ≤0.50 and stability ≥0.90), compare original, identity-hook, NNLS-reconstructed, and zero FFN activation interventions on the first 32 GP7 windows ordered by sample ID. Use the existing validated AO-3 activation hook and frozen checkpoint. Report output relative L2 deviation and coordinate MAE from the original model prediction. Treat reconstruction as adequate only if output relative L2 deviation ≤0.05; this is also an engineering screen, not a universal standard. Otherwise stop without Sobol analysis.

## Stop conditions

Run only the fixed grid and one selected-rank refit. Do not search further if results are poor. Preserve the Stage 1 and Stage 2 artifacts. Save configuration, results, coefficients, banks, runtime, input hashes, and the conditional-gate decision in this versioned directory.
