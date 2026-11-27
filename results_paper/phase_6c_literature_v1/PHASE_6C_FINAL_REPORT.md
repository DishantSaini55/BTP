# Phase 6C final report

## Status

Complete: literature/claim/novelty/feasibility review only. No model training,
inference, hyperparameter choice, downloads, architecture changes, manuscript
edit, commit or push occurred.

## Canonical empirical record

Phase 6B v2/v3 is canonical: 2,615 windows, 13 videos, 861 track clusters,
GP3/5/8/9 and five neural seeds. MSE: CV 3997.237; AO-1 8516.523 +/- 1145.123;
AO-2 3975.058 +/- 82.442; AO-3 3962.073 +/- 82.551; AO-RIDGE 2889.718; CP
11558.828. Ridge improves 27.707% over CV. Neural mean differences versus CV
are marginal and their reported seed-level t intervals include zero.

## Review conclusion

Only the AO models/baselines are directly comparable. MTN/PIETraj/BiTraP/SGNet
are relevant IDD-PeD/first-person literature but not matched to the annotation-
only protocol without a separately frozen reproduction. Fixed-camera,
multi-agent, world-coordinate or best-of-K methods are methodological context,
not numerical SOTA competitors.

## Novelty conclusion and direction

CV anchoring is supported as a formulation in this study; ridge, not the neural
transformer, is the best frozen predictor. SATIE is a compactness variant, not
an established accuracy advance. SAGE, MASC and ROUTE lack reliable empirical
support and should not be headline contributions. Recommend the comparative
empirical paper direction specified in `FINAL_RESEARCH_DIRECTION_RECOMMENDATION.md`.

## Remaining submission work

Authorised manuscript rewrite; protocol-safe published-results context table;
release/readme review; and, only if separately approved, qualitative pilot
materials or a fair external-baseline reproduction. Phase 6D was not started.
