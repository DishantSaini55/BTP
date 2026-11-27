# Phase 6D CRAFT recovery status

Status: implementation recovery incomplete; no experimental outputs are valid.

The four earlier partial directories were inspected and preserved. None had a
complete stage marker or independently verifiable completed results. The frozen
AO checkpoint hashes were verified:

- AO-2 seed 20260928: `1d933f60c0201c30e75bbfd765ad450e4347c33516067d8864d48398cd304cdf`
- AO-3 seed 20260928: `5f96001dd3b9b9e5f87561f0444c845c6f441c6dd45a2129ef0bcaa9d893114f`

The development split manifest declares 988 train and 432 validation windows,
four train videos and two GP7 validation videos, with `official_test_sets_read`
empty. No active Python processes were present at inspection.

The previous warnings were sklearn NMF `ConvergenceWarning`s at fixed caps of
160/300 iterations. Earlier attempts ran NMF on 14,820 temporal activation rows
without feature scaling. The new v2 draft adds column-norm scaling and a
deterministic video-balanced row sample, but it has only passed syntax
compilation. It has not passed the requested regression suite or end-to-end
smoke tests and has not been executed on data. Do not treat it as validated.

Remaining before execution: finish mathematically correct inverse scaling in
all reconstruction and intervention paths; implement and pass the 13 required
focused tests; add per-stage atomic resume markers and heartbeat/budget guards;
verify synthetic Sobol and ordering cases; then run a smoke and the sequential
AO-3-first experiment under one process. No experiment claim is supported by
this recovery attempt.
