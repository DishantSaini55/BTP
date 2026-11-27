# Complete Phase 3 Comparison

## Evidence reviewed

This comparison uses the Phase 3 reports, all 14 rows in
`P3_initial_screen_v1/P3_VALIDATION_SUMMARY.csv`, all 602 per-window policy
rows, `P3_PER_WINDOW_AGGREGATES.json`, all seven run configurations and status
files, and all six trainable run histories and selected-checkpoint metadata.
The history files each contain 10 epochs. The P3-B/P3-C initialization
SHA-256 values match exactly; the P3-D/P3-E values also match exactly. Their
residual anchor is the sole declared experimental difference within each
pair. The historical E7 CSV independently records 133,700 MTN parameters,
139,142 PACE-Net-without-SATIE parameters, and 130,694 PACE-Net-with-SATIE
parameters.

The evaluation is validation-only: 43 overlapping windows from
`gp_set_0006_vid_0001`, one seed (`20260928`). Results do not support
statistical significance or generalization claims.

## Selected-checkpoint comparison

Raw MSE, C-MSE, and CF-MSE are pixel-space quantities. The normalized loss
is used for checkpoint selection; raw pixel MSE is the primary descriptive
comparison here. “Mean correction RMS” is the mean over windows of the RMS
prediction difference from deterministic CV. For P3-B, which uses a
constant-position anchor, that difference includes both the anchor
difference and the learned residual; it is not a pure neural correction.

| Model / policy | Norm. train MSE | Norm. val MSE | Raw val MSE | C-MSE | CF-MSE | Params | Selected epoch | Train-loop seconds | Mean correction RMS px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic CV | 0.342556 | 1.218765 | 3,630.315 | 3,074.488 | 15,769.100 | 0 | n/a | 0.00 | 0.00 |
| Original MTN, CP anchor (P3-B) | 0.133812 | 1.006105 | 7,106.317 | 6,554.383 | 20,871.902 | 133,700 | 9 | 414.57 | 65.73 |
| MTN-CV (P3-C) | 0.257023 | 1.009737 | 3,341.441 | 2,596.374 | 11,865.537 | 133,700 | 8 | 278.89 | 34.90 |
| Original PACE-Net, CP anchor, hard (P3-D) | 0.112770 | 0.999297 | 8,010.100 | 7,344.731 | 17,571.076 | 130,694 | 10 | 267.60 | 78.22 |
| PACE-Net-CV, hard (P3-E) | 0.251217 | 1.011788 | 3,275.395 | 2,414.860 | 11,814.230 | 130,694 | 8 | 294.84 | 33.50 |
| Always-open PACE-Net-CV (P3-F) | 0.249132 | 1.035673 | 3,379.496 | 2,585.675 | 11,596.665 | 130,068 | 8 | 262.82 | 37.82 |
| Context-only SAGE V2-CV, soft (P3-G) | 0.256166 | 0.996157 | 3,106.417 | 2,421.371 | 11,851.799 | 130,630 | 8 | 270.82 | 31.66 |
| Context-only SAGE V2-CV, hard (P3-G) | 0.256166 | 0.992688 | 3,082.826 | 2,394.186 | 11,889.572 | 130,630 | 8 | 270.82 | 31.04 |

P3-G hard has the lowest raw and normalized validation MSE in this table.
That is a factual ranking for this one seed and validation video, not a
general “best model” conclusion. Its hard gate admitted every validation
window. The hard result is exactly its forced-open result; the ranking
therefore does not show that sample-selective hard admission produced the
gain.

## Per-window CV comparison and correction distributions

Counts are exact pre-rounding comparisons from the finalized aggregate JSON;
they avoid false differences caused by the CSV's finite decimal formatting.
Each per-window correction statistic is based on 43 windows.

| Model / policy | Better than CV | Worse than CV | Equal | Correction RMS mean ± SD | Median [Q1, Q3] | Range |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic CV | 0 | 0 | 43 | 0.00 ± 0.00 | 0.00 [0.00, 0.00] | 0.00-0.00 |
| P3-B | 5 | 38 | 0 | 65.73 ± 27.53 | 68.07 [46.28, 84.12] | 11.12-149.26 |
| P3-C | 21 | 22 | 0 | 34.90 ± 17.88 | 35.36 [21.67, 48.44] | 7.98-70.75 |
| P3-D hard | 4 | 39 | 0 | 78.22 ± 33.52 | 78.92 [52.29, 104.29] | 14.47-149.90 |
| P3-E hard | 27 | 16 | 0 | 33.50 ± 17.23 | 30.67 [18.78, 45.38] | 9.86-74.77 |
| P3-F always open | 25 | 18 | 0 | 37.82 ± 18.27 | 35.99 [21.28, 49.90] | 9.58-75.43 |
| P3-G soft | 26 | 17 | 0 | 31.66 ± 14.44 | 29.96 [20.13, 42.19] | 9.34-66.42 |
| P3-G hard | 26 | 17 | 0 | 31.04 ± 13.81 | 29.00 [20.67, 41.22] | 9.95-66.59 |

## What is attributable and what remains confounded

- **P3-B versus P3-C: attributable to the residual anchor in this run.**
  They have byte-identical initial model state, the same architecture and
  optimizer, the same seed/order, and differ only in CP versus CV anchor.
  Switching to CV lowered raw validation MSE from 7,106.32 to 3,341.44
  (53.0%); normalized MSE rose slightly from 1.006105 to 1.009737.
- **P3-D versus P3-E: attributable to the residual anchor in this run.**
  They likewise share an identical initial state and protocol. Switching
  from CP to CV lowered raw validation MSE from 8,010.10 to 3,275.40
  (59.1%), while normalized MSE rose from 0.999297 to 1.011788. Both
  metrics are reported; this is not a reason to change the pre-established
  normalized checkpoint-selection rule.
- **P3-C versus deterministic CV: learned MTN-CV correction comparison.**
  The anchor remains CV, so the 7.96% lower mean raw MSE is a comparison of
  the learned correction against its deterministic anchor. It is still
  single-seed validation evidence: 21 windows improved and 22 worsened.
- **P3-E versus P3-F: gate-training/inference strategy comparison.**
  Both use CV, context, MASC, and SATIE; P3-E has the original hard SAGE gate,
  while P3-F is independently trained always-open context. P3-E hard is
  3.08% lower in mean raw MSE, but the per-window gate intervention below
  does not support useful selective decisions.
- **P3-F versus P3-C is confounded.** It changes context availability and
  SATIE as well as the model family, so their small raw-MSE difference cannot
  be attributed to context alone.
- **P3-G versus P3-F is the intended differentiable-gate versus
  independently trained always-open comparison**, with identical initial
  base/context weights and the same data order. P3-G hard is 8.78% lower in
  mean raw MSE on this validation set, but all hard decisions are open. This
  is an exploratory training-strategy comparison, not evidence that
  sample-dependent hard admission helped.
- **P3-D/E versus P3-C** also changes context, MASC, SATIE, and gate
  architecture; it does not isolate a single one of those factors.

## Gate utility interventions

For each SAGE checkpoint, forced-open and forced-closed predictions are
paired on the same validation window. “Open beneficial” means forced-open
raw MSE is lower than forced-closed raw MSE. The natural hard decision is
then compared with that sign; these interventions do not represent separately
trained models.

| Model | Mean p (SD) | Probability range | Hard-open | Open beneficial | Closed beneficial | Decision matches beneficial intervention |
|---|---:|---:|---:|---:|---:|---:|
| P3-D | 0.9164 (0.0768) | 0.5805-0.9915 | 43/43 | 27/43 | 16/43 | 27/43 (62.8%) |
| P3-E | 0.5525 (0.0924) | 0.3193-0.6856 | 32/43 | 20/43 | 23/43 | 15/43 (34.9%) |
| P3-G | 0.5524 (0.0058) | 0.5422-0.5583 | 43/43 | 23/43 | 20/43 | 23/43 (53.5%) |

Mean binary gate entropy was 0.2575 nats for P3-D, 0.6707 for P3-E, and
0.6876 for P3-G (maximum `ln(2) ~= 0.6931`). P3-E's hard decisions agreed
with the lower-error context intervention on only 15 windows. P3-G's
all-open decision has the same 23/43 agreement as always-open would. P3-D
also opened every case. The results do not show reliable learned selective
context use.

The P3-E hard mean raw MSE (3,275.40) is higher than its forced-open value
(3,250.78), and lower than its forced-closed value (3,370.40). For P3-G,
hard equals forced-open exactly (3,082.83), while forced-closed is 3,177.00.
These aggregate values and paired windows should be interpreted together;
an average improvement does not imply per-window benefit.

## Resource and data-integrity notes

P3-B through P3-G each completed 10 epochs. Recorded train-loop durations
range from 262.82 to 414.57 seconds; maximum sampled RSS is 349,159,424 bytes.
P3-G soft/hard rows refer to one checkpoint and one training run. Full epoch
times, checkpoint sizes, histories, model configurations, and status files
remain in `P3_initial_screen_v1/`.

The Phase 3 runner only constructed train and validation datasets and
performed no test model forward or test metric calculation. The monolithic
pickle did deserialize its test role into memory, however. This is the
documented strict-isolation deviation; it is not evidence that test values
influenced training or selection. The Phase 4 role-cache migration and loader
are designed to prevent test-cache access during screening.
