# Phase 4 Gate Utility Report

For each learned-gate checkpoint, forced-open and forced-closed
predictions are post-hoc validation interventions. `open_beneficial`
means forced-open raw MSE is lower than forced-closed raw MSE for that
window. These interventions are not independent models and do not
establish causal utility or generalization.

| Model | Seed | Mean p | Hard-open | Gate grad L2 | Context grad L2 | Open beneficial | Closed beneficial | Decision matches | Corr(p, open benefit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p4_e | 20260928 | 0.5525 | 0.744 | 0.00148433 | 0.0164062 | 20 | 23 | 15 | 0.20366449240517837 |
| p4_e | 20260929 | 0.5006 | 0.419 | 0.000322814 | 0.0104735 | 40 | 3 | 21 | 0.46600254242786077 |
| p4_e | 20260930 | 0.5116 | 0.860 | 0.00029414 | 0.00954707 | 33 | 10 | 31 | -0.2982419309473939 |
| p4_f | 20260928 | 0.5524 | 1.000 | 0.00166151 | 0.0213847 | 23 | 20 | 23 | -0.46018938988576535 |
| p4_f | 20260929 | 0.5550 | 1.000 | 0.0011902 | 0.027369 | 36 | 7 | 36 | 0.06896646943774509 |
| p4_f | 20260930 | 0.7655 | 1.000 | 0.00104545 | 0.0326641 | 30 | 13 | 30 | 0.2550531903866631 |

Only 43 overlapping validation windows from a single video are available
per seed. Treat all gate-utility results as exploratory.
