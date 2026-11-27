# Phase 4 Parameter / Accuracy Efficiency

Parameter counts are unique trainable parameters audited by module
identity, not repeated state-dictionary entries. Accuracy is the mean
selected-policy raw validation MSE across completed seeds. The table
does not infer statistical equivalence or generalization.

| Model | Parameters | Mean validation raw MSE | SD across seeds | Delta vs CV | Pareto efficient |
|---|---:|---:|---:|---:|---:|
| p4_a | 133700 | 4566.179 | 2300.545759458322 | 935.863 | False |
| p4_b | 125252 | 3256.154 | 174.5003254854075 | -374.162 | True |
| p4_c | 138516 | 3552.724 | 481.13066171604896 | -77.591 | False |
| p4_d | 130068 | 3819.907 | 810.1965713453011 | 189.592 | False |
| p4_e | 130694 | 5095.549 | 2936.3977032361063 | 1465.234 | False |
| p4_f | 130630 | 5774.419 | 4301.222378988605 | 2144.104 | False |
| cv_baseline | 0 | 3630.315 | 0.0 | 0.000 | True |

The CV point has zero learned parameters and is included as a baseline
anchor. The validation split is one video; this efficiency frontier is
descriptive only.
