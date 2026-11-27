# Phase 4 Runtime Report

All rows are completed train/validation-only runs. `training_seconds` is
the runner's accumulated training-loop time; per-epoch duration includes
the validation pass. Final diagnostic evaluations and report writing are
not included. RSS is sampled process RSS, not a hardware-counter peak.

| Configuration | Seeds | Mean epochs | Mean seconds | Mean seconds/epoch | Max sampled RSS (bytes) | Max checkpoint (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| P4_A | 3 | 16.33 | 591.03 | 34.69 | 336777216 | 5058969 |
| P4_B | 3 | 20.67 | 554.33 | 26.21 | 329457664 | 4977705 |
| P4_C | 3 | 18.33 | 580.19 | 31.15 | 277061632 | 5162373 |
| P4_D | 3 | 19.33 | 528.77 | 26.79 | 286281728 | 5080789 |
| P4_E | 3 | 17.33 | 875.29 | 49.59 | 294993920 | 5102969 |
| P4_F | 3 | 16.33 | 611.59 | 37.53 | 263413760 | 5100857 |

The three-seed validation set is one video with 43 overlapping windows.
No test split was loaded or evaluated. Runs used the exact settings saved
in each `run_config.json`; seeds and epoch counts are recorded in
`P4_RUNTIME_SUMMARY.csv`.

Reproduction command (run from the repository root in PowerShell):

```powershell
python -m MTN.run_p4_screen --configurations p4_a p4_b p4_c p4_d p4_e p4_f --seeds 20260928 20260929 20260930 --max-epochs 30 --patience 10 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --cache-root results_paper\splits\gp6_from_scratch_v1\roles_v1 --output-root results_paper\P4_initial_screen_v1
```
