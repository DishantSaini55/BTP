# Phase 6B Execution and Integrity Addendum v1

## Close-out status

**Phase 6B is complete.** This addendum supersedes neither the frozen
pre-execution documents nor any historical result artifact. It records the
completed held-out execution and identifies the corrected result bundle that
must be used for reporting.

Canonical official-test outputs are `AO_confirmation_v1/OFFICIAL_TEST_RESULTS_v2.json`
and its sibling `OFFICIAL_TEST_RESULTS_v2/` CSV directory. All v3 report
derivatives were checked against that corrected bundle. The original v1
official-test bundle remains preserved as a historical, superseded artifact.

## Frozen protocol and membership verification

- The corrected bundle contains 2,615 eligible windows from 13 videos and
  861 `(video_id, track_id)` clusters across GP3, GP5, GP8 and GP9.
- The v1 and v2 bundles have identical source-video membership, source XML
  SHA-256 mapping, sample ordering and all non-ridge neural/baseline results.
- The sole changed seed record is `AO-RIDGE`, seed 20260928: pooled raw-pixel
  MSE changed from 3,986.642578 in v1 to 2,889.718262 in v2.
- All 15 neural checkpoint hashes and the frozen ridge-coefficient hash match
  `FINAL_CHECKPOINT_HASHES.csv`; ridge alpha remains 100.
- No inference, training, checkpoint selection, normalization refit,
  hyperparameter selection or raw-video access was performed in this close-out.

## v1 ridge defect and correction

The ridge coefficients predict residuals in the frozen train-normalized
coordinate system. In the v1 evaluator path, those normalized residuals were
combined with a raw-pixel constant-velocity forecast, mixing coordinate
systems before metrics were calculated.

The corrected evaluator constructs constant velocity in normalized coordinates,
adds the frozen normalized ridge residual there, then applies the single frozen
train-only inverse normalization to obtain raw-pixel predictions. This is a
unit-consistency correction, not a changed model, metric, split, membership,
seed, or selected hyperparameter. The equal membership/XML hashes and the fact
that only the deterministic ridge record changed verify that no test-driven
tuning occurred.

## Derived-artifact verification

The corrected JSON/CSV seed records reconcile exactly (18 records: 15 neural
seed evaluations plus AO-0, AO-CP and AO-RIDGE). The v3 model summary matches
their per-model means; all 18 macro rows exactly recompute from the corrected
per-video and per-track CSVs. The final report v3, paired-comparison v3 and
track-bootstrap v3 therefore refer to the corrected v2 results.

| Artifact | SHA-256 |
|---|---|
| `OFFICIAL_TEST_RESULTS_v2.json` | `6a96d14b857452a454ceeab76954bdb5f9dfc15d11de18c4ff6500e7d8558b03` |
| `OFFICIAL_TEST_MODEL_SUMMARY_v3.csv` | `a6bbb80b6b8bd45664735a728dbd382e9a1f1c2cb6d44e106f6c3dd56e136aec` |
| `OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv` | `938f90352fc25c27d093fdca961583c0c6d374db0867a68854e687d0f6d0e922` |
| `OFFICIAL_TEST_TRACK_BOOTSTRAP_v3.csv` | `9afd804d026d91e4a746d08696cb06a8e3e542f5461d333544f877853b15ed68` |
| `OFFICIAL_TEST_MACRO_RESULTS.csv` | `272acc76b445ca477351445c52b9af63424af4f9f2d03f134547086bd1721698` |
| `OFFICIAL_TEST_FINAL_REPORT_v3.md` | `ec5252286a50efbcc8dc168d568600deceecd9f04c54884346f1461ff7faa773` |
| `OFFICIAL_TEST_RUNTIME_REPORT.md` | `93e0b10f2095396ac078db0be2fc5ebf39715aa99c4deeb21f481c0763e1134a` |

## Close-out regression tests

`MTN/finalize_official_test_v1.py` had a post-reporting regression in which
the neural paired-comparison path referenced `n_seeds` before assignment. It
now derives the count from the explicitly paired seed vector. The macro
calculation is also exposed as a pure helper for regression coverage; its
existing outputs were not regenerated or overwritten.

Focused close-out validation was run without bytecode writes:

```powershell
python -B -m unittest tests.test_official_test_evaluator_v1 tests.test_ao_confirmation_finalize_v1 tests.test_phase6b_closeout_v1 -v
```

All 11 tests passed. They cover evaluator MSE guards, seed-level intervals,
the repaired five-seed and deterministic comparisons, macro aggregation, v1/v2
membership preservation, the ridge-only correction, and v2-to-v3/macro
artifact reconciliation.

## Interpretation boundary

The official-test result supports AO-RIDGE as the lowest-error frozen candidate
on this annotation-only protocol. AO-2 and AO-3 are close to constant velocity;
their seed-level intervals against AO-0 include zero. All inference is
conditional on this fixed test membership; the five neural seeds are repeated
training realizations, not independent test datasets. Ridge alpha was chosen
on the fixed GP7 validation data and remains descriptive rather than
test-selected.
