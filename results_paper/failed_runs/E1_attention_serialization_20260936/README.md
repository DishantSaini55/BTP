# E1 attention-serialization recovery record

`PACE-Net without SATIE (SAGE + MASC)`, seed `20260936`, trained to its
validation-selected checkpoint but failed during test-output serialization on
2026-11-22. No `metrics.json`, `per_sample.csv`, or prediction artifact was
written for this run.

Cause: exact all-closed SAGE batches use the motion-only 15-token memory path,
whereas a batch containing an admitted sample has a 16-token memory. The
evaluator tried to concatenate those attention tensors directly.

The original checkpoint and run configuration are retained here unchanged for
audit. The evaluator was repaired to pad only the absent context-token column
with `NaN` and to save `attention_memory_tokens` per sample. The seed is then
retrained from the same fixed seed and train/validation-only protocol into a
new standard run directory; it is not an overwrite or a test-driven rerun.
