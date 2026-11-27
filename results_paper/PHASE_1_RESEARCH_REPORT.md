# PACE-Net Phase 1 research report

## Scope and preservation

Phase 1 added a resumable E8 runner, deterministic SAGE gradient diagnostics,
and a read-only audit of the E8 gate inputs. It did not modify the frozen E1
protocol, historical E1/E2/E3/E7 outputs, original model architecture, or
existing checkpoints. It did not launch a full ten-seed experiment, evaluate
test performance for E8, or select a hyperparameter from test outcomes.

The original `MTN/run_e8.py` remains unchanged. The new runner is
`MTN/run_e8_v2.py`; it writes to a versioned `E8_v2` output tree and requires
run status/configuration validation rather than treating a directory as proof
of completion.

## Findings

### SAGE gradient pathway

The all-closed batch early return makes prediction loss independent of both
the SAGE gate MLP and context encoder: their parameter gradients are absent
(`None`), not present-but-zero. In mixed batches the open examples provide
nonzero gradients, while per-example loss for a closed example gives exactly
zero gradient for those branch parameters. The hard mask comparison blocks
gradients through the mask decision and the masked attention key. Numerical
context-token scaling remains differentiable when the token is admitted; it
does not rescue a closed token once attention masks it.

Actual deterministic gradient values and interpretation are in
`results_paper/SAGE_GRADIENT_AUDIT.md` and
`results_paper/diagnostics/phase1/sage_gradient_measurements.json`. These
mechanism tests are not proof that this is the only cause of historical gate
collapse.

### Normalization and coordinate-shift audit

The E1/E8 loader computes box and speed mean/std from concatenated **training
windows only**. Recorded box stats are mean
`[883.170044, 921.353699, 904.123657, 977.436157]` and std
`[213.529541, 20.202641, 205.581223, 28.961599]`; speed mean/std are
`26.418589 / 6.800016`. Thus normalization leakage was not found.

A separate range-calculation bug was found in the prior E3 analysis:
`run_e3.py:x_t_ranges` used each 60-frame sample's `bbox[-1]`, the final
frame, rather than `bbox[14]`, the last observed frame actually passed to the
SAGE gate. That old measurement includes future-target coordinates and is not
the gate's `x_T`. This does not invalidate saved E1 model metrics, but it
means the reported historical `[-38.396156, 4.739763]` range should not be
described as the gate input.

The corrected read-only range audit computes actual normalized gate inputs
from observed frame 15:

| Split | Normalized input global min..max | Components outside ±5 |
|---|---:|---:|
| Train | -2.406800 .. 3.370805 | 0 / 848 (0%) |
| Validation | -8.360971 .. 3.145600 | 32 / 172 (18.60%) |
| Test (inspection only; no tuning) | -30.811996 .. 4.941484 | 432 / 868 (49.77%) |

Most of the large test shift is in the second box coordinate, whose last
observed normalized value spans `[-30.8120, -18.8443]` over test windows.
The test inputs were inspected only to correct the historical range claim;
they were not used to set a scale, choose a model, or assess E8.

The existing E8 `clipped_normalized_x_t` transformation is preserved. Its
fixed ±5 clipping leaves train inputs unchanged, clips 18.60% of validation
components, and would clip 49.77% of test components; the latter is descriptive
only. This is a material test-domain shift, not evidence that clipping
improves prediction.

As an alternative representation diagnostic, box deltas from the last
observed box were scaled using the same training-only box standard deviations.
The training 99th-percentile absolute delta per coordinate was
`[0.296293, 0.504102, 0.285560, 0.494917]`. Applying a `tanh(delta / scale)`
transform bounded the observed training and validation deltas approximately
within `[-0.938, 0.977]`. These values motivate a validation-only comparison
of relative motion/center-size features; they do not justify changing E1,
choosing a test-optimized transform, or claiming better accuracy.

Full coordinate ranges, historical-vs-corrected definitions, and the bounded
delta calculation are in
`results_paper/diagnostics/phase1/e8_input_ranges.json`.

### E8 runner and smoke evidence

`MTN/run_e8_v2.py` adds flushed epoch logging, train/validation losses,
per-epoch elapsed and process CPU time, resident-memory sampling, periodic
heartbeats, atomic CSV/JSON and checkpoint writes, model/optimizer/Python/
NumPy/PyTorch/DataLoader RNG state resume, early-stopping state, and explicit
`running`, `completed`, `interrupted`, and `failed` states. It retains
`clipped_normalized_x_t` and `context_only`, offers validation-only mode, and
configures seeds, variants, models, epochs, batch size, and CPU threads.

Smoke configurations use one seed, a maximum of two epochs, and capped train
and validation batches. They are infrastructure checks only; their
validation losses are based on two batches and must not be interpreted as
full-split selection or model comparison. Machine-readable per-run history and
metrics are written under
`results_paper/runs/E8_v2_phase1_smoke/`; consolidated measurements are in
`results_paper/diagnostics/phase1/phase1_diagnostics.json`.

No E8 test metrics are produced in validation-only mode. Do not claim a
forecast accuracy improvement from these smoke runs.

## Prior experiment status

- E1, E2, E3, and E7 remain as historically reported in their original files.
- The prior E3 range calculation used a future frame for the quantity labelled
  `x_T`; corrected observed-input ranges are now separately reported.
- E8 was previously stopped after an apparent stall. The Phase 1 run used only
  capped two-epoch validation-screening configurations, not the full E8
  experiment.
- E4/E5/E6/E9 and qualitative work remain paused per prior user direction.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python -m pytest -q tests\test_sage_gradients.py tests\test_e8_runner_v2.py
python -m MTN.sage_gradient_diagnostics --output results_paper\diagnostics\phase1\sage_gradient_measurements_repro.json
python -m MTN.audit_e8_input_ranges --output results_paper\diagnostics\phase1\e8_input_ranges_repro.json
python -m MTN.run_e8_v2 --models without_satie with_satie --variants clipped_normalized_x_t context_only --seeds 20260928 --max-epochs 2 --validation-only --max-train-batches 2 --max-validation-batches 2 --cpu-threads 4 --output-root results_paper\runs\E8_v2_phase1_reproduction
```

The smoke command above uses a fresh output directory and deliberately
processes only two train and two validation batches per epoch. Existing
directories are never silently reused as successful runs.

## Phase 2 recommendations

1. Keep E1 results and protocol frozen. Correct the E3 gate-input range in
   future documentation with a versioned analysis rather than rewriting the
   historical output in place.
2. Before a full E8 run, use validation-only screening and complete split
   validation metrics with the new runner; inspect gate gradient
   presence/zero rates and progress/resource heartbeats.
3. Treat relative displacement or center/size normalization as a candidate
   input representation for a pre-registered validation-only comparison.
   Define its train-only fit and mapping before looking at test metrics.
4. Do not infer a remedy from one seed or these short capped smoke runs.
   Any full E8 seeds, loss changes, or architecture changes require an
   explicit next-phase decision.
