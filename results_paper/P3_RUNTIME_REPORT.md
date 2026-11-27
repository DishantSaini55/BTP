# Phase 3 CPU Runtime Report

## Environment

The environment information below is from the repository's existing CPU
runtime audit; it was not re-probed as part of the Phase 3 screen.

- CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 4 physical cores / 8 logical
  processors.
- Installed RAM: 16,985,051,136 bytes (15.82 GiB).
- Graphics: Intel UHD Graphics 620 integrated graphics; no dedicated GPU.
- OS / Python: Windows 11 Pro, build 26200 / Python 3.12.10.
- PyTorch: `2.14.1+cpu`; CUDA unavailable.
- Training thread limit: 4 CPU threads.

## Screening workload and measured runtime

P3-B through P3-G each trained sequentially for 10 epochs on the 212-window
training split and evaluated each epoch on the 43-window validation split.
Each run used batch size 8, Adam at `1e-3`, and MSE. The elapsed values below
are the runner's recorded training-loop time, which includes epoch validation
and loop/checkpoint work; they are not a separate end-to-end command wall
time. Per-epoch duration includes train plus validation.

| Run | Epochs | Selected epoch | Mean epoch seconds | Min-max epoch seconds | Recorded elapsed seconds | Peak sampled RSS bytes | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3-B | 10 | 9 | 40.94 | 22.77-107.37 | 414.57 | 332,496,896 | 5,058,009 |
| P3-C | 10 | 8 | 27.37 | 23.09-38.90 | 278.89 | 335,376,384 | 5,058,009 |
| P3-D | 10 | 10 | 26.26 | 24.15-30.08 | 267.60 | 341,000,192 | 5,090,745 |
| P3-E | 10 | 8 | 28.96 | 23.07-42.60 | 294.84 | 344,625,152 | 5,090,745 |
| P3-F | 10 | 8 | 25.81 | 21.26-29.34 | 262.82 | 347,725,824 | 5,074,709 |
| P3-G | 10 | 8 | 26.55 | 22.41-34.65 | 270.82 | 349,159,424 | 5,089,721 |

Across the six unique trainable runs, summed recorded elapsed time was
1,789.535 seconds (29 minutes 49.5 seconds). Maximum sampled process RSS was
349,159,424 bytes (332.98 MiB). RSS is a sampled process value, not a
hardware-level peak-memory measurement. P3-A was a deterministic baseline:
zero training epochs and zero trainable parameters.

Per-epoch training and validation losses, elapsed seconds, batch counts, and
sampled RSS are in each run's `history.csv` and `history.json`. Atomic
checkpoints, RNG/optimizer state, progress heartbeats, and explicit status
files were retained in each run directory. The status files for P3-A through
P3-G all read `completed`; the six trainable runs each have 10 recorded
epochs. P3-G's soft/hard validation rows share one checkpoint and one
training duration, so that duration is counted once in the total.

The validation-only runner did not construct or iterate a test dataset. The
cache limitation is material: the shared `flow_splits.pkl` is deserialized
as a whole before the runner selects train and validation roles, so the
serialized test role is present in process memory. No test values were
normalized, passed to the model, evaluated, or used to select checkpoints.
See the test-data isolation caveat in `PHASE_3_RESEARCH_REPORT.md`.

The first post-training plotting attempt failed because `matplotlib` was not
installed. All run states were already complete. The exporter was changed to
standard-library SVG generation; finalization then succeeded and produced
both validation-only SVGs. No package was installed and no training was
repeated for this plotting correction.

## Reproduction commands

Run from the repository root in PowerShell. Use a new output directory; the
runner intentionally refuses to overwrite an existing run:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python -m MTN.run_p3_screen --configurations p3_a p3_b p3_c p3_d p3_e p3_f p3_g --seeds 20260928 --max-epochs 10 --patience 10 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --output-root results_paper\P3_reproduction_v1
```

The command trains on the fixed training split and performs validation after
each epoch; it does not evaluate test predictions. For a strict
no-test-deserialization reproduction, first provide a train/validation-only
cache rather than the current monolithic `flow_splits.pkl`.

Run the final regression suite with:

```powershell
python -m unittest discover -s tests -v
```

The Phase 3 smoke screen, full validation summary, per-window metrics, and
plots are retained separately under `P3_smoke_v1/` and
`P3_initial_screen_v1/`; capped smoke losses are infrastructure checks, not
full-split accuracy results.
