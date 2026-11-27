# Phase 5B annotation-only runtime report

## Environment and protocol

- CPU: Intel Core i5-8365U; Windows 11 Pro; CPU-only PyTorch.
- Four CPU threads, sequential model execution, batch size 8, Adam at `1e-3`.
- Seed `20260928`; maximum ten epochs; patience 10; per-run wall-clock budget 1,800 seconds.
- Full frozen split: 988 training windows / 4 videos / 261 tracks and 432 validation windows / 2 videos / 120 tracks.
- PyTorch `2.14.1+cpu`, NumPy `2.4.6`.
- Checkpoint selection used validation normalized MSE. No test split was loaded or evaluated.

## Measured runs

| Model | Status | Epochs | Best epoch | Params | Training-loop seconds | Process seconds | Validation seconds | Peak sampled RSS (bytes) | Checkpoint (bytes) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AO-0 | completed | 0 | — | 0 | 0.000 | Not recorded | 0.037 | 220,684,288 | — |
| AO-1 | completed | 10 | 4 | 76,804 | 1,496.509 | 1,509.431 | 2.457 | 310,083,584 | 3,968,719 |
| AO-2 | completed | 10 | 4 | 76,804 | 811.209 | 826.591 | 3.684 | 249,135,104 | 3,968,719 |
| AO-3 | completed | 10 | 4 | 68,356 | 771.348 | 788.784 | 8.518 | 250,593,280 | 3,887,183 |

AO-0 is the deterministic last-three-observed-box constant-velocity baseline. Its reported 0.037 seconds is validation inference only; startup and output-generation time were not measured as a comparable training process.

The learned runs used 3,079.067 seconds of training-loop time and 3,124.806 seconds of recorded process time combined. Maximum sampled RSS was about 295.6 MiB (AO-1). Every learned run completed ten epochs below the 1,800-second cap. Training time includes epoch loops; process time includes final validation and per-run finalization measured by the runner. OS load varied during execution, and some epoch/validation timings had large spikes, so these one-machine observations are descriptive rather than benchmark claims.

The one-epoch AO-2 timing-only benchmark in `AO_epoch_benchmark_v1/` is separate and is not mixed with the ten-epoch pilot. See `AO_COMPUTATIONAL_BUDGET.md` for the pre-screen projection and its comparison with measured totals.

## Reproduction

The source files and frozen caches are separate from the historical experiments. The commands below use a fresh output directory so they cannot overwrite the delivered screen:

```powershell
python MTN\ao_dataset_v1.py prepare --output-root results_paper\splits\ao_annotation_only_v1_reproduction
python MTN\run_ao_v1.py --cache-dir results_paper\splits\ao_annotation_only_v1_reproduction --output-root results_paper\AO_initial_screen_v1_reproduction --models AO-0,AO-1,AO-2,AO-3 --seed 20260928 --max-epochs 10 --patience 10 --batch-size 8 --learning-rate 0.001 --cpu-threads 4 --wall-clock-budget-seconds 1800
python MTN\run_ao_v1.py --finalize-only --output-root results_paper\AO_initial_screen_v1_reproduction
```

`prepare` refuses to overwrite an existing split directory. The runner refuses to reuse an existing run without `--resume`; only an `incomplete` run with a compatible saved configuration and checkpoint can be resumed. The original frozen cache and delivered pilot output remain untouched by the reproduction commands.
