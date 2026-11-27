# CPU runtime report

## Machine and software

- CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 4 physical cores / 8 logical
  processors.
- Installed memory: 16,985,051,136 bytes (15.82 GiB).
- Graphics: Intel UHD Graphics 620 integrated graphics; no dedicated GPU.
- PyTorch: `2.14.1+cpu`; CUDA unavailable.
- OS / Python: Windows 11 Pro, build 26200 / Python 3.12.10.
- E8 runner thread limit: 4. Deterministic SAGE unit probes used one thread.

The E7 inference timing below is reproduced from the existing
`results_paper/summary/E7_summary.csv`; E7 was not rerun for Phase 1.

| Model family | Median ms, batch 1 | IQR ms, batch 1 | Median ms, batch 8 | IQR ms, batch 8 |
|---|---:|---:|---:|---:|
| MTN | 50.685 | 5.721 | 83.910 | 6.049 |
| PACE-Net without SATIE | 53.849 | 6.482 | 88.005 | 7.241 |
| PACE-Net with SATIE | 57.562 | 5.432 | 95.039 | 12.720 |
| PACE-Net with ROUTE | 56.744 | 5.748 | 88.499 | 7.766 |
| PACE-Net with ROUTE and SATIE | 53.154 | 5.587 | 88.129 | 6.397 |

E7 used 4 threads, 50 warm-ups, and 500 timed iterations. Its corrected MAC
column is encoder self-attention only: 227,520 per sample / 1,820,160 for
batch 8. These are inference timings and are not directly comparable to a
training epoch.

## Phase 1 capped training smoke

Four configurations were run sequentially, all with seed `20260928`, 2
epochs maximum, 2 training batches and 2 validation batches per epoch, CPU
threads 4, and validation-only mode. No test split was evaluated. Each epoch's
loss is the mean over the two capped batches, not the complete split.

| Model | Variant | Train loss, epoch 1 -> 2 | Validation loss, epoch 1 -> 2 | Epoch seconds, 1 -> 2 | Admission rate, 1 -> 2 | Gate grad L2 mean, 1 -> 2 | Context grad L2 mean, 1 -> 2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Without SATIE | clipped normalized x_T | 2.155768 -> 0.953974 | 3.410111 -> 2.061301 | 3.294 -> 61.780 | 0.4375 -> 0.5625 | 0.00137348 -> 0.00038797 | 0.0810362 -> 0.0551214 |
| Without SATIE | context only | 2.192304 -> 0.811001 | 3.451703 -> 2.019459 | 1.697 -> 39.699 | 0.5000 -> 0.3750 | 0.00171815 -> 0.00008953 | 0.1065180 -> 0.0307078 |
| With SATIE | clipped normalized x_T | 2.132345 -> 0.870956 | 3.460143 -> 2.116428 | 3.531 -> 57.095 | 0.4375 -> 0.5625 | 0.00148746 -> 0.00030906 | 0.0858935 -> 0.0490138 |
| With SATIE | context only | 2.174714 -> 0.791702 | 3.501227 -> 2.048963 | 1.710 -> 127.502 | 0.5000 -> 0.3750 | 0.00191076 -> 0.00014710 | 0.1125957 -> 0.0255704 |

Each run wrote explicit `completed` status and states `test_evaluated=false`.
The runner was resumed from its one-epoch checkpoint to epoch 2; model,
optimizer, RNG, loader-generator, best-checkpoint, stale-epoch, and history
state were restored. One initial attempt was marked `failed` after exposing a
history CSV field-list defect; after fixing it, resume completed the affected
run, and all final smoke run states are completed.

Across the 8 measured epochs:

- Summed epoch wall time: 296.309 seconds (4.94 minutes).
- Total command wall time: 457 seconds (7.62 minutes), including startup,
  sequential model initialization, checkpointing and progress I/O.
- Summed epoch process CPU time: 180.219 seconds.
- Maximum sampled resident working set: 353,394,688 bytes (337.1 MiB).
- All four runs used two train and two validation batches per epoch.
- The final validation-only losses are smoke diagnostics only; not comparable
  full-split metrics and not evidence for selecting a variant.

The substantial epoch-time variation, including 39.7–127.5 seconds for
two-batch second epochs, was observed on this machine. Periodic heartbeats
continued to report progress and sampled memory while a batch was executing;
no freeze or completed-epoch stall was seen. This does not establish that the
prior E8 stall was fixed in every environment. The one-second heartbeat was
used for observability during this smoke; it adds frequent log flushes, so use
the default 30 seconds for ordinary runs.

RSS sampling was unavailable in the first epoch records because the first
sampler implementation did not return a valid Windows process handle. The
Windows API signature was corrected before the resumed epochs; RSS values
above are from successfully sampled second epochs only.

Full per-epoch gradients, gate values, elapsed/process CPU time, RSS, histories
and run state are in
`results_paper/diagnostics/phase1/phase1_diagnostics.json` and the individual
run directories under `results_paper/runs/E8_v2_phase1_smoke/`.

## Reproduction commands

From the repository root in PowerShell, with a fresh output directory:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python -m MTN.run_e8_v2 --models without_satie with_satie --variants clipped_normalized_x_t context_only --seeds 20260928 --max-epochs 2 --validation-only --max-train-batches 2 --max-validation-batches 2 --cpu-threads 4 --heartbeat-seconds 30 --output-root results_paper\runs\E8_v2_cpu_reproduction
```

That command deliberately runs only capped validation batches and does not
produce test metrics. It is not a full E8 experiment.
