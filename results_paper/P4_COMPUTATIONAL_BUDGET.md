# Phase 4 Computational Budget

## Fixed environment and run limits

The recorded environment is the CPU-only Windows 11 machine documented in
`P3_RUNTIME_REPORT.md`: Intel Core i5-8365U, 16 GB RAM, Intel UHD Graphics 620,
and CPU PyTorch. Phase 4 runs use four CPU threads by default, batch size 8,
Adam at `1e-3`, MSE, and the full 212-sample train / 43-sample validation
splits. The runner caps training at 30 epochs, uses validation-only early
stopping with patience 10, and defaults to a 3,600-second per-run wall-clock
budget. Runs are sequential. A budget-expired run is marked `incomplete` and
retains a resumable checkpoint; it is not treated as a completed result.

## Measured full-split epoch benchmark

One full train-and-validation epoch was measured per configuration at four
threads with seed `20260928`. Each measurement is exploratory runtime data,
not an accuracy comparison. `epoch_seconds` includes training batches and
that epoch's validation pass; it excludes final policy evaluations,
checkpoint/report finalization, and process startup.

| Configuration | Parameters | Epoch seconds | Train normalized MSE | Validation normalized MSE | Peak sampled RSS (bytes) | Checkpoint (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| P4-A | 133,700 | 24.97 | 0.692956 | 1.246203 | 328,376,320 | 5,057,753 |
| P4-B | 125,252 | 25.74 | 0.677308 | 1.228143 | 319,021,056 | 4,976,169 |
| P4-C | 138,516 | 28.28 | 0.717900 | 1.217981 | 321,671,168 | 5,156,293 |
| P4-D | 130,068 | 28.34 | 0.703179 | 1.196893 | 323,178,496 | 5,074,709 |
| P4-E | 130,694 | 32.20 | 0.684803 | 1.203630 | 323,469,312 | 5,091,001 |
| P4-F | 130,630 | 44.90 | 0.677080 | 1.223263 | 320,806,912 | 5,089,977 |

The observed one-epoch total is 184.43 seconds across the six models. A simple
linear projection for all six configurations, three seeds, and 30 epochs is
about 16,598 seconds (4.61 hours), before final diagnostic evaluations and
file I/O. This is a planning estimate from one epoch per configuration, not a
runtime guarantee; actual early stopping may reduce it, and later epochs may
run at different speeds. The slowest observed run, P4-F, projects to about
22.5 minutes for 30 epochs, below its per-run one-hour budget under the
measured conditions.

## Completed three-seed screen

The completed sequential screen used 18 runs across P4-A–F and seeds
20260928–20260930. All runs completed with validation-only early stopping;
the runs used 325 epochs in total (18.06 mean epochs per run). Summing the
runner's recorded training-loop time gives 11,223.612 seconds (3.118 hours),
excluding process startup and final report generation. This is the measured
aggregate training time, not wall-clock elapsed time for the whole session.
Per-configuration and per-run timings, sampled RSS, and checkpoint sizes are
in `P4_initial_screen_v1/P4_RUNTIME_REPORT.md` and
`P4_initial_screen_v1/P4_RUNTIME_SUMMARY.csv`. The longest run took
1,114.306 seconds, below the configured 3,600-second budget. Maximum sampled
RSS was 336,777,216 bytes.

For comparison only, the same P4-F seed and full-split one-epoch command took
18.77 seconds at one thread and 44.90 seconds at four threads. This is one
paired observation, not a stable thread-scaling benchmark. The four-thread
setting remains the Phase 4 default; no CPU performance improvement is
claimed.

Peak RSS is sampled process RSS, not a hardware-counter peak. The maximum in
this benchmark batch was 328,376,320 bytes. The role-separated loader loads
only train and validation caches; no test cache was included in the temporary
benchmark cache.

## Capped smoke checks

All six P4 configurations completed one capped epoch using two train batches
and two validation batches, with a temporary cache containing only train and
validation files. These checks verified runner operation, checkpoint/status
creation, expected parameter counts, prediction serialization, and summary
CSV generation. Their loss values are not full-split results and are not
used for model selection.

## Reproduction

From the repository root in PowerShell, create a temporary cache containing
only `train.pkl`, `train_role_manifest.json`, `validation.pkl`, and
`validation_role_manifest.json`, then run:

```powershell
python -m MTN.run_p4_screen --configurations p4_a p4_b p4_c p4_d p4_e p4_f --seeds 20260928 --max-epochs 1 --patience 1 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --cache-root $cache --output-root results_paper\P4_epoch_benchmark_4threads_v1
```

The benchmark outputs retain `history.csv`, `validation_metrics.json`,
`run_config.json`, `status.json`, `progress.log`, and atomic checkpoints for
each configuration under `P4_epoch_benchmark_4threads_v1/`. The source
`roles_v1/test.pkl` was not copied to the temporary cache.
