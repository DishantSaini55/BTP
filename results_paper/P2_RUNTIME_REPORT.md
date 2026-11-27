# Phase 2 CPU runtime and execution record

## Environment and protocol

- CPU: Intel Core i5-8365U, 4 physical / 8 logical cores.
- RAM: 16 GB.
- GPU: Intel UHD Graphics 620 integrated graphics; no dedicated GPU; runs used CPU-only PyTorch.
- OS / runtime recorded in the existing Phase 1 reports: Windows 11 Pro, Python 3.12, PyTorch 2.14.1+cpu.
- Training threads: 4; batch size: 8; Adam learning rate: `1e-3`.
- Split: 212 train windows and 43 validation windows from the frozen GP6 split. The P2 runner loads only these two roles; it does not load test samples.
- Flow input: existing cached optical-flow features.

## Infrastructure smoke checks

Five sequential diagnostic configurations (P2-A through P2-E) completed one capped epoch each with two training batches, the full validation split, and no test evaluation. Those losses are infrastructure-only and are not full-split accuracy results. Machine-readable outputs are kept under `results_paper/P2_smoke_v3/`.

A separate P2-C run exercised the wall-budget path: after one training batch it was explicitly marked `incomplete`, then resumed from its model/optimizer/RNG/epoch/batch checkpoint and completed the capped run. The record is under `results_paper/P2_resume_smoke_v1/`. Earlier failed smoke attempts are preserved in separately versioned directories and are not part of the screening results.

## Full initial screening

The one-seed full-train/full-validation screen completed sequentially with seed `20260928`, up to 10 epochs/configuration, 4 CPU threads, and a 3600-second cumulative per-run wall-clock budget. Each run completed all 10 epochs; no test split was loaded or evaluated. Per-run status, flushed progress, history CSV/JSON, resumable atomic checkpoints, validation predictions, gate logits/decisions, and decoder context-attention weights are preserved under `results_paper/P2_initial_screen_v1/`.

| Configuration | State | Epochs | Selected epoch | Recorded run time (s) | Peak sampled RSS (bytes) |
|---|---|---:|---:|---:|---:|
| P2-A | completed | 10 | 10 | 1,016.71 | 340,217,856 |
| P2-B | completed | 10 | 6 | 374.35 | 346,865,664 |
| P2-C | completed | 10 | 5 | 264.23 | 344,051,712 |
| P2-D | completed | 10 | 5 | 268.74 | 343,351,296 |
| P2-E | completed | 10 | 9 | 286.24 | 303,640,576 |

The sum of the runner-recorded per-run times is 2,210.27 seconds (36 minutes 50.27 seconds). Per-run elapsed time includes training and epoch validation/checkpoint overhead, as recorded by the runner. The largest sampled process RSS was about 331 MiB. No GPU was used.

Exact full-screen reproduction command:

```powershell
python MTN\run_p2_screen.py --configurations p2_a p2_b p2_c p2_d p2_e --seeds 20260928 --max-epochs 10 --patience 10 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --checkpoint-every-batches 5
```

Budget-stop and resume reproduction (the second command resumes the same explicitly incomplete capped diagnostic run):

```powershell
python MTN\run_p2_screen.py --configurations p2_c --seeds 20260928 --max-epochs 1 --max-train-batches 1 --wall-clock-budget-seconds 0.01 --output-root results_paper\P2_resume_smoke_v1
python MTN\run_p2_screen.py --configurations p2_c --seeds 20260928 --max-epochs 1 --max-train-batches 1 --wall-clock-budget-seconds 900 --output-root results_paper\P2_resume_smoke_v1 --resume
```
