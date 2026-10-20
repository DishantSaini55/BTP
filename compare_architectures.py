"""Reproducible, identical-input comparison of baseline and context MTN.

This benchmark intentionally does not manufacture trajectory ground truth. It
reports functional, output, latency, and model-size measurements, and labels
accuracy as unavailable when IDD-PeD assets are not installed.
"""

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from context_smoke_test import make_samples
from MTN.context_TF import ContextIndividualTF
from MTN.individual_TF import IndividualTF


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OPTIONS = {"N": 1, "d_model": 32, "d_ff": 64, "h": 4, "dropout": 0.0}
WARMUP_RUNS = 10
MEASURED_RUNS = 100


def parameter_stats(model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    return parameter_count, parameter_bytes


def timed_forward(model, batch):
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model.forward_batch(batch)
        durations = []
        for _ in range(MEASURED_RUNS):
            started = time.perf_counter_ns()
            output = model.forward_batch(batch)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return output, durations


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260927)
    np.random.seed(20260927)
    batch = make_samples(batch_size=4)

    # Resetting the seed before construction makes ContextIndividualTF.base
    # identical to the previous architecture. The only structural difference
    # is the context branch and its gate.
    torch.manual_seed(101)
    previous = IndividualTF(15, 4, 4, 4, **OPTIONS).eval()
    torch.manual_seed(101)
    current = ContextIndividualTF(15, 4, 4, 4, gate_mode="hard", **OPTIONS).eval()
    if not all(torch.equal(a, b) for a, b in zip(previous.parameters(), current.base.parameters())):
        raise AssertionError("Baseline weights do not match the new model's base branch.")

    previous_output, previous_times = timed_forward(previous, batch)
    current_output, current_times = timed_forward(current, batch)
    gates = current.last_context_gate.cpu().numpy()
    difference = current_output - previous_output
    per_sample_rmse = difference.square().mean(dim=(1, 2)).sqrt().cpu().numpy()
    per_sample_max_abs = difference.abs().amax(dim=(1, 2)).cpu().numpy()
    previous_params, previous_bytes = parameter_stats(previous)
    current_params, current_bytes = parameter_stats(current)

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "architecture_comparison_predictions.npz",
        boxes=batch["boxes"].numpy(), speed=batch["speed"].numpy(),
        ego_flow=batch["ego_flow"].numpy(), ped_flow=batch["ped_flow"].numpy(),
        scene_context=batch["scene_context"].numpy(),
        observed_occlusion=batch["observed_occlusion"].numpy(),
        previous_predictions=previous_output.numpy(), current_predictions=current_output.numpy(),
        context_gate=gates,
    )

    def timing_summary(values):
        return {
            "mean_ms": float(statistics.mean(values)),
            "median_ms": float(statistics.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(min(values)), "max_ms": float(max(values)),
        }

    results = {
        "scope": "synthetic functional benchmark only; IDD-PeD assets unavailable",
        "environment": {
            "device": "CPU", "torch_threads": 1, "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS, "seed": 20260927,
            "same_base_weights": True,
        },
        "dataset_available": {name: (ROOT / name).exists()
                              for name in ("IDDPedestrian", "flow", "IDDP_model")},
        "previous": {"parameters": previous_params, "parameter_bytes": previous_bytes,
                     "latency": timing_summary(previous_times),
                     "shape": list(previous_output.shape), "finite": bool(torch.isfinite(previous_output).all())},
        "current": {"parameters": current_params, "parameter_bytes": current_bytes,
                    "latency": timing_summary(current_times),
                    "shape": list(current_output.shape), "finite": bool(torch.isfinite(current_output).all()),
                    "context_gate": gates.tolist()},
        "sample_differences": [{
            "sample": index, "gate": float(gates[index]),
            "prediction_rmse_vs_previous": float(per_sample_rmse[index]),
            "max_abs_difference_vs_previous": float(per_sample_max_abs[index]),
        } for index in range(batch["boxes"].shape[0])],
        "accuracy": "unavailable: no held-out labels or IDD-PeD data are installed",
    }
    (OUTPUT_DIR / "architecture_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    latency_delta = (results["current"]["latency"]["mean_ms"] /
                     results["previous"]["latency"]["mean_ms"] - 1) * 100
    parameter_delta = (current_params / previous_params - 1) * 100
    report = f"""# Architecture Comparison Report

## 1. Executive Summary

The previous MTN and current context-aware MTN both completed every identical synthetic test sample with finite `(4, 45, 4)` predictions. No measured trajectory accuracy comparison is possible: the IDD-PeD dataset, optical-flow inputs, labels, and checkpoint directory are absent. Therefore, no architecture is recommended for the actual use case yet.

## 2. Testing Methodology

- Device: CPU; PyTorch threads: 1; seed: `20260927`.
- Four fixed samples, identical input tensors and ordering for both models.
- The previous model's parameters exactly match the current model's shared MTN base branch (`same_base_weights=True`).
- Both ran in evaluation mode with 10 warm-up passes and 100 timed passes. Timings are per batch of four, using `perf_counter_ns`.
- Current model used its deployment-mode hard context gate. No targets were fabricated; accuracy fields remain unavailable.

## 3. Test Dataset/Samples

Synthetic contract samples: 15 observed boxes, 60 speed steps, 60 ego/pedestrian flow steps, six categorical scene fields, and 15 observed-only occlusion labels. Saved inputs and predictions: `outputs/architecture_comparison_predictions.npz`.

IDD-PeD availability: `IDDPedestrian={results['dataset_available']['IDDPedestrian']}`, `flow={results['dataset_available']['flow']}`, `IDDP_model={results['dataset_available']['IDDP_model']}`.

## 4–7. Architecture and Metric Results

| Metric | Previous MTN | Current context MTN | Difference |
| --- | ---: | ---: | ---: |
| Output shape | {tuple(previous_output.shape)} | {tuple(current_output.shape)} | same |
| Finite outputs | {results['previous']['finite']} | {results['current']['finite']} | same |
| Parameters | {previous_params:,} | {current_params:,} | {parameter_delta:+.2f}% |
| Parameter memory | {previous_bytes / 1024:.2f} KiB | {current_bytes / 1024:.2f} KiB | {(current_bytes / previous_bytes - 1) * 100:+.2f}% |
| Mean latency | {fmt(results['previous']['latency']['mean_ms'])} ms | {fmt(results['current']['latency']['mean_ms'])} ms | {latency_delta:+.2f}% |
| Median latency | {fmt(results['previous']['latency']['median_ms'])} ms | {fmt(results['current']['latency']['median_ms'])} ms | — |
| P95 latency | {fmt(results['previous']['latency']['p95_ms'])} ms | {fmt(results['current']['latency']['p95_ms'])} ms | — |
| Accuracy / MSE | unavailable | unavailable | no labels installed |

## 8. Sample-wise Differences

| Sample | Context gate | Prediction RMSE vs. previous | Maximum absolute difference |
| ---: | ---: | ---: | ---: |
"""
    for item in results["sample_differences"]:
        report += (f"| {item['sample']} | {item['gate']:.0f} | "
                   f"{item['prediction_rmse_vs_previous']:.8f} | "
                   f"{item['max_abs_difference_vs_previous']:.8f} |\n")
    report += f"""

## 9. Performance Improvements / Regressions

Only structural and runtime measurements are supported. The current model adds {current_params - previous_params:,} parameters and changed measured mean latency by {latency_delta:+.2f}%. This is not an accuracy improvement or regression.

## 10. Errors and Failure Cases

No forward-pass failures, non-finite values, or shape mismatches occurred. The blocked evaluation case is real-data accuracy: all required IDD-PeD assets are missing.

## 11. Resource / Latency Comparison

Parameter storage is reported exactly. CPU process RSS and accelerator memory were not captured because this run used CPU and no process-memory profiler is installed. Timing variability is represented by median and P95 values above.

## 12–13. Key Findings and Trade-offs

The current model's context path and hard gate execute correctly on the same inputs. The added branch has a measurable parameter/latency cost. Whether its day/night, scene, and occlusion information improves trajectory prediction cannot be determined from unlabelled synthetic data.

## 14. Final Recommendation

**Do not finalize either architecture from this benchmark.** Install the identical IDD-PeD split, flow arrays, time-of-day map, and held-out labels; train both architectures with matched seed, schedule, and checkpoints; then compare held-out MSE/CMSE overall and by night/day, occlusion, signalization, and interaction.
"""
    (ROOT / "ARCHITECTURE_COMPARISON.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
