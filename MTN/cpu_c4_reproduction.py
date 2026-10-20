"""Reproduce the matched one-epoch B-vs-C4 CPU POC with dedicated C4 tying."""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import cpu_targeted_weight_sharing as common
from c4_encoder_shared import verify_c4_sharing

OUT = common.OUT

def run_variant(name, strategy, train_ds, valid_ds, test_ds, mean, std, batches):
    started = time.perf_counter()
    model, training = common.train(strategy, train_ds, valid_ds)
    training_seconds = time.perf_counter() - started
    path = OUT / f"C4_reproduction_{name}.pth"
    torch.save(model.state_dict(), path)
    metrics, per_sample = common.evaluate(model, test_ds, mean, std)
    return {
        "strategy": "No sharing" if strategy == "none" else "C4: encoder self-attention Q/K/V/O shared across depth",
        "parameters": common.count(model),
        "trainable_parameters": common.count(model),
        "attention_parameters": common.attention_count(model),
        "checkpoint_bytes": path.stat().st_size,
        "training_seconds": training_seconds,
        "cpu_memory_measurement": "Not collected: psutil is not installed in the locked POC environment.",
        "training": training,
        "metrics": metrics,
        "per_sample": per_sample,
        "latency": common.warm_latency(model, batches),
        "implementation_verification": verify_c4_sharing(model) if strategy != "none" else None,
    }

def main():
    common.seed()
    train_raw, valid_raw, test_raw = common.load("train"), common.load("validation"), common.load("test")
    mean, std, mean_speed, std_speed = common.stats(train_raw)
    train_ds = common.build(train_raw, "train", mean, std, mean_speed, std_speed)
    valid_ds = common.build(valid_raw, "validation", mean, std, mean_speed, std_speed)
    test_ds = common.build(test_raw, "test", mean, std, mean_speed, std_speed)
    batches = list(DataLoader(test_ds, batch_size=1, shuffle=False))
    baseline = run_variant("B", "none", train_ds, valid_ds, test_ds, mean, std, batches)
    c4 = run_variant("C4", "encoder_self_attn", train_ds, valid_ds, test_ds, mean, std, batches)
    comparison = {
        "parameter_reduction": baseline["parameters"] - c4["parameters"],
        "parameter_reduction_percent": 100 * (baseline["parameters"] - c4["parameters"]) / baseline["parameters"],
        "mse_delta": c4["metrics"]["mse"] - baseline["metrics"]["mse"],
        "mse_delta_percent": 100 * (c4["metrics"]["mse"] - baseline["metrics"]["mse"]) / baseline["metrics"]["mse"],
        "samples_c4_better_mse": sum(c["mse"] < b["mse"] for b, c in zip(baseline["per_sample"], c4["per_sample"])),
        "samples_c4_worse_mse": sum(c["mse"] > b["mse"] for b, c in zip(baseline["per_sample"], c4["per_sample"])),
    }
    result = {"protocol": "Exact prior targeted POC protocol: cached identical CPU Farneback flow; 32/8/10 fixed video-separated split; seed 20260928; official checkpoint initialization; Adam lr 1e-3; batch 8; MSE; one epoch; four CPU threads; 5 warm-up passes and 200 latency forwards.", "B": baseline, "C4": c4, "comparison": comparison}
    target = OUT / "C4_reproduction_results.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"B": baseline["metrics"], "C4": c4["metrics"], "comparison": comparison}, indent=2))
    print(target)
if __name__ == "__main__": main()
