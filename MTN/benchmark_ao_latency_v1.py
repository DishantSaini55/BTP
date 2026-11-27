"""Matched CPU latency and encoder-attention MAC audit for AO-2/AO-3."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import run_ao_v1
    from .ao_models_v1 import (
        MODEL_CONFIG,
        build_ao_model,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )
except ImportError:
    import run_ao_v1
    from ao_models_v1 import (
        MODEL_CONFIG,
        build_ao_model,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )


WARMUPS = 50
TIMED_ITERATIONS = 500
BATCH_SIZES = (1, 8)
MODEL_IDS = ("AO-2", "AO-3")


def _cpu_name() -> str:
    try:
        result = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
            ],
            text=True,
            timeout=10,
        ).strip()
        if result:
            return result
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.platform()


def encoder_self_attention_macs(batch_size: int, sequence_length: int) -> int:
    d_model = MODEL_CONFIG["d_model"]
    layer_count = MODEL_CONFIG["layers"]
    per_layer = (
        4 * batch_size * sequence_length * d_model * d_model
        + 2 * batch_size * sequence_length * sequence_length * d_model
    )
    return layer_count * per_layer


def verify_attention_calls(model: torch.nn.Module, example: torch.Tensor) -> int:
    modules = []
    seen: set[int] = set()
    for layer in model.encoder_layers:
        module = layer.self_attn
        if id(module) not in seen:
            modules.append(module)
            seen.add(id(module))
    calls = 0

    def count_call(*arguments):
        nonlocal calls
        if len(arguments) != 3:
            raise AssertionError("Unexpected forward-hook arguments.")
        calls += 1

    handles = [module.register_forward_hook(count_call) for module in modules]
    try:
        with torch.no_grad():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
    expected = MODEL_CONFIG["layers"]
    if calls != expected:
        raise AssertionError(
            f"Expected {expected} encoder self-attention executions, got {calls}."
        )
    return calls


def benchmark(
    cache_dir: Path,
    checkpoint_root: Path,
    output_csv: Path,
    warmups: int = WARMUPS,
    timed_iterations: int = TIMED_ITERATIONS,
) -> list[dict[str, Any]]:
    if output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite latency results: {output_csv}")
    if warmups < 1 or timed_iterations < 2:
        raise ValueError("Use at least one warm-up and two timed iterations.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    train, validation, _ = run_ao_v1._read_split(cache_dir.resolve())
    normalization = run_ao_v1._train_only_normalization(train)
    mean = torch.from_numpy(normalization["mean"])
    std = torch.from_numpy(normalization["std"])
    template = torch.from_numpy(validation["observed_boxes"][:1])
    template = (template - mean) / std
    cpu = _cpu_name()
    rows = []

    for model_id in MODEL_IDS:
        checkpoint_path = (
            checkpoint_root / model_id / "seed_20260928" / "best_model.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = build_ao_model(model_id).eval()
        model.load_state_dict(payload["model_state"])
        if int(payload["parameter_count"]) != unique_trainable_parameter_count(model):
            raise AssertionError(f"Checkpoint parameter count mismatch for {model_id}.")
        if model_id == "AO-3" and not verify_satie_identity(model):
            raise AssertionError("AO-3 attention module identity is not shared.")
        calls_per_forward = verify_attention_calls(model, template)

        for batch_size in BATCH_SIZES:
            batch = template.repeat(batch_size, 1, 1).contiguous()
            with torch.no_grad():
                for _ in range(warmups):
                    model(batch)

            stop_sampler = threading.Event()
            peak_memory = [run_ao_v1._memory_bytes() or 0]

            def sample_memory() -> None:
                while not stop_sampler.wait(0.02):
                    peak_memory[0] = max(
                        peak_memory[0], run_ao_v1._memory_bytes() or 0
                    )

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            durations = []
            try:
                with torch.no_grad():
                    for _ in range(timed_iterations):
                        started = time.perf_counter()
                        model(batch)
                        durations.append(
                            (time.perf_counter() - started) * 1000.0
                        )
            finally:
                stop_sampler.set()
                sampler.join()
            peak_memory[0] = max(
                peak_memory[0], run_ao_v1._memory_bytes() or 0
            )
            values = np.asarray(durations, dtype=np.float64)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            rows.append({
                "model": model_id,
                "checkpoint_source": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "batch_size": batch_size,
                "cpu": cpu,
                "platform": platform.platform(),
                "thread_count": 4,
                "warmups": warmups,
                "timed_iterations": timed_iterations,
                "median_ms": float(median),
                "q25_ms": float(q25),
                "q75_ms": float(q75),
                "iqr_ms": float(q75 - q25),
                "parameter_count": unique_trainable_parameter_count(model),
                "encoder_attention_calls_per_forward": calls_per_forward,
                "encoder_self_attention_macs": encoder_self_attention_macs(
                    batch_size, template.shape[1]
                ),
                "peak_sampled_process_memory_bytes": peak_memory[0],
                "weights_only": True,
                "official_test_data_loaded": False,
            })
            print(json.dumps(rows[-1], indent=2), flush=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_csv.with_suffix(".json")
    json_path.write_text(json.dumps({
        "schema": "ao_cpu_latency_v1",
        "cache_sha256": {
            name: run_ao_v1._sha256(cache_dir / name)
            for name in ("train.npz", "validation.npz", "split_manifest.json")
        },
        "warmups": warmups,
        "timed_iterations": timed_iterations,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results_paper/splits/ao_annotation_only_v1"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results_paper/AO_initial_screen_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_paper/AO_confirmation_preflight/latency.csv"),
    )
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--timed-iterations", type=int, default=TIMED_ITERATIONS)
    args = parser.parse_args()
    benchmark(
        args.cache_dir,
        args.checkpoint_root,
        args.output,
        args.warmups,
        args.timed_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
