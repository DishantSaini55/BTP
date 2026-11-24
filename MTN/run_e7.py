"""E7 CPU timing and FLOP benchmark for the five E1 model families."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
    from .run_from_scratch_e1 import NAMES, RESULTS, SEEDS, load_data, model_for, set_seed, forward
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared
    from run_from_scratch_e1 import NAMES, RESULTS, SEEDS, load_data, model_for, set_seed, forward

THREAD_COUNT = 4
WARMUPS = 50
TIMED_ITERATIONS = 500
BATCH_SIZES = (1, 8)


def cpu_info() -> str:
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()"],
            text=True,
        ).strip()
        if output:
            return output
    except Exception:
        pass
    return platform.processor() or platform.platform()


def make_batch(template: dict, batch_size: int) -> dict:
    batch = {}
    for key, value in template.items():
        if torch.is_tensor(value):
            batch[key] = value[:1].repeat((batch_size,) + (1,) * (value.ndim - 1))
        elif isinstance(value, list):
            batch[key] = value[:1] * batch_size
        else:
            batch[key] = value
    return batch


class ProfilingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> torch.Tensor:
        return forward(self.model, batch)


def main() -> None:
    torch.set_num_threads(THREAD_COUNT)
    torch.set_num_interop_threads(THREAD_COUNT)
    datasets, stats = load_data()
    template = next(iter(DataLoader(datasets["train"], batch_size=1, shuffle=False)))
    cpu = cpu_info()
    rows = []

    for name in NAMES:
        set_seed(SEEDS[0])
        model = model_for(name).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint = (
            RESULTS / "runs" / "E1" / {
                NAMES[0]: "mtn_baseline",
                NAMES[1]: "pace_net_without_satie_sage_plus_masc",
                NAMES[2]: "pace_net_sage_plus_masc_plus_satie",
                NAMES[3]: "pace_net_with_route",
                NAMES[4]: "pace_net_with_route_and_satie",
            }[name] / "20260928" / "checkpoint.pth"
        )
        checkpoint_bytes = checkpoint.stat().st_size if checkpoint.exists() else None
        for batch_size in BATCH_SIZES:
            batch = make_batch(template, batch_size)
            with torch.no_grad():
                for _ in range(WARMUPS):
                    forward(model, batch)
                durations = []
                for _ in range(TIMED_ITERATIONS):
                    started = time.perf_counter()
                    forward(model, batch)
                    durations.append((time.perf_counter() - started) * 1000)
            profile_model = model_for(name).eval()
            profile_batch = make_batch(template, batch_size)
            macs, params = profile(ProfilingWrapper(profile_model), inputs=(profile_batch,), verbose=False)
            durations = np.asarray(durations)
            median = float(np.median(durations))
            q25, q75 = np.quantile(durations, [0.25, 0.75])
            rows.append({
                "model": name,
                "batch_size": batch_size,
                "cpu": cpu,
                "thread_count": THREAD_COUNT,
                "warmups": WARMUPS,
                "timed_iterations": TIMED_ITERATIONS,
                "median_ms": median,
                "iqr_ms": float(q75 - q25),
                "q25_ms": float(q25),
                "q75_ms": float(q75),
                "thop_macs": int(macs),
                "thop_params": int(params),
                "parameter_count": parameter_count,
                "checkpoint_seed_20260928_bytes": checkpoint_bytes,
                "platform": platform.platform(),
            })
            print(json.dumps(rows[-1], indent=2))

    out = RESULTS / "runs" / "E7"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    with (out / "timing_flops.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "timing_flops.json").write_text(json.dumps({
        "cpu": cpu,
        "thread_count": THREAD_COUNT,
        "warmups": WARMUPS,
        "timed_iterations": TIMED_ITERATIONS,
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
