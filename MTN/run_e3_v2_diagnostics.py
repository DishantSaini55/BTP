"""Versioned E3 gate-input range audit using the final observed bbox."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_from_scratch_e1 import SPLIT
except ImportError:
    from run_from_scratch_e1 import SPLIT


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
COORDINATE_NAMES = ("bbox_coord_0", "bbox_coord_1", "bbox_coord_2", "bbox_coord_3")


def _statistics(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Gate-input values must be non-empty and finite.")
    quantile_values = np.quantile(flat, QUANTILES, method="linear")
    return {
        "count": int(flat.size),
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "standard_deviation_population": float(np.std(flat, ddof=0)),
        "quantiles": {
            f"q{int(q * 100):02d}": float(value)
            for q, value in zip(QUANTILES, quantile_values)
        },
        "fraction_outside_minus5_plus5": float(np.mean((flat < -5.0) | (flat > 5.0))),
        "fraction_below_minus5": float(np.mean(flat < -5.0)),
        "fraction_above_plus5": float(np.mean(flat > 5.0)),
    }


def analyze(observe_length: int = 15) -> dict[str, Any]:
    if observe_length < 1:
        raise ValueError("observe_length must be positive.")
    flow_path = SPLIT / "flow_splits.pkl"
    if not flow_path.exists():
        raise FileNotFoundError(f"Cached split not found: {flow_path}")
    with flow_path.open("rb") as handle:
        split = pickle.load(handle)

    train_boxes = np.concatenate([sample["bbox"] for sample in split["train"]], axis=0)
    train_mean = train_boxes.mean(axis=0)
    train_std = np.maximum(train_boxes.std(axis=0), 1e-6)
    results: dict[str, Any] = {
        "diagnostic_version": "E3_v2_observed_gate_input",
        "observe_length": observe_length,
        "gate_input_frame_index_zero_based": observe_length - 1,
        "gate_input_frame_index_one_based": observe_length,
        "normalization_fit_role": "train",
        "train_box_mean": [float(x) for x in train_mean],
        "train_box_std": [float(x) for x in train_std],
        "quantile_method": "NumPy linear",
        "standard_deviation": "population (ddof=0)",
        "outside_interval": "strictly less than -5 or strictly greater than 5",
        "sample_counts": {},
        "splits": {},
        "test_usage": "descriptive range only; not used to select any threshold, input, or hyperparameter",
    }

    for role in ("train", "validation", "test"):
        samples = split[role]
        if not samples:
            raise ValueError(f"Split has no samples: {role}")
        if any(sample["bbox"].shape[0] < observe_length for sample in samples):
            raise ValueError(f"An observation is shorter than observe_length in {role}.")
        raw = np.stack([sample["bbox"][observe_length - 1] for sample in samples]).astype(np.float64)
        normalized = (raw - train_mean) / train_std
        per_coordinate = {}
        for index, name in enumerate(COORDINATE_NAMES):
            per_coordinate[name] = {
                "raw_pixels": _statistics(raw[:, index]),
                "normalized_gate_input": _statistics(normalized[:, index]),
            }
        results["sample_counts"][role] = len(samples)
        results["splits"][role] = {
            "raw_pixels": _statistics(raw),
            "normalized_gate_input": _statistics(normalized),
            "per_coordinate": per_coordinate,
        }
    return results


def _markdown_report(data: dict[str, Any]) -> str:
    lines = [
        "# Corrected E3 observed gate-input ranges",
        "",
        "Versioned descriptive diagnostic. The SAGE gate receives the last observed normalized bbox,",
        f"frame index {data['gate_input_frame_index_zero_based']} (zero-based; frame {data['gate_input_frame_index_one_based']} of the observation).",
        "Normalization mean/std are calculated from train boxes only.",
        "",
        "## Per-coordinate statistics",
        "",
        "Quantiles are q00, q01, q05, q25, q50, q75, q95, q99, q100; SD is population SD (`ddof=0`).",
        "Outside fraction counts values strictly outside `[-5, 5]`.",
        "",
        "| Split | Coordinate | Raw min..max px | Normalized min..max | Mean | SD | q01 | q05 | q50 | q95 | q99 | Outside ±5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("train", "validation", "test"):
        split = data["splits"][role]
        for name in COORDINATE_NAMES:
            raw = split["per_coordinate"][name]["raw_pixels"]
            norm = split["per_coordinate"][name]["normalized_gate_input"]
            lines.append(
                f"| {role} | {name} | {raw['minimum']:.6f}..{raw['maximum']:.6f} | "
                f"{norm['minimum']:.6f}..{norm['maximum']:.6f} | {norm['mean']:.6f} | "
                f"{norm['standard_deviation_population']:.6f} | "
                f"{norm['quantiles']['q01']:.6f} | {norm['quantiles']['q05']:.6f} | "
                f"{norm['quantiles']['q50']:.6f} | {norm['quantiles']['q95']:.6f} | "
                f"{norm['quantiles']['q99']:.6f} | {norm['fraction_outside_minus5_plus5']:.6f} |"
            )
    lines += [
        "",
        "## Aggregate normalized ranges",
        "",
        "| Split | n windows | Min | Max | Mean | SD | q01 | q05 | q50 | q95 | q99 | Outside ±5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("train", "validation", "test"):
        stats = data["splits"][role]["normalized_gate_input"]
        q = stats["quantiles"]
        lines.append(
            f"| {role} | {data['sample_counts'][role]} | {stats['minimum']:.6f} | "
            f"{stats['maximum']:.6f} | {stats['mean']:.6f} | "
            f"{stats['standard_deviation_population']:.6f} | {q['q01']:.6f} | "
            f"{q['q05']:.6f} | {q['q50']:.6f} | {q['q95']:.6f} | "
            f"{q['q99']:.6f} | {stats['fraction_outside_minus5_plus5']:.6f} |"
        )
    lines += [
        "",
        "## Historical diagnostic correction",
        "",
        "The historical `MTN/run_e3.py:x_t_ranges` indexed `sample['bbox'][-1]`, the final",
        "frame of the 60-frame sample. The gate input is `inp[:, -1]` after the 15-frame",
        "observation slice, corresponding to `sample['bbox'][observe_length - 1]`.",
        "Thus the old range included future-target coordinates and was not a measurement of",
        "the gate input. This is a diagnostic frame-selection error; by itself it does not",
        "establish training leakage or invalidate the saved model metrics.",
        "",
        "The test split is included only to report descriptive observed-input ranges. Its",
        "statistics were not used to choose clipping thresholds, gate inputs, or",
        "hyperparameters. The historical E3 report and output files are preserved.",
        "",
        "Machine-readable detail: `results_paper/diagnostics/E3_v2_observed_input_ranges.json`.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(data: dict[str, Any], json_path: Path) -> None:
    report_path = json_path.parent.parent / "E3_CORRECTED_REPORT.md"
    csv_path = json_path.with_suffix(".csv")
    for path in (json_path, report_path, csv_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(data), encoding="utf-8")
    rows = []
    for role in ("train", "validation", "test"):
        for coordinate in COORDINATE_NAMES:
            for scale in ("raw_pixels", "normalized_gate_input"):
                item = data["splits"][role]["per_coordinate"][coordinate][scale]
                rows.append({
                    "split": role,
                    "coordinate": coordinate,
                    "scale": scale,
                    "count": item["count"],
                    "minimum": item["minimum"],
                    "maximum": item["maximum"],
                    "mean": item["mean"],
                    "standard_deviation_population": item["standard_deviation_population"],
                    **item["quantiles"],
                    "fraction_outside_minus5_plus5": item["fraction_outside_minus5_plus5"],
                })
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe-length", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_paper/diagnostics/E3_v2_observed_input_ranges.json"),
    )
    args = parser.parse_args()
    result = analyze(args.observe_length)
    write_outputs(result, args.output)
    print(json.dumps({
        "json": str(args.output),
        "report": str(args.output.parent.parent / "E3_CORRECTED_REPORT.md"),
        "sample_counts": result["sample_counts"],
        "normalized_ranges": {
            role: result["splits"][role]["normalized_gate_input"]
            for role in ("train", "validation", "test")
        },
    }, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
