"""Audit the exact observed-frame SAGE input and training-only box statistics."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_from_scratch_e1 import SPLIT
except ImportError:
    from run_from_scratch_e1 import SPLIT


OBSERVED_STEPS = 15
GATE_OBSERVED_INDEX = OBSERVED_STEPS - 1


def _range(values: np.ndarray) -> dict[str, Any]:
    flat = values.reshape(-1, values.shape[-1])
    return {
        "coordinate_min": [float(value) for value in flat.min(axis=0)],
        "coordinate_max": [float(value) for value in flat.max(axis=0)],
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "abs_max": float(np.abs(flat).max()),
        "components_outside_abs_5_fraction": float(np.mean(np.abs(flat) > 5.0)),
    }


def audit() -> dict[str, Any]:
    path = SPLIT / "flow_splits.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Required cached split missing: {path}")
    with path.open("rb") as handle:
        split = pickle.load(handle)
    train_boxes = np.concatenate([sample["bbox"] for sample in split["train"]])
    mean = train_boxes.mean(axis=0)
    std = np.maximum(train_boxes.std(axis=0), 1e-6)

    observed_ranges = {}
    historical_endpoint_ranges = {}
    observed_relative = {}
    clip_fractions = {}
    relative_by_role = {}
    for role in ("train", "validation", "test"):
        observed_raw = np.stack([sample["bbox"][GATE_OBSERVED_INDEX] for sample in split[role]])
        observed_z = (observed_raw - mean) / std
        observed_ranges[role] = _range(observed_z)
        clip_fractions[role] = {
            "components_below_minus_5": float(np.mean(observed_z < -5.0)),
            "components_above_plus_5": float(np.mean(observed_z > 5.0)),
            "components_clipped_total": float(np.mean(np.abs(observed_z) > 5.0)),
        }

        # This reproduces the old E3 expression for diagnosis only. bbox[-1]
        # is frame 60 (45 steps after the gate's frame 15), not the gate input.
        endpoint_raw = np.stack([sample["bbox"][-1] for sample in split[role]])
        historical_endpoint_ranges[role] = _range((endpoint_raw - mean) / std)

        relative = np.stack([
            (sample["bbox"][:OBSERVED_STEPS] - sample["bbox"][GATE_OBSERVED_INDEX]) / std
            for sample in split[role]
        ])
        relative_by_role[role] = relative
        observed_relative[role] = _range(relative)

    train_scale = np.quantile(np.abs(relative_by_role["train"]), 0.99, axis=(0, 1))
    train_scale = np.maximum(train_scale, 1e-6)
    bounded_train = np.tanh(relative_by_role["train"] / train_scale)
    bounded_validation = np.tanh(relative_by_role["validation"] / train_scale)

    return {
        "split_version": "gp6_from_scratch_v1",
        "normalization_fit_role": "train",
        "normalization_box_mean": [float(value) for value in mean],
        "normalization_box_std": [float(value) for value in std],
        "sample_counts": {role: len(split[role]) for role in ("train", "validation", "test")},
        "gate_observed_frame_index_zero_based": GATE_OBSERVED_INDEX,
        "gate_observed_frame_number_within_window_one_based": OBSERVED_STEPS,
        "observed_gate_input_normalized_ranges": observed_ranges,
        "observed_gate_input_clip_fractions_at_abs_5": clip_fractions,
        "historical_e3_endpoint_ranges_diagnostic_only": historical_endpoint_ranges,
        "historical_e3_expression": "sample['bbox'][-1] (frame 60), not sample['bbox'][14] (last observed frame)",
        "historical_endpoint_is_future_target": True,
        "relative_observed_box_delta_ranges_normalized_by_train_box_std": observed_relative,
        "relative_motion_tanh_scale_train_abs_q99": [float(value) for value in train_scale],
        "bounded_relative_motion_ranges": {
            "train": _range(bounded_train),
            "validation": _range(bounded_validation),
            "test_not_used_for_scale_selection": True,
        },
        "note": (
            "The clip fractions describe the fixed ±5 E8 transform; they do not select a threshold. "
            "The test split is inspected only to verify the historical gate-input range, not to choose "
            "a representation, scale, or training hyperparameter."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
