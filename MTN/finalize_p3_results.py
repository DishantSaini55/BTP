"""Add validation-window comparisons and descriptive CV/motion summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from .run_e8_v2 import atomic_write_csv, atomic_write_json
except ImportError:
    from run_e8_v2 import atomic_write_csv, atomic_write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def enrich(output_root: Path) -> dict:
    path = output_root / "P3_PER_WINDOW_COMPARISON.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_csv(path)
    baseline = {
        row["sample_id"]: float(row["raw_pixel_mse"])
        for row in rows
        if row.get("configuration") == "p3_a"
    }
    if not baseline:
        raise RuntimeError("P3-A validation rows are missing from the comparison.")
    fieldnames = list(rows[0])
    derived_fields = (
        "cv_pixel_mse",
        "mse_improvement_vs_cv",
        "improved_over_cv",
        "worsened_vs_cv",
        "cv_motion_class",
    )
    fieldnames.extend(field for field in derived_fields if field not in fieldnames)
    aggregates: dict[str, dict] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in baseline:
            raise RuntimeError(f"No paired P3-A CV row for validation window {sample_id}.")
        cv_mse = baseline[sample_id]
        model_mse = float(row["raw_pixel_mse"])
        difference = cv_mse - model_mse
        speed = float(row["observed_center_speed_px_per_frame"])
        row.update({
            "cv_pixel_mse": f"{cv_mse:.12g}",
            "mse_improvement_vs_cv": f"{difference:.12g}",
            "improved_over_cv": str(model_mse < cv_mse).lower(),
            "worsened_vs_cv": str(model_mse > cv_mse).lower(),
            "cv_motion_class": "stationary" if speed <= 1e-6 else "moving",
        })
        key = f"{row['configuration']}:{row['policy']}"
        group = aggregates.setdefault(key, {
            "configuration": row["configuration"],
            "policy": row["policy"],
            "sample_count": 0,
            "improved_count": 0,
            "worsened_count": 0,
            "unchanged_count": 0,
            "correction_rms_pixels": [],
            "gate_probability": [],
            "motion": {},
        })
        group["sample_count"] += 1
        group["improved_count"] += int(model_mse < cv_mse)
        group["worsened_count"] += int(model_mse > cv_mse)
        group["unchanged_count"] += int(model_mse == cv_mse)
        group["correction_rms_pixels"].append(float(row["correction_rms_pixels"]))
        group["gate_probability"].append(float(row["admission_probability"]))
        motion = group["motion"].setdefault(row["cv_motion_class"], {
            "sample_count": 0,
            "mean_raw_pixel_mse": 0.0,
            "mean_cv_pixel_mse": 0.0,
        })
        motion["sample_count"] += 1
        motion["mean_raw_pixel_mse"] += model_mse
        motion["mean_cv_pixel_mse"] += cv_mse
    for group in aggregates.values():
        n = group["sample_count"]
        group["fraction_improved_over_cv"] = group["improved_count"] / n
        group["fraction_worsened_vs_cv"] = group["worsened_count"] / n
        group["mean_correction_rms_pixels"] = float(np.mean(group.pop("correction_rms_pixels")))
        group["mean_admission_probability"] = float(np.mean(group.pop("gate_probability")))
        for motion in group["motion"].values():
            count = motion["sample_count"]
            motion["mean_raw_pixel_mse"] /= count
            motion["mean_cv_pixel_mse"] /= count
    atomic_write_csv(path, rows, fieldnames)
    payload = {
        "validation_only": True,
        "one_seed_exploratory": True,
        "stationary_definition": "observed center displacement <= 1e-6 pixel per source-frame interval",
        "groups": list(aggregates.values()),
    }
    atomic_write_json(output_root / "P3_PER_WINDOW_AGGREGATES.json", payload)
    return payload


if __name__ == "__main__":
    root = Path("results_paper/P3_initial_screen_v1")
    print(json.dumps(enrich(root), indent=2))
