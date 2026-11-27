"""Validation-only aggregation for the Phase 6A AO confirmation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import run_ao_v1
except ImportError:
    import run_ao_v1


SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)
MODELS = ("AO-0", "AO-1", "AO-2", "AO-3")
METRICS = ("mse", "normalized_mse", "cmse", "cfmse")
PAIRS = (
    ("AO-1", "AO-0", "Constant-position transformer vs CV"),
    ("AO-2", "AO-0", "Unshared CV-residual transformer vs CV"),
    ("AO-3", "AO-0", "SATIE CV-residual transformer vs CV"),
    ("AO-2", "AO-1", "CV-residual vs constant-position transformer"),
    ("AO-3", "AO-2", "SATIE vs unshared encoder attention"),
)
T_CRITICAL_95_DF4 = 2.7764451051977987
DEFAULT_CACHE = Path("results_paper/splits/ao_annotation_only_v1")
DEFAULT_ROOT = Path("results_paper/AO_confirmation_v1")
DEFAULT_MANIFEST = Path("results_paper/AO_CONFIRMATION_MANIFEST_v2.json")
DEFAULT_CLASSICAL_ROOT = Path("results_paper/AO_confirmation_v1/classical")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty summary: {path}")
    run_ao_v1._atomic_csv(path, rows, list(rows[0]))


def _paired_stats(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if not values.size:
        raise ValueError("Paired summary requires at least one difference.")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else None
    half_width = (
        T_CRITICAL_95_DF4 * sd / math.sqrt(5)
        if values.size == 5 and sd is not None else None
    )
    return {
        "n_seeds": int(values.size),
        "mean_delta_first_minus_second": mean,
        "sd_delta": sd,
        "ci95_t_low": mean - half_width if half_width is not None else None,
        "ci95_t_high": mean + half_width if half_width is not None else None,
        "degrees_of_freedom": int(values.size - 1) if values.size > 1 else None,
        "dz": mean / sd if sd not in (None, 0.0) else None,
        "first_model_wins": int(np.sum(values < 0)),
    }


def _track_bootstrap(
    root: Path,
    first: str,
    second: str,
    seeds: tuple[int, ...],
    metric: str,
    repetitions: int,
    random_seed: int,
    first_root: Path | None = None,
    second_root: Path | None = None,
) -> dict[str, Any]:
    by_track: dict[tuple[str, str], list[float]] = {}
    for seed in seeds:
        left_rows = _read_csv(
            (first_root or root)
            / first / f"seed_{seed}" / "validation_per_sample.csv"
        )
        right_rows = _read_csv(
            (second_root or root)
            / second / f"seed_{seed}" / "validation_per_sample.csv"
        )
        left = {row["sample_id"]: row for row in left_rows}
        right = {row["sample_id"]: row for row in right_rows}
        if left.keys() != right.keys():
            raise ValueError(f"Paired sample IDs differ for {first}/{second}/{seed}.")
        for sample_id, left_row in left.items():
            right_row = right[sample_id]
            if (
                left_row["video_id"] != right_row["video_id"]
                or left_row["track_id"] != right_row["track_id"]
            ):
                raise ValueError(f"Track identity differs for paired sample {sample_id}.")
            cluster = (left_row["video_id"], left_row["track_id"])
            by_track.setdefault(cluster, []).append(
                float(left_row[metric]) - float(right_row[metric])
            )
    clusters = sorted(by_track)
    if len(clusters) < 2:
        raise ValueError("Track-cluster bootstrap requires at least two tracks.")
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        values = [
            value
            for cluster_index in draw
            for value in by_track[clusters[int(cluster_index)]]
        ]
        estimates[index] = float(np.mean(values))
    return {
        "n_track_clusters": len(clusters),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": random_seed,
        "bootstrap_ci95_low": float(np.quantile(estimates, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_caveat": (
            "Conditional on two fixed GP7 validation videos; windows overlap; "
            "does not establish population-level video generalization."
        ),
    }


def _correction_rms(
    prediction: np.ndarray,
    observed: np.ndarray,
    model_id: str,
) -> np.ndarray:
    if prediction.shape != (len(observed), 45, 4):
        raise ValueError(f"Unexpected prediction shape for {model_id}: {prediction.shape}")
    if model_id in ("AO-1", "AO-CP"):
        anchor = np.repeat(observed[:, -1:, :], 45, axis=1)
    else:
        velocity = (observed[:, -1, :] - observed[:, -3, :]) / 2.0
        steps = np.arange(1, 46, dtype=np.float32).reshape(1, 45, 1)
        anchor = observed[:, -1:, :] + steps * velocity[:, None, :]
    delta = prediction - anchor
    return np.sqrt(np.mean(np.sum(delta ** 2, axis=2), axis=1))


def _check_provenance(
    root: Path,
    cache: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(manifest_path)
    state = _read_json(root / "orchestration_state.json")
    provenance = _read_json(root / "split_provenance.json")
    if manifest.get("schema") != "ao_confirmation_protocol_v1":
        raise ValueError("Unexpected confirmation manifest schema.")
    if state.get("manifest_sha256") != run_ao_v1._sha256(manifest_path):
        raise ValueError("Orchestration state does not match the frozen manifest.")
    expected_state = {
        "seeds": list(SEEDS),
        "models": ["AO-1", "AO-2", "AO-3"],
        "max_epochs": 30,
        "patience": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "cpu_threads": 4,
        "global_budget_seconds": 14 * 60 * 60,
        "per_run_budget_seconds": 2 * 60 * 60,
        "final_two_seed_reserve_seconds": 6 * 60 * 60,
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise ValueError("Orchestration settings differ from the frozen protocol.")
    expected_hashes = {
        name: run_ao_v1._sha256(cache / name)
        for name in ("train.npz", "validation.npz", "split_manifest.json")
    }
    if manifest.get("cache_sha256") != expected_hashes:
        raise ValueError("Current caches differ from the frozen manifest.")
    if provenance.get("cache_sha256") != expected_hashes:
        raise ValueError("Run provenance does not match the frozen caches.")
    if provenance.get("official_test_sets_read") != []:
        raise ValueError("Run provenance does not certify test isolation.")
    if manifest.get("official_test_sets_read") != []:
        raise ValueError("Frozen manifest does not certify test isolation.")
    repository_root = manifest_path.parent.parent
    for relative_path, expected_hash in manifest.get("source_sha256", {}).items():
        source = repository_root / Path(relative_path)
        if not source.is_file() or run_ao_v1._sha256(source) != expected_hash:
            raise ValueError(f"Frozen source changed after confirmation: {relative_path}")
    return manifest, state


def _collect_completed_runs(
    root: Path,
    seeds: tuple[int, ...],
    cache_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    incomplete = []
    for model_id in MODELS:
        for seed in seeds:
            run = root / model_id / f"seed_{seed}"
            status_path = run / "status.json"
            metrics_path = run / "metrics.json"
            if not status_path.is_file() or not metrics_path.is_file():
                incomplete.append(f"{model_id}/seed_{seed}: missing status or metrics")
                continue
            status = _read_json(status_path)
            metrics = _read_json(metrics_path)
            if (
                status.get("state") != "completed"
                or metrics.get("status") != "completed"
                or not metrics.get("validation_complete")
            ):
                incomplete.append(
                    f"{model_id}/seed_{seed}: status={status.get('state')}, "
                    f"metrics={metrics.get('status')}"
                )
                continue
            if metrics.get("cache_sha256") != cache_hashes:
                raise ValueError(f"Cache provenance mismatch for {model_id}/seed_{seed}.")
            if metrics.get("seed") != seed or metrics.get("model_id") != model_id:
                raise ValueError(f"Run identity mismatch for {model_id}/seed_{seed}.")
            if model_id != "AO-0":
                config = _read_json(run / "run_config.json")
                expected_config = {
                    "model_id": model_id,
                    "seed": seed,
                    "max_epochs": 30,
                    "patience": 10,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "cpu_threads": 4,
                    "validation_only": True,
                    "cache_sha256": cache_hashes,
                }
                if any(
                    config.get(key) != value
                    for key, value in expected_config.items()
                ):
                    raise ValueError(
                        f"Run config differs from the frozen protocol: {model_id}/seed_{seed}."
                    )
            if len(_read_csv(run / "validation_per_sample.csv")) != 432:
                raise ValueError(f"Run is not full-validation-split: {model_id}/seed_{seed}.")
            rows.append({
                "model_id": model_id,
                "seed": seed,
                "run": run,
                "metrics": metrics,
            })
    return rows, incomplete


def finalize(
    root: Path = DEFAULT_ROOT,
    cache: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
    classical_root: Path = DEFAULT_CLASSICAL_ROOT,
    seeds: tuple[int, ...] = SEEDS,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260928,
) -> dict[str, Any]:
    root, cache, manifest_path = root.resolve(), cache.resolve(), manifest_path.resolve()
    if seeds != SEEDS:
        raise ValueError(f"Finalization requires frozen seeds {SEEDS}.")
    if bootstrap_repetitions < 1000:
        raise ValueError("Use at least 1,000 track-bootstrap repetitions.")
    output_names = (
        "AO_CONFIRMATION_SEED_SUMMARY.csv",
        "AO_CONFIRMATION_MODEL_SUMMARY.csv",
        "AO_CONFIRMATION_PAIRED.csv",
        "AO_CONFIRMATION_PER_VIDEO.csv",
        "AO_CONFIRMATION_PER_TRACK.csv",
        "AO_CONFIRMATION_CORRECTION_MAGNITUDE.csv",
        "AO_CONFIRMATION_CORRECTION_PER_WINDOW.csv",
        "AO_CONFIRMATION_FINAL_REPORT.md",
        "AO_CONFIRMATION_FINAL_METADATA.json",
        "AO_CONFIRMATION_RUNTIME_REPORT.md",
    )
    existing = [name for name in output_names if (root / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite final outputs: {existing}")
    manifest, orchestration = _check_provenance(root, cache, manifest_path)
    run_provenance = _read_json(root / "split_provenance.json")
    completed, incomplete = _collect_completed_runs(
        root, seeds, manifest["cache_sha256"]
    )
    classical_report_path = (
        classical_root.resolve() / "classical_baseline_report.json"
    )
    if not classical_report_path.is_file():
        raise FileNotFoundError(
            f"Required classical preflight is missing: {classical_report_path}"
        )
    classical_report = _read_json(classical_report_path)
    if (
        classical_report.get("official_test_sets_read") != []
        or classical_report.get("cache_sha256") != manifest["cache_sha256"]
    ):
        raise ValueError("Classical preflight provenance does not match this freeze.")
    for model_id in ("AO-CP", "AO-RIDGE"):
        recorded_metrics = classical_report.get(model_id)
        run = classical_root.resolve() / model_id / "seed_20260928"
        if (
            not isinstance(recorded_metrics, dict)
            or recorded_metrics.get("status") != "completed"
        ):
            raise ValueError(f"Missing completed classical baseline {model_id}.")
        metrics = dict(recorded_metrics)
        if model_id == "AO-RIDGE":
            metrics["ridge_alpha_selected"] = (
                classical_report["ridge_alpha_selected"]
            )
            metrics["model_selection_seconds"] = (
                classical_report["ridge_model_selection_seconds"]
            )
        completed.append({
            "model_id": model_id,
            "seed": 20260928,
            "run": run,
            "metrics": metrics,
        })
    if orchestration.get("status") == "completed" and incomplete:
        raise RuntimeError(
            "Orchestration reports complete but run records are incomplete:\n"
            + "\n".join(incomplete)
        )
    completed_map = {
        (row["model_id"], row["seed"]): row["metrics"]
        for row in completed
    }
    baseline_seeds = [
        seed for seed in seeds if ("AO-0", seed) in completed_map
    ]
    if baseline_seeds:
        reference_seed = baseline_seeds[0]
        reference = completed_map[("AO-0", reference_seed)]
        reference_rows = {
            row["sample_id"]: row
            for row in _read_csv(
                root / "AO-0" / f"seed_{reference_seed}" / "validation_per_sample.csv"
            )
        }
        for seed in baseline_seeds[1:]:
            candidate = completed_map[("AO-0", seed)]
            if any(
                not np.isclose(reference[metric], candidate[metric], rtol=0, atol=1e-12)
                for metric in METRICS
            ):
                raise ValueError("Deterministic AO-0 metrics differ across seeds.")
            candidate_rows = {
                row["sample_id"]: row
                for row in _read_csv(
                    root / "AO-0" / f"seed_{seed}" / "validation_per_sample.csv"
                )
            }
            if reference_rows.keys() != candidate_rows.keys():
                raise ValueError("Deterministic AO-0 validation sample IDs differ.")
            for sample_id in reference_rows:
                if any(
                    not np.isclose(
                        float(reference_rows[sample_id][metric]),
                        float(candidate_rows[sample_id][metric]),
                        rtol=0,
                        atol=1e-12,
                    )
                    for metric in METRICS
                ):
                    raise ValueError(
                        f"Deterministic AO-0 outputs differ for {sample_id}."
                    )
    seed_rows = []
    for row in completed:
        model_id, seed, metrics = row["model_id"], row["seed"], row["metrics"]
        run = row["run"]
        checkpoint = run / "best_model.pt"
        latest_checkpoint = run / "latest_checkpoint.pt"
        ridge_coefficients = run / "ridge_coefficients.npz"
        seed_rows.append({
            "model_id": model_id,
            "seed": seed,
            "status": metrics["status"],
            "selection_split": metrics.get(
                "selection_split",
                "validation" if model_id in ("AO-1", "AO-2", "AO-3")
                else "none",
            ),
            "epochs_completed": metrics["epochs_completed"],
            "best_epoch": metrics.get("best_epoch"),
            "train_normalized_mse": metrics.get("train_normalized_mse"),
            "validation_normalized_mse": metrics["normalized_mse"],
            "validation_raw_pixel_mse": metrics["mse"],
            "validation_cmse": metrics["cmse"],
            "validation_cfmse": metrics["cfmse"],
            "validation_horizon_1_mse": metrics.get("horizon_1_mse"),
            "validation_horizon_15_mse": metrics.get("horizon_15_mse"),
            "validation_horizon_30_mse": metrics.get("horizon_30_mse"),
            "validation_horizon_45_mse": metrics.get("horizon_45_mse"),
            "macro_video_mse": metrics.get("macro_video_mse"),
            "macro_video_cmse": metrics.get("macro_video_cmse"),
            "macro_video_cfmse": metrics.get("macro_video_cfmse"),
            "macro_track_mse": metrics.get("macro_track_mse"),
            "macro_track_cmse": metrics.get("macro_track_cmse"),
            "macro_track_cfmse": metrics.get("macro_track_cfmse"),
            "parameter_count": metrics["parameter_count"],
            "ridge_alpha_selected": metrics.get("ridge_alpha_selected"),
            "training_seconds": metrics["training_seconds"],
            "model_selection_seconds": metrics.get("model_selection_seconds"),
            "run_wall_clock_seconds": metrics.get("run_wall_clock_seconds"),
            "peak_process_memory_bytes": metrics.get("peak_process_memory_bytes"),
            "best_checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
            "latest_checkpoint_bytes": (
                latest_checkpoint.stat().st_size if latest_checkpoint.is_file() else 0
            ),
            "best_checkpoint_sha256": (
                run_ao_v1._sha256(checkpoint) if checkpoint.is_file() else ""
            ),
            "latest_checkpoint_sha256": (
                run_ao_v1._sha256(latest_checkpoint)
                if latest_checkpoint.is_file() else ""
            ),
            "ridge_coefficients_sha256": (
                run_ao_v1._sha256(ridge_coefficients)
                if ridge_coefficients.is_file() else ""
            ),
        })

    model_rows = []
    for model_id in (*MODELS, "AO-CP", "AO-RIDGE"):
        values = [
            completed_map[(model_id, seed)]
            for seed in seeds if (model_id, seed) in completed_map
        ]
        if not values:
            continue
        output: dict[str, Any] = {
            "model_id": model_id,
            "n_completed_seeds": len(values),
            "unique_parameter_count": values[0]["parameter_count"],
        }
        for metric in (
            "epochs_completed", "best_epoch", "train_normalized_mse",
            "normalized_mse", "mse", "cmse", "cfmse", "training_seconds",
            "horizon_1_mse", "horizon_15_mse", "horizon_30_mse",
            "horizon_45_mse",
            "ridge_alpha_selected",
            "macro_video_mse", "macro_video_cmse", "macro_video_cfmse",
            "macro_track_mse", "macro_track_cmse", "macro_track_cfmse",
            "model_selection_seconds", "run_wall_clock_seconds",
            "peak_process_memory_bytes",
        ):
            numbers = [
                float(item[metric]) for item in values if item.get(metric) is not None
            ]
            output[f"{metric}_mean"] = float(np.mean(numbers)) if numbers else None
            output[f"{metric}_sd"] = (
                float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
            )
        model_rows.append(output)

    paired_rows = []
    complete_seeds = tuple(
        seed for seed in seeds
        if all((model, seed) in completed_map for model in MODELS)
    )
    if len(complete_seeds) == len(seeds):
        for first, second, label in PAIRS:
            for metric in METRICS:
                differences = [
                    float(completed_map[(first, seed)][metric])
                    - float(completed_map[(second, seed)][metric])
                    for seed in seeds
                ]
                row = {
                    "comparison": label,
                    "first_model": first,
                    "second_model": second,
                    "metric": metric,
                    **_paired_stats(differences),
                    "t_ci_caveat": (
                        "Seed stochasticity on the same two GP7 validation videos; "
                        "does not measure video-level generalization."
                    ),
                    "n_track_clusters": None,
                    "bootstrap_repetitions": None,
                    "bootstrap_seed": None,
                    "bootstrap_ci95_low": None,
                    "bootstrap_ci95_high": None,
                    "bootstrap_caveat": "not applicable to normalized MSE",
                }
                if metric in ("mse", "cmse", "cfmse"):
                    row.update(_track_bootstrap(
                        root, first, second, seeds, metric,
                        bootstrap_repetitions,
                        bootstrap_seed + len(paired_rows),
                    ))
                paired_rows.append(row)
    if ("AO-0", 20260928) in completed_map:
        for first in ("AO-CP", "AO-RIDGE"):
            for metric in METRICS:
                difference = (
                    float(completed_map[(first, 20260928)][metric])
                    - float(completed_map[("AO-0", 20260928)][metric])
                )
                row = {
                    "comparison": f"{first} vs AO-0; single-seed validation-only",
                    "first_model": first,
                    "second_model": "AO-0",
                    "metric": metric,
                    **_paired_stats([difference]),
                    "t_ci_caveat": (
                        "Not computed: one deterministic fit on the fixed "
                        "validation split."
                    ),
                    "n_track_clusters": None,
                    "bootstrap_repetitions": None,
                    "bootstrap_seed": None,
                    "bootstrap_ci95_low": None,
                    "bootstrap_ci95_high": None,
                    "bootstrap_caveat": "not applicable to normalized MSE",
                }
                if metric in ("mse", "cmse", "cfmse"):
                    row.update(_track_bootstrap(
                        root,
                        first,
                        "AO-0",
                        (20260928,),
                        metric,
                        bootstrap_repetitions,
                        bootstrap_seed + len(paired_rows),
                        first_root=classical_root.resolve(),
                    ))
                    if first == "AO-RIDGE":
                        row["bootstrap_caveat"] += (
                            " Alpha was selected on this same validation split; "
                            "interval is descriptive and not selection-adjusted."
                        )
                paired_rows.append(row)

    _, validation, _ = run_ao_v1._read_split(cache)
    observed = validation["observed_boxes"]
    correction_rows = []
    per_window_rows = []
    per_video_rows = []
    per_track_rows = []
    for row in completed:
        model_id, seed, run = row["model_id"], row["seed"], row["run"]
        for filename, destination in (
            ("validation_per_video.csv", per_video_rows),
            ("validation_per_track.csv", per_track_rows),
        ):
            for grouped in _read_csv(run / filename):
                destination.append({
                    "model_id": model_id,
                    "seed": seed,
                    **grouped,
                })
        payload_path = run / "validation_predictions.npz"
        with np.load(payload_path, allow_pickle=False) as payload:
            if not np.array_equal(
                payload["sample_id"].astype(str),
                validation["sample_id"].astype(str),
            ):
                raise ValueError(f"Prediction sample count mismatch: {model_id}/{seed}.")
            predictions = payload["prediction_boxes"]
            sample_ids = payload["sample_id"].astype(str)
        csv_sample_ids = [
            row["sample_id"] for row in _read_csv(
                run / "validation_per_sample.csv"
            )
        ]
        if csv_sample_ids != sample_ids.tolist():
            raise ValueError(f"CSV/NPZ sample order differs for {model_id}/{seed}.")
        values = _correction_rms(predictions, observed, model_id)
        correction_rows.append({
            "model_id": model_id,
            "seed": seed,
            "anchor": (
                "constant position" if model_id in ("AO-1", "AO-CP")
                else "constant velocity"
            ),
            "mean_correction_rms_xyxy_pixels": float(np.mean(values)),
            "median_correction_rms_xyxy_pixels": float(np.median(values)),
            "p90_correction_rms_xyxy_pixels": float(np.quantile(values, 0.9)),
            "validation_window_count": len(values),
        })
        for sample_id, value in zip(sample_ids, values):
            per_window_rows.append({
                "model_id": model_id,
                "seed": seed,
                "sample_id": sample_id,
                "correction_rms_xyxy_pixels": float(value),
            })

    complete = orchestration.get("status") == "completed" and not incomplete
    report_lines = [
        "# Phase 6A AO confirmation report",
        "",
        f"Run matrix status: **{'complete' if complete else 'incomplete'}**.",
        "Completed confirmation runs: "
        f"{sum(row['model_id'] in MODELS for row in completed)} / "
        f"{len(MODELS) * len(seeds)}.",
        f"Frozen split: {run_provenance['train_samples']} train windows across "
        f"{len(run_provenance['train_videos'])} videos; "
        f"{run_provenance['validation_samples']} validation windows across "
        f"{len(run_provenance['validation_videos'])} GP7 videos.",
        "",
        "All predictive results are validation-only on two fixed GP7 videos. "
        "The five seeds measure training stochasticity on the same split; they "
        "are not five independent datasets. Track-cluster bootstrap intervals "
        "are conditional on those two videos and overlapping windows.",
        "",
        "The official test partition was not read or evaluated. No raw video or "
        "optical-flow feature was used. No statistical significance or broad "
        "video-generalization claim is made.",
        "",
        "## Completed model means",
        "",
        "| Model | Seeds | Parameters | Epochs | Norm. MSE | Pooled MSE | Macro-video MSE | Macro-track MSE | C-MSE | CF-MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        epochs = (
            f"{row['epochs_completed_mean']:.2f}"
            if row["model_id"] in ("AO-1", "AO-2", "AO-3") else "n/a"
        )
        report_lines.append(
            f"| {row['model_id']} | {row['n_completed_seeds']} | "
            f"{row['unique_parameter_count']} | "
            f"{epochs} | "
            f"{row['normalized_mse_mean']:.6f} | {row['mse_mean']:.3f} | "
            f"{row['macro_video_mse_mean']:.3f} | "
            f"{row['macro_track_mse_mean']:.3f} | "
            f"{row['cmse_mean']:.3f} | {row['cfmse_mean']:.3f} |"
        )
    report_lines.append(
        "Macro-video and macro-track MSE give equal weight to each video or "
        "track, respectively; the accompanying per-video and per-track CSVs "
        "retain the underlying subgroup values."
    )
    report_lines.extend([
        "",
        "## Raw-pixel MSE by forecast horizon",
        "",
        "| Model | Step 1 | Step 15 | Step 30 | Step 45 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in model_rows:
        report_lines.append(
            f"| {row['model_id']} | {row['horizon_1_mse_mean']:.3f} | "
            f"{row['horizon_15_mse_mean']:.3f} | "
            f"{row['horizon_30_mse_mean']:.3f} | "
            f"{row['horizon_45_mse_mean']:.3f} |"
        )
    report_lines.extend(["", "## Architecture selection", ""])
    if complete:
        neural_rows = [
            row for row in model_rows
            if row["model_id"] in ("AO-1", "AO-2", "AO-3")
        ]
        best_neural = min(neural_rows, key=lambda row: row["mse_mean"])
        cv_mean = next(
            row["mse_mean"] for row in model_rows if row["model_id"] == "AO-0"
        )
        delta = best_neural["mse_mean"] - cv_mean
        relation = (
            f"better than AO-0 by {abs(delta):.3f}"
            if delta < 0 else (
                "equal to AO-0"
                if delta == 0 else f"worse than AO-0 by {delta:.3f}"
            )
        )
        report_lines.append(
            f"By the predeclared mean pooled raw-pixel validation MSE rule, "
            f"{best_neural['model_id']} is the best neural candidate "
            f"({best_neural['mse_mean']:.3f}), {relation} "
            f"({cv_mean:.3f}). This is a "
            "validation-only ranking, not a significance or generalization claim."
        )
        ao2 = next(
            (row for row in model_rows if row["model_id"] == "AO-2"), None
        )
        ao3 = next(
            (row for row in model_rows if row["model_id"] == "AO-3"), None
        )
        if ao2 is not None and ao3 is not None:
            reduction = (
                100.0
                * (ao2["unique_parameter_count"] - ao3["unique_parameter_count"])
                / ao2["unique_parameter_count"]
            )
            sharing_delta = ao3["mse_mean"] - ao2["mse_mean"]
            direction = (
                "lower" if sharing_delta < 0
                else "equal" if sharing_delta == 0 else "higher"
            )
            report_lines.append(
                f"AO-3 uses {reduction:.1f}% fewer unique parameters than AO-2; "
                f"its mean raw-pixel MSE is {abs(sharing_delta):.3f} {direction}. "
                "The paired uncertainty is reported below; parameter savings "
                "do not establish predictive equivalence."
            )
        ridge = next(
            (row for row in model_rows if row["model_id"] == "AO-RIDGE"), None
        )
        if ridge is not None:
            report_lines.append(
                f"AO-RIDGE is reported separately (one deterministic fit; "
                f"validation-selected alpha "
                f"{ridge['ridge_alpha_selected_mean']:.3g}; "
                f"MSE {ridge['mse_mean']:.3f}) "
                "and is not a five-seed architectural comparison."
            )
    else:
        report_lines.append(
            "No final architecture selection is made because the frozen "
            "five-seed matrix is incomplete."
        )
    report_lines.extend([
        "",
        "## Paired seed comparisons",
        "",
        "Negative first-minus-second deltas favor the first-named model. "
        "t-intervals use df = n-1; intervals are descriptive. Track bootstrap "
        "intervals are secondary and conditional on the fixed validation videos.",
        "",
        "| Comparison | Metric | Mean delta | t 95% CI | d_z | First wins | Track bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if paired_rows:
        for row in paired_rows:
            t_ci = (
                f"[{row['ci95_t_low']:.4g}, {row['ci95_t_high']:.4g}]"
                if row["ci95_t_low"] is not None else "n/a"
            )
            b_ci = (
                f"[{row['bootstrap_ci95_low']:.4g}, {row['bootstrap_ci95_high']:.4g}]"
                if row.get("bootstrap_ci95_low") is not None else "n/a"
            )
            report_lines.append(
                f"| {row['comparison']} | {row['metric']} | "
                f"{row['mean_delta_first_minus_second']:.5g} | {t_ci} | "
                f"{row['dz'] if row['dz'] is not None else 'n/a'} | "
                f"{row['first_model_wins']}/{row['n_seeds']} | {b_ci} |"
            )
    else:
        report_lines.append("| No complete five-seed paired matrix | n/a | n/a | n/a | n/a | n/a | n/a |")
    report_lines.extend(["", "## Incomplete runs", ""])
    report_lines.extend([f"- {item}" for item in incomplete] or ["- None."])
    report_lines.extend([
        "",
        "A model recommendation must be based on the completed validation "
        "matrix, parameter counts, measured runtime, and declared criteria. "
        "AO-RIDGE's predefined alpha was selected on this same validation split "
        "and is exploratory; it is not an independent holdout result.",
        "",
        "## Final-test readiness",
        "",
        "The confirmation data and predictions are limited to development "
        "training and GP7 validation. Before any official-test evaluation, "
        "freeze the model-selection decision, checkpoint hashes, metric code, "
        "and evaluation protocol; obtain explicit approval to access the "
        "official test partition. Model and coefficient SHA-256 values are "
        "recorded in the final metadata. This phase did not perform that evaluation.",
        "",
    ])

    latency_path = root / "AO_FINAL_LATENCY.csv"
    runtime_lines = [
        "# Phase 6A AO confirmation runtime report",
        "",
        "Training and memory measurements are CPU-only run observations. They "
        "are not GPU comparisons or performance claims beyond this machine.",
        "",
        "| Model | Seeds | Mean training/fit seconds | Mean alpha-selection seconds | Mean wall seconds | Max sampled process bytes | Mean best-checkpoint bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in (*MODELS, "AO-CP", "AO-RIDGE"):
        rows = [row for row in seed_rows if row["model_id"] == model_id]
        if not rows:
            continue
        training_values = [float(row["training_seconds"]) for row in rows]
        selection_values = [
            float(row["model_selection_seconds"])
            for row in rows if row["model_selection_seconds"] is not None
        ]
        wall_values = [
            float(row["run_wall_clock_seconds"])
            for row in rows if row["run_wall_clock_seconds"] is not None
        ]
        memory_values = [
            int(row["peak_process_memory_bytes"])
            for row in rows if row["peak_process_memory_bytes"] is not None
        ]
        checkpoint_values = [int(row["best_checkpoint_bytes"]) for row in rows]
        selection_text = (
            f"{np.mean(selection_values):.2f}" if selection_values else "n/a"
        )
        wall_text = f"{np.mean(wall_values):.2f}" if wall_values else "n/a"
        memory_text = str(max(memory_values)) if memory_values else "n/a"
        runtime_lines.append(
            f"| {model_id} | {len(rows)} | {np.mean(training_values):.2f} | "
            f"{selection_text} | {wall_text} | {memory_text} | "
            f"{np.mean(checkpoint_values):.0f} |"
        )
    runtime_lines.extend(["", "## Final-checkpoint inference latency", ""])
    if latency_path.is_file():
        latency_rows = _read_csv(latency_path)
        expected_latency = {
            ("AO-2", "1"): (76804, 227520),
            ("AO-2", "8"): (76804, 1820160),
            ("AO-3", "1"): (68356, 227520),
            ("AO-3", "8"): (68356, 1820160),
        }
        observed_latency = {
            (row["model"], row["batch_size"]): row for row in latency_rows
        }
        if set(observed_latency) != set(expected_latency):
            raise ValueError("Final latency table must cover AO-2/AO-3 at batches 1 and 8.")
        for key, (parameters, macs) in expected_latency.items():
            row = observed_latency[key]
            if (
                int(row["parameter_count"]) != parameters
                or int(row["encoder_self_attention_macs"]) != macs
                or int(row["thread_count"]) != 4
                or int(row["warmups"]) != 50
                or int(row["timed_iterations"]) != 500
                or row["weights_only"] != "True"
                or row["official_test_data_loaded"] != "False"
            ):
                raise ValueError(f"Final latency protocol mismatch in row {key}.")
        cpus = {row["cpu"] for row in latency_rows}
        if len(cpus) != 1:
            raise ValueError("Final latency rows do not share one CPU identity.")
        runtime_lines.extend([
            "| Model | Batch | CPU | Threads | Warm-ups | Iterations | Median ms | IQR ms | MACs |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in latency_rows:
            runtime_lines.append(
                f"| {row['model']} | {row['batch_size']} | {row['cpu']} | "
                f"{row['thread_count']} | {row['warmups']} | "
                f"{row['timed_iterations']} | {float(row['median_ms']):.3f} | "
                f"{float(row['iqr_ms']):.3f} | "
                f"{row['encoder_self_attention_macs']} |"
            )
        runtime_lines.append("")
    else:
        runtime_lines.append(
            "Final-checkpoint latency has not yet been measured; only the "
            "separate preflight timings are available."
        )
    runtime_lines.append("")

    _write_csv(root / output_names[0], seed_rows)
    _write_csv(root / output_names[1], model_rows)
    if paired_rows:
        _write_csv(root / output_names[2], paired_rows)
    _write_csv(root / output_names[3], per_video_rows)
    _write_csv(root / output_names[4], per_track_rows)
    _write_csv(root / output_names[5], correction_rows)
    _write_csv(root / output_names[6], per_window_rows)
    run_ao_v1._atomic_bytes(
        root / output_names[7], "\n".join(report_lines).encode("utf-8")
    )
    metadata = {
        "schema": "ao_confirmation_final_v1",
        "run_matrix_complete": complete,
        "completed_run_count": sum(row["model_id"] in MODELS for row in completed),
        "expected_run_count": len(MODELS) * len(seeds),
        "seeds": list(seeds),
        "models": [*MODELS, "AO-CP", "AO-RIDGE"],
        "manifest_sha256": run_ao_v1._sha256(manifest_path),
        "finalizer_sha256": run_ao_v1._sha256(Path(__file__)),
        "cache_sha256": manifest["cache_sha256"],
        "train_samples": run_provenance["train_samples"],
        "train_videos": run_provenance["train_videos"],
        "validation_samples": run_provenance["validation_samples"],
        "validation_videos": run_provenance["validation_videos"],
        "official_test_sets_read": [],
        "model_artifact_hashes": {
            f"{row['model_id']}/seed_{row['seed']}": {
                "best_checkpoint_sha256": row["best_checkpoint_sha256"],
                "latest_checkpoint_sha256": row["latest_checkpoint_sha256"],
                "ridge_coefficients_sha256": row["ridge_coefficients_sha256"],
            }
            for row in seed_rows
        },
        "bootstrap_repetitions": bootstrap_repetitions if paired_rows else None,
        "bootstrap_seed": bootstrap_seed if paired_rows else None,
        "video_generalization_caveat": (
            "Two fixed GP7 validation videos; overlapping windows."
        ),
        "incomplete_runs": incomplete,
    }
    run_ao_v1._atomic_json(root / output_names[8], metadata)
    run_ao_v1._atomic_bytes(
        root / output_names[9], "\n".join(runtime_lines).encode("utf-8")
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--classical-root", type=Path, default=DEFAULT_CLASSICAL_ROOT
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    args = parser.parse_args()
    result = finalize(
        args.output_root,
        args.cache_dir,
        args.manifest,
        args.classical_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
