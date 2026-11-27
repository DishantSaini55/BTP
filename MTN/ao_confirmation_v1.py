"""Classical observation-only baselines for Phase 6A confirmation."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import run_ao_v1 as runner
    from .ao_models_v1 import constant_velocity_forecast
except ImportError:
    import run_ao_v1 as runner
    from ao_models_v1 import constant_velocity_forecast


ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PREDICTION_FRAMES = 45
RIDGE_MODEL_ID = "AO-RIDGE"
CONSTANT_POSITION_ID = "AO-CP"


def observation_features(
    observed_boxes: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> np.ndarray:
    if observed_boxes.ndim != 3 or observed_boxes.shape[1:] != (15, 4):
        raise ValueError("Expected observed boxes with shape [samples, 15, 4].")
    normalized = (
        observed_boxes - normalization_mean.reshape(1, 1, 4)
    ) / normalization_std.reshape(1, 1, 4)
    relative = normalized - normalized[:, -1:, :]
    return np.concatenate(
        (relative.reshape(len(normalized), -1), normalized[:, -1, :]),
        axis=1,
    ).astype(np.float64, copy=False)


def fit_ridge_cv_residual(
    train_features: np.ndarray,
    train_observed: np.ndarray,
    train_future: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    if len(train_features) != len(train_observed) or len(train_features) != len(train_future):
        raise ValueError("Training feature, observation, and target counts differ.")
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    x = (train_features - feature_mean) / feature_std

    mean = normalization_mean.reshape(1, 1, 4)
    std = normalization_std.reshape(1, 1, 4)
    observed_normalized = (train_observed - mean) / std
    target_normalized = (train_future - mean) / std
    cv_normalized = constant_velocity_forecast(
        _as_tensor(observed_normalized)
    ).cpu().numpy()
    residual = (target_normalized - cv_normalized).reshape(len(x), -1)
    residual_mean = residual.mean(axis=0)
    centered_residual = residual - residual_mean
    gram = x.T @ x
    coefficient = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0]), x.T @ centered_residual
    )
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "residual_mean": residual_mean,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def predict_ridge_cv_residual(
    model: dict[str, np.ndarray | float],
    observed_boxes: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> np.ndarray:
    features = observation_features(
        observed_boxes, normalization_mean, normalization_std
    )
    standardized = (
        features - np.asarray(model["feature_mean"])
    ) / np.asarray(model["feature_std"])
    residual = (
        standardized @ np.asarray(model["coefficient"])
        + np.asarray(model["residual_mean"])
    ).reshape(len(features), PREDICTION_FRAMES, 4)
    normalized_observed = (
        observed_boxes - normalization_mean.reshape(1, 1, 4)
    ) / normalization_std.reshape(1, 1, 4)
    return (
        constant_velocity_forecast(_as_tensor(normalized_observed)).cpu().numpy()
        + residual
    ).astype(np.float32)


def _as_tensor(values: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _read_cv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return [
            {**row, "mse": float(row["mse"])}
            for row in csv.DictReader(source)
        ]


def _normalize_predictions(
    predictions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (
        predictions - mean.reshape(1, 1, 4)
    ) / std.reshape(1, 1, 4)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    runner._atomic_bytes(path, buffer.getvalue())


def run_classical_baselines(
    cache_dir: Path,
    output_root: Path,
    confirmation_root: Path,
) -> dict[str, Any]:
    train, validation, _ = runner._read_split(cache_dir.resolve())
    normalization = runner._train_only_normalization(train)
    mean = normalization["mean"]
    std = normalization["std"]
    cache_hashes = {
        name: runner._sha256(cache_dir / name)
        for name in ("train.npz", "validation.npz", "split_manifest.json")
    }
    provenance_path = confirmation_root / "split_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "Confirmation split provenance is missing; run the frozen AO-0 baseline first."
        )
    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    if provenance.get("cache_sha256") != cache_hashes:
        raise ValueError("Classical baselines do not match frozen cache checksums.")
    if provenance.get("official_test_sets_read") != []:
        raise ValueError("Confirmation provenance does not certify test isolation.")

    cv_path = (
        confirmation_root / "AO-0" / "seed_20260928"
        / "validation_per_sample.csv"
    )
    if not cv_path.is_file():
        raise FileNotFoundError(
            "Run the frozen confirmation CV baseline before fitting ridge."
        )
    cv_rows = _read_cv_rows(cv_path)

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite classical baseline outputs: {output_root}"
        )
    output_root.mkdir(parents=True)

    results: dict[str, Any] = {
        "schema": "ao_classical_baselines_v1",
        "cache_sha256": cache_hashes,
        "normalization_fit_role": "train",
        "official_test_sets_read": [],
        "input_features": (
            "15 observed xyxy boxes only; normalized with train-only "
            "coordinate statistics"
        ),
        "constant_position": None,
        "ridge_alpha_grid": list(ALPHA_GRID),
        "ridge_validation_selection_metric": "pooled raw-pixel MSE",
        "ridge_alpha_scores": [],
    }
    for model_id in (CONSTANT_POSITION_ID, RIDGE_MODEL_ID):
        run = output_root / model_id / "seed_20260928"
        run.mkdir(parents=True)
        if model_id == CONSTANT_POSITION_ID:
            predictions = np.repeat(
                validation["observed_boxes"][:, -1:, :],
                PREDICTION_FRAMES,
                axis=1,
            ).astype(np.float32)
        else:
            ridge_selection_started = time.perf_counter()
            train_features = observation_features(
                train["observed_boxes"], mean, std
            )
            models = []
            for alpha in ALPHA_GRID:
                fit_started = time.perf_counter()
                model = fit_ridge_cv_residual(
                    train_features,
                    train["observed_boxes"],
                    train["future_boxes"],
                    mean,
                    std,
                    alpha,
                )
                fit_seconds = time.perf_counter() - fit_started
                candidate_normalized = predict_ridge_cv_residual(
                    model, validation["observed_boxes"], mean, std
                )
                candidate_raw = (
                    candidate_normalized * std.reshape(1, 1, 4)
                    + mean.reshape(1, 1, 4)
                ).astype(np.float32)
                candidate_rows = runner._sample_metrics(
                    candidate_raw,
                    validation,
                    candidate_normalized,
                    normalization,
                )
                candidate_mse = float(np.mean([
                    row["mse"] for row in candidate_rows
                ]))
                results["ridge_alpha_scores"].append({
                    "alpha": alpha,
                    "validation_raw_pixel_mse": candidate_mse,
                    "coefficient_fit_seconds": fit_seconds,
                })
                models.append((candidate_mse, model, fit_seconds))
            _, selected_model, selected_fit_seconds = min(
                models, key=lambda item: item[0]
            )
            normalized_predictions = predict_ridge_cv_residual(
                selected_model, validation["observed_boxes"], mean, std
            )
            predictions = (
                normalized_predictions * std.reshape(1, 1, 4)
                + mean.reshape(1, 1, 4)
            ).astype(np.float32)
            _write_npz(run / "ridge_coefficients.npz", {
                name: np.asarray(value)
                for name, value in selected_model.items()
            })
            results["ridge_alpha_selected"] = selected_model["alpha"]
            results["ridge_selected_fit_seconds"] = selected_fit_seconds
            results["ridge_model_selection_seconds"] = (
                time.perf_counter() - ridge_selection_started
            )
            results["ridge_parameter_count"] = int(
                np.asarray(selected_model["coefficient"]).size
                + np.asarray(selected_model["residual_mean"]).size
            )
            results["ridge_input_feature_count"] = int(
                train_features.shape[1]
            )
            results["ridge_output_count"] = int(
                np.asarray(selected_model["coefficient"]).shape[1]
            )
            results["ridge_intercept_is_unpenalized_centered_target_mean"] = True

        normalized_predictions = _normalize_predictions(predictions, mean, std)
        summary, _ = runner._write_prediction_outputs(
            run, model_id, predictions, validation, normalized_predictions,
            normalization, cv_rows,
        )
        summary.update({
            "model_id": model_id,
            "seed": 20260928,
            "status": "completed",
            "epochs_completed": 0,
            "parameter_count": (
                0 if model_id == CONSTANT_POSITION_ID
                else results["ridge_parameter_count"]
            ),
            "training_seconds": (
                float(results["ridge_selected_fit_seconds"])
                if model_id == RIDGE_MODEL_ID else 0.0
            ),
            "training_time_status": (
                "measured_train_only_closed_form_fit"
                if model_id == RIDGE_MODEL_ID else "not_applicable_deterministic_baseline"
            ),
            "cache_sha256": cache_hashes,
            "input_features": results["input_features"],
            "selection_split": (
                "validation" if model_id == RIDGE_MODEL_ID else "none"
            ),
        })
        runner._atomic_json(run / "metrics.json", summary)
        runner._atomic_json(run / "status.json", {
            "state": "completed",
            "model_id": model_id,
            "seed": 20260928,
            "epochs_completed": 0,
        })
        results[model_id] = summary
    runner._atomic_json(output_root / "classical_baseline_report.json", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results_paper/splits/ao_annotation_only_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results_paper/AO_confirmation_v1/classical"),
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=Path("results_paper/AO_confirmation_v1"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_classical_baselines(
        args.cache_dir, args.output_root, args.confirmation_root
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
