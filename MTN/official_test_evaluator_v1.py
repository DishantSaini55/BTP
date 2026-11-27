"""Evaluation-only Phase 6B official-test preparer and runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import ao_models_v1
    from . import run_ao_v1
    from . import ao_dataset_v1
except ImportError:
    import ao_models_v1
    import run_ao_v1
    import ao_dataset_v1


SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)
NEURAL_MODELS = ("AO-1", "AO-2", "AO-3")
TEST_SETS = ("gp_set_0003", "gp_set_0005", "gp_set_0008", "gp_set_0009")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_checkpoint(checkpoint_root: Path, model_id: str, seed: int) -> torch.nn.Module:
    if model_id not in NEURAL_MODELS:
        raise ValueError(f"Evaluation-only neural loader rejects model {model_id}.")
    checkpoint_path = checkpoint_root / model_id / f"seed_{seed}" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_id") != model_id or payload.get("seed") != seed:
        raise ValueError("Checkpoint model/seed identity mismatch.")
    anchor = "constant_velocity"
    if model_id == "AO-1":
        anchor = "constant_position"
    model = ao_models_v1.build_ao_model(model_id).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    if model.anchor != anchor:
        raise ValueError("Checkpoint residual anchor mismatch.")
    if model_id == "AO-3" and not ao_models_v1.verify_satie_identity(model):
        raise AssertionError("AO-3 SATIE sharing identity mismatch.")
    expected_parameters = {"AO-1": 76804, "AO-2": 76804, "AO-3": 68356}[model_id]
    if ao_models_v1.unique_trainable_parameter_count(model) != expected_parameters:
        raise ValueError("Frozen parameter count mismatch.")
    if int(payload["parameter_count"]) != expected_parameters:
        raise ValueError("Checkpoint parameter metadata mismatch.")
    return model


def pooled_raw_pixel_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ.")
    return float(np.mean((predictions - targets) ** 2))


def build_official_test_arrays(annotation_root: Path) -> dict[str, np.ndarray]:
    samples: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    video_rows: list[dict[str, Any]] = []
    pedestrian_root = annotation_root / "annotations" / "gopro"
    for set_id in TEST_SETS:
        set_root = pedestrian_root / set_id
        if not set_root.is_dir():
            raise FileNotFoundError(set_root)
        paths = sorted(set_root.glob(f"{set_id}_vid_[0-9][0-9][0-9][0-9].xml"))
        if not paths:
            raise FileExistsError(f"No official-test pedestrian XML in {set_root}")
        for path in paths:
            video_id = path.stem
            report, video_samples = ao_dataset_v1.audit_video(
                annotation_root, set_id, video_id
            )
            if report["task_frame_count"] < 60:
                raise ValueError(f"Video too short for 60-frame windows: {video_id}")
            if report["pedestrian_tracks"] < 1:
                raise ValueError(f"No pedestrian tracks: {video_id}")
            if report["eligible_windows_stride30"] < 1:
                raise ValueError(f"No eligible 60-frame windows: {video_id}")
            source_hashes[video_id] = report["pedestrian_xml_sha256"]
            video_rows.append({
                "set_id": set_id,
                "video_id": video_id,
                "task_frame_count": report["task_frame_count"],
                "frame_width": report["frame_width"],
                "frame_height": report["frame_height"],
                "eligible_tracks": report["tracks_with_60_frame_windows"],
                "eligible_windows": report["eligible_windows_stride30"],
                "xml_sha256": report["pedestrian_xml_sha256"],
            })
            samples.extend(video_samples)
    samples.sort(key=lambda item: (
        item["video_id"], int(item["track_id"]), item["frame_start"]
    ))
    if not samples:
        raise ValueError("Official-test protocol produced no eligible windows.")
    arrays = {
        "observed_boxes": np.stack(
            [sample["boxes"][:15] for sample in samples]
        ).astype(np.float32, copy=False),
        "future_boxes": np.stack(
            [sample["boxes"][15:] for sample in samples]
        ).astype(np.float32, copy=False),
        "sample_id": np.asarray([sample["sample_id"] for sample in samples], dtype=np.str_),
        "video_id": np.asarray([sample["video_id"] for sample in samples], dtype=np.str_),
        "set_id": np.asarray([sample["set_id"] for sample in samples], dtype=np.str_),
        "track_id": np.asarray([sample["track_id"] for sample in samples], dtype=np.str_),
        "frame_start": np.asarray([sample["frame_start"] for sample in samples], dtype=np.int32),
        "frame_end": np.asarray([sample["frame_end"] for sample in samples], dtype=np.int32),
    }
    arrays["_video_rows"] = video_rows
    arrays["_source_hashes"] = source_hashes
    return arrays


def evaluate_development_cache(
    cache_dir: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    train, validation, _ = run_ao_v1._read_split(cache_dir.resolve())
    normalization = run_ao_v1._train_only_normalization(train)
    mean = torch.from_numpy(normalization["mean"])
    std = torch.from_numpy(normalization["std"])
    observed = torch.from_numpy(validation["observed_boxes"])
    targets = validation["future_boxes"]
    normalized_observed = (observed - mean) / std
    sample_ids = [str(value) for value in validation["sample_id"].tolist()]
    results: dict[str, Any] = {}

    for model_id in NEURAL_MODELS:
        seed_results = {}
        for seed in SEEDS:
            model = validate_frozen_checkpoint(checkpoint_root, model_id, seed)
            model.eval()
            predictions = []
            with torch.no_grad():
                for start in range(0, len(normalized_observed), 8):
                    batch = normalized_observed[start:start + 8]
                    predictions.append(model(batch).cpu().numpy())
            normalized_predictions = np.concatenate(predictions, axis=0)
            raw_predictions = (
                normalized_predictions * normalization["std"].reshape(1, 1, 4)
                + normalization["mean"].reshape(1, 1, 4)
            ).astype(np.float32)
            seed_results[str(seed)] = {
                "pooled_raw_pixel_mse": pooled_raw_pixel_mse(raw_predictions, targets),
                "sample_ids": sample_ids,
            }
        results[model_id] = seed_results
    return {
        "schema": "phase6b_evaluation_smoke_v1",
        "official_test_data_accessed": False,
        "model_results": results,
    }


def per_window_metrics(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ.")
    return np.mean((predictions - targets) ** 2, axis=(1, 2))


def center_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred_x = (predictions[..., 0] + predictions[..., 2]) / 2.0
    pred_y = (predictions[..., 1] + predictions[..., 3]) / 2.0
    target_x = (targets[..., 0] + targets[..., 2]) / 2.0
    target_y = (targets[..., 1] + targets[..., 3]) / 2.0
    return float(np.mean((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2))


def final_center_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    px = (predictions[:, -1, 0] + predictions[:, -1, 2]) / 2.0
    py = (predictions[:, -1, 1] + predictions[:, -1, 3]) / 2.0
    tx = (targets[:, -1, 0] + targets[:, -1, 2]) / 2.0
    ty = (targets[:, -1, 1] + targets[:, -1, 3]) / 2.0
    return float(np.mean((px - tx) ** 2 + (py - ty) ** 2))


def evaluate_official_test(
    annotation_root: Path,
    cache_dir: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    test = build_official_test_arrays(annotation_root)
    train, _, _ = run_ao_v1._read_split(cache_dir.resolve())
    normalization = run_ao_v1._train_only_normalization(train)
    mean = torch.from_numpy(normalization["mean"])
    std = torch.from_numpy(normalization["std"])
    observed = torch.from_numpy(test["observed_boxes"])
    targets = test["future_boxes"]
    normalized_observed = (observed - mean) / std
    sample_ids = test["sample_id"].tolist()
    video_ids = test["video_id"].tolist()
    track_ids = test["track_id"].tolist()

    cv_predictions = ao_models_v1.constant_velocity_forecast(
        observed
    ).numpy()
    cp_predictions = np.repeat(
        test["observed_boxes"][:, -1:, :], 45, axis=1
    ).astype(np.float32)

    ridge_path = (
        checkpoint_root / "classical" / "AO-RIDGE" / "seed_20260928"
        / "ridge_coefficients.npz"
    )
    if not ridge_path.is_file():
        raise FileNotFoundError(ridge_path)
    with np.load(ridge_path, allow_pickle=False) as ridge:
        alpha = float(ridge["alpha"])
        feature_mean = ridge["feature_mean"]
        feature_std = ridge["feature_std"]
        residual_mean = ridge["residual_mean"]
        coefficient = ridge["coefficient"]
    if alpha != 100.0:
        raise ValueError("Frozen ridge alpha is not 100.")
    normalized = (
        test["observed_boxes"] - normalization["mean"].reshape(1, 1, 4)
    ) / normalization["std"].reshape(1, 1, 4)
    relative = normalized - normalized[:, -1:, :]
    features = np.concatenate(
        (relative.reshape(len(normalized), -1), normalized[:, -1, :]),
        axis=1,
    ).astype(np.float64)
    standardized = (features - feature_mean) / feature_std
    residual = standardized @ coefficient + residual_mean
    residual = residual.reshape(len(features), 45, 4).astype(np.float32)
    normalized_observed_numpy = (
        test["observed_boxes"] - normalization["mean"].reshape(1, 1, 4)
    ) / normalization["std"].reshape(1, 1, 4)
    cv_normalized = ao_models_v1.constant_velocity_forecast(
        torch.from_numpy(normalized_observed_numpy.astype(np.float32))
    ).numpy()
    ridge_normalized = cv_normalized.astype(np.float32) + residual
    ridge_predictions = (
        ridge_normalized * normalization["std"].reshape(1, 1, 4)
        + normalization["mean"].reshape(1, 1, 4)
    ).astype(np.float32)

    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    cv_windows = per_window_metrics(cv_predictions, targets)
    cp_windows = per_window_metrics(cp_predictions, targets)
    ridge_windows = per_window_metrics(ridge_predictions, targets)

    for model_id in NEURAL_MODELS:
        model_predictions[model_id] = {}
        for seed in SEEDS:
            model = validate_frozen_checkpoint(checkpoint_root, model_id, seed)
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(normalized_observed), 8):
                    chunks.append(model(normalized_observed[start:start + 8]).cpu().numpy())
            normalized_predictions = np.concatenate(chunks, axis=0)
            raw_predictions = (
                normalized_predictions * normalization["std"].reshape(1, 1, 4)
                + normalization["mean"].reshape(1, 1, 4)
            ).astype(np.float32)
            model_predictions[model_id][str(seed)] = raw_predictions

    per_window_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    model_predictions["AO-0"] = {"20260928": cv_predictions}
    model_predictions["AO-CP"] = {"20260928": cp_predictions}
    model_predictions["AO-RIDGE"] = {"20260928": ridge_predictions}

    def summarize(model_id: str, seed: int, predictions: np.ndarray) -> dict[str, Any]:
        horizon = {}
        for step in (1, 15, 30, 45):
            index = step - 1
            horizon[f"horizon_{step}_mse"] = float(
                np.mean((predictions[:, index, :] - targets[:, index, :]) ** 2)
            )
        mse = pooled_raw_pixel_mse(predictions, targets)
        return {
            "model_id": model_id,
            "seed": seed,
            "mse": mse,
            "normalized_mse": float(
                np.mean(
                    (
                        (predictions - targets)
                        / normalization["std"].reshape(1, 1, 4)
                    ) ** 2
                )
            ),
            "cmse": center_mse(predictions, targets),
            "cfmse": final_center_mse(predictions, targets),
            **horizon,
            "fraction_improved_vs_cv": float(
                np.mean(per_window_metrics(predictions, targets) < cv_windows)
            ),
        }

    for model_id in ("AO-0", "AO-CP", "AO-RIDGE", *NEURAL_MODELS):
        seeds = (SEEDS if model_id in NEURAL_MODELS else (20260928,))
        for seed in seeds:
            predictions = model_predictions[model_id][str(seed)]
            row = summarize(model_id, seed, predictions)
            seed_rows.append(row)
            window_mse = per_window_metrics(predictions, targets)
            cv_window_mse = cv_windows if model_id != "AO-0" else cp_windows
            for index, sample_id in enumerate(sample_ids):
                per_window_rows.append({
                    "model_id": model_id,
                    "seed": seed,
                    "sample_id": sample_id,
                    "video_id": video_ids[index],
                    "track_id": track_ids[index],
                    "mse": float(window_mse[index]),
                    "improved_vs_cv": bool(window_mse[index] < cv_windows[index]),
                })

    per_video_rows = []
    per_track_rows = []
    for model_id in ("AO-0", "AO-CP", "AO-RIDGE", *NEURAL_MODELS):
        seeds = (SEEDS if model_id in NEURAL_MODELS else (20260928,))
        for seed in seeds:
            predictions = model_predictions[model_id][str(seed)]
            window_mse = per_window_metrics(predictions, targets)
            for video_id in sorted(set(video_ids)):
                indexes = [i for i, value in enumerate(video_ids) if value == video_id]
                if indexes:
                    per_video_rows.append({
                        "model_id": model_id,
                        "seed": seed,
                        "video_id": video_id,
                        "window_count": len(indexes),
                        "track_count": len({
                            track_ids[i] for i in indexes
                        }),
                        "mse": float(np.mean(window_mse[indexes])),
                    })
            for cluster in sorted(set(zip(video_ids, track_ids))):
                indexes = [
                    i for i, value in enumerate(zip(video_ids, track_ids))
                    if value == cluster
                ]
                per_track_rows.append({
                    "model_id": model_id,
                    "seed": seed,
                    "video_id": cluster[0],
                    "track_id": cluster[1],
                    "window_count": len(indexes),
                    "mse": float(np.mean(window_mse[indexes])),
                })

    return {
        "schema": "phase6b_official_test_v1",
        "official_test_data_accessed": True,
        "test_sets": list(TEST_SETS),
        "sample_count": len(sample_ids),
        "video_count": len(set(video_ids)),
        "track_count": len(set(zip(video_ids, track_ids))),
        "source_video_membership": test["_video_rows"],
        "source_xml_sha256": test["_source_hashes"],
        "normalization": {
            "fit_role": "train",
            "mean": normalization["mean"].tolist(),
            "std": normalization["std"].tolist(),
        },
        "ridge_alpha": alpha,
        "seed_rows": seed_rows,
        "per_window_rows": per_window_rows,
        "per_video_rows": per_video_rows,
        "per_track_rows": per_track_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path(r"D:\IDD-PeD-Evaluation\annotations\IDDPedestrian"),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-official-test",
        action="store_true",
        help="Authorize this invocation to evaluate frozen official-test annotations.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.allow_official_test:
        result = evaluate_official_test(
            args.annotation_root.resolve(),
            args.cache_dir.resolve(),
            args.checkpoint_root.resolve(),
        )
    else:
        result = evaluate_development_cache(args.cache_dir, args.checkpoint_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        if args.output_json.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output_json}")
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, key in (
            ("OFFICIAL_TEST_SEED_RESULTS.csv", "seed_rows"),
            ("OFFICIAL_TEST_PER_WINDOW.csv", "per_window_rows"),
            ("OFFICIAL_TEST_PER_VIDEO.csv", "per_video_rows"),
            ("OFFICIAL_TEST_PER_TRACK.csv", "per_track_rows"),
        ):
            path = args.output_dir / name
            rows = result.get(key, [])
            if not rows:
                continue
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({
        "schema": result["schema"],
        "official_test_data_accessed": result["official_test_data_accessed"],
        "sample_count": result.get("sample_count"),
        "video_count": result.get("video_count"),
        "track_count": result.get("track_count"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
