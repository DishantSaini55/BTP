"""Resumable, validation-only runner for the annotation-only Phase 5B pilot."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import random
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .ao_models_v1 import (
        build_ao_model,
        constant_velocity_forecast,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )
except ImportError:
    from ao_models_v1 import (
        build_ao_model,
        constant_velocity_forecast,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results_paper" / "splits" / "ao_annotation_only_v1"
DEFAULT_OUTPUT = ROOT / "results_paper" / "AO_initial_screen_v1"
SEED = 20260928
CONFIRMATION_SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3
VALID_MODEL_IDS = ("AO-0", "AO-1", "AO-2", "AO-3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        .encode("utf-8"),
    )


def _atomic_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb+") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def _progress(run: Path, message: str) -> None:
    print(message, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8", newline="\n") as log:
        log.write(message + "\n")
        log.flush()
        os.fsync(log.fileno())


def _status(run: Path, state: str, **details: Any) -> None:
    _atomic_json(
        run / "status.json",
        {"state": state, "updated_at": _utc_now(), **details},
    )


def _read_split(cache_root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = cache_root / "split_manifest.json"
    train_path = cache_root / "train.npz"
    validation_path = cache_root / "validation.npz"
    for path in (manifest_path, train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required annotation-only file missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ao_annotation_only_v1":
        raise ValueError("Unexpected annotation-only cache schema.")
    if manifest.get("official_test_sets_read") != []:
        raise ValueError("Cache manifest does not certify test-set isolation.")
    if manifest.get("historical_gp6_test_video_used") is not False:
        raise ValueError("Historical GP6 test video is not allowed in this cache.")
    if manifest.get("model_input_fields") != ["observed_boxes[0:15]"]:
        raise ValueError("Unexpected model input fields in the split manifest.")
    role_data: list[dict[str, np.ndarray]] = []
    for role, filename in (("train", train_path), ("validation", validation_path)):
        with np.load(filename, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        required = {
            "observed_boxes", "future_boxes", "sample_id",
            "video_id", "set_id", "track_id", "frame_start",
        }
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"{role} cache is missing required fields: {sorted(missing)}")
        expected = manifest.get("roles", {}).get(role, {}).get("sample_count")
        if expected != len(arrays["sample_id"]):
            raise ValueError(f"{role} cache count differs from its manifest.")
        if arrays["observed_boxes"].shape[1:] != (15, 4):
            raise ValueError(f"{role} observed box shape is invalid.")
        if arrays["future_boxes"].shape[1:] != (45, 4):
            raise ValueError(f"{role} future box shape is invalid.")
        role_data.append(arrays)
    train, validation = role_data
    train_videos = set(train["video_id"].tolist())
    validation_videos = set(validation["video_id"].tolist())
    if train_videos.intersection(validation_videos):
        raise ValueError("Train and validation contain a shared source video.")
    if not train_videos or not validation_videos:
        raise ValueError("Train and validation roles must both be nonempty.")
    if not set(train["set_id"].tolist()).issubset({
        "gp_set_0001", "gp_set_0002", "gp_set_0004", "gp_set_0006"
    }):
        raise ValueError("Unexpected non-training set in train cache.")
    if set(validation["set_id"].tolist()) != {"gp_set_0007"}:
        raise ValueError("Validation cache must contain only GP7 videos.")
    return train, validation, manifest


def _train_only_normalization(train: dict[str, np.ndarray]) -> dict[str, Any]:
    unique_boxes: dict[tuple[str, str, int], np.ndarray] = {}
    for index in range(len(train["sample_id"])):
        video_id = str(train["video_id"][index])
        track_id = str(train["track_id"][index])
        start = int(train["frame_start"][index])
        boxes = np.concatenate(
            (train["observed_boxes"][index], train["future_boxes"][index]),
            axis=0,
        )
        for offset, box in enumerate(boxes):
            unique_boxes.setdefault((video_id, track_id, start + offset), box)
    if not unique_boxes:
        raise ValueError("Cannot fit normalization on an empty training role.")
    values = np.stack(list(unique_boxes.values())).astype(np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "unique_training_frame_count": int(len(unique_boxes)),
        "fit_role": "train",
    }


def _initial_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _seed_everything(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(cpu_threads)


def _evaluate(
    model: torch.nn.Module | None,
    validation: dict[str, np.ndarray],
    normalization: dict[str, Any],
    batch_size: int,
    model_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = normalization["mean"]
    std = normalization["std"]
    observed = validation["observed_boxes"]
    started = time.perf_counter()
    normalized_predictions: list[np.ndarray] = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for start in range(0, len(observed), batch_size):
            batch = torch.from_numpy(observed[start:start + batch_size])
            normalized_batch = (batch - torch.from_numpy(mean)) / torch.from_numpy(std)
            if model_id == "AO-0":
                normalized_prediction = constant_velocity_forecast(normalized_batch)
            else:
                normalized_prediction = model(normalized_batch)
            normalized_predictions.append(
                normalized_prediction.detach().cpu().numpy()
            )
    elapsed = time.perf_counter() - started
    prediction = np.concatenate(normalized_predictions, axis=0)
    raw_prediction = prediction * std.reshape(1, 1, 4) + mean.reshape(1, 1, 4)
    metadata = {
        "validation_seconds": elapsed,
        "validation_milliseconds_per_sample": (
            1000.0 * elapsed / len(observed) if len(observed) else None
        ),
    }
    return raw_prediction.astype(np.float32, copy=False), metadata


def _sample_metrics(
    raw_predictions: np.ndarray,
    validation: dict[str, np.ndarray],
    normalized_predictions: np.ndarray,
    normalization: dict[str, Any],
) -> list[dict[str, Any]]:
    mean = normalization["mean"]
    std = normalization["std"]
    raw_targets = validation["future_boxes"]
    normalized_targets = (raw_targets - mean.reshape(1, 1, 4)) / std.reshape(1, 1, 4)
    rows = []
    for index, (prediction, target) in enumerate(zip(raw_predictions, raw_targets)):
        pred_center = (prediction[:, :2] + prediction[:, 2:]) / 2.0
        target_center = (target[:, :2] + target[:, 2:]) / 2.0
        error = prediction - target
        normalized_error = normalized_predictions[index] - normalized_targets[index]
        rows.append({
            "sample_id": str(validation["sample_id"][index]),
            "video_id": str(validation["video_id"][index]),
            "track_id": str(validation["track_id"][index]),
            "frame_start": int(validation["frame_start"][index]),
            "mse": float(np.mean(error ** 2)),
            "normalized_mse": float(np.mean(normalized_error ** 2)),
            "cmse": float(np.mean((pred_center - target_center) ** 2)),
            "cfmse": float(np.mean(
                (pred_center[-1] - target_center[-1]) ** 2
            )),
            "horizon_1_mse": float(np.mean(error[0] ** 2)),
            "horizon_15_mse": float(np.mean(error[14] ** 2)),
            "horizon_30_mse": float(np.mean(error[29] ** 2)),
            "horizon_45_mse": float(np.mean(error[44] ** 2)),
        })
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty validation set.")
    summary: dict[str, Any] = {"sample_count": len(rows)}
    for key in (
        "mse", "normalized_mse", "cmse", "cfmse", "horizon_1_mse",
        "horizon_15_mse", "horizon_30_mse", "horizon_45_mse",
    ):
        summary[key] = float(np.mean([row[key] for row in rows]))
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_track: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
        by_track[(row["video_id"], row["track_id"])].append(row)
    summary["video_count"] = len(by_video)
    summary["track_count"] = len(by_track)
    for metric in ("mse", "normalized_mse", "cmse", "cfmse"):
        summary[f"macro_video_{metric}"] = float(np.mean([
            np.mean([row[metric] for row in group])
            for group in by_video.values()
        ]))
        summary[f"macro_track_{metric}"] = float(np.mean([
            np.mean([row[metric] for row in group])
            for group in by_track.values()
        ]))
    return summary


def _write_prediction_outputs(
    run: Path,
    model_id: str,
    raw_predictions: np.ndarray,
    validation: dict[str, np.ndarray],
    normalized_predictions: np.ndarray,
    normalization: dict[str, Any],
    baseline_rows: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _sample_metrics(
        raw_predictions, validation, normalized_predictions, normalization
    )
    baseline_by_id = {
        row["sample_id"]: row for row in (baseline_rows or [])
    }
    for row in rows:
        baseline = baseline_by_id.get(row["sample_id"])
        row["cv_baseline_mse"] = baseline["mse"] if baseline else ""
        row["improved_vs_cv"] = (
            row["mse"] < baseline["mse"] if baseline else ""
        )
    summary = _summarize_rows(rows)
    if baseline_rows:
        summary["fraction_windows_improved_vs_cv"] = float(np.mean([
            bool(row["improved_vs_cv"]) for row in rows
        ]))
    else:
        summary["fraction_windows_improved_vs_cv"] = (
            0.0 if model_id == "AO-0" else None
        )
    _atomic_bytes(
        run / "validation_predictions.npz",
        _npz_buffer({
            "sample_id": validation["sample_id"],
            "video_id": validation["video_id"],
            "track_id": validation["track_id"],
            "frame_start": validation["frame_start"],
            "prediction_boxes": raw_predictions,
            "target_boxes": validation["future_boxes"],
        }),
    )
    fields = [
        "sample_id", "video_id", "track_id", "frame_start", "mse",
        "normalized_mse", "cmse", "cfmse", "horizon_1_mse",
        "horizon_15_mse", "horizon_30_mse", "horizon_45_mse",
        "cv_baseline_mse", "improved_vs_cv",
    ]
    _atomic_csv(run / "validation_per_sample.csv", rows, fields)
    per_video = _group_rows(
        rows, lambda row: row["video_id"], model_id
    )
    per_track = _group_rows(
        rows, lambda row: (row["video_id"], row["track_id"]), model_id
    )
    _atomic_csv(
        run / "validation_per_video.csv", per_video,
        ["model", "video_id", "window_count", "track_count", "mse",
         "normalized_mse", "cmse", "cfmse"],
    )
    _atomic_csv(
        run / "validation_per_track.csv", per_track,
        ["model", "video_id", "track_id", "window_count", "mse",
         "normalized_mse", "cmse", "cfmse"],
    )
    return summary, rows


def _npz_buffer(arrays: dict[str, np.ndarray]) -> bytes:
    import io

    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _group_rows(rows, key_fn, model_id):
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output = []
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        entry = {
            "model": model_id,
            "window_count": len(group),
            "mse": float(np.mean([row["mse"] for row in group])),
            "normalized_mse": float(np.mean(
                [row["normalized_mse"] for row in group]
            )),
            "cmse": float(np.mean([row["cmse"] for row in group])),
            "cfmse": float(np.mean([row["cfmse"] for row in group])),
        }
        if isinstance(group_key, tuple):
            entry["video_id"], entry["track_id"] = group_key
        else:
            entry["video_id"] = group_key
            entry["track_id"] = ""
            entry["track_count"] = len({row["track_id"] for row in group})
        output.append(entry)
    return output


def _save_run_summary(output_root: Path) -> None:
    rows = []
    for status_path in output_root.glob("AO-*/seed_*/status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            continue
        run = status_path.parent
        metrics_path = status_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("benchmark_only") or not metrics.get("validation_complete"):
            continue
        history_path = status_path.parent / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            best_epoch = metrics.get("best_epoch")
            best_record = next(
                (record for record in history
                 if record["epoch"] == best_epoch),
                None,
            )
            metrics["train_normalized_mse"] = (
                best_record["train_normalized_mse"] if best_record else None
            )
            metrics["train_normalized_mse_last_epoch"] = (
                history[-1]["train_normalized_mse"] if history else None
            )
        for filename, prefix in (
            ("validation_per_video.csv", "macro_video"),
            ("validation_per_track.csv", "macro_track"),
        ):
            table = run / filename
            if not table.exists():
                continue
            with table.open("r", encoding="utf-8", newline="") as source:
                grouped_rows = list(csv.DictReader(source))
            for metric in ("mse", "normalized_mse", "cmse", "cfmse"):
                values = [float(row[metric]) for row in grouped_rows]
                metrics[f"{prefix}_{metric}"] = (
                    float(np.mean(values)) if values else None
                )
        metrics.setdefault(
            "run_wall_clock_seconds",
            metrics.get("total_process_seconds_this_invocation"),
        )
        _atomic_json(metrics_path, metrics)
        rows.append({
            "model": metrics["model_id"],
            "seed": metrics["seed"],
            "status": status["state"],
            "epochs_completed": metrics["epochs_completed"],
            "train_normalized_mse": metrics.get("train_normalized_mse"),
            "train_normalized_mse_last_epoch": metrics.get(
                "train_normalized_mse_last_epoch"
            ),
            "validation_normalized_mse": metrics["normalized_mse"],
            "validation_raw_pixel_mse": metrics["mse"],
            "validation_cmse": metrics["cmse"],
            "validation_cfmse": metrics["cfmse"],
            "macro_video_mse": metrics["macro_video_mse"],
            "macro_video_normalized_mse": metrics["macro_video_normalized_mse"],
            "macro_video_cmse": metrics["macro_video_cmse"],
            "macro_video_cfmse": metrics["macro_video_cfmse"],
            "macro_track_mse": metrics["macro_track_mse"],
            "macro_track_normalized_mse": metrics["macro_track_normalized_mse"],
            "macro_track_cmse": metrics["macro_track_cmse"],
            "macro_track_cfmse": metrics["macro_track_cfmse"],
            "fraction_windows_improved_vs_cv": metrics.get(
                "fraction_windows_improved_vs_cv"
            ),
            "parameter_count": metrics["parameter_count"],
            "training_seconds": metrics["training_seconds"],
            "run_wall_clock_seconds": metrics.get("run_wall_clock_seconds"),
            "validation_seconds": metrics["validation_seconds"],
            "peak_process_memory_bytes": metrics.get(
                "peak_process_memory_bytes"
            ),
        })
    rows.sort(key=lambda row: (row["model"], row["seed"]))
    fields = [
        "model", "seed", "status", "epochs_completed",
        "train_normalized_mse", "train_normalized_mse_last_epoch",
        "validation_normalized_mse",
        "validation_raw_pixel_mse", "validation_cmse", "validation_cfmse",
        "macro_video_mse", "macro_video_normalized_mse", "macro_video_cmse",
        "macro_video_cfmse", "macro_track_mse",
        "macro_track_normalized_mse", "macro_track_cmse", "macro_track_cfmse",
        "fraction_windows_improved_vs_cv", "parameter_count",
        "training_seconds", "run_wall_clock_seconds", "validation_seconds",
        "peak_process_memory_bytes",
    ]
    _atomic_csv(output_root / "AO_VALIDATION_SUMMARY.csv", rows, fields)


def _save_detailed_group_summaries(output_root: Path) -> None:
    video_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    for status_path in output_root.glob("AO-*/seed_*/status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            continue
        run = status_path.parent
        metrics_path = run / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("benchmark_only") or not metrics.get("validation_complete"):
            continue
        for filename, destination in (
            ("validation_per_video.csv", video_rows),
            ("validation_per_track.csv", track_rows),
        ):
            path = run / filename
            with path.open("r", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    destination.append({
                        "model": metrics["model_id"],
                        "seed": metrics["seed"],
                        **row,
                    })
    video_rows.sort(key=lambda row: (row["model"], row["seed"], row["video_id"]))
    track_rows.sort(key=lambda row: (
        row["model"], row["seed"], row["video_id"], row["track_id"]
    ))
    _atomic_csv(
        output_root / "AO_PER_VIDEO_RESULTS.csv",
        video_rows,
        ["model", "seed", "video_id", "track_count", "window_count",
         "mse", "normalized_mse", "cmse", "cfmse"],
    )
    _atomic_csv(
        output_root / "AO_PER_TRACK_RESULTS.csv",
        track_rows,
        ["model", "seed", "video_id", "track_id", "window_count",
         "mse", "normalized_mse", "cmse", "cfmse"],
    )


def _run_baseline(
    run: Path,
    validation: dict[str, np.ndarray],
    normalization: dict[str, Any],
    batch_size: int,
    seed: int,
    cache_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    mean, std = normalization["mean"], normalization["std"]
    normalized_input = (
        validation["observed_boxes"] - mean.reshape(1, 1, 4)
    ) / std.reshape(1, 1, 4)
    normalized_prediction = constant_velocity_forecast(
        torch.from_numpy(normalized_input)
    ).numpy()
    prediction = normalized_prediction * std.reshape(1, 1, 4) + mean.reshape(1, 1, 4)
    _atomic_json(run / "normalization.json", {
        **normalization,
        "mean": mean.tolist(),
        "std": std.tolist(),
    })
    metadata = _evaluate(None, validation, normalization, batch_size, "AO-0")[1]
    metrics, rows = _write_prediction_outputs(
        run, "AO-0", prediction.astype(np.float32), validation,
        normalized_prediction, normalization, None,
    )
    metrics.update({
        "schema": "ao_run_metrics_v1",
        "model_id": "AO-0",
        "seed": seed,
        "status": "completed",
        "epochs_completed": 0,
        "parameter_count": 0,
        "training_seconds": 0.0,
        "training_time_status": "not_applicable_deterministic_baseline",
        "peak_process_memory_bytes": _memory_bytes(),
        "cache_sha256": cache_hashes,
        **metadata,
        "validation_complete": True,
        "benchmark_only": False,
    })
    _atomic_json(run / "metrics.json", metrics)
    _status(run, "completed", model_id="AO-0", seed=seed, epochs_completed=0)
    _progress(run, "COMPLETE model=AO-0 validation_only=true")
    return rows


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    order: np.ndarray,
    next_offset: int,
    epoch_loss_total: float,
    epoch_sample_count: int,
    history: list[dict[str, Any]],
    best_loss: float,
    best_epoch: int,
    best_model_state: dict[str, torch.Tensor] | None,
    stale_epochs: int,
    total_training_seconds: float,
    total_run_wall_seconds: float,
    epoch_elapsed_seconds: float,
    peak_memory: int | None,
) -> dict[str, Any]:
    return {
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "order": order,
        "next_offset": next_offset,
        "epoch_loss_total": epoch_loss_total,
        "epoch_sample_count": epoch_sample_count,
        "history": history,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "stale_epochs": stale_epochs,
        "total_training_seconds": total_training_seconds,
        "total_run_wall_seconds": total_run_wall_seconds,
        "epoch_elapsed_seconds": epoch_elapsed_seconds,
        "peak_memory": peak_memory,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_history(run: Path, history: list[dict[str, Any]]) -> None:
    _atomic_json(run / "history.json", history)
    if not history:
        return
    _atomic_csv(
        run / "history.csv", history,
        ["epoch", "train_normalized_mse", "validation_normalized_mse",
         "train_epoch_seconds", "validation_seconds", "epoch_seconds",
         "train_batches", "process_memory_bytes"],
    )


def _run_trainable(
    run: Path,
    model_id: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    normalization: dict[str, Any],
    args: argparse.Namespace,
    cache_hashes: dict[str, str],
    baseline_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if run.exists() and not args.resume:
        raise FileExistsError(
            f"Run directory already exists; use --resume only for an incomplete run: {run}"
        )
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "runner_version": "AO_v1",
        "model_id": model_id,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "cpu_threads": args.cpu_threads,
        "wall_clock_budget_seconds": args.wall_clock_budget_seconds,
        "cache_sha256": cache_hashes,
        "normalization_mean": normalization["mean"].tolist(),
        "normalization_std": normalization["std"].tolist(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "validation_only": True,
    }
    config_path = run / "run_config.json"
    status_path = run / "status.json"
    checkpoint_path = run / "latest_checkpoint.pt"
    resuming = config_path.exists()
    if resuming:
        if not args.resume:
            raise FileExistsError(f"Refusing to reuse prior run without --resume: {run}")
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        mutable = {"max_epochs", "wall_clock_budget_seconds"}
        if any(
            old_config.get(key) != value
            for key, value in config.items()
            if key not in mutable
        ):
            raise ValueError(f"Saved run configuration is incompatible: {run}")
        old_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists() else {}
        )
        if old_status.get("state") in {"completed", "benchmark_complete"}:
            raise RuntimeError(f"Run is already finalized: {run}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint_path}")
        _atomic_json(config_path, config)
    else:
        if any(run.iterdir()):
            raise RuntimeError(
                f"Run directory contains files but no saved configuration: {run}"
            )
        _atomic_json(config_path, config)
        _atomic_json(run / "normalization.json", {
            **normalization,
            "mean": normalization["mean"].tolist(),
            "std": normalization["std"].tolist(),
        })

    _seed_everything(args.seed, args.cpu_threads)
    model = build_ao_model(model_id)
    parameter_count = unique_trainable_parameter_count(model)
    if model_id == "AO-3" and not verify_satie_identity(model):
        raise AssertionError("AO-3 encoder self-attention is not shared by identity.")
    initial_digest = _initial_state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, Any]] = []
    first_epoch = 1
    order = np.empty(0, dtype=np.int64)
    next_offset = 0
    epoch_loss_total = 0.0
    epoch_sample_count = 0
    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    stale_epochs = 0
    total_training_seconds = 0.0
    total_run_wall_seconds = 0.0
    epoch_elapsed_seconds = 0.0
    peak_memory = _memory_bytes()
    if resuming:
        checkpoint = _load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        first_epoch = int(checkpoint["epoch"])
        order = np.asarray(checkpoint["order"], dtype=np.int64)
        next_offset = int(checkpoint["next_offset"])
        epoch_loss_total = float(checkpoint["epoch_loss_total"])
        epoch_sample_count = int(checkpoint["epoch_sample_count"])
        history = checkpoint["history"]
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_model_state = checkpoint["best_model_state"]
        stale_epochs = int(checkpoint["stale_epochs"])
        total_training_seconds = float(checkpoint["total_training_seconds"])
        total_run_wall_seconds = float(
            checkpoint.get("total_run_wall_seconds", total_training_seconds)
        )
        epoch_elapsed_seconds = float(
            checkpoint.get("epoch_elapsed_seconds", 0.0)
        )
        peak_memory = checkpoint["peak_memory"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])

    mean_t = torch.from_numpy(normalization["mean"])
    std_t = torch.from_numpy(normalization["std"])
    train_observed = torch.from_numpy(train["observed_boxes"])
    train_targets = torch.from_numpy(train["future_boxes"])
    validation_observed = torch.from_numpy(validation["observed_boxes"])
    validation_targets = torch.from_numpy(validation["future_boxes"])
    criterion = torch.nn.MSELoss()
    invocation_started = time.perf_counter()

    def budget_reached() -> bool:
        return (
            args.wall_clock_budget_seconds > 0
            and total_run_wall_seconds
            + time.perf_counter() - invocation_started
            >= args.wall_clock_budget_seconds
        )

    completed_epoch = len(history)
    _status(
        run, "running", model_id=model_id, seed=args.seed,
        epoch=first_epoch, resumed=resuming,
    )
    _progress(
        run,
        f"START model={model_id} seed={args.seed} parameters={parameter_count} "
        f"train={len(train_observed)} validation={len(validation_observed)}",
    )

    try:
        for epoch in range(first_epoch, args.max_epochs + 1):
            if len(order) == 0:
                order = np.random.RandomState(args.seed + epoch).permutation(
                    len(train_observed)
                ).astype(np.int64)
                next_offset = 0
                epoch_loss_total = 0.0
                epoch_sample_count = 0
                epoch_elapsed_seconds = 0.0
            model.train()
            train_epoch_started = time.perf_counter()
            batch_count = 0
            exhausted_budget = False
            while next_offset < len(order):
                if budget_reached():
                    exhausted_budget = True
                    break
                indices = order[next_offset:next_offset + args.batch_size]
                index_tensor = torch.from_numpy(indices.copy()).long()
                inputs = (train_observed[index_tensor] - mean_t) / std_t
                targets = (train_targets[index_tensor] - mean_t) / std_t
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite training loss at {model_id} epoch {epoch}."
                    )
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(
                        f"Non-finite gradient at {model_id} epoch {epoch}."
                    )
                optimizer.step()
                batch_samples = len(indices)
                epoch_loss_total += float(loss.detach()) * batch_samples
                epoch_sample_count += batch_samples
                next_offset += len(indices)
                batch_count += 1
                memory = _memory_bytes()
                if memory is not None:
                    peak_memory = max(peak_memory or 0, memory)
                if batch_count % 10 == 0:
                    _progress(
                        run,
                        f"HEARTBEAT state=running model={model_id} epoch={epoch} "
                        f"batches={batch_count} "
                        f"partial_train_loss={epoch_loss_total / epoch_sample_count:.8f} "
                        f"elapsed_seconds={time.perf_counter() - invocation_started:.1f} "
                        f"rss_bytes={memory}",
                    )
                budget_hit = budget_reached()
                if batch_count % 10 == 0 or budget_hit:
                    _atomic_torch_save(
                        checkpoint_path,
                        _checkpoint_state(
                            model, optimizer, epoch, order, next_offset,
                            epoch_loss_total, epoch_sample_count, history,
                            best_loss, best_epoch, best_model_state, stale_epochs,
                            total_training_seconds,
                            total_run_wall_seconds
                            + time.perf_counter() - invocation_started,
                            epoch_elapsed_seconds
                            + time.perf_counter() - train_epoch_started,
                            peak_memory,
                        ),
                    )
                if budget_hit:
                    exhausted_budget = True
                    break

            if exhausted_budget:
                _atomic_torch_save(
                    checkpoint_path,
                    _checkpoint_state(
                        model, optimizer, epoch, order, next_offset,
                        epoch_loss_total, epoch_sample_count, history,
                        best_loss, best_epoch, best_model_state, stale_epochs,
                        total_training_seconds,
                        total_run_wall_seconds
                        + time.perf_counter() - invocation_started,
                        epoch_elapsed_seconds
                        + time.perf_counter() - train_epoch_started,
                        peak_memory,
                    ),
                )
                _status(
                    run, "incomplete", model_id=model_id, seed=args.seed,
                    epoch=epoch, reason="wall_clock_budget_reached",
                    resume_checkpoint=str(checkpoint_path),
                )
                _progress(
                    run,
                    f"INCOMPLETE model={model_id} epoch={epoch} "
                    "reason=wall_clock_budget_reached",
                )
                return None

            train_epoch_seconds = (
                epoch_elapsed_seconds
                + time.perf_counter() - train_epoch_started
            )
            model.eval()
            validation_started = time.perf_counter()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for start in range(0, len(validation_observed), args.batch_size):
                    end = start + args.batch_size
                    inputs = (validation_observed[start:end] - mean_t) / std_t
                    targets = (validation_targets[start:end] - mean_t) / std_t
                    val_loss = criterion(model(inputs), targets)
                    if not bool(torch.isfinite(val_loss)):
                        raise FloatingPointError(
                            f"Non-finite validation loss at {model_id} epoch {epoch}."
                        )
                    count = len(inputs)
                    val_loss_total += float(val_loss) * count
                    val_count += count
            validation_seconds = time.perf_counter() - validation_started
            normalized_validation_loss = val_loss_total / val_count
            normalized_train_loss = epoch_loss_total / epoch_sample_count
            epoch_seconds = train_epoch_seconds + validation_seconds
            total_training_seconds += epoch_seconds
            completed_epoch = epoch
            current_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if normalized_validation_loss < best_loss:
                best_loss = normalized_validation_loss
                best_epoch = epoch
                best_model_state = current_state
                stale_epochs = 0
                _atomic_torch_save(
                    run / "best_model.pt",
                    {
                        "schema": "ao_best_checkpoint_v1",
                        "model_id": model_id,
                        "seed": args.seed,
                        "epoch": epoch,
                        "model_state": current_state,
                        "parameter_count": parameter_count,
                        "initial_state_sha256": initial_digest,
                        "cache_sha256": cache_hashes,
                        "normalization": {
                            "mean": normalization["mean"],
                            "std": normalization["std"],
                            "fit_role": "train",
                        },
                    },
                )
            else:
                stale_epochs += 1
            record = {
                "epoch": epoch,
                "train_normalized_mse": normalized_train_loss,
                "validation_normalized_mse": normalized_validation_loss,
                "train_epoch_seconds": train_epoch_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "train_batches": batch_count,
                "process_memory_bytes": _memory_bytes(),
            }
            history.append(record)
            _write_history(run, history)
            _atomic_torch_save(
                checkpoint_path,
                _checkpoint_state(
                    model, optimizer, epoch + 1, np.empty(0, dtype=np.int64),
                    0, 0.0, 0, history, best_loss, best_epoch,
                    best_model_state, stale_epochs, total_training_seconds,
                    total_run_wall_seconds
                    + time.perf_counter() - invocation_started,
                    0.0,
                    peak_memory,
                ),
            )
            order = np.empty(0, dtype=np.int64)
            next_offset = 0
            epoch_loss_total = 0.0
            epoch_sample_count = 0
            _progress(
                run,
                f"EPOCH model={model_id} epoch={epoch} "
                f"train_normalized_mse={normalized_train_loss:.8f} "
                f"validation_normalized_mse={normalized_validation_loss:.8f} "
                f"epoch_seconds={epoch_seconds:.2f} best_epoch={best_epoch}",
            )
            budget_hit_after_validation = budget_reached()
            if budget_hit_after_validation:
                _atomic_torch_save(
                    checkpoint_path,
                    _checkpoint_state(
                        model, optimizer, epoch + 1,
                        np.empty(0, dtype=np.int64), 0, 0.0, 0,
                        history, best_loss, best_epoch, best_model_state,
                        stale_epochs, total_training_seconds,
                        total_run_wall_seconds
                        + time.perf_counter() - invocation_started,
                        0.0, peak_memory,
                    ),
                )
                _status(
                    run, "incomplete", model_id=model_id, seed=args.seed,
                    epoch=epoch, reason="wall_clock_budget_reached",
                    resume_checkpoint=str(checkpoint_path),
                )
                return None
            if stale_epochs >= args.patience:
                break

        if best_model_state is None:
            raise RuntimeError("Training ended without a finite validation checkpoint.")
        model.load_state_dict(best_model_state)
        normalized_predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(validation_observed), args.batch_size):
                end = start + args.batch_size
                inputs = (validation_observed[start:end] - mean_t) / std_t
                normalized_predictions.append(model(inputs).cpu().numpy())
        normalized_predictions_array = np.concatenate(normalized_predictions)
        raw_predictions = (
            normalized_predictions_array * normalization["std"].reshape(1, 1, 4)
            + normalization["mean"].reshape(1, 1, 4)
        ).astype(np.float32)
        _, eval_metadata = _evaluate(
            model, validation, normalization, args.batch_size, model_id
        )
        metric_summary, rows = _write_prediction_outputs(
            run, model_id, raw_predictions, validation,
            normalized_predictions_array, normalization, baseline_rows,
        )
        benchmark_only = bool(args.benchmark_only)
        metrics = {
            "schema": "ao_run_metrics_v1",
            "model_id": model_id,
            "seed": args.seed,
            "status": "benchmark_complete" if benchmark_only else "completed",
            "epochs_completed": completed_epoch,
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_loss,
            "train_normalized_mse": (
                history[best_epoch - 1]["train_normalized_mse"]
                if history and best_epoch > 0 else None
            ),
            "train_normalized_mse_last_epoch": (
                history[-1]["train_normalized_mse"] if history else None
            ),
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_digest,
            "satie_identity_verified": (
                verify_satie_identity(model) if model_id == "AO-3" else None
            ),
            "training_seconds": total_training_seconds,
            "peak_process_memory_bytes": peak_memory,
            "cache_sha256": cache_hashes,
            "normalization_unique_train_frames": normalization[
                "unique_training_frame_count"
            ],
            "validation_complete": True,
            "benchmark_only": benchmark_only,
            "wall_clock_budget_seconds": args.wall_clock_budget_seconds,
            "run_wall_clock_seconds": (
                total_run_wall_seconds
                + time.perf_counter() - invocation_started
            ),
            "total_process_seconds_this_invocation": (
                time.perf_counter() - invocation_started
            ),
            **metric_summary,
            **eval_metadata,
        }
        _atomic_json(run / "metrics.json", metrics)
        final_state = "benchmark_complete" if benchmark_only else "completed"
        _status(
            run, final_state, model_id=model_id, seed=args.seed,
            epoch=completed_epoch, best_epoch=best_epoch,
            training_seconds=total_training_seconds,
            run_wall_clock_seconds=metrics["run_wall_clock_seconds"],
        )
        _progress(
            run,
            f"COMPLETE state={final_state} model={model_id} "
            f"epochs={completed_epoch} validation_mse={metrics['mse']:.6f}",
        )
        return rows
    except BaseException as error:
        _status(
            run, "failed", model_id=model_id, seed=args.seed,
            error_type=type(error).__name__, error=str(error),
        )
        _progress(run, f"FAILED model={model_id} error={type(error).__name__}: {error}")
        (run / "failure_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not models or any(model not in VALID_MODEL_IDS for model in models):
        raise ValueError(f"Models must be selected from {VALID_MODEL_IDS}.")
    return models


def _run_validation_only(
    args: argparse.Namespace,
    model_id: str,
    validation: dict[str, np.ndarray],
    normalization: dict[str, Any],
    cache_hashes: dict[str, str],
) -> None:
    run = args.output_root.resolve() / model_id / f"validation_only_seed_{args.seed}"
    if model_id == "AO-0":
        _run_baseline(
            run, validation, normalization, args.batch_size,
            args.seed, cache_hashes,
        )
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(
            "--validation-only for AO-1..AO-3 requires --checkpoint."
        )
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint.get("model_id") not in {None, model_id}:
        raise ValueError("Checkpoint model ID does not match --models.")
    if checkpoint.get("seed") not in {None, args.seed}:
        raise ValueError("Checkpoint seed does not match the fixed pilot seed.")
    checkpoint_hashes = checkpoint.get("cache_sha256")
    if checkpoint_hashes is None:
        config_path = args.checkpoint.parent / "run_config.json"
        if config_path.is_file():
            checkpoint_hashes = json.loads(
                config_path.read_text(encoding="utf-8")
            ).get("cache_sha256")
    if checkpoint_hashes is not None and checkpoint_hashes != cache_hashes:
        raise ValueError("Checkpoint cache hashes differ from the selected split.")
    if run.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation-only output: {run}"
        )
    run.mkdir(parents=True)
    _seed_everything(args.seed, args.cpu_threads)
    model = build_ao_model(model_id)
    model.load_state_dict(checkpoint["model_state"])
    if model_id == "AO-3" and not verify_satie_identity(model):
        raise AssertionError("AO-3 encoder self-attention is not shared by identity.")
    mean_t = torch.from_numpy(normalization["mean"])
    std_t = torch.from_numpy(normalization["std"])
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation["observed_boxes"]), args.batch_size):
            batch = torch.from_numpy(
                validation["observed_boxes"][start:start + args.batch_size]
            )
            normalized = (batch - mean_t) / std_t
            predictions.append(model(normalized).cpu().numpy())
    normalized_predictions = np.concatenate(predictions)
    raw_predictions = (
        normalized_predictions * normalization["std"].reshape(1, 1, 4)
        + normalization["mean"].reshape(1, 1, 4)
    ).astype(np.float32)
    eval_metadata = _evaluate(
        model, validation, normalization, args.batch_size, model_id
    )[1]
    metrics, _ = _write_prediction_outputs(
        run, model_id, raw_predictions, validation,
        normalized_predictions, normalization, None,
    )
    _atomic_json(run / "normalization.json", {
        **normalization,
        "mean": normalization["mean"].tolist(),
        "std": normalization["std"].tolist(),
    })
    metrics.update({
        "schema": "ao_run_metrics_v1",
        "model_id": model_id,
        "seed": args.seed,
        "status": "completed",
        "validation_only": True,
        "validation_complete": True,
        "parameter_count": unique_trainable_parameter_count(model),
        "checkpoint": str(args.checkpoint.resolve()),
        "cache_sha256": cache_hashes,
        **eval_metadata,
    })
    _atomic_json(run / "metrics.json", metrics)
    _status(
        run, "completed", model_id=model_id, seed=args.seed,
        validation_only=True,
    )


def run(args: argparse.Namespace) -> None:
    if args.finalize_only:
        output_root = args.output_root.resolve()
        if not output_root.is_dir():
            raise FileNotFoundError(f"Screening output root is missing: {output_root}")
        _save_run_summary(output_root)
        _save_detailed_group_summaries(output_root)
        return
    if args.seed not in CONFIRMATION_SEEDS:
        raise ValueError(
            f"Seed must be one of the predefined values: {CONFIRMATION_SEEDS}."
        )
    if args.max_epochs < 1 or args.max_epochs > 30:
        raise ValueError("Confirmation runs are capped at thirty epochs.")
    if args.cpu_threads < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Threads, batch size, and patience must be positive.")
    cache_root = args.cache_dir.resolve()
    output_root = args.output_root.resolve()
    train, validation, manifest = _read_split(cache_root)
    normalization = _train_only_normalization(train)
    cache_hashes = {
        name: _sha256(cache_root / name)
        for name in ("train.npz", "validation.npz", "split_manifest.json")
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "split_provenance.json", {
        "cache_root": str(cache_root),
        "cache_sha256": cache_hashes,
        "manifest_schema": manifest["schema"],
        "official_test_sets_read": [],
        "train_samples": len(train["sample_id"]),
        "validation_samples": len(validation["sample_id"]),
        "train_videos": sorted(set(train["video_id"].tolist())),
        "validation_videos": sorted(set(validation["video_id"].tolist())),
        "normalization_fit_role": "train",
        "normalization_unique_train_frames": normalization[
            "unique_training_frame_count"
        ],
    })

    selected_models = _parse_models(args.models)
    if args.validation_only:
        if len(selected_models) != 1:
            raise ValueError("--validation-only requires exactly one model.")
        _run_validation_only(
            args, selected_models[0], validation, normalization,
            cache_hashes,
        )
        return
    cv_rows: list[dict[str, Any]] | None = None
    baseline_run = output_root / "AO-0" / f"seed_{args.seed}"
    if "AO-0" in selected_models:
        cv_rows = _run_baseline(
            baseline_run, validation, normalization, args.batch_size,
            args.seed, cache_hashes,
        )
    else:
        prior = baseline_run / "validation_per_sample.csv"
        if prior.exists():
            with prior.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                cv_rows = [
                    {**row, "mse": float(row["mse"])} for row in reader
                ]

    for model_id in selected_models:
        if model_id == "AO-0":
            continue
        run_directory = output_root / model_id / f"seed_{args.seed}"
        _run_trainable(
            run_directory, model_id, train, validation, normalization, args,
            cache_hashes, cv_rows,
        )
    _save_run_summary(output_root)
    _save_detailed_group_summaries(output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default="AO-0,AO-1,AO-2,AO-3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wall-clock-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
