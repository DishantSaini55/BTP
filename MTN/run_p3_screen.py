"""Resumable, validation-only Phase 3 constant-velocity residual screening."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

try:
    from .cv_residual_v1 import (
        P3_CONFIGURATIONS,
        ResidualAnchorModel,
        build_p3_model,
        configure_validation_policy,
        constant_position_forecast,
        constant_velocity_forecast,
        trainable_parameter_count,
        unwrap_predictor,
    )
    from .run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_bytes,
        atomic_write_csv,
        atomic_write_json,
        load_split,
        set_seed,
    )
    from .run_from_scratch_e1 import RESULTS, SPLIT
    from .sage_v2 import (
        AlwaysOpenContextIndividualTF,
        DifferentiableSAGEV2,
    )
    from .context_TF import ContextIndividualTF
    from .individual_TF import IndividualTF
except ImportError:
    from cv_residual_v1 import (
        P3_CONFIGURATIONS,
        ResidualAnchorModel,
        build_p3_model,
        configure_validation_policy,
        constant_position_forecast,
        constant_velocity_forecast,
        trainable_parameter_count,
        unwrap_predictor,
    )
    from run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_bytes,
        atomic_write_csv,
        atomic_write_json,
        load_split,
        set_seed,
    )
    from run_from_scratch_e1 import RESULTS, SPLIT
    from sage_v2 import AlwaysOpenContextIndividualTF, DifferentiableSAGEV2
    from context_TF import ContextIndividualTF
    from individual_TF import IndividualTF


DEFAULT_OUTPUT_ROOT = RESULTS / "P3_initial_screen_v1"
SUMMARY_FIELDS = (
    "configuration", "seed", "status", "policy", "epochs_completed",
    "selected_epoch", "train_normalized_mse", "validation_normalized_mse",
    "validation_raw_pixel_mse", "validation_cmse", "validation_cfmse",
    "validation_mse_h01", "validation_mse_h05", "validation_mse_h15",
    "validation_mse_h30", "validation_mse_h45", "unique_parameters",
    "training_seconds", "peak_sampled_rss_bytes", "checkpoint_bytes",
    "mean_admission_probability", "hard_admission_rate",
    "context_attention_weight",
)


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    return default_collate([dataset[index] for index in indices])


def _status(run: Path, state: str, **fields: Any) -> None:
    if state not in {"running", "completed", "interrupted", "incomplete", "failed"}:
        raise ValueError(f"Unknown P3 run state {state!r}.")
    atomic_write_json(run / "status.json", {
        "state": state, "updated_at": _utc_now(), **fields,
    })


def _progress(run: Path, message: str) -> None:
    print(message, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator,
    state: dict[str, Any],
) -> None:
    payload = dict(state)
    payload.update({
        "model_state": _copy_state(model),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "data_order_generator": generator.get_state(),
        },
    })
    atomic_torch_save(path, payload)


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["data_order_generator"])


def _config(
    configuration: str, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "runner_version": "P3_screen_v1",
        "configuration": configuration,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss": "MSE",
        "cpu_threads": args.cpu_threads,
        "wall_clock_budget_seconds": args.wall_clock_budget_seconds,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "validation_only": True,
        "test_split_loaded_as_dataset": False,
        "normalization": "train-only per-coordinate affine bbox statistics",
        "zero_correction_head": args.zero_correction_head,
        "split_cache_sha256": _sha256_file(SPLIT / "flow_splits.pkl"),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "source_sha256": {
            "run_p3_screen.py": _sha256_file(Path(__file__)),
            "cv_residual_v1.py": _sha256_file(Path(__file__).with_name("cv_residual_v1.py")),
            "run_from_scratch_e1.py": _sha256_file(
                Path(__file__).with_name("run_from_scratch_e1.py")
            ),
        },
    }


def _compatible(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    mutable = {"max_epochs", "wall_clock_budget_seconds", "source_sha256"}
    return all(
        previous.get(key) == value
        for key, value in current.items()
        if key not in mutable
    )


def _prepare_run(
    run: Path, config: dict[str, Any], resume: bool
) -> dict[str, Any] | None:
    if not run.exists():
        run.mkdir(parents=True)
        atomic_write_json(run / "run_config.json", config)
        return None
    if not resume:
        raise FileExistsError(
            f"Run path exists; pass --resume only for a compatible incomplete run: {run}"
        )
    config_path, checkpoint_path = run / "run_config.json", run / "latest_checkpoint.pt"
    status_path = run / "status.json"
    if not config_path.exists() or not checkpoint_path.exists() or not status_path.exists():
        raise RuntimeError(f"Existing P3 run is missing config, status, or checkpoint: {run}")
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if not _compatible(previous, config):
        raise ValueError(f"Saved P3 configuration is incompatible: {run}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") == "completed":
        raise RuntimeError(f"Completed P3 run will not be overwritten: {run}")
    if status.get("state") == "failed":
        raise RuntimeError(f"P3 run is marked failed; inspect {status_path} before resuming.")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _policy_list(configuration: str) -> tuple[str, ...]:
    return {
        "p3_b": ("base",),
        "p3_c": ("base",),
        "p3_d": ("hard", "forced_open", "forced_closed"),
        "p3_e": ("hard", "forced_open", "forced_closed"),
        "p3_f": ("always_open",),
        "p3_g": ("soft", "hard", "forced_open", "forced_closed"),
    }[configuration]


def _gate_values(
    predictor: torch.nn.Module, policy: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(predictor, ContextIndividualTF):
        gate = predictor.last_context_gate
        logits = predictor.last_context_gate_logits
        if gate is None or logits is None:
            raise RuntimeError("Historical SAGE did not record its gate output.")
        if policy == "forced_open":
            probability = torch.ones_like(gate)
        elif policy == "forced_closed":
            probability = torch.zeros_like(gate)
        else:
            probability = logits.softmax(dim=-1)[:, 1]
        decision = gate >= 0.5
    elif isinstance(predictor, DifferentiableSAGEV2):
        probability = predictor.last_admission_probability
        logits = predictor.last_gate_logits
        decision = predictor.last_gate_decision
        if probability is None or logits is None:
            raise RuntimeError("SAGE V2 did not record admission diagnostics.")
        if decision is None:
            decision = probability >= 0.5
    else:
        probability = torch.ones(batch_size)
        decision = torch.ones(batch_size, dtype=torch.bool)
        logits = torch.empty(batch_size, 0)
    return (
        probability.detach().cpu().numpy(),
        decision.detach().cpu().numpy().astype(bool),
        logits.detach().cpu().numpy(),
    )


def _context_attention(predictor: torch.nn.Module, batch_size: int) -> np.ndarray:
    if isinstance(predictor, DifferentiableSAGEV2):
        value = predictor.last_context_attention
        if value is not None:
            return value.detach().cpu().numpy()
    base = predictor if isinstance(predictor, IndividualTF) else getattr(predictor, "base", None)
    if base is None:
        return np.ones(batch_size, dtype=np.float32)
    per_layer = []
    for layer in base.model.decoder.layers:
        attention = layer.src_attn.attn
        if attention is not None and attention.shape[0] == batch_size:
            if attention.shape[-1] == 16:
                per_layer.append(attention[..., -1].mean(dim=(1, 2)).detach().cpu())
            elif attention.shape[-1] == 15:
                per_layer.append(torch.zeros(batch_size))
    if not per_layer:
        return np.zeros(batch_size, dtype=np.float32)
    return torch.stack(per_layer).mean(dim=0).numpy()


def _sample_metrics(
    raw_prediction: np.ndarray,
    raw_target: np.ndarray,
    raw_cv: np.ndarray,
) -> dict[str, float]:
    center_prediction = (raw_prediction[:, :2] + raw_prediction[:, 2:]) / 2.0
    center_target = (raw_target[:, :2] + raw_target[:, 2:]) / 2.0
    return {
        "raw_pixel_mse": float(np.mean((raw_prediction - raw_target) ** 2)),
        "cmse": float(np.mean((center_prediction - center_target) ** 2)),
        "cfmse": float(np.mean((center_prediction[-1] - center_target[-1]) ** 2)),
        "mse_h01": float(np.mean((raw_prediction[0] - raw_target[0]) ** 2)),
        "mse_h05": float(np.mean((raw_prediction[4] - raw_target[4]) ** 2)),
        "mse_h15": float(np.mean((raw_prediction[14] - raw_target[14]) ** 2)),
        "mse_h30": float(np.mean((raw_prediction[29] - raw_target[29]) ** 2)),
        "mse_h45": float(np.mean((raw_prediction[44] - raw_target[44]) ** 2)),
        "cv_raw_pixel_mse": float(np.mean((raw_cv - raw_target) ** 2)),
        "correction_rms_pixels": float(np.sqrt(np.mean((raw_prediction - raw_cv) ** 2))),
    }


def _evaluate(
    model: ResidualAnchorModel,
    dataset: Any,
    stats: tuple[np.ndarray, ...],
    policy: str,
    batch_size: int,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mean, std = stats[0], stats[1]
    predictor = unwrap_predictor(model)
    model.eval()
    normalized_predictions, normalized_targets, raw_predictions = [], [], []
    raw_targets, raw_cv_predictions, observed_raw, sample_ids = [], [], [], []
    sample_rows: list[dict[str, Any]] = []
    probabilities, decisions, logits_values, attentions = [], [], [], []
    normalized_sum = 0.0
    count = 0
    with configure_validation_policy(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and start // batch_size >= max_batches:
                break
            batch = _batch(dataset, list(range(start, min(start + batch_size, len(dataset)))))
            prediction = model.forward_batch(batch)
            target = batch["pred_target"]
            n = int(target.shape[0])
            normalized_sum += float(torch.nn.functional.mse_loss(prediction, target)) * n
            count += n
            prediction_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            raw_prediction = prediction_np * std + mean
            raw_target = target_np * std + mean
            boxes_np = batch["boxes"].detach().cpu().numpy()
            raw_observed = boxes_np * std + mean
            cv_normalized = constant_velocity_forecast(batch["boxes"])
            raw_cv = cv_normalized.detach().cpu().numpy() * std + mean
            normalized_predictions.append(prediction_np)
            normalized_targets.append(target_np)
            raw_predictions.append(raw_prediction)
            raw_targets.append(raw_target)
            raw_cv_predictions.append(raw_cv)
            observed_raw.append(raw_observed)
            sample_ids.extend(batch["sample_id"])
            batch_probs, batch_decisions, batch_logits = _gate_values(predictor, policy, n)
            probabilities.append(batch_probs)
            decisions.append(batch_decisions)
            logits_values.append(batch_logits)
            attentions.append(_context_attention(predictor, n))
            for index in range(n):
                item = _sample_metrics(
                    raw_prediction[index], raw_target[index], raw_cv[index]
                )
                item.update({
                    "sample_id": batch["sample_id"][index],
                    "pid": batch["pid"][index],
                    "video": batch["video"][index],
                    "frame_start": int(batch["frame_start"][index]),
                    "frame_end": int(batch["frame_end"][index]),
                    "admission_probability": float(batch_probs[index]),
                    "hard_admission": bool(batch_decisions[index]),
                    "context_attention_weight": float(
                        _context_attention(predictor, n)[index]
                    ),
                    "observed_center_speed_px_per_frame": float(
                        np.linalg.norm(
                            (
                                (raw_observed[index, -1, :2] + raw_observed[index, -1, 2:]) / 2
                                - (raw_observed[index, -3, :2] + raw_observed[index, -3, 2:]) / 2
                            ) / 2
                        )
                    ),
                })
                sample_rows.append(item)
    if count == 0:
        raise RuntimeError(f"Validation/evaluation produced no samples for policy {policy}.")
    raw_p = np.concatenate(raw_predictions)
    raw_t = np.concatenate(raw_targets)
    normalized_p = np.concatenate(normalized_predictions)
    normalized_t = np.concatenate(normalized_targets)
    probs = np.concatenate(probabilities)
    hard = np.concatenate(decisions)
    attention = np.concatenate(attentions)
    metric_values = [_sample_metrics(raw_p[i], raw_t[i], np.concatenate(raw_cv_predictions)[i])
                     for i in range(len(raw_p))]
    metrics: dict[str, Any] = {
        "policy": policy,
        "normalized_mse": normalized_sum / count,
        "raw_pixel_mse": float(np.mean([row["raw_pixel_mse"] for row in metric_values])),
        "cmse": float(np.mean([row["cmse"] for row in metric_values])),
        "cfmse": float(np.mean([row["cfmse"] for row in metric_values])),
        "mse_h01": float(np.mean([row["mse_h01"] for row in metric_values])),
        "mse_h05": float(np.mean([row["mse_h05"] for row in metric_values])),
        "mse_h15": float(np.mean([row["mse_h15"] for row in metric_values])),
        "mse_h30": float(np.mean([row["mse_h30"] for row in metric_values])),
        "mse_h45": float(np.mean([row["mse_h45"] for row in metric_values])),
        "mean_admission_probability": float(np.mean(probs)),
        "hard_admission_rate": float(np.mean(hard)),
        "context_attention_weight": float(np.mean(attention)),
        "sample_count": count,
    }
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype=str),
        "predictions_normalized": normalized_p,
        "targets_normalized": normalized_t,
        "predictions_raw": raw_p,
        "targets_raw": raw_t,
        "cv_predictions_raw": np.concatenate(raw_cv_predictions),
        "observed_boxes_raw": np.concatenate(observed_raw),
        "admission_probability": probs,
        "hard_admission": hard,
        "gate_logits": np.concatenate(logits_values),
        "context_attention_weight": attention,
    }
    return {
        **metrics,
        "sample_rows": sample_rows,
        "normalized_loss_sum": normalized_sum,
        "sample_count": count,
    }, arrays


def _dataset_loss(
    model: ResidualAnchorModel, dataset: Any, batch_size: int,
    policy: str, max_batches: int | None = None,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with configure_validation_policy(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and start // batch_size >= max_batches:
                break
            batch = _batch(dataset, list(range(start, min(start + batch_size, len(dataset)))))
            prediction = model.forward_batch(batch)
            n = int(batch["pred_target"].shape[0])
            total += float(torch.nn.functional.mse_loss(
                prediction, batch["pred_target"]
            )) * n
            count += n
    if not count:
        raise RuntimeError("Training metric has no samples.")
    return total / count


def _append_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    keys = {(row["configuration"], row["policy"], int(row["seed"])) for row in existing}
    for row in rows:
        key = (row["configuration"], row["policy"], int(row["seed"]))
        if key in keys:
            raise FileExistsError(f"Summary already contains {key}; refusing to replace it.")
        existing.append(row)
        keys.add(key)
    atomic_write_csv(path, existing, list(fields))


def _run_cv_baseline(
    datasets: dict[str, Any], stats: tuple[np.ndarray, ...], args: argparse.Namespace
) -> None:
    run = args.output_root / "p3_a" / str(args.seeds[0])
    if run.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite deterministic baseline output: {run}")
        status_path = run / "status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("state") != "completed":
            raise RuntimeError(f"Cannot resume incomplete deterministic baseline: {run}")
        return
    run.mkdir(parents=True)
    _status(run, "running", configuration="p3_a", seed=args.seeds[0])
    mean, std = stats[0], stats[1]
    rows_by_role = {}
    summary_rows = []
    archive: dict[str, np.ndarray] = {}
    for role in ("train", "validation"):
        dataset = datasets[role]
        predictions, targets, observed, sample_ids, role_rows = [], [], [], [], []
        normalized_sum = 0.0
        count = 0
        for start in range(0, len(dataset), args.batch_size):
            if args.max_validation_batches is not None and role == "validation":
                if start // args.batch_size >= args.max_validation_batches:
                    break
            batch = _batch(dataset, list(range(start, min(start + args.batch_size, len(dataset)))))
            boxes = batch["boxes"].float()
            prediction = constant_velocity_forecast(boxes)
            target = batch["pred_target"].float()
            normalized_sum += float(torch.nn.functional.mse_loss(prediction, target)) * len(boxes)
            count += len(boxes)
            pred_raw = prediction.numpy() * std + mean
            target_raw = target.numpy() * std + mean
            obs_raw = boxes.numpy() * std + mean
            predictions.append(pred_raw)
            targets.append(target_raw)
            observed.append(obs_raw)
            sample_ids.extend(batch["sample_id"])
            for i, sample_id in enumerate(batch["sample_id"]):
                row = _sample_metrics(pred_raw[i], target_raw[i], pred_raw[i])
                row.update({
                    "configuration": "p3_a",
                    "policy": "exact_b1_formula",
                    "sample_id": sample_id,
                    "pid": batch["pid"][i],
                    "video": batch["video"][i],
                    "frame_start": int(batch["frame_start"][i]),
                    "frame_end": int(batch["frame_end"][i]),
                    "admission_probability": 1.0,
                    "hard_admission": True,
                    "context_attention_weight": 0.0,
                    "observed_center_speed_px_per_frame": float(np.linalg.norm(
                        (((obs_raw[i, -1, :2] + obs_raw[i, -1, 2:]) / 2)
                         - ((obs_raw[i, -3, :2] + obs_raw[i, -3, 2:]) / 2)) / 2
                    )),
                })
                role_rows.append(row)
        metric = {
            "policy": "exact_b1_formula",
            "normalized_mse": normalized_sum / count,
            "raw_pixel_mse": float(np.mean([row["raw_pixel_mse"] for row in role_rows])),
            "cmse": float(np.mean([row["cmse"] for row in role_rows])),
            "cfmse": float(np.mean([row["cfmse"] for row in role_rows])),
            "mse_h01": float(np.mean([row["mse_h01"] for row in role_rows])),
            "mse_h05": float(np.mean([row["mse_h05"] for row in role_rows])),
            "mse_h15": float(np.mean([row["mse_h15"] for row in role_rows])),
            "mse_h30": float(np.mean([row["mse_h30"] for row in role_rows])),
            "mse_h45": float(np.mean([row["mse_h45"] for row in role_rows])),
            "sample_count": count,
        }
        rows_by_role[role] = role_rows
        summary_rows.append({
            "configuration": "p3_a", "seed": args.seeds[0], "status": "completed",
            "policy": "exact_b1_formula", "epochs_completed": 0, "selected_epoch": 0,
            "train_normalized_mse": metric["normalized_mse"] if role == "train" else None,
            "validation_normalized_mse": metric["normalized_mse"] if role == "validation" else None,
            "validation_raw_pixel_mse": metric["raw_pixel_mse"] if role == "validation" else None,
            "validation_cmse": metric["cmse"] if role == "validation" else None,
            "validation_cfmse": metric["cfmse"] if role == "validation" else None,
            "validation_mse_h01": metric["mse_h01"] if role == "validation" else None,
            "validation_mse_h05": metric["mse_h05"] if role == "validation" else None,
            "validation_mse_h15": metric["mse_h15"] if role == "validation" else None,
            "validation_mse_h30": metric["mse_h30"] if role == "validation" else None,
            "validation_mse_h45": metric["mse_h45"] if role == "validation" else None,
            "unique_parameters": 0, "training_seconds": 0.0,
            "peak_sampled_rss_bytes": _process_memory_bytes() or 0,
            "checkpoint_bytes": 0,
            "mean_admission_probability": None, "hard_admission_rate": None,
            "context_attention_weight": None,
        })
        if role == "validation":
            archive.update({
                f"{role}_sample_id": np.asarray(sample_ids, dtype=str),
                f"{role}_predictions_raw": np.concatenate(predictions),
                f"{role}_targets_raw": np.concatenate(targets),
                f"{role}_observed_boxes_raw": np.concatenate(observed),
            })
        atomic_write_json(run / f"{role}_metrics.json", metric)
    _append_rows(args.output_root / "P3_VALIDATION_SUMMARY.csv",
                 [row for row in summary_rows if row["validation_normalized_mse"] is not None],
                 SUMMARY_FIELDS)
    atomic_write_bytes(
        run / "validation_predictions.npz",
        _npz_bytes(archive),
    )
    _write_sample_csv(run / "validation_per_sample.csv", rows_by_role["validation"])
    atomic_write_json(run / "run_config.json", {
        "configuration": "p3_a",
        "seed": args.seeds[0],
        "training": False,
        "baseline_formula": "(x_t - x_(t-2))/2 per [x1,y1,x2,y2], steps 1..45",
        "validation_only": True,
        "test_split_loaded_as_dataset": False,
    })
    _status(run, "completed", configuration="p3_a", epochs_completed=0)


def _npz_bytes(values: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["sample_id"]
    atomic_write_csv(path, rows, fields)


def _run_one(
    configuration: str,
    seed: int,
    datasets: dict[str, Any],
    stats: tuple[np.ndarray, ...],
    args: argparse.Namespace,
) -> None:
    run = args.output_root / configuration / str(seed)
    config = _config(configuration, seed, args)
    checkpoint_path = run / "latest_checkpoint.pt"
    set_seed(seed, args.cpu_threads)
    model = build_p3_model(configuration, zero_correction_head=args.zero_correction_head)
    initial_digest = _state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    resumed = _prepare_run(run, config, args.resume)
    if resumed is None:
        state: dict[str, Any] = {
            "completed_epochs": 0,
            "epoch_order": None,
            "next_batch_index": 0,
            "epoch_loss_sum": 0.0,
            "epoch_sample_count": 0,
            "history": [],
            "best_validation_loss": float("inf"),
            "best_epoch": 0,
            "stale_epochs": 0,
            "best_model_state": None,
            "total_training_seconds": 0.0,
            "peak_sampled_rss_bytes": 0,
            "initial_state_sha256": initial_digest,
        }
        _status(run, "running", configuration=configuration, seed=seed, completed_epochs=0)
        _checkpoint(checkpoint_path, model, optimizer, generator, state)
    else:
        state = resumed
        if state.get("initial_state_sha256") != initial_digest:
            raise RuntimeError("Recreated initialization differs; refusing resume.")
        model.load_state_dict(state["model_state"])
        if state["optimizer_state"] is None:
            raise RuntimeError("Resumable P3 checkpoint lacks optimizer state.")
        optimizer.load_state_dict(state["optimizer_state"])
        _restore_rng(state["rng_state"], generator)
        _progress(run, f"RESUME {configuration} seed={seed} epoch={state['completed_epochs'] + 1}")
    atomic_write_json(run / "run_config.json", config)
    _progress(
        run,
        f"INITIALIZED {configuration} seed={seed} parameters={trainable_parameter_count(model)} "
        f"anchor={model.anchor} initial_state_sha256={initial_digest}",
    )
    _status(
        run, "running", configuration=configuration, seed=seed,
        completed_epochs=state["completed_epochs"], maximum_epochs=args.max_epochs,
    )
    train_dataset, validation_dataset = datasets["train"], datasets["validation"]
    full_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    batches_per_epoch = min(full_batches, args.max_train_batches) if args.max_train_batches else full_batches
    started_invocation = time.perf_counter()
    if float(state["total_training_seconds"]) >= args.wall_clock_budget_seconds:
        _status(run, "incomplete", reason="wall_clock_budget_exhausted",
                completed_epochs=state["completed_epochs"])
        return
    try:
        while state["completed_epochs"] < args.max_epochs:
            epoch = state["completed_epochs"] + 1
            if state["epoch_order"] is None:
                state["epoch_order"] = torch.randperm(
                    len(train_dataset), generator=generator
                ).tolist()
                state["next_batch_index"] = 0
                state["epoch_loss_sum"] = 0.0
                state["epoch_sample_count"] = 0
            epoch_started = time.perf_counter()
            model.train()
            while state["next_batch_index"] < batches_per_epoch:
                batch_index = state["next_batch_index"]
                indices = state["epoch_order"][
                    batch_index * args.batch_size:(batch_index + 1) * args.batch_size
                ]
                batch = _batch(train_dataset, indices)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_batch(batch)
                loss = torch.nn.functional.mse_loss(prediction, batch["pred_target"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite training loss at {configuration}, epoch {epoch}.")
                loss.backward()
                optimizer.step()
                n = int(batch["pred_target"].shape[0])
                state["epoch_loss_sum"] += float(loss.detach()) * n
                state["epoch_sample_count"] += n
                state["next_batch_index"] += 1
                rss = _process_memory_bytes() or 0
                state["peak_sampled_rss_bytes"] = max(
                    int(state["peak_sampled_rss_bytes"]), rss
                )
                if state["next_batch_index"] % args.checkpoint_every_batches == 0:
                    _checkpoint(checkpoint_path, model, optimizer, generator, state)
                elapsed = (
                    float(state["total_training_seconds"])
                    + time.perf_counter() - started_invocation
                )
                if elapsed >= args.wall_clock_budget_seconds:
                    state["total_training_seconds"] = elapsed
                    _checkpoint(checkpoint_path, model, optimizer, generator, state)
                    _status(
                        run, "incomplete", reason="wall_clock_budget_exhausted",
                        completed_epochs=state["completed_epochs"],
                        epoch_in_progress=epoch,
                        next_batch_index=state["next_batch_index"],
                        elapsed_seconds=elapsed,
                    )
                    _progress(run, f"INCOMPLETE {configuration} epoch={epoch} batch={state['next_batch_index']}")
                    return
                if state["next_batch_index"] % 5 == 0:
                    _progress(
                        run, f"HEARTBEAT {configuration} epoch={epoch} "
                        f"batches={state['next_batch_index']}/{batches_per_epoch} "
                        f"rss_bytes={rss}",
                    )
            train_loss = state["epoch_loss_sum"] / state["epoch_sample_count"]
            validation_policy = {
                "p3_d": "hard", "p3_e": "hard", "p3_f": "always_open",
                "p3_g": "soft",
            }.get(configuration, "base")
            validation, _ = _evaluate(
                model, validation_dataset, stats, validation_policy,
                args.batch_size, args.max_validation_batches,
            )
            validation_loss = float(validation["normalized_mse"])
            epoch_seconds = time.perf_counter() - epoch_started
            state["history"].append({
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation_loss,
                "epoch_seconds": epoch_seconds,
                "training_batches": batches_per_epoch,
                "validation_samples": validation["sample_count"],
                "process_rss_bytes": _process_memory_bytes(),
            })
            state["completed_epochs"] = epoch
            state["epoch_order"] = None
            state["next_batch_index"] = 0
            if validation_loss < state["best_validation_loss"]:
                state["best_validation_loss"] = validation_loss
                state["best_epoch"] = epoch
                state["stale_epochs"] = 0
                state["best_model_state"] = _copy_state(model)
            else:
                state["stale_epochs"] += 1
            state["total_training_seconds"] += time.perf_counter() - started_invocation
            started_invocation = time.perf_counter()
            atomic_write_json(run / "history.json", state["history"])
            atomic_write_csv(run / "history.csv", state["history"], [
                "epoch", "train_normalized_mse", "validation_normalized_mse",
                "epoch_seconds", "training_batches", "validation_samples",
                "process_rss_bytes",
            ])
            _checkpoint(checkpoint_path, model, optimizer, generator, state)
            _progress(
                run, f"EPOCH {configuration} {epoch}/{args.max_epochs} "
                f"train_mse={train_loss:.8f} validation_mse={validation_loss:.8f} "
                f"seconds={epoch_seconds:.2f}",
            )
            _status(
                run, "running", configuration=configuration,
                completed_epochs=epoch, maximum_epochs=args.max_epochs,
                best_epoch=state["best_epoch"], stale_epochs=state["stale_epochs"],
            )
            if state["stale_epochs"] >= args.patience:
                break
        if state["best_model_state"] is None:
            raise RuntimeError("No validation checkpoint was selected.")
        model.load_state_dict(state["best_model_state"])
        train_policy = {
            "p3_d": "hard", "p3_e": "hard", "p3_f": "always_open",
            "p3_g": "soft",
        }.get(configuration, "base")
        train_loss = _dataset_loss(
            model, train_dataset, args.batch_size, train_policy,
            args.max_train_batches,
        )
        validation_results, arrays_by_policy = [], {}
        sample_rows_by_policy = {}
        for policy in _policy_list(configuration):
            result, arrays = _evaluate(
                model, validation_dataset, stats, policy, args.batch_size,
                args.max_validation_batches,
            )
            sample_rows_by_policy[policy] = result.pop("sample_rows")
            arrays_by_policy[policy] = arrays
            validation_results.append(result)
        all_archive: dict[str, np.ndarray] = {}
        combined_rows = []
        for policy, arrays in arrays_by_policy.items():
            for name, value in arrays.items():
                all_archive[f"{policy}__{name}"] = value
            for row in sample_rows_by_policy[policy]:
                combined_rows.append({"configuration": configuration, "policy": policy, **row})
        atomic_write_bytes(run / "validation_predictions.npz", _npz_bytes(all_archive))
        if combined_rows:
            _write_sample_csv(run / "validation_per_sample.csv", combined_rows)
        rows = []
        for metrics in validation_results:
            row = {
                "configuration": configuration,
                "seed": seed,
                "status": "completed",
                "policy": metrics["policy"],
                "epochs_completed": state["completed_epochs"],
                "selected_epoch": state["best_epoch"],
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": metrics["normalized_mse"],
                "validation_raw_pixel_mse": metrics["raw_pixel_mse"],
                "validation_cmse": metrics["cmse"],
                "validation_cfmse": metrics["cfmse"],
                "validation_mse_h01": metrics["mse_h01"],
                "validation_mse_h05": metrics["mse_h05"],
                "validation_mse_h15": metrics["mse_h15"],
                "validation_mse_h30": metrics["mse_h30"],
                "validation_mse_h45": metrics["mse_h45"],
                "unique_parameters": trainable_parameter_count(model),
                "training_seconds": state["total_training_seconds"],
                "peak_sampled_rss_bytes": state["peak_sampled_rss_bytes"],
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "mean_admission_probability": metrics["mean_admission_probability"],
                "hard_admission_rate": metrics["hard_admission_rate"],
                "context_attention_weight": metrics["context_attention_weight"],
            }
            rows.append(row)
        atomic_write_json(run / "validation_metrics.json", {
            "status": "completed",
            "configuration": configuration,
            "seed": seed,
            "selected_epoch": state["best_epoch"],
            "epochs_completed": state["completed_epochs"],
            "initial_state_sha256": initial_digest,
            "unique_parameters": trainable_parameter_count(model),
            "training_seconds": state["total_training_seconds"],
            "peak_sampled_rss_bytes": state["peak_sampled_rss_bytes"],
            "policies": validation_results,
        })
        _append_rows(args.output_root / "P3_VALIDATION_SUMMARY.csv", rows, SUMMARY_FIELDS)
        _status(
            run, "completed", configuration=configuration, seed=seed,
            completed_epochs=state["completed_epochs"], selected_epoch=state["best_epoch"],
            training_seconds=state["total_training_seconds"],
        )
        _progress(
            run, f"COMPLETED {configuration} seed={seed} "
            f"epochs={state['completed_epochs']} best_epoch={state['best_epoch']}",
        )
    except KeyboardInterrupt:
        _checkpoint(checkpoint_path, model, optimizer, generator, state)
        _status(run, "interrupted", configuration=configuration,
                completed_epochs=state["completed_epochs"])
        raise
    except Exception as error:
        _status(run, "failed", configuration=configuration,
                completed_epochs=state.get("completed_epochs", 0),
                error=repr(error), traceback=traceback.format_exc())
        raise


def _write_per_window_comparison(output_root: Path) -> None:
    source_files = list(output_root.glob("p3_*/20260928/validation_per_sample.csv"))
    new_rows = []
    for source in sorted(source_files):
        with source.open("r", newline="", encoding="utf-8") as handle:
            new_rows.extend(csv.DictReader(handle))
    if not new_rows:
        return
    path = output_root / "P3_PER_WINDOW_COMPARISON.csv"
    old_rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
    existing = {
        (row.get("configuration"), row.get("policy"), row.get("sample_id"))
        for row in old_rows
    }
    merged = list(old_rows)
    for row in new_rows:
        key = (row.get("configuration"), row.get("policy"), row.get("sample_id"))
        if key not in existing:
            merged.append(row)
            existing.add(key)
    fields = list(merged[0])
    atomic_write_csv(path, merged, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", nargs="+",
                        default=("p3_a",) + P3_CONFIGURATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wall-clock-budget-seconds", type=float, default=3600.0)
    parser.add_argument("--checkpoint-every-batches", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--zero-correction-head", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("Epoch, batch-size, and thread limits must be positive.")
    if any(key not in {"p3_a", *P3_CONFIGURATIONS} for key in args.configurations):
        raise ValueError("Unknown Phase 3 configuration.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets, stats = load_split(validation_only=True)
    if set(datasets) != {"train", "validation"}:
        raise RuntimeError("Phase 3 must receive only train and validation datasets.")
    if "p3_a" in args.configurations:
        _run_cv_baseline(datasets, stats, args)
    for configuration in args.configurations:
        if configuration == "p3_a":
            continue
        for seed in args.seeds:
            existing_status = args.output_root / configuration / str(seed) / "status.json"
            if args.resume and existing_status.exists():
                if json.loads(existing_status.read_text()).get("state") == "completed":
                    continue
            _run_one(configuration, seed, datasets, stats, args)
            status_path = args.output_root / configuration / str(seed) / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("state") != "completed":
                _write_per_window_comparison(args.output_root)
                return
    _write_per_window_comparison(args.output_root)
    try:
        from .finalize_p3_results import enrich
    except ImportError:
        from finalize_p3_results import enrich
    enrich(args.output_root)
    if args.max_train_batches is None and args.max_validation_batches is None and not args.skip_figures:
        try:
            from .plot_p3_qualitative import make_figures
        except ImportError:
            from plot_p3_qualitative import make_figures
        make_figures(args.output_root)


if __name__ == "__main__":
    main()
