"""Resumable, validation-only Phase 4 screening on independent role caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

try:
    from .context_TF import ContextIndividualTF
    from .cv_residual_v1 import (
        ResidualAnchorModel,
        constant_velocity_forecast,
        unwrap_predictor,
    )
    from .dataset_roles_v1 import ROLE_CACHE_ROOT, load_train_validation
    from .p4_models_v1 import (
        P4_CONFIGURATIONS,
        P4_EVALUATION_POLICIES,
        P4_SELECTION_POLICY,
        build_p4_model,
        parameter_sharing_audit,
        unique_parameter_count,
    )
    from .individual_TF import IndividualTF
    from .run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_csv,
        atomic_write_json,
        set_seed,
    )
    from .run_from_scratch_e1 import RESULTS
    from .sage_v2 import AlwaysOpenContextIndividualTF, DifferentiableSAGEV2
except ImportError:
    from context_TF import ContextIndividualTF
    from cv_residual_v1 import ResidualAnchorModel, constant_velocity_forecast, unwrap_predictor
    from dataset_roles_v1 import ROLE_CACHE_ROOT, load_train_validation
    from p4_models_v1 import (
        P4_CONFIGURATIONS,
        P4_EVALUATION_POLICIES,
        P4_SELECTION_POLICY,
        build_p4_model,
        parameter_sharing_audit,
        unique_parameter_count,
    )
    from individual_TF import IndividualTF
    from run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_csv,
        atomic_write_json,
        set_seed,
    )
    from run_from_scratch_e1 import RESULTS
    from sage_v2 import AlwaysOpenContextIndividualTF, DifferentiableSAGEV2


DEFAULT_OUTPUT_ROOT = RESULTS / "P4_initial_screen_v1"
DEFAULT_SEEDS = (20260928, 20260929, 20260930)
SUMMARY_FIELDS = (
    "configuration", "seed", "status", "policy", "epochs_completed",
    "selected_epoch", "train_normalized_mse", "train_raw_pixel_mse",
    "validation_normalized_mse", "validation_raw_pixel_mse",
    "validation_cmse", "validation_cfmse", "validation_mse_h01",
    "validation_mse_h05", "validation_mse_h15", "validation_mse_h30",
    "validation_mse_h45", "unique_parameters", "training_seconds",
    "peak_sampled_rss_bytes", "checkpoint_bytes", "gate_applicable",
    "mean_admission_probability", "admission_probability_sd",
    "admission_probability_q05", "admission_probability_q25",
    "admission_probability_median", "admission_probability_q75",
    "admission_probability_q95", "hard_admission_rate",
    "effective_admission_rate", "gate_entropy_nats",
    "context_attention_weight", "mean_correction_rms_pixels",
    "correction_rms_sd_pixels", "mean_gate_gradient_l2",
    "mean_context_encoder_gradient_l2",
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
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    return default_collate([dataset[index] for index in indices])


def _progress(run: Path, message: str) -> None:
    print(message, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _status(run: Path, state: str, **fields: Any) -> None:
    if state not in {"running", "completed", "interrupted", "incomplete", "failed"}:
        raise ValueError(f"Unknown Phase 4 run state: {state}")
    atomic_write_json(run / "status.json", {
        "state": state,
        "updated_at": _utc_now(),
        **fields,
    })


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    state: dict[str, Any],
) -> None:
    payload = dict(state)
    payload.update({
        "model_state": _copy_state(model),
        "optimizer_state": optimizer.state_dict(),
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


def _make_config(
    configuration: str,
    seed: int,
    args: argparse.Namespace,
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    module_dir = Path(__file__).resolve().parent
    source_names = (
        "run_p4_screen.py",
        "p4_models_v1.py",
        "cv_residual_v1.py",
        "dataset_roles_v1.py",
        "sage_v2.py",
        "context_TF.py",
        "individual_TF.py",
        "c4_encoder_shared.py",
    )
    return {
        "runner_version": "P4_screen_v1",
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
        "checkpoint_every_batches": args.checkpoint_every_batches,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "validation_only": True,
        "cache_schema": cache_metadata["schema"],
        "loaded_roles": ["train", "validation"],
        "normalization_fit_role": "train",
        "role_cache_hashes": {
            role: dict(values)
            for role, values in cache_metadata["roles_loaded"].items()
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "source_sha256": {
            name: _sha256_file(module_dir / name) for name in source_names
        },
    }


def _prepare_run(
    run: Path,
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Any] | None:
    if not run.exists():
        run.mkdir(parents=True)
        atomic_write_json(run / "run_config.json", config)
        return None
    if not resume:
        raise FileExistsError(
            f"Run path exists; use --resume only for a compatible incomplete run: {run}"
        )
    config_path = run / "run_config.json"
    checkpoint_path = run / "latest_checkpoint.pt"
    status_path = run / "status.json"
    if not all(path.is_file() for path in (config_path, checkpoint_path, status_path)):
        raise RuntimeError(f"Existing Phase 4 run lacks config, status, or checkpoint: {run}")
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    mutable = {"wall_clock_budget_seconds"}
    if any(
        previous.get(key) != value
        for key, value in config.items()
        if key not in mutable
    ):
        raise ValueError(f"Saved Phase 4 run configuration is incompatible: {run}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") == "completed":
        raise RuntimeError(f"Completed Phase 4 run cannot be overwritten: {run}")
    if status.get("state") == "failed":
        raise RuntimeError(f"Failed Phase 4 run requires inspection before resuming: {run}")
    atomic_write_json(config_path, config)
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _gradient_stats(module: torch.nn.Module | None) -> dict[str, Any]:
    if module is None:
        return {"l2_norm": None, "missing": 0, "zero": 0, "total": 0}
    parameters = [item for item in module.parameters() if item.requires_grad]
    missing = sum(item.grad is None for item in parameters)
    zero = 0
    squared_norm = 0.0
    present = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Non-finite Phase 4 gradient.")
        norm_squared = float(torch.sum(gradient * gradient).item())
        zero += int(norm_squared == 0.0)
        squared_norm += norm_squared
        present += 1
    return {
        "l2_norm": math.sqrt(squared_norm) if present else None,
        "missing": missing,
        "zero": zero,
        "total": len(parameters),
    }


def _gradient_modules(model: ResidualAnchorModel):
    predictor = model.predictor
    if isinstance(predictor, (ContextIndividualTF, DifferentiableSAGEV2)):
        return predictor.gate, predictor.context_encoder
    if isinstance(predictor, AlwaysOpenContextIndividualTF):
        return None, predictor.context_encoder
    return None, None


@contextmanager
def _validation_policy(model: ResidualAnchorModel, policy: str):
    predictor = model.predictor
    old_gate_mode = None
    old_gate_fn = None
    old_inference_mode = None
    old_forced_admission = None
    if isinstance(predictor, ContextIndividualTF):
        if policy in {"hard", "soft"}:
            old_gate_mode = predictor.gate_mode
            predictor.gate_mode = policy
        elif policy in {"forced_open", "forced_closed"}:
            old_gate_fn = predictor._context_gate
            forced = 1.0 if policy == "forced_open" else 0.0

            def gate_fn(context_token, observed_boxes):
                logits = predictor.gate(torch.cat(
                    (context_token, observed_boxes[:, -1]), dim=-1
                ))
                probability = logits.softmax(dim=-1)[:, 1]
                gate = torch.full_like(probability, forced)
                predictor.last_context_gate = gate.detach()
                predictor.last_context_gate_logits = logits.detach()
                predictor._context_gate_for_loss = gate
                predictor._p4_soft_admission_probability = probability.detach()
                return gate

            predictor._context_gate = gate_fn
        else:
            raise ValueError(f"Unsupported original SAGE policy: {policy}")
    elif isinstance(predictor, DifferentiableSAGEV2):
        old_inference_mode = predictor.inference_mode
        old_forced_admission = predictor.forced_admission
        if policy in {"soft", "hard"}:
            predictor.inference_mode = policy
            predictor.forced_admission = None
        elif policy in {"forced_open", "forced_closed"}:
            predictor.inference_mode = "hard"
            predictor.forced_admission = 1.0 if policy == "forced_open" else 0.0
        else:
            raise ValueError(f"Unsupported SAGE V2 policy: {policy}")
    elif isinstance(predictor, AlwaysOpenContextIndividualTF):
        if policy != "always_open":
            raise ValueError(f"Unsupported always-open policy: {policy}")
    elif policy != "base":
        raise ValueError(f"Unsupported base-model policy: {policy}")

    try:
        yield
    finally:
        if old_gate_mode is not None:
            predictor.gate_mode = old_gate_mode
        if old_gate_fn is not None:
            predictor._context_gate = old_gate_fn
        if old_inference_mode is not None:
            predictor.inference_mode = old_inference_mode
            predictor.forced_admission = old_forced_admission


def _context_attention(predictor: torch.nn.Module, batch_size: int) -> np.ndarray:
    if isinstance(predictor, DifferentiableSAGEV2):
        value = predictor.last_context_attention
        if value is not None:
            return value.detach().cpu().numpy()
    base = predictor if isinstance(predictor, IndividualTF) else getattr(predictor, "base", None)
    if base is None:
        return np.zeros(batch_size, dtype=np.float32)
    values = []
    for layer in base.model.decoder.layers:
        attention = layer.src_attn.attn
        if attention is not None:
            if attention.shape[-1] == 16:
                values.append(attention[..., -1].mean(dim=(1, 2)).detach().cpu())
            elif attention.shape[-1] == 15:
                values.append(torch.zeros(batch_size))
    if not values:
        return np.zeros(batch_size, dtype=np.float32)
    return torch.stack(values).mean(dim=0).numpy()


def _gate_diagnostics(
    predictor: torch.nn.Module,
    policy: str,
    batch_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    if isinstance(predictor, ContextIndividualTF):
        logits = predictor.last_context_gate_logits
        effective = predictor.last_context_gate
        if logits is None or effective is None:
            raise RuntimeError("Historical SAGE did not retain gate diagnostics.")
        probabilities = logits.softmax(dim=-1)[:, 1]
        decisions = logits.argmax(dim=-1).eq(1)
        return (
            probabilities.detach().cpu().numpy(),
            effective.detach().cpu().numpy(),
            decisions.detach().cpu().numpy(),
            logits.detach().cpu().numpy(),
        )
    if isinstance(predictor, DifferentiableSAGEV2):
        logits = predictor.last_gate_logits
        effective = predictor.last_admission_probability
        if logits is None or effective is None:
            raise RuntimeError("SAGE V2 did not retain gate diagnostics.")
        probabilities = logits.softmax(dim=-1)[:, 1]
        decisions = logits.argmax(dim=-1).eq(1)
        return (
            probabilities.detach().cpu().numpy(),
            effective.detach().cpu().numpy(),
            decisions.detach().cpu().numpy(),
            logits.detach().cpu().numpy(),
        )
    if isinstance(predictor, AlwaysOpenContextIndividualTF):
        ones = np.ones(batch_size, dtype=np.float32)
        return None, ones, np.ones(batch_size, dtype=bool), np.empty((batch_size, 0))
    return None, None, np.zeros(batch_size, dtype=bool), np.empty((batch_size, 0))


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


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


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
    normalized_predictions, normalized_targets = [], []
    raw_predictions, raw_targets, cv_predictions, observed_raw = [], [], [], []
    sample_ids, pids, videos = [], [], []
    probabilities, effective_admissions, hard_decisions = [], [], []
    logits_values, attentions, sample_rows = [], [], []
    normalized_loss_sum = 0.0
    count = 0
    with _validation_policy(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and start // batch_size >= max_batches:
                break
            batch = _batch(
                dataset, list(range(start, min(start + batch_size, len(dataset))))
            )
            prediction = model.forward_batch(batch)
            target = batch["pred_target"]
            n = int(target.shape[0])
            normalized_loss_sum += float(F.mse_loss(prediction, target)) * n
            count += n

            prediction_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            boxes_np = batch["boxes"].detach().cpu().numpy()
            raw_prediction = prediction_np * std + mean
            raw_target = target_np * std + mean
            raw_observed = boxes_np * std + mean
            raw_cv = constant_velocity_forecast(batch["boxes"]).detach().cpu().numpy()
            raw_cv = raw_cv * std + mean

            normalized_predictions.append(prediction_np)
            normalized_targets.append(target_np)
            raw_predictions.append(raw_prediction)
            raw_targets.append(raw_target)
            cv_predictions.append(raw_cv)
            observed_raw.append(raw_observed)
            sample_ids.extend(batch["sample_id"])
            pids.extend(batch["pid"])
            videos.extend(batch["video"])

            gate_probability, effective, decisions, logits = _gate_diagnostics(
                predictor, policy, n
            )
            if gate_probability is None:
                probabilities.extend([None] * n)
                entropies = [None] * n
            else:
                probabilities.extend(gate_probability.tolist())
                entropies = _binary_entropy(gate_probability).tolist()
            if effective is None:
                effective_admissions.extend([None] * n)
            else:
                effective_admissions.extend(effective.tolist())
            if gate_probability is None:
                hard_decisions.extend([None] * n)
            else:
                hard_decisions.extend(decisions.tolist())
            logits_values.append(logits)
            batch_attention = _context_attention(predictor, n)
            attentions.append(batch_attention)

            for index in range(n):
                row = _sample_metrics(
                    raw_prediction[index], raw_target[index], raw_cv[index]
                )
                row.update({
                    "sample_id": batch["sample_id"][index],
                    "pid": batch["pid"][index],
                    "video": batch["video"][index],
                    "configuration": "",
                    "policy": policy,
                    "admission_probability": (
                        "" if gate_probability is None else float(gate_probability[index])
                    ),
                    "effective_admission": (
                        "" if effective is None else float(effective[index])
                    ),
                    "hard_gate_decision": (
                        "" if gate_probability is None else bool(decisions[index])
                    ),
                    "gate_entropy_nats": (
                        "" if gate_probability is None else float(entropies[index])
                    ),
                    "context_attention_weight": float(batch_attention[index]),
                })
                sample_rows.append(row)

    if count == 0:
        raise RuntimeError(f"No samples evaluated for Phase 4 policy {policy}.")
    raw_p = np.concatenate(raw_predictions)
    raw_t = np.concatenate(raw_targets)
    raw_cv = np.concatenate(cv_predictions)
    normalized_p = np.concatenate(normalized_predictions)
    normalized_t = np.concatenate(normalized_targets)
    sample_metrics = [
        _sample_metrics(raw_p[index], raw_t[index], raw_cv[index])
        for index in range(count)
    ]
    metric_names = (
        "raw_pixel_mse", "cmse", "cfmse", "mse_h01", "mse_h05",
        "mse_h15", "mse_h30", "mse_h45", "correction_rms_pixels",
    )
    metrics: dict[str, Any] = {
        "policy": policy,
        "sample_count": count,
        "normalized_mse": normalized_loss_sum / count,
        **{
            name: float(np.mean([row[name] for row in sample_metrics]))
            for name in metric_names
        },
        "context_attention_weight": float(np.mean(np.concatenate(attentions))),
    }
    valid_probabilities = np.asarray(
        [value for value in probabilities if value is not None], dtype=np.float64
    )
    valid_effective = np.asarray(
        [value for value in effective_admissions if value is not None], dtype=np.float64
    )
    valid_decisions = np.asarray(
        [value for value in hard_decisions if value is not None], dtype=np.float64
    )
    if valid_probabilities.size:
        quantiles = np.quantile(valid_probabilities, [0.05, 0.25, 0.5, 0.75, 0.95])
        metrics.update({
            "gate_applicable": True,
            "mean_admission_probability": float(np.mean(valid_probabilities)),
            "admission_probability_sd": float(np.std(valid_probabilities, ddof=0)),
            "admission_probability_q05": float(quantiles[0]),
            "admission_probability_q25": float(quantiles[1]),
            "admission_probability_median": float(quantiles[2]),
            "admission_probability_q75": float(quantiles[3]),
            "admission_probability_q95": float(quantiles[4]),
            "hard_admission_rate": float(np.mean(valid_decisions)),
            "effective_admission_rate": float(np.mean(valid_effective)),
            "gate_entropy_nats": float(np.mean(_binary_entropy(valid_probabilities))),
        })
    else:
        metrics.update({
            "gate_applicable": False,
            "mean_admission_probability": None,
            "admission_probability_sd": None,
            "admission_probability_q05": None,
            "admission_probability_q25": None,
            "admission_probability_median": None,
            "admission_probability_q75": None,
            "admission_probability_q95": None,
            "hard_admission_rate": None,
            "effective_admission_rate": (
                float(np.mean(valid_effective)) if valid_effective.size else None
            ),
            "gate_entropy_nats": None,
        })
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype=str),
        "pid": np.asarray(pids, dtype=str),
        "video": np.asarray(videos, dtype=str),
        "predictions_normalized": normalized_p,
        "targets_normalized": normalized_t,
        "predictions_raw": raw_p,
        "targets_raw": raw_t,
        "cv_predictions_raw": raw_cv,
        "observed_boxes_raw": np.concatenate(observed_raw),
        "admission_probability": np.asarray([
            np.nan if value is None else value for value in probabilities
        ], dtype=np.float64),
        "effective_admission": np.asarray([
            np.nan if value is None else value for value in effective_admissions
        ], dtype=np.float64),
        "hard_gate_decision": np.asarray([
            np.nan if value is None else float(value) for value in hard_decisions
        ], dtype=np.float64),
        "gate_logits": np.concatenate(logits_values),
        "context_attention_weight": np.concatenate(attentions),
    }
    return {**metrics, "sample_rows": sample_rows}, arrays


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    atomic_write_csv(path, rows, fields)


def _train_loss(
    model: ResidualAnchorModel,
    dataset: Any,
    batch_size: int,
    policy: str,
    max_batches: int | None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with _validation_policy(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and start // batch_size >= max_batches:
                break
            batch = _batch(
                dataset, list(range(start, min(start + batch_size, len(dataset))))
            )
            prediction = model.forward_batch(batch)
            n = int(batch["pred_target"].shape[0])
            total += float(F.mse_loss(prediction, batch["pred_target"])) * n
            count += n
    if count == 0:
        raise RuntimeError("Training metric has no training samples.")
    return total / count


def _baseline_role(
    dataset: Any,
    stats: tuple[np.ndarray, ...],
    batch_size: int,
    max_batches: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean, std = stats[0], stats[1]
    normalized_loss_sum = 0.0
    count = 0
    rows = []
    for start in range(0, len(dataset), batch_size):
        if max_batches is not None and start // batch_size >= max_batches:
            break
        batch = _batch(dataset, list(range(start, min(start + batch_size, len(dataset)))))
        prediction = constant_velocity_forecast(batch["boxes"])
        target = batch["pred_target"]
        n = int(target.shape[0])
        normalized_loss_sum += float(F.mse_loss(prediction, target)) * n
        count += n
        raw_prediction = prediction.detach().cpu().numpy() * std + mean
        raw_target = target.detach().cpu().numpy() * std + mean
        for index in range(n):
            row = _sample_metrics(
                raw_prediction[index], raw_target[index], raw_prediction[index]
            )
            row.update({
                "configuration": "cv_baseline",
                "policy": "deterministic",
                "sample_id": batch["sample_id"][index],
                "pid": batch["pid"][index],
                "video": batch["video"][index],
                "admission_probability": "",
                "effective_admission": "",
                "hard_gate_decision": "",
                "gate_entropy_nats": "",
                "context_attention_weight": 0.0,
            })
            rows.append(row)
    if count == 0:
        raise RuntimeError("Deterministic CV baseline received no samples.")
    return {
        "sample_count": count,
        "normalized_mse": normalized_loss_sum / count,
        **{
            name: float(np.mean([row[name] for row in rows]))
            for name in (
                "raw_pixel_mse", "cmse", "cfmse", "mse_h01", "mse_h05",
                "mse_h15", "mse_h30", "mse_h45",
            )
        },
    }, rows


def _ensure_cv_baseline(
    output_root: Path,
    datasets: dict[str, Any],
    stats: tuple[np.ndarray, ...],
    cache_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    json_path = output_root / "P4_CV_BASELINE.json"
    csv_path = output_root / "P4_CV_BASELINE_PER_WINDOW.csv"
    if json_path.exists() or csv_path.exists():
        if not json_path.is_file() or not csv_path.is_file():
            raise RuntimeError("Existing deterministic CV baseline output is incomplete.")
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if previous.get("role_cache_hashes") != cache_metadata["roles_loaded"]:
            raise ValueError("Existing CV baseline was made from different role caches.")
        return
    train, train_rows = _baseline_role(
        datasets["train"], stats, args.batch_size, args.max_train_batches
    )
    validation, validation_rows = _baseline_role(
        datasets["validation"], stats, args.batch_size, args.max_validation_batches
    )
    record = {
        "status": "completed",
        "configuration": "cv_baseline",
        "policy": "deterministic",
        "normalization_fit_role": "train",
        "role_cache_hashes": cache_metadata["roles_loaded"],
        "train": train,
        "validation": validation,
        "unique_parameters": 0,
        "training_seconds": 0.0,
        "validation_only": True,
    }
    atomic_write_json(json_path, record)
    _write_sample_csv(csv_path, validation_rows)


def _save_predictions(path: Path, arrays_by_policy: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{
            f"{policy}__{name}": value
            for policy, arrays in arrays_by_policy.items()
            for name, value in arrays.items()
        })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_one(
    configuration: str,
    seed: int,
    datasets: dict[str, Any],
    stats: tuple[np.ndarray, ...],
    cache_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    run = args.output_root / configuration / str(seed)
    config = _make_config(configuration, seed, args, cache_metadata)
    checkpoint_path = run / "latest_checkpoint.pt"
    set_seed(seed, args.cpu_threads)
    model = build_p4_model(configuration)
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
            "gate_gradient_l2": [],
            "context_gradient_l2": [],
            "gate_gradient_all_missing_batches": 0,
            "context_gradient_all_missing_batches": 0,
            "current_epoch_gate_norms": [],
            "current_epoch_context_norms": [],
            "initial_state_sha256": initial_digest,
        }
        _status(run, "running", configuration=configuration, seed=seed, completed_epochs=0)
        _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
    else:
        state = resumed
        if state.get("initial_state_sha256") != initial_digest:
            raise RuntimeError("Recreated initialization differs; refusing resume.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _restore_rng(state["rng_state"], generator)
        _progress(run, f"RESUME {configuration} seed={seed} epoch={state['completed_epochs'] + 1}")

    atomic_write_json(run / "run_config.json", config)
    sharing = parameter_sharing_audit(model)
    if not sharing["verified"]:
        raise RuntimeError(f"Attention-sharing audit failed for {configuration}: {sharing}")
    _progress(
        run, f"INITIALIZED {configuration} seed={seed} "
        f"parameters={unique_parameter_count(model)} anchor=cv "
        f"initial_state_sha256={initial_digest}",
    )
    _status(
        run, "running", configuration=configuration, seed=seed,
        completed_epochs=state["completed_epochs"], maximum_epochs=args.max_epochs,
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]
    full_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    batches_per_epoch = (
        min(full_batches, args.max_train_batches)
        if args.max_train_batches is not None else full_batches
    )
    if batches_per_epoch < 1:
        raise ValueError("Training must contain at least one batch per epoch.")
    started_invocation = time.perf_counter()
    if state["total_training_seconds"] >= args.wall_clock_budget_seconds:
        _status(
            run, "incomplete", reason="wall_clock_budget_exhausted",
            completed_epochs=state["completed_epochs"],
        )
        return "incomplete"
    gate_module, context_module = _gradient_modules(model)

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
            epoch_gate_norms = state["current_epoch_gate_norms"]
            epoch_context_norms = state["current_epoch_context_norms"]
            while state["next_batch_index"] < batches_per_epoch:
                batch_index = state["next_batch_index"]
                indices = state["epoch_order"][
                    batch_index * args.batch_size:(batch_index + 1) * args.batch_size
                ]
                batch = _batch(train_dataset, indices)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_batch(batch)
                loss = F.mse_loss(prediction, batch["pred_target"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite training loss for {configuration}, epoch {epoch}."
                    )
                loss.backward()
                gate_gradient = _gradient_stats(gate_module)
                context_gradient = _gradient_stats(context_module)
                if gate_module is not None:
                    if gate_gradient["l2_norm"] is None:
                        state["gate_gradient_all_missing_batches"] += 1
                    else:
                        epoch_gate_norms.append(gate_gradient["l2_norm"])
                        state["gate_gradient_l2"].append(gate_gradient["l2_norm"])
                if context_module is not None:
                    if context_gradient["l2_norm"] is None:
                        state["context_gradient_all_missing_batches"] += 1
                    else:
                        epoch_context_norms.append(context_gradient["l2_norm"])
                        state["context_gradient_l2"].append(context_gradient["l2_norm"])
                optimizer.step()
                n = int(batch["pred_target"].shape[0])
                state["epoch_loss_sum"] += float(loss.detach()) * n
                state["epoch_sample_count"] += n
                state["next_batch_index"] += 1
                sampled_rss = _process_memory_bytes() or 0
                state["peak_sampled_rss_bytes"] = max(
                    int(state["peak_sampled_rss_bytes"]), sampled_rss
                )

                elapsed = (
                    float(state["total_training_seconds"])
                    + time.perf_counter() - started_invocation
                )
                if state["next_batch_index"] % args.checkpoint_every_batches == 0:
                    _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
                if state["next_batch_index"] % args.heartbeat_batches == 0:
                    _progress(
                        run, f"HEARTBEAT {configuration} seed={seed} epoch={epoch} "
                        f"batches={state['next_batch_index']}/{batches_per_epoch} "
                        f"elapsed={elapsed:.1f}s rss_bytes={sampled_rss}",
                    )
                if elapsed >= args.wall_clock_budget_seconds:
                    state["total_training_seconds"] = elapsed
                    _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
                    _status(
                        run, "incomplete", reason="wall_clock_budget_exhausted",
                        completed_epochs=state["completed_epochs"],
                        epoch_in_progress=epoch,
                        next_batch_index=state["next_batch_index"],
                        elapsed_seconds=elapsed,
                    )
                    _progress(run, f"INCOMPLETE {configuration} seed={seed} epoch={epoch}")
                    return "incomplete"

            train_loss = state["epoch_loss_sum"] / state["epoch_sample_count"]
            selection_policy = P4_SELECTION_POLICY[configuration]
            validation, _ = _evaluate(
                model, validation_dataset, stats, selection_policy,
                args.batch_size, args.max_validation_batches,
            )
            validation_loss = float(validation["normalized_mse"])
            epoch_seconds = time.perf_counter() - epoch_started
            history_row = {
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation_loss,
                "epoch_seconds": epoch_seconds,
                "training_batches": batches_per_epoch,
                "validation_samples": validation["sample_count"],
                "gate_gradient_l2_mean": (
                    float(np.mean(epoch_gate_norms)) if epoch_gate_norms else None
                ),
                "context_gradient_l2_mean": (
                    float(np.mean(epoch_context_norms)) if epoch_context_norms else None
                ),
                "process_rss_bytes": _process_memory_bytes(),
            }
            state["history"].append(history_row)
            state["current_epoch_gate_norms"] = []
            state["current_epoch_context_norms"] = []
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
            atomic_write_csv(run / "history.csv", state["history"], list(history_row))
            _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
            _progress(
                run, f"EPOCH {configuration} seed={seed} {epoch}/{args.max_epochs} "
                f"train_mse={train_loss:.8f} validation_mse={validation_loss:.8f} "
                f"seconds={epoch_seconds:.2f}",
            )
            _status(
                run, "running", configuration=configuration, seed=seed,
                completed_epochs=epoch, maximum_epochs=args.max_epochs,
                selected_epoch=state["best_epoch"], stale_epochs=state["stale_epochs"],
            )
            if state["total_training_seconds"] >= args.wall_clock_budget_seconds:
                _status(
                    run, "incomplete", reason="wall_clock_budget_exhausted",
                    completed_epochs=state["completed_epochs"],
                    selected_epoch=state["best_epoch"],
                    elapsed_seconds=state["total_training_seconds"],
                )
                _progress(
                    run, f"INCOMPLETE {configuration} seed={seed} "
                    f"after validation epoch={epoch}",
                )
                return "incomplete"
            if state["stale_epochs"] >= args.patience:
                break

        if state["best_model_state"] is None:
            raise RuntimeError("No validation-selected checkpoint was created.")
        model.load_state_dict(state["best_model_state"])
        selection_policy = P4_SELECTION_POLICY[configuration]
        train_metrics, _ = _evaluate(
            model, train_dataset, stats, selection_policy,
            args.batch_size, args.max_train_batches,
        )
        validation_results = []
        arrays_by_policy = {}
        sample_rows = []
        for policy in P4_EVALUATION_POLICIES[configuration]:
            metrics, arrays = _evaluate(
                model, validation_dataset, stats, policy,
                args.batch_size, args.max_validation_batches,
            )
            rows = metrics.pop("sample_rows")
            for row in rows:
                row["configuration"] = configuration
                row["seed"] = seed
            sample_rows.extend(rows)
            arrays_by_policy[policy] = arrays
            validation_results.append(metrics)

        run_metrics = {
            "status": "completed",
            "configuration": configuration,
            "seed": seed,
            "selected_epoch": state["best_epoch"],
            "epochs_completed": state["completed_epochs"],
            "selection_policy": selection_policy,
            "unique_parameters": unique_parameter_count(model),
            "training_seconds": state["total_training_seconds"],
            "peak_sampled_rss_bytes": state["peak_sampled_rss_bytes"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "train": {
                key: value for key, value in train_metrics.items()
                if key != "sample_rows"
            },
            "policies": validation_results,
            "gate_gradient_l2_mean_present_batches": (
                float(np.mean(state["gate_gradient_l2"]))
                if state["gate_gradient_l2"] else None
            ),
            "gate_gradient_batches_all_missing": state["gate_gradient_all_missing_batches"],
            "context_gradient_l2_mean_present_batches": (
                float(np.mean(state["context_gradient_l2"]))
                if state["context_gradient_l2"] else None
            ),
            "context_gradient_batches_all_missing": state["context_gradient_all_missing_batches"],
            "initial_state_sha256": initial_digest,
            "parameter_sharing_audit": sharing,
            "cache_roles_loaded": ["train", "validation"],
        }
        _save_predictions(run / "validation_predictions.npz", arrays_by_policy)
        _write_sample_csv(run / "validation_per_sample.csv", sample_rows)
        atomic_write_json(run / "validation_metrics.json", run_metrics)
        _status(
            run, "completed", configuration=configuration, seed=seed,
            completed_epochs=state["completed_epochs"],
            selected_epoch=state["best_epoch"],
            training_seconds=state["total_training_seconds"],
        )
        _progress(
            run, f"COMPLETED {configuration} seed={seed} "
            f"epochs={state['completed_epochs']} selected_epoch={state['best_epoch']}",
        )
        return "completed"
    except KeyboardInterrupt:
        state["total_training_seconds"] += time.perf_counter() - started_invocation
        _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
        _status(
            run, "interrupted", configuration=configuration, seed=seed,
            completed_epochs=state.get("completed_epochs", 0),
        )
        raise
    except Exception as error:
        state["total_training_seconds"] += time.perf_counter() - started_invocation
        _save_checkpoint(checkpoint_path, model, optimizer, generator, state)
        _status(
            run, "failed", configuration=configuration, seed=seed,
            completed_epochs=state.get("completed_epochs", 0),
            error=repr(error), traceback=traceback.format_exc(),
        )
        raise


def _summary_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("p4_*/202609*/validation_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for policy_metrics in data["policies"]:
            row = {
                "configuration": data["configuration"],
                "seed": data["seed"],
                "status": data["status"],
                "policy": policy_metrics["policy"],
                "epochs_completed": data["epochs_completed"],
                "selected_epoch": data["selected_epoch"],
                "train_normalized_mse": data["train"]["normalized_mse"],
                "train_raw_pixel_mse": data["train"]["raw_pixel_mse"],
                "unique_parameters": data["unique_parameters"],
                "training_seconds": data["training_seconds"],
                "peak_sampled_rss_bytes": data["peak_sampled_rss_bytes"],
                "checkpoint_bytes": data["checkpoint_bytes"],
                "mean_gate_gradient_l2": data["gate_gradient_l2_mean_present_batches"],
                "mean_context_encoder_gradient_l2": data["context_gradient_l2_mean_present_batches"],
            }
            metric_columns = {
                "validation_normalized_mse": "normalized_mse",
                "validation_raw_pixel_mse": "raw_pixel_mse",
                "validation_cmse": "cmse",
                "validation_cfmse": "cfmse",
                "validation_mse_h01": "mse_h01",
                "validation_mse_h05": "mse_h05",
                "validation_mse_h15": "mse_h15",
                "validation_mse_h30": "mse_h30",
                "validation_mse_h45": "mse_h45",
                "gate_applicable": "gate_applicable",
                "mean_admission_probability": "mean_admission_probability",
                "admission_probability_sd": "admission_probability_sd",
                "admission_probability_q05": "admission_probability_q05",
                "admission_probability_q25": "admission_probability_q25",
                "admission_probability_median": "admission_probability_median",
                "admission_probability_q75": "admission_probability_q75",
                "admission_probability_q95": "admission_probability_q95",
                "hard_admission_rate": "hard_admission_rate",
                "effective_admission_rate": "effective_admission_rate",
                "gate_entropy_nats": "gate_entropy_nats",
                "context_attention_weight": "context_attention_weight",
                "mean_correction_rms_pixels": "correction_rms_pixels",
            }
            row.update({
                column: policy_metrics.get(key)
                for column, key in metric_columns.items()
            })
            row["correction_rms_sd_pixels"] = float(np.std(
                [float(sample["correction_rms_pixels"])
                 for sample in _read_sample_rows(path.parent / "validation_per_sample.csv")
                 if sample["policy"] == policy_metrics["policy"]],
                ddof=0,
            ))
            rows.append(row)
    return rows


def _read_sample_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summaries(output_root: Path, configurations, seeds) -> None:
    rows = _summary_rows(output_root)
    if rows:
        atomic_write_csv(
            output_root / "P4_PER_SEED_RESULTS.csv",
            rows,
            list(SUMMARY_FIELDS),
        )
    statuses = []
    for configuration in configurations:
        for seed in seeds:
            status_path = output_root / configuration / str(seed) / "status.json"
            state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            statuses.append({
                "configuration": configuration,
                "seed": seed,
                "state": state.get("state", "not_started"),
                "completed_epochs": state.get("completed_epochs", 0),
                "selected_epoch": state.get("selected_epoch"),
                "training_seconds": state.get("training_seconds"),
            })
    atomic_write_json(output_root / "P4_RUN_MATRIX.json", {
        "runner_version": "P4_screen_v1",
        "status_counts": {
            status: sum(row["state"] == status for row in statuses)
            for status in ("completed", "incomplete", "interrupted", "failed", "not_started")
        },
        "runs": statuses,
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", nargs="+", default=P4_CONFIGURATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wall-clock-budget-seconds", type=float, default=3600.0)
    parser.add_argument("--checkpoint-every-batches", type=int, default=5)
    parser.add_argument("--heartbeat-batches", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--cache-root", type=Path, default=ROLE_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.max_epochs > 30:
        raise ValueError("Phase 4 maximum epochs must be in [1, 30].")
    if args.patience < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("Patience, batch size, and CPU thread count must be positive.")
    if args.checkpoint_every_batches < 1 or args.heartbeat_batches < 1:
        raise ValueError("Checkpoint and heartbeat batch intervals must be positive.")
    unknown = set(args.configurations) - set(P4_CONFIGURATIONS)
    if unknown:
        raise ValueError(f"Unknown Phase 4 configurations: {sorted(unknown)}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed list contains duplicates.")
    if not args.configurations or not args.seeds:
        raise ValueError("At least one configuration and seed are required.")

    torch.set_num_threads(args.cpu_threads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets, stats, cache_metadata = load_train_validation(args.cache_root)
    if set(datasets) != {"train", "validation"}:
        raise RuntimeError("Phase 4 loader must expose only train and validation.")
    _ensure_cv_baseline(args.output_root, datasets, stats, cache_metadata, args)
    stopped_incomplete = False
    for configuration in args.configurations:
        for seed in args.seeds:
            status_path = args.output_root / configuration / str(seed) / "status.json"
            if args.resume and status_path.is_file():
                if json.loads(status_path.read_text(encoding="utf-8")).get("state") == "completed":
                    continue
            state = _run_one(
                configuration, seed, datasets, stats, cache_metadata, args
            )
            _write_summaries(args.output_root, P4_CONFIGURATIONS, args.seeds)
            if state != "completed":
                stopped_incomplete = True
                break
        if stopped_incomplete:
            break
    _write_summaries(args.output_root, P4_CONFIGURATIONS, args.seeds)
    if stopped_incomplete:
        print("Batch stopped at an incomplete run; use --resume after inspecting its status.", flush=True)


if __name__ == "__main__":
    main()
