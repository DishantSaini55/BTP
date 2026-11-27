"""Resumable, validation-only Phase 2 screening on the frozen GP6 split."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

try:
    from .context_TF import ContextIndividualTF
    from .run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_bytes,
        atomic_write_csv,
        atomic_write_json,
        load_split,
    )
    from .run_from_scratch_e1 import RESULTS, SPLIT
    from .sage_v2 import (
        DifferentiableSAGEV2,
        build_satie_model,
        unique_trainable_parameter_count,
    )
except ImportError:
    from context_TF import ContextIndividualTF
    from run_e8_v2 import (
        _process_memory_bytes,
        atomic_torch_save,
        atomic_write_bytes,
        atomic_write_csv,
        atomic_write_json,
        load_split,
    )
    from run_from_scratch_e1 import RESULTS, SPLIT
    from sage_v2 import (
        DifferentiableSAGEV2,
        build_satie_model,
        unique_trainable_parameter_count,
    )


CONFIGURATIONS = ("p2_a", "p2_b", "p2_c", "p2_d", "p2_e")
POLICIES = {
    "p2_a": ("hard", "soft", "forced_open", "forced_closed"),
    "p2_b": ("always_open",),
    "p2_c": ("soft", "hard", "forced_open", "forced_closed"),
    "p2_d": ("soft", "hard", "forced_open", "forced_closed"),
    "p2_e": ("soft", "hard", "forced_open", "forced_closed"),
}
DEFAULT_OUTPUT_ROOT = RESULTS / "P2_initial_screen_v1"
CSV_FIELDS = (
    "configuration",
    "seed",
    "status",
    "training_scope",
    "policy",
    "epochs_completed",
    "train_normalized_mse",
    "validation_normalized_mse",
    "validation_raw_pixel_mse",
    "validation_cmse",
    "validation_cfmse",
    "gate_gradient_l2_mean",
    "context_gradient_l2_mean",
    "mean_admission_probability",
    "admission_probability_sd",
    "admission_probability_q05",
    "admission_probability_q50",
    "admission_probability_q95",
    "hard_admission_rate",
    "gate_entropy",
    "gate_logit_mean",
    "gate_logit_sd",
    "context_attention_weight",
    "context_benefit_pearson_r",
    "admitted_context_benefit_mean",
    "rejected_context_benefit_mean",
    "context_benefit_sample_count",
    "unique_trainable_parameters",
    "training_seconds",
    "peak_sampled_rss_bytes",
    "checkpoint_bytes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_seed(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(cpu_threads)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_l2(module: torch.nn.Module) -> tuple[float | None, int, int]:
    squared_norm = 0.0
    present = 0
    zero = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            continue
        present += 1
        norm_squared = float(parameter.grad.detach().float().square().sum())
        zero += int(norm_squared == 0.0)
        squared_norm += norm_squared
    return (float(np.sqrt(squared_norm)) if present else None, present, zero)


def _copy_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "data_order_generator": generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["data_order_generator"])


def _checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    state: dict[str, Any],
) -> None:
    payload = dict(state)
    payload.update({
        "model_state": _copy_state_to_cpu(model),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(generator),
    })
    atomic_torch_save(path, payload)


def _run_status(run: Path, state: str, **fields: Any) -> None:
    if state not in {"running", "completed", "interrupted", "incomplete", "failed"}:
        raise ValueError(f"Unknown P2 run state: {state}")
    atomic_write_json(run / "status.json", {
        "state": state,
        "updated_at": _utc_now(),
        **fields,
    })


def _append_progress(run: Path, message: str) -> None:
    print(message, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _history_write(run: Path, history: list[dict[str, Any]]) -> None:
    atomic_write_json(run / "history.json", history)
    fields = [
        "epoch", "train_normalized_mse", "validation_normalized_mse",
        "epoch_seconds", "training_batches", "gate_gradient_l2_mean",
        "gate_gradient_present_parameters", "gate_gradient_zero_parameters",
        "context_gradient_l2_mean", "context_gradient_present_parameters",
        "context_gradient_zero_parameters", "mean_admission_probability",
        "hard_admission_rate", "process_rss_bytes",
    ]
    atomic_write_csv(run / "history.csv", history, fields)


def _batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    return default_collate([dataset[index] for index in indices])


def _attention_weights(model: torch.nn.Module, batch_size: int) -> torch.Tensor:
    if isinstance(model, DifferentiableSAGEV2) and model.last_context_attention is not None:
        return model.last_context_attention.detach().cpu()
    base = model if isinstance(model, torch.nn.Module) and not hasattr(model, "base") else model.base
    layer_values = []
    for layer in base.model.decoder.layers:
        attention = layer.src_attn.attn
        if attention is not None and attention.shape[0] == batch_size and attention.shape[-1] == 16:
            layer_values.append(attention[..., -1].mean(dim=(1, 2)).detach().cpu())
    if not layer_values:
        return torch.zeros(batch_size)
    return torch.stack(layer_values).mean(dim=0)


def _policy_context(model: torch.nn.Module, policy: str):
    class PolicyScope:
        def __enter__(self):
            self.original_gate_mode = None
            self.original_inference_mode = None
            self.original_forced_admission = None
            self.original_gate_function = None
            if isinstance(model, ContextIndividualTF):
                self.original_gate_mode = model.gate_mode
                if policy in {"hard", "soft"}:
                    model.gate_mode = policy
                elif policy in {"forced_open", "forced_closed"}:
                    value = 1.0 if policy == "forced_open" else 0.0

                    def forced_gate(context_token, _observed_boxes):
                        gate = context_token.new_full(
                            (context_token.shape[0],), value
                        )
                        logits = context_token.new_zeros(
                            context_token.shape[0], 2
                        )
                        model.last_context_gate = gate.detach()
                        model.last_context_gate_logits = logits
                        model._context_gate_for_loss = gate
                        return gate

                    self.original_gate_function = model._context_gate
                    model._context_gate = forced_gate
            elif isinstance(model, DifferentiableSAGEV2):
                self.original_inference_mode = model.inference_mode
                self.original_forced_admission = model.forced_admission
                if policy in {"hard", "soft"}:
                    model.inference_mode = policy
                    model.forced_admission = None
                elif policy in {"forced_open", "forced_closed"}:
                    model.inference_mode = "hard"
                    model.forced_admission = (
                        1.0 if policy == "forced_open" else 0.0
                    )
            elif policy != "always_open":
                raise ValueError(
                    f"Policy {policy} is invalid for {type(model).__name__}."
                )
            return self

        def __exit__(self, _exc_type, _exc_value, _tb):
            if self.original_gate_mode is not None:
                model.gate_mode = self.original_gate_mode
            if self.original_gate_function is not None:
                model._context_gate = self.original_gate_function
            if self.original_inference_mode is not None:
                model.inference_mode = self.original_inference_mode
                model.forced_admission = self.original_forced_admission
            return False

    return PolicyScope()


def _evaluate(
    model: torch.nn.Module,
    dataset: Any,
    stats: tuple[np.ndarray, ...],
    policy: str,
    batch_size: int,
) -> tuple[dict[str, Any], float]:
    box_mean, box_std, _, _ = stats
    model.eval()
    normalized_loss_sum = 0.0
    sample_count = 0
    sample_mse: list[float] = []
    sample_cmse: list[float] = []
    sample_cfmse: list[float] = []
    gate_probabilities: list[np.ndarray] = []
    gate_decisions: list[np.ndarray] = []
    gate_entropies: list[np.ndarray] = []
    gate_logits: list[np.ndarray] = []
    context_attention: list[np.ndarray] = []
    normalized_predictions: list[np.ndarray] = []
    normalized_targets: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    with _policy_context(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = _batch(dataset, list(range(start, min(start + batch_size, len(dataset)))))
            prediction = model.forward_batch(batch)
            target = batch["pred_target"]
            n = int(target.shape[0])
            normalized_loss_sum += float(torch.nn.functional.mse_loss(
                prediction, target, reduction="mean"
            )) * n
            sample_count += n
            normalized_predictions.append(prediction.detach().cpu().numpy())
            normalized_targets.append(target.detach().cpu().numpy())
            raw_prediction = prediction.detach().cpu().numpy() * box_std + box_mean
            raw_target = target.detach().cpu().numpy() * box_std + box_mean
            raw_predictions.append(raw_prediction)
            raw_targets.append(raw_target)
            sample_ids.extend(batch["sample_id"])
            for index in range(n):
                prediction_center = (
                    raw_prediction[index, :, :2] + raw_prediction[index, :, 2:]
                ) / 2.0
                target_center = (
                    raw_target[index, :, :2] + raw_target[index, :, 2:]
                ) / 2.0
                sample_mse.append(float(np.mean(
                    (raw_prediction[index] - raw_target[index]) ** 2
                )))
                sample_cmse.append(float(np.mean(
                    (prediction_center - target_center) ** 2
                )))
                sample_cfmse.append(float(np.mean(
                    (prediction_center[-1] - target_center[-1]) ** 2
                )))
            if isinstance(model, ContextIndividualTF):
                gate = model.last_context_gate
                logits = model.last_context_gate_logits
                gate_logits.append(logits.detach().cpu().numpy())
                probability = (
                    logits.softmax(dim=-1)[:, 1]
                    if policy in {"hard", "soft"}
                    else gate
                )
                decision = gate >= 0.5
            elif isinstance(model, DifferentiableSAGEV2):
                probability = model.last_admission_probability
                gate_logits.append(model.last_gate_logits.detach().cpu().numpy())
                decision = model.last_gate_decision
                if decision is None:
                    decision = probability >= 0.5
            else:
                probability = torch.ones(n)
                decision = torch.ones(n, dtype=torch.bool)
                gate_logits.append(np.empty((n, 0), dtype=np.float32))
            gate_probabilities.append(probability.detach().cpu().numpy())
            gate_decisions.append(decision.detach().cpu().numpy())
            probability_array = probability.detach().cpu().numpy().astype(np.float64)
            clipped = np.clip(probability_array, 1e-12, 1.0 - 1e-12)
            entropy = -(
                clipped * np.log(clipped)
                + (1.0 - clipped) * np.log(1.0 - clipped)
            )
            entropy[(probability_array <= 0.0) | (probability_array >= 1.0)] = 0.0
            gate_entropies.append(entropy)
            context_attention.append(_attention_weights(model, n).numpy())

    if sample_count != len(dataset):
        raise RuntimeError(
            f"Validation evaluated {sample_count} of {len(dataset)} windows."
        )
    probabilities = np.concatenate(gate_probabilities)
    decisions = np.concatenate(gate_decisions).astype(bool)
    entropy_values = np.concatenate(gate_entropies)
    attention_values = np.concatenate(context_attention)
    metrics = {
        "policy": policy,
        "validation_normalized_mse": normalized_loss_sum / sample_count,
        "validation_raw_pixel_mse": float(np.mean(sample_mse)),
        "validation_cmse": float(np.mean(sample_cmse)),
        "validation_cfmse": float(np.mean(sample_cfmse)),
        "mean_admission_probability": float(np.mean(probabilities)),
        "admission_probability_sd": float(np.std(probabilities, ddof=0)),
        "admission_probability_q05": float(np.quantile(probabilities, 0.05)),
        "admission_probability_q50": float(np.quantile(probabilities, 0.50)),
        "admission_probability_q95": float(np.quantile(probabilities, 0.95)),
        "hard_admission_rate": float(np.mean(decisions)),
        "gate_entropy": float(np.mean(entropy_values)),
        "gate_logit_mean": (
            float(np.mean(np.concatenate(gate_logits)))
            if gate_logits and gate_logits[0].shape[-1]
            else None
        ),
        "gate_logit_sd": (
            float(np.std(np.concatenate(gate_logits), ddof=0))
            if gate_logits and gate_logits[0].shape[-1]
            else None
        ),
        "context_attention_weight": float(np.mean(attention_values)),
        "context_attention_weight_sd": float(np.std(attention_values, ddof=0)),
    }
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype=str),
        "predictions_normalized": np.concatenate(normalized_predictions),
        "targets_normalized": np.concatenate(normalized_targets),
        "predictions_raw": np.concatenate(raw_predictions),
        "targets_raw": np.concatenate(raw_targets),
        "gate_probability": probabilities,
        "gate_logits": np.concatenate(gate_logits),
        "gate_decision": decisions,
        "context_attention": attention_values,
    }
    return metrics, normalized_loss_sum / sample_count, arrays


def _evaluate_train_loss(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    policy: str,
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    with _policy_context(model, policy), torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = _batch(dataset, list(range(start, min(start + batch_size, len(dataset)))))
            value = torch.nn.functional.mse_loss(
                model.forward_batch(batch), batch["pred_target"], reduction="mean"
            )
            n = int(batch["pred_target"].shape[0])
            loss_sum += float(value) * n
            count += n
    if not count:
        raise RuntimeError("Training split has no windows.")
    return loss_sum / count


def _load_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _run_config(
    configuration: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    flow_path = SPLIT / "flow_splits.pkl"
    return {
        "runner_version": "P2_screen_v1",
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
        "training_scope": (
            "full_train_split"
            if args.max_train_batches is None
            else f"capped_at_{args.max_train_batches}_batches_per_epoch"
        ),
        "validation_only": True,
        "test_split_loaded": False,
        "train_samples": 212,
        "validation_samples": 43,
        "split_cache_sha256": _sha256_file(flow_path),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "source_sha256": {
            "run_p2_screen.py": _sha256_file(Path(__file__)),
            "sage_v2.py": _sha256_file(Path(__file__).with_name("sage_v2.py")),
            "context_TF.py": _sha256_file(Path(__file__).with_name("context_TF.py")),
            "run_from_scratch_e1.py": _sha256_file(
                Path(__file__).with_name("run_from_scratch_e1.py")
            ),
        },
    }


def _compatible_config(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    mutable = {"max_epochs", "wall_clock_budget_seconds"}
    return all(
        previous.get(key) == value
        for key, value in current.items()
        if key not in mutable
    )


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
            f"Run path already exists; pass --resume only for a compatible incomplete run: {run}"
        )
    config_path = run / "run_config.json"
    checkpoint_path = run / "latest_checkpoint.pt"
    status_path = run / "status.json"
    if not config_path.exists() or not checkpoint_path.exists() or not status_path.exists():
        raise RuntimeError(
            f"Existing run lacks config, status, or a resumable checkpoint: {run}"
        )
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if not _compatible_config(previous, config):
        raise ValueError(f"Saved P2 run configuration is incompatible: {run}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") == "completed":
        raise RuntimeError(f"Run is already completed and will not be overwritten: {run}")
    if status.get("state") == "failed":
        raise RuntimeError(
            f"Run is marked failed; inspect {run / 'status.json'} before resuming."
        )
    atomic_write_json(config_path, config)
    return _load_state(checkpoint_path)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        old_rows: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            old_rows.extend(csv.DictReader(handle))
        existing = {
            (row["configuration"], int(row["seed"]), row["policy"])
            for row in old_rows
        }
        for row in rows:
            key = (row["configuration"], int(row["seed"]), row["policy"])
            if key in existing:
                raise FileExistsError(
                    f"Summary already contains {key}; refusing to replace prior results."
                )
        all_rows = old_rows + rows
    else:
        all_rows = rows
    atomic_write_csv(path, all_rows, list(CSV_FIELDS))


def run_one(
    configuration: str,
    seed: int,
    datasets: dict[str, Any],
    stats: tuple[np.ndarray, ...],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    run = args.output_root / configuration / str(seed)
    config = _run_config(configuration, seed, args)
    checkpoint_path = run / "latest_checkpoint.pt"
    _set_seed(seed, args.cpu_threads)
    model = build_satie_model(configuration)
    initial_digest = _state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    resume_state = _prepare_run(run, config, args.resume)
    if resume_state is None:
        _run_status(
            run,
            "running",
            configuration=configuration,
            seed=seed,
            completed_epochs=0,
            maximum_epochs=args.max_epochs,
        )
        state: dict[str, Any] = {
            "completed_epochs": 0,
            "epoch_order": None,
            "next_batch_index": 0,
            "epoch_train_loss_sum": 0.0,
            "epoch_train_sample_count": 0,
            "epoch_gate_gradient_sum": 0.0,
            "epoch_gate_gradient_batches": 0,
            "epoch_gate_gradient_present_parameters": 0,
            "epoch_gate_gradient_zero_parameters": 0,
            "epoch_context_gradient_sum": 0.0,
            "epoch_context_gradient_batches": 0,
            "epoch_context_gradient_present_parameters": 0,
            "epoch_context_gradient_zero_parameters": 0,
            "epoch_admission_values": [],
            "epoch_elapsed_seconds": 0.0,
            "history": [],
            "best_validation_loss": float("inf"),
            "best_epoch": 0,
            "stale_epochs": 0,
            "best_model_state": None,
            "total_training_seconds": 0.0,
            "peak_sampled_rss_bytes": 0,
        }
        _checkpoint(checkpoint_path, model, optimizer, generator, state)
    else:
        state = resume_state
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _restore_rng_state(state["rng_state"], generator)
        _append_progress(
            run,
            f"RESUME state={json.loads((run / 'status.json').read_text())['state']} "
            f"epoch={state['completed_epochs'] + 1} "
            f"next_batch={state['next_batch_index']}",
        )
    if state.get("initial_state_sha256", initial_digest) != initial_digest:
        raise RuntimeError(
            "Recreated initialization differs from the saved initial model; refusing resume."
        )
    state["initial_state_sha256"] = initial_digest
    if not state["history"]:
        _append_progress(
            run,
            f"INITIALIZED configuration={configuration} seed={seed} "
            f"parameters={unique_trainable_parameter_count(model)} "
            f"initial_state_sha256={initial_digest}",
        )
    else:
        _append_progress(
            run,
            f"CONTINUING configuration={configuration} seed={seed} "
            f"epochs_completed={state['completed_epochs']}",
        )

    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]
    full_batches_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    batches_per_epoch = (
        min(full_batches_per_epoch, args.max_train_batches)
        if args.max_train_batches is not None
        else full_batches_per_epoch
    )
    invocation_started = time.perf_counter()
    budget = args.wall_clock_budget_seconds
    _run_status(
        run,
        "running",
        configuration=configuration,
        seed=seed,
        completed_epochs=state["completed_epochs"],
        maximum_epochs=args.max_epochs,
        initial_state_sha256=initial_digest,
    )
    if float(state["total_training_seconds"]) >= budget:
        _run_status(
            run,
            "incomplete",
            reason="cumulative_wall_clock_budget_exhausted",
            completed_epochs=state["completed_epochs"],
            elapsed_seconds=state["total_training_seconds"],
        )
        return []
    try:
        while state["completed_epochs"] < args.max_epochs:
            epoch_number = state["completed_epochs"] + 1
            if state["epoch_order"] is None:
                state["epoch_order"] = torch.randperm(
                    len(train_dataset), generator=generator
                ).tolist()
                state["next_batch_index"] = 0
                state["epoch_train_loss_sum"] = 0.0
                state["epoch_train_sample_count"] = 0
                state["epoch_gate_gradient_sum"] = 0.0
                state["epoch_gate_gradient_batches"] = 0
                state["epoch_gate_gradient_present_parameters"] = 0
                state["epoch_gate_gradient_zero_parameters"] = 0
                state["epoch_context_gradient_sum"] = 0.0
                state["epoch_context_gradient_batches"] = 0
                state["epoch_context_gradient_present_parameters"] = 0
                state["epoch_context_gradient_zero_parameters"] = 0
                state["epoch_admission_values"] = []
                state["epoch_elapsed_seconds"] = 0.0
            epoch_started = time.perf_counter()
            prior_epoch_seconds = float(state["epoch_elapsed_seconds"])
            model.train()
            while state["next_batch_index"] < batches_per_epoch:
                batch_index = state["next_batch_index"]
                batch_indices = state["epoch_order"][
                    batch_index * args.batch_size:(batch_index + 1) * args.batch_size
                ]
                batch = _batch(train_dataset, batch_indices)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_batch(batch)
                loss = torch.nn.functional.mse_loss(
                    prediction, batch["pred_target"], reduction="mean"
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch_number}, batch {batch_index + 1}."
                    )
                loss.backward()
                if hasattr(model, "gate"):
                    gate_norm, gate_present, gate_zero = _gradient_l2(model.gate)
                    if gate_norm is not None:
                        state["epoch_gate_gradient_sum"] += gate_norm
                        state["epoch_gate_gradient_batches"] += 1
                    state["epoch_gate_gradient_present_parameters"] += gate_present
                    state["epoch_gate_gradient_zero_parameters"] += gate_zero
                context_norm, context_present, context_zero = _gradient_l2(
                    model.context_encoder
                )
                if context_norm is not None:
                    state["epoch_context_gradient_sum"] += context_norm
                    state["epoch_context_gradient_batches"] += 1
                state["epoch_context_gradient_present_parameters"] += context_present
                state["epoch_context_gradient_zero_parameters"] += context_zero
                optimizer.step()

                batch_n = int(batch["pred_target"].shape[0])
                state["epoch_train_loss_sum"] += float(loss.detach()) * batch_n
                state["epoch_train_sample_count"] += batch_n
                if isinstance(model, ContextIndividualTF):
                    state["epoch_admission_values"].extend(
                        model.last_context_gate.detach().cpu().tolist()
                    )
                elif isinstance(model, DifferentiableSAGEV2):
                    state["epoch_admission_values"].extend(
                        model.last_admission_probability.detach().cpu().tolist()
                    )
                state["next_batch_index"] += 1
                state["epoch_elapsed_seconds"] = (
                    prior_epoch_seconds + time.perf_counter() - epoch_started
                )
                rss = _process_memory_bytes() or 0
                state["peak_sampled_rss_bytes"] = max(
                    int(state["peak_sampled_rss_bytes"]), rss
                )
                if (
                    state["next_batch_index"] % args.checkpoint_every_batches == 0
                    or state["next_batch_index"] == batches_per_epoch
                ):
                    _checkpoint(checkpoint_path, model, optimizer, generator, state)
                elapsed = (
                    float(state["total_training_seconds"])
                    + time.perf_counter() - invocation_started
                )
                if time.perf_counter() - invocation_started >= budget:
                    state["total_training_seconds"] = elapsed
                    _checkpoint(checkpoint_path, model, optimizer, generator, state)
                    _run_status(
                        run,
                        "incomplete",
                        reason="wall_clock_budget_exhausted",
                        completed_epochs=state["completed_epochs"],
                        epoch_in_progress=epoch_number,
                        next_batch_index=state["next_batch_index"],
                        elapsed_seconds=elapsed,
                    )
                    _append_progress(
                        run,
                        f"INCOMPLETE reason=wall_clock_budget_exhausted "
                        f"epoch={epoch_number} next_batch={state['next_batch_index']} "
                        f"elapsed_seconds={elapsed:.2f}",
                    )
                    return []
                if state["next_batch_index"] % 5 == 0:
                    partial_train_loss = (
                        state["epoch_train_loss_sum"]
                        / state["epoch_train_sample_count"]
                    )
                    _append_progress(
                        run,
                        f"HEARTBEAT state=running epoch={epoch_number} "
                        f"batch={state['next_batch_index']}/{batches_per_epoch} "
                        f"partial_train_mse={partial_train_loss:.8f} "
                        f"rss_bytes={rss}",
                    )

            validation_policy = {
                "p2_a": "hard",
                "p2_b": "always_open",
            }.get(configuration, "soft")
            validation_loss = _evaluate(
                model,
                validation_dataset,
                stats,
                validation_policy,
                args.batch_size,
            )[1]
            epoch_seconds = prior_epoch_seconds + time.perf_counter() - epoch_started
            admission = np.asarray(state["epoch_admission_values"], dtype=np.float64)
            history_row = {
                "epoch": epoch_number,
                "train_normalized_mse": (
                    state["epoch_train_loss_sum"] / state["epoch_train_sample_count"]
                ),
                "validation_normalized_mse": validation_loss,
                "epoch_seconds": epoch_seconds,
                "training_batches": batches_per_epoch,
                "gate_gradient_l2_mean": (
                    state["epoch_gate_gradient_sum"]
                    / state["epoch_gate_gradient_batches"]
                    if state["epoch_gate_gradient_batches"] else None
                ),
                "gate_gradient_present_parameters":
                    state["epoch_gate_gradient_present_parameters"],
                "gate_gradient_zero_parameters":
                    state["epoch_gate_gradient_zero_parameters"],
                "context_gradient_l2_mean": (
                    state["epoch_context_gradient_sum"]
                    / state["epoch_context_gradient_batches"]
                    if state["epoch_context_gradient_batches"] else None
                ),
                "context_gradient_present_parameters":
                    state["epoch_context_gradient_present_parameters"],
                "context_gradient_zero_parameters":
                    state["epoch_context_gradient_zero_parameters"],
                "mean_admission_probability": (
                    float(admission.mean()) if admission.size else None
                ),
                "hard_admission_rate": (
                    float(np.mean(admission >= 0.5)) if admission.size else None
                ),
                "process_rss_bytes": _process_memory_bytes(),
            }
            state["history"].append(history_row)
            state["completed_epochs"] = epoch_number
            state["epoch_order"] = None
            state["next_batch_index"] = 0
            state["epoch_elapsed_seconds"] = 0.0
            state["total_training_seconds"] += time.perf_counter() - invocation_started
            invocation_started = time.perf_counter()
            if validation_loss < state["best_validation_loss"]:
                state["best_validation_loss"] = validation_loss
                state["best_epoch"] = epoch_number
                state["stale_epochs"] = 0
                state["best_model_state"] = _copy_state_to_cpu(model)
            else:
                state["stale_epochs"] += 1
            _history_write(run, state["history"])
            _checkpoint(checkpoint_path, model, optimizer, generator, state)
            _append_progress(
                run,
                f"EPOCH configuration={configuration} seed={seed} epoch={epoch_number} "
                f"train_mse={history_row['train_normalized_mse']:.8f} "
                f"validation_mse={validation_loss:.8f} "
                f"epoch_seconds={epoch_seconds:.2f} "
                f"gate_grad_l2={history_row['gate_gradient_l2_mean']} "
                f"context_grad_l2={history_row['context_gradient_l2_mean']} "
                f"admission_mean={history_row['mean_admission_probability']}",
            )
            _run_status(
                run,
                "running",
                completed_epochs=state["completed_epochs"],
                maximum_epochs=args.max_epochs,
                best_epoch=state["best_epoch"],
                stale_epochs=state["stale_epochs"],
            )
            if state["stale_epochs"] >= args.patience:
                break

        if state["best_model_state"] is None:
            raise RuntimeError("No validation checkpoint was selected.")
        model.load_state_dict(state["best_model_state"])
        selected_policy = {
            "p2_a": "hard",
            "p2_b": "always_open",
        }.get(configuration, "soft")
        train_loss = _evaluate_train_loss(
            model, train_dataset, args.batch_size, selected_policy
        )
        validation_results = []
        validation_arrays: dict[str, dict[str, np.ndarray]] = {}
        for policy in POLICIES[configuration]:
            result, _, arrays = _evaluate(
                model, validation_dataset, stats, policy, args.batch_size
            )
            validation_results.append(result)
            validation_arrays[policy] = arrays

        context_benefit = None
        benefit_correlation = None
        admitted_benefit_mean = None
        rejected_benefit_mean = None
        if {"forced_open", "forced_closed"}.issubset(validation_arrays):
            open_arrays = validation_arrays["forced_open"]
            closed_arrays = validation_arrays["forced_closed"]
            open_errors = np.mean(
                (open_arrays["predictions_raw"] - open_arrays["targets_raw"]) ** 2,
                axis=(1, 2),
            )
            closed_errors = np.mean(
                (closed_arrays["predictions_raw"] - closed_arrays["targets_raw"]) ** 2,
                axis=(1, 2),
            )
            context_benefit = closed_errors - open_errors
            learned_policy = (
                "soft" if "soft" in validation_arrays
                else "hard" if "hard" in validation_arrays
                else None
            )
            if learned_policy is not None:
                learned_arrays = validation_arrays[learned_policy]
                learned_probability = learned_arrays["gate_probability"]
                if (
                    np.std(learned_probability) > 0.0
                    and np.std(context_benefit) > 0.0
                ):
                    benefit_correlation = float(np.corrcoef(
                        learned_probability, context_benefit
                    )[0, 1])
                admitted_mask = learned_arrays["gate_decision"].astype(bool)
                admitted_benefit_mean = (
                    float(np.mean(context_benefit[admitted_mask]))
                    if np.any(admitted_mask) else None
                )
                rejected_benefit_mean = (
                    float(np.mean(context_benefit[~admitted_mask]))
                    if np.any(~admitted_mask) else None
                )

        elapsed_total = float(state["total_training_seconds"])
        checkpoint_bytes = checkpoint_path.stat().st_size
        row_output: list[dict[str, Any]] = []
        for result in validation_results:
            row = {
                "configuration": configuration,
                "seed": seed,
                "status": "completed",
                "training_scope": config["training_scope"],
                "policy": result["policy"],
                "epochs_completed": state["completed_epochs"],
                "train_normalized_mse": train_loss,
                **result,
                "gate_gradient_l2_mean": (
                    float(np.mean([
                        item["gate_gradient_l2_mean"]
                        for item in state["history"]
                        if item["gate_gradient_l2_mean"] is not None
                    ]))
                    if any(item["gate_gradient_l2_mean"] is not None for item in state["history"])
                    else None
                ),
                "context_gradient_l2_mean": float(np.mean([
                    item["context_gradient_l2_mean"]
                    for item in state["history"]
                    if item["context_gradient_l2_mean"] is not None
                ])),
                "context_benefit_pearson_r": benefit_correlation,
                "admitted_context_benefit_mean": admitted_benefit_mean,
                "rejected_context_benefit_mean": rejected_benefit_mean,
                "context_benefit_sample_count": (
                    int(context_benefit.size) if context_benefit is not None else 0
                ),
                "unique_trainable_parameters": unique_trainable_parameter_count(model),
                "training_seconds": elapsed_total,
                "peak_sampled_rss_bytes": state["peak_sampled_rss_bytes"],
                "checkpoint_bytes": checkpoint_bytes,
            }
            row_output.append({key: row.get(key) for key in CSV_FIELDS})
        summary_path = (
            RESULTS / "P2_VALIDATION_SUMMARY.csv"
            if args.output_root == DEFAULT_OUTPUT_ROOT
            else args.output_root / "P2_VALIDATION_SUMMARY.csv"
        )
        archive: dict[str, np.ndarray] = {}
        per_sample: dict[str, dict[str, Any]] = {}
        for policy, arrays in validation_arrays.items():
            per_sample[policy] = {
                "mse_raw": np.mean(
                    (arrays["predictions_raw"] - arrays["targets_raw"]) ** 2,
                    axis=(1, 2),
                ),
                "gate_probability": arrays["gate_probability"],
                "gate_decision": arrays["gate_decision"],
                "context_attention": arrays["context_attention"],
            }
            for name, value in arrays.items():
                archive[f"{policy}__{name}"] = value
        if context_benefit is not None:
            archive["forced_context_benefit_closed_minus_open_mse"] = context_benefit
        archive_buffer = io.BytesIO()
        np.savez_compressed(archive_buffer, **archive)
        atomic_write_bytes(
            run / "validation_predictions.npz", archive_buffer.getvalue()
        )
        sample_ids = next(iter(validation_arrays.values()))["sample_id"]
        sample_rows = []
        for index, sample_id in enumerate(sample_ids):
            row = {"sample_id": sample_id}
            for policy, values in per_sample.items():
                row[f"{policy}_raw_mse"] = float(values["mse_raw"][index])
                row[f"{policy}_admission_probability"] = float(
                    values["gate_probability"][index]
                )
                row[f"{policy}_hard_admission"] = bool(
                    values["gate_decision"][index]
                )
                row[f"{policy}_context_attention"] = float(
                    values["context_attention"][index]
                )
            if context_benefit is not None:
                row["forced_context_benefit_closed_minus_open_mse"] = float(
                    context_benefit[index]
                )
            sample_rows.append(row)
        if sample_rows:
            atomic_write_csv(
                run / "validation_per_sample.csv",
                sample_rows,
                list(sample_rows[0]),
            )
        atomic_write_json(run / "validation_metrics.json", {
            "status": "completed",
            "configuration": configuration,
            "seed": seed,
            "selected_epoch": state["best_epoch"],
            "epochs_completed": state["completed_epochs"],
            "selected_train_normalized_mse": train_loss,
            "selected_policy": selected_policy,
            "initial_state_sha256": initial_digest,
            "unique_trainable_parameters": unique_trainable_parameter_count(model),
            "training_seconds": elapsed_total,
            "peak_sampled_rss_bytes": state["peak_sampled_rss_bytes"],
            "checkpoint_bytes": checkpoint_bytes,
            "training_scope": config["training_scope"],
            "forced_context_benefit": {
                "definition": "forced-closed per-window raw MSE minus forced-open per-window raw MSE",
                "mean": (
                    float(np.mean(context_benefit))
                    if context_benefit is not None else None
                ),
                "pearson_r_with_learned_admission_probability": benefit_correlation,
                "mean_when_learned_gate_admits": admitted_benefit_mean,
                "mean_when_learned_gate_rejects": rejected_benefit_mean,
                "sample_count": (
                    int(context_benefit.size) if context_benefit is not None else 0
                ),
                "note": "Descriptive 43-window validation analysis; not used for model selection beyond validation early stopping.",
            },
            "validation_policies": validation_results,
        })
        _write_summary(summary_path, row_output)
        _run_status(
            run,
            "completed",
            completed_epochs=state["completed_epochs"],
            selected_epoch=state["best_epoch"],
            validation_only=True,
        )
        _append_progress(
            run,
            f"COMPLETED configuration={configuration} seed={seed} "
            f"selected_epoch={state['best_epoch']} epochs={state['completed_epochs']} "
            f"elapsed_seconds={elapsed_total:.2f}",
        )
        return row_output
    except KeyboardInterrupt:
        state["total_training_seconds"] += time.perf_counter() - invocation_started
        _checkpoint(checkpoint_path, model, optimizer, generator, state)
        _run_status(
            run,
            "interrupted",
            completed_epochs=state["completed_epochs"],
            next_batch_index=state["next_batch_index"],
        )
        _append_progress(run, "INTERRUPTED checkpoint_saved=true")
        return []
    except Exception as error:
        _run_status(
            run,
            "failed",
            completed_epochs=state["completed_epochs"],
            error_type=type(error).__name__,
            error=str(error),
        )
        (run / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260928])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wall-clock-budget-seconds", type=float, default=3600.0)
    parser.add_argument("--checkpoint-every-batches", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("max epochs, patience, and batch size must be positive.")
    if args.cpu_threads < 1 or args.wall_clock_budget_seconds <= 0:
        parser.error("CPU threads and wall-clock budget must be positive.")
    if args.checkpoint_every_batches < 1:
        parser.error("checkpoint interval must be positive.")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("maximum training batches must be positive when supplied.")
    if not args.seeds:
        parser.error("at least one seed is required.")

    datasets, stats = load_split(validation_only=True)
    if set(datasets) != {"train", "validation"}:
        raise AssertionError("Validation-only load unexpectedly includes another split.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for configuration in args.configurations:
        for seed in args.seeds:
            rows = run_one(configuration, seed, datasets, stats, args)
            all_rows.extend(rows)
            if json.loads((args.output_root / configuration / str(seed) / "status.json").read_text())["state"] in {
                "incomplete", "interrupted", "failed"
            }:
                return
    if all_rows:
        print(json.dumps({
            "summary_csv": str(
                RESULTS / "P2_VALIDATION_SUMMARY.csv"
                if args.output_root == DEFAULT_OUTPUT_ROOT
                else args.output_root / "P2_VALIDATION_SUMMARY.csv"
            ),
            "completed_rows_this_invocation": len(all_rows),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
