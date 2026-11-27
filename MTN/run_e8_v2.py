"""Versioned, resumable CPU runner for the E8 SAGE gate diagnostics."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import pickle
import random
import signal
import tempfile
import time
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
    from .context_TF import ContextIndividualTF
    from .run_from_scratch_e1 import (
        RESULTS,
        SEEDS,
        SPLIT,
        WindowDataset,
        forward,
    )
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared
    from context_TF import ContextIndividualTF
    from run_from_scratch_e1 import (
        RESULTS,
        SEEDS,
        SPLIT,
        WindowDataset,
        forward,
    )


MODEL_SPECS = {
    "without_satie": {
        "name": "PACE-Net without SATIE (SAGE + MASC)",
        "slug": "pace_net_without_satie_sage_plus_masc",
        "satie": False,
    },
    "with_satie": {
        "name": "PACE-Net (SAGE + MASC + SATIE)",
        "slug": "pace_net_sage_plus_masc_plus_satie",
        "satie": True,
    },
}
VARIANTS = ("clipped_normalized_x_t", "context_only")
DEFAULT_OUTPUT_ROOT = RESULTS / "runs" / "E8_v2"
LEARNING_RATE = 1e-3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temp_name)
        with open(temp_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def set_seed(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(cpu_threads)


def build_model(model_key: str) -> ContextIndividualTF:
    spec = MODEL_SPECS[model_key]
    model = ContextIndividualTF(
        15, 4, 4, 4, N=3, d_model=32, d_ff=128, h=4, dropout=0.1
    )
    if spec["satie"]:
        make_c4_encoder_self_attention_shared(model)
    return model


def _record_gate(base: ContextIndividualTF, logits: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    base.last_context_gate = gate.detach()
    base.last_context_gate_logits = logits.detach()
    base._context_gate_for_loss = gate
    return gate


def configure_variant(model: ContextIndividualTF, variant: str) -> None:
    if variant == "context_only":
        model.gate = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 2),
        )

        def context_only_gate(context_token: torch.Tensor, observed_boxes: torch.Tensor) -> torch.Tensor:
            logits = model.gate(context_token)
            if model.training:
                gate = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
            else:
                gate = logits.argmax(dim=-1).eq(1).to(context_token.dtype)
            return _record_gate(model, logits, gate)

        model._context_gate = context_only_gate
        return

    if variant == "clipped_normalized_x_t":
        def clipped_gate(context_token: torch.Tensor, observed_boxes: torch.Tensor) -> torch.Tensor:
            x_t = observed_boxes[:, -1].clamp(-5.0, 5.0)
            logits = model.gate(torch.cat((context_token, x_t), dim=-1))
            if model.training:
                gate = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
            else:
                gate = logits.argmax(dim=-1).eq(1).to(context_token.dtype)
            return _record_gate(model, logits, gate)

        model._context_gate = clipped_gate
        return

    raise ValueError(f"Unknown E8 variant: {variant}")


def load_split(validation_only: bool) -> tuple[dict[str, WindowDataset], tuple[np.ndarray, ...]]:
    flow_path = SPLIT / "flow_splits.pkl"
    if not flow_path.exists():
        raise FileNotFoundError(f"Required cached flow split not found: {flow_path}")
    with flow_path.open("rb") as handle:
        split = pickle.load(handle)
    train_boxes = np.concatenate([sample["bbox"] for sample in split["train"]])
    train_speed = np.concatenate([sample["obd_speed"] for sample in split["train"]])
    stats = (
        train_boxes.mean(0),
        np.maximum(train_boxes.std(0), 1e-6),
        train_speed.mean(0),
        np.maximum(train_speed.std(0), 1e-6),
    )
    roles = ("train", "validation") if validation_only else ("train", "validation", "test")
    datasets = {role: WindowDataset(split[role], stats) for role in roles}
    return datasets, stats


def gradient_summary(module: torch.nn.Module) -> dict[str, Any]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    missing = sum(parameter.grad is None for parameter in parameters)
    zero = 0
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not bool(torch.isfinite(grad).all()):
            return {
                "l2_norm": None,
                "parameter_count": len(parameters),
                "missing_parameter_count": missing,
                "zero_parameter_count": zero,
                "nonzero_parameter_count": len(parameters) - missing - zero,
                "finite": False,
            }
        norm_sq = float(torch.sum(grad * grad).item())
        zero += int(norm_sq == 0.0)
        squared_norm += norm_sq
    present = len(parameters) - missing
    return {
        "l2_norm": float(np.sqrt(squared_norm)) if present else None,
        "parameter_count": len(parameters),
        "missing_parameter_count": missing,
        "zero_parameter_count": zero,
        "nonzero_parameter_count": present - zero,
        "finite": True,
    }


def _process_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
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

    try:
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
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    except (AttributeError, OSError):
        return None


def _write_progress(run: Path, line: str) -> None:
    print(line, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ProgressHeartbeat:
    def __init__(self, run: Path, interval_seconds: float):
        self.run = run
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = {"epoch": 0, "batches": 0, "started": time.perf_counter()}
        self.thread = threading.Thread(target=self._run, name="e8-progress-heartbeat", daemon=True)

    def start(self) -> None:
        if self.interval_seconds > 0:
            self.thread.start()

    def update(self, epoch: int, batches: int, started: float) -> None:
        with self.lock:
            self.state = {"epoch": epoch, "batches": batches, "started": started}

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            with self.lock:
                state = dict(self.state)
            _write_progress(
                self.run,
                f"HEARTBEAT state=running epoch={state['epoch']} "
                f"completed_batches={state['batches']} "
                f"epoch_elapsed_seconds={time.perf_counter() - state['started']:.2f} "
                f"process_cpu_seconds={time.process_time():.2f} "
                f"rss_bytes={_process_memory_bytes()}",
            )

    def close(self) -> None:
        self.stop_event.set()
        if self.interval_seconds > 0:
            self.thread.join(timeout=max(1.0, self.interval_seconds + 1.0))


def _status(run: Path, state: str, **fields: Any) -> None:
    if state not in {"running", "completed", "interrupted", "failed"}:
        raise ValueError(f"Invalid run state: {state}")
    atomic_write_json(
        run / "status.json",
        {"state": state, "updated_at": utc_now(), **fields},
    )


def _rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    loader_generator.set_state(state["loader_generator"])


def _module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    epoch: int,
    history: list[dict[str, Any]],
    best_validation_loss: float,
    best_epoch: int,
    stale_epochs: int,
    best_model_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    return {
        "model_state": _module_state(model),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(loader_generator),
        "epoch": epoch,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "best_model_state": best_model_state,
    }


def _history_write(run: Path, history: list[dict[str, Any]]) -> None:
    atomic_write_json(run / "history.json", history)
    fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "epoch_seconds",
        "train_batches",
        "validation_batches",
        "validation_gate_mean",
        "train_admission_rate",
        "train_gate_mean",
        "gate_grad_l2_mean",
        "gate_grad_l2_max",
        "gate_grad_missing_parameters",
        "gate_grad_zero_parameters",
        "context_grad_l2_mean",
        "context_grad_l2_max",
        "context_grad_missing_parameters",
        "context_grad_zero_parameters",
        "process_cpu_seconds",
        "process_rss_bytes",
    ]
    atomic_write_csv(run / "history.csv", history, fields)


def _batch_limit(loader: DataLoader, maximum: int | None):
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def _train_epoch(
    model: ContextIndividualTF,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epoch: int,
    run: Path,
    heartbeat_seconds: float,
    max_batches: int | None,
    progress_heartbeat: ProgressHeartbeat | None = None,
) -> dict[str, Any]:
    model.train()
    criterion = torch.nn.MSELoss()
    loss_total = 0.0
    sample_count = 0
    gate_values: list[np.ndarray] = []
    gate_grad_norms: list[float] = []
    context_grad_norms: list[float] = []
    gate_missing = gate_zero = context_missing = context_zero = 0
    batch_count = 0
    started = time.perf_counter()
    process_started = time.process_time()
    last_heartbeat = started
    for batch in _batch_limit(loader, max_batches):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, batch)
        loss = criterion(prediction, batch["pred_target"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_count + 1}")
        loss.backward()
        gate_grad = gradient_summary(model.gate)
        context_grad = gradient_summary(model.context_encoder)
        if not gate_grad["finite"] or not context_grad["finite"]:
            raise FloatingPointError(f"Non-finite SAGE gradients at epoch {epoch}, batch {batch_count + 1}")
        if gate_grad["l2_norm"] is not None:
            gate_grad_norms.append(gate_grad["l2_norm"])
        if context_grad["l2_norm"] is not None:
            context_grad_norms.append(context_grad["l2_norm"])
        gate_missing += gate_grad["missing_parameter_count"]
        gate_zero += gate_grad["zero_parameter_count"]
        context_missing += context_grad["missing_parameter_count"]
        context_zero += context_grad["zero_parameter_count"]
        if model.last_context_gate is not None:
            gate_values.append(model.last_context_gate.detach().cpu().numpy())
        optimizer.step()

        batch_size = int(batch["pred_target"].shape[0])
        loss_total += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        batch_count += 1
        if progress_heartbeat is not None:
            progress_heartbeat.update(epoch, batch_count, started)
        now = time.perf_counter()
        if heartbeat_seconds > 0 and now - last_heartbeat >= heartbeat_seconds:
            _write_progress(
                run,
                f"HEARTBEAT state=running epoch={epoch} batches={batch_count} "
                f"partial_train_loss={loss_total / sample_count:.8f} "
                f"elapsed_seconds={now - started:.2f}",
            )
            last_heartbeat = now
    if batch_count == 0:
        raise RuntimeError("Training loader produced no batches.")
    gate_array = np.concatenate(gate_values) if gate_values else np.array([], dtype=np.float32)
    return {
        "train_loss": loss_total / sample_count,
        "train_batches": batch_count,
        "train_admission_rate": float(np.mean(gate_array > 0.5)) if gate_array.size else None,
        "train_gate_mean": float(np.mean(gate_array)) if gate_array.size else None,
        "gate_grad_l2_mean": float(np.mean(gate_grad_norms)) if gate_grad_norms else None,
        "gate_grad_l2_max": float(np.max(gate_grad_norms)) if gate_grad_norms else None,
        "gate_grad_missing_parameters": gate_missing,
        "gate_grad_zero_parameters": gate_zero,
        "context_grad_l2_mean": float(np.mean(context_grad_norms)) if context_grad_norms else None,
        "context_grad_l2_max": float(np.max(context_grad_norms)) if context_grad_norms else None,
        "context_grad_missing_parameters": context_missing,
        "context_grad_zero_parameters": context_zero,
        "process_cpu_seconds": time.process_time() - process_started,
        "process_rss_bytes": _process_memory_bytes(),
    }


def _validation_epoch(
    model: ContextIndividualTF,
    loader: DataLoader,
    max_batches: int | None,
) -> tuple[float, int, float | None]:
    model.eval()
    criterion = torch.nn.MSELoss()
    loss_total = 0.0
    sample_count = 0
    batch_count = 0
    gate_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_limit(loader, max_batches):
            prediction = forward(model, batch)
            loss = criterion(prediction, batch["pred_target"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite validation loss.")
            batch_size = int(batch["pred_target"].shape[0])
            loss_total += float(loss.item()) * batch_size
            sample_count += batch_size
            batch_count += 1
            if model.last_context_gate is not None:
                gate_values.append(model.last_context_gate.detach().cpu().numpy())
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no batches.")
    gate_array = np.concatenate(gate_values) if gate_values else np.array([], dtype=np.float32)
    gate_mean = float(np.mean(gate_array)) if gate_array.size else None
    return loss_total / sample_count, batch_count, gate_mean


def _evaluate_test(model: ContextIndividualTF, loader: DataLoader, stats: tuple[np.ndarray, ...]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    mean, std, _, _ = stats
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = forward(model, batch).cpu()
            target = batch["pred_target"].cpu()
            raw_prediction = prediction.numpy() * std + mean
            raw_target = target.numpy() * std + mean
            for index in range(prediction.shape[0]):
                center_prediction = (raw_prediction[index, :, :2] + raw_prediction[index, :, 2:]) / 2
                center_target = (raw_target[index, :, :2] + raw_target[index, :, 2:]) / 2
                rows.append({
                    "sample_id": batch["sample_id"][index],
                    "video": batch["video"][index],
                    "pid": batch["pid"][index],
                    "frame_start": int(batch["frame_start"][index]),
                    "frame_end": int(batch["frame_end"][index]),
                    "mse": float(np.mean((raw_prediction[index] - raw_target[index]) ** 2)),
                    "cmse": float(np.mean((center_prediction - center_target) ** 2)),
                    "cfmse": float(np.mean((center_prediction[-1] - center_target[-1]) ** 2)),
                })
    if not rows:
        raise RuntimeError("Test loader produced no samples.")
    return {
        "mse": float(np.mean([row["mse"] for row in rows])),
        "cmse": float(np.mean([row["cmse"] for row in rows])),
        "cfmse": float(np.mean([row["cfmse"] for row in rows])),
    }, rows


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("run_e8.py"),
        Path(__file__).with_name("context_TF.py"),
        Path(__file__).with_name("run_from_scratch_e1.py"),
        Path(__file__).parent / "transformer" / "functional.py",
        Path(__file__).parent / "transformer" / "encoder_decoder.py",
    )
    return {
        str(path.relative_to(RESULTS.parents[0])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def _config_for(args: argparse.Namespace, model_key: str, variant: str, seed: int) -> dict[str, Any]:
    return {
        "runner_version": "E8_v2",
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["name"],
        "variant": variant,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "cpu_threads": args.cpu_threads,
        "validation_only": args.validation_only,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "source_sha256": _source_hashes(),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _compatible_config(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    mutable = {"max_epochs", "source_sha256"}
    return all(previous.get(key) == value for key, value in current.items() if key not in mutable)


def _prepare_run(run: Path, config: dict[str, Any], resume: bool) -> tuple[bool, dict[str, Any] | None]:
    if not run.exists():
        run.mkdir(parents=True)
        atomic_write_json(run / "run_config.json", config)
        return False, None

    status_path = run / "status.json"
    config_path = run / "run_config.json"
    metrics_path = run / "metrics.json"
    if not status_path.exists() or not config_path.exists():
        if any(run.iterdir()):
            raise RuntimeError(
                f"Existing run directory has no complete status/config; refusing to treat it as successful or overwrite it: {run}"
            )
        atomic_write_json(config_path, config)
        return False, None

    previous_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not _compatible_config(previous_config, config):
        raise ValueError(f"Run configuration differs from saved configuration: {run}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") == "completed" and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") == "completed":
            old_max = int(previous_config["max_epochs"])
            latest = run / "latest_checkpoint.pt"
            if (
                resume
                and config["validation_only"]
                and config["max_epochs"] > old_max
                and latest.exists()
                and int(status.get("epoch", 0)) < config["max_epochs"]
                and metrics.get("stopping_reason") == "max_epochs"
            ):
                atomic_write_json(config_path, config)
                return False, _load_checkpoint(latest)
            return True, metrics
    if not resume:
        raise RuntimeError(f"Run exists in state {status.get('state')!r}; pass --resume to continue: {run}")
    latest = run / "latest_checkpoint.pt"
    if not latest.exists():
        raise FileNotFoundError(f"Cannot resume without a valid checkpoint: {latest}")
    atomic_write_json(config_path, config)
    return False, _load_checkpoint(latest)


def _restore_checkpoint(
    payload: dict[str, Any],
    model: ContextIndividualTF,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
) -> tuple[int, list[dict[str, Any]], float, int, int, dict[str, torch.Tensor] | None]:
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_state(payload["rng_state"], loader_generator)
    return (
        int(payload["epoch"]) + 1,
        list(payload["history"]),
        float(payload["best_validation_loss"]),
        int(payload["best_epoch"]),
        int(payload["stale_epochs"]),
        payload.get("best_model_state"),
    )


def run_one(args: argparse.Namespace, model_key: str, variant: str, seed: int) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    run = args.output_root / f"{spec['slug']}_{variant}" / str(seed)
    config = _config_for(args, model_key, variant, seed)
    finished, payload = _prepare_run(run, config, args.resume)
    if finished:
        print(f"COMPLETED run={run}; validated status and metrics, not directory existence.", flush=True)
        return payload

    _status(
        run,
        "running",
        epoch=0,
        best_epoch=0,
        max_epochs=args.max_epochs,
        validation_only=args.validation_only,
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_model_state = None
    previous_sigterm = None
    progress_heartbeat = None
    try:
        set_seed(seed, args.cpu_threads)
        model = build_model(model_key)
        configure_variant(model, variant)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loader_generator = torch.Generator().manual_seed(seed)
        datasets, stats = load_split(args.validation_only)
        train_loader = DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True, generator=loader_generator
        )
        validation_loader = DataLoader(
            datasets["validation"], batch_size=args.batch_size, shuffle=False
        )
        if payload is not None:
            start_epoch, history, best_validation_loss, best_epoch, stale_epochs, best_model_state = _restore_checkpoint(
                payload, model, optimizer, loader_generator
            )
            if start_epoch > args.max_epochs:
                raise ValueError(
                    f"Checkpoint already completed epoch {start_epoch - 1}, beyond requested max {args.max_epochs}."
                )
        if not (run / "latest_checkpoint.pt").exists():
            initial = _checkpoint_payload(
                model, optimizer, loader_generator, 0, history, best_validation_loss,
                best_epoch, stale_epochs, best_model_state,
            )
            atomic_torch_save(run / "latest_checkpoint.pt", initial)

        def handle_sigterm(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        previous_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
        progress_heartbeat = ProgressHeartbeat(run, args.heartbeat_seconds)
        progress_heartbeat.start()
        _write_progress(
            run,
            f"START state=running model={model_key} variant={variant} seed={seed} "
            f"epoch={start_epoch}/{args.max_epochs} threads={args.cpu_threads} "
            f"validation_only={args.validation_only}",
        )
        for epoch in range(start_epoch, args.max_epochs + 1):
            epoch_started = time.perf_counter()
            progress_heartbeat.update(epoch, 0, epoch_started)
            train_record = _train_epoch(
                model, optimizer, train_loader, epoch, run,
                args.heartbeat_seconds, args.max_train_batches, progress_heartbeat,
            )
            validation_loss, validation_batches, validation_gate_mean = _validation_epoch(
                model, validation_loader, args.max_validation_batches
            )
            epoch_seconds = time.perf_counter() - epoch_started
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                best_model_state = _module_state(model)
                atomic_torch_save(
                    run / "best_checkpoint.pt",
                    {"model_state": best_model_state, "epoch": best_epoch, "validation_loss": best_validation_loss},
                )
            else:
                stale_epochs += 1
            record = {
                "epoch": epoch,
                "train_loss": train_record["train_loss"],
                "validation_loss": validation_loss,
                "epoch_seconds": epoch_seconds,
                "train_batches": train_record["train_batches"],
                "validation_batches": validation_batches,
                "validation_gate_mean": validation_gate_mean,
                "process_rss_bytes": _process_memory_bytes(),
                **{key: value for key, value in train_record.items() if key != "train_loss"},
            }
            history.append(record)
            latest = _checkpoint_payload(
                model, optimizer, loader_generator, epoch, history,
                best_validation_loss, best_epoch, stale_epochs, best_model_state,
            )
            atomic_torch_save(run / "latest_checkpoint.pt", latest)
            _history_write(run, history)
            _status(
                run,
                "running",
                epoch=epoch,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                max_epochs=args.max_epochs,
                validation_only=args.validation_only,
            )
            _write_progress(
                run,
                f"EPOCH state=running epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"validation_loss={validation_loss:.8f} epoch_seconds={epoch_seconds:.2f} "
                f"admission_rate={record['train_admission_rate']} "
                f"gate_grad_l2={record['gate_grad_l2_mean']} "
                f"context_grad_l2={record['context_grad_l2_mean']} "
                f"rss_bytes={record['process_rss_bytes']}",
            )
            if stale_epochs >= args.patience:
                break

        if best_model_state is None:
            raise RuntimeError("Training ended without a validation-selected checkpoint.")
        model.load_state_dict(best_model_state)
        result: dict[str, Any] = {
            "status": "completed",
            "model": spec["name"],
            "model_key": model_key,
            "variant": variant,
            "seed": seed,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_loss,
            "stopping_reason": "early_stopping" if stale_epochs >= args.patience else "max_epochs",
            "training_seconds": float(sum(row["epoch_seconds"] for row in history)),
            "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "validation_only": args.validation_only,
            "test_evaluated": False,
            "history_rows": len(history),
            "max_train_batches": args.max_train_batches,
            "max_validation_batches": args.max_validation_batches,
        }
        if not args.validation_only:
            # Test data is accessed exactly here, after validation-only checkpoint selection.
            test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)
            test_metrics, test_rows = _evaluate_test(model, test_loader, stats)
            result.update(test_metrics)
            result["test_evaluated"] = True
            fields = list(test_rows[0])
            atomic_write_csv(run / "per_sample.csv", test_rows, fields)
        atomic_write_json(run / "metrics.json", result)
        _status(
            run,
            "completed",
            epoch=result["epochs_completed"],
            best_epoch=best_epoch,
            validation_only=args.validation_only,
            test_evaluated=result["test_evaluated"],
        )
        _write_progress(
            run,
            f"COMPLETE state=completed epochs={result['epochs_completed']} "
            f"best_epoch={best_epoch} best_validation_loss={best_validation_loss:.8f} "
            f"test_evaluated={result['test_evaluated']} elapsed_seconds={result['training_seconds']:.2f}",
        )
        return result
    except KeyboardInterrupt as exc:
        completed_epoch = history[-1]["epoch"] if history else start_epoch - 1
        _status(run, "interrupted", epoch=completed_epoch, reason=str(exc))
        _write_progress(run, f"INTERRUPTED state=interrupted reason={exc}")
        raise
    except Exception as exc:
        _status(
            run,
            "failed",
            epoch=(history[-1]["epoch"] if history else start_epoch - 1),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        _write_progress(run, f"FAILED state=failed error={type(exc).__name__}: {exc}")
        raise
    finally:
        if progress_heartbeat is not None:
            progress_heartbeat.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=["without_satie"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEEDS[0]])
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    if args.patience < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("--patience, --batch-size, and --cpu-threads must be positive")
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds cannot be negative")
    for option in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    for model_key in args.models:
        for variant in args.variants:
            for seed in args.seeds:
                run_one(args, model_key, variant, seed)


if __name__ == "__main__":
    main()
