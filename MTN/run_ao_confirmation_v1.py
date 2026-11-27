"""Sequential, resumable Phase 6A AO multi-seed confirmation runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results_paper" / "splits" / "ao_annotation_only_v1"
DEFAULT_OUTPUT = ROOT / "results_paper" / "AO_confirmation_v1"
DEFAULT_MANIFEST = ROOT / "results_paper" / "AO_CONFIRMATION_MANIFEST_v2.json"
SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)
MODELS = ("AO-1", "AO-2", "AO-3")
DEFAULT_GLOBAL_BUDGET_SECONDS = 14 * 60 * 60
DEFAULT_PER_RUN_BUDGET_SECONDS = 2 * 60 * 60
DEFAULT_FINAL_TWO_RESERVE_SECONDS = 6 * 60 * 60


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fork_seed_20260928(
    model_id: str,
    output_root: Path,
    manifest: dict[str, Any],
) -> Path:
    import shutil

    source = (
        ROOT / "results_paper" / "AO_initial_screen_v1"
        / model_id / "seed_20260928"
    )
    destination = output_root / model_id / "seed_20260928"
    if destination.exists():
        return destination
    expected = manifest["seed_20260928_resumable_checkpoint_sources"][model_id]
    source_checkpoint = source / "latest_checkpoint.pt"
    source_config = source / "run_config.json"
    if (
        not source_checkpoint.is_file()
        or not source_config.is_file()
        or _sha256(source_checkpoint) != expected["optimizer_and_rng_state_sha256"]
        or _sha256(source_config) != expected["checkpoint_config_sha256"]
    ):
        raise ValueError(f"Frozen seed-20260928 continuation source changed: {source}")
    destination.mkdir(parents=True)
    for name in (
        "run_config.json", "latest_checkpoint.pt", "history.json",
        "history.csv", "normalization.json", "best_model.pt",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    status = _read_json(source / "status.json")
    status.update({
        "state": "incomplete",
        "reason": "forked_checkpoint_for_phase6a_extension",
        "source_run": str(source.relative_to(ROOT)),
    })
    _write_state(destination / "status.json", status)
    return destination


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_for(run_dir: Path) -> str | None:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return None
    return _read_json(status_path).get("state")


def _run_command(
    command: list[str],
    log_path: Path,
    global_deadline: float,
    per_run_budget_seconds: int,
    state: dict[str, Any],
    state_path: Path,
) -> bool:
    remaining = global_deadline - time.monotonic()
    if remaining <= 1:
        return False
    run_budget = max(1, min(per_run_budget_seconds, math.floor(remaining)))
    command.extend([
        "--max-epochs", "30",
        "--patience", "10",
        "--batch-size", "8",
        "--learning-rate", "0.001",
        "--cpu-threads", "4",
        "--wall-clock-budget-seconds", str(run_budget),
    ])
    started = time.monotonic()
    state["inflight_started_epoch_seconds"] = time.time()
    state["last_command"] = command
    _write_state(state_path, state)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    elapsed = time.monotonic() - started
    state["cumulative_wall_seconds"] = (
        float(state.get("cumulative_wall_seconds", 0.0)) + elapsed
    )
    state["last_command"] = command
    state["last_command_seconds"] = elapsed
    state["updated_at_epoch_seconds"] = time.time()
    state["last_return_code"] = code
    state.pop("inflight_started_epoch_seconds", None)
    _write_state(state_path, state)
    if code != 0:
        raise RuntimeError(f"Experiment subprocess failed with exit code {code}.")
    return True


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "ao_confirmation_protocol_v1":
        raise ValueError("Missing or incompatible frozen confirmation manifest.")
    if manifest.get("cache_sha256") != {
        name: _sha256(cache_dir / name)
        for name in ("train.npz", "validation.npz", "split_manifest.json")
    }:
        raise ValueError("Confirmation caches no longer match frozen hashes.")
    if manifest.get("official_test_sets_read") != []:
        raise ValueError("Frozen manifest does not certify test isolation.")

    state_path = output_root / "orchestration_state.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Output root already contains data; pass --resume: {output_root}"
            )
        if not state_path.is_file():
            raise FileNotFoundError(
                "Existing output has no orchestration state to resume."
            )
        state = _read_json(state_path)
        expected = {
            "manifest_sha256": _sha256(manifest_path),
            "cache_sha256": manifest["cache_sha256"],
            "seeds": list(SEEDS),
            "models": list(MODELS),
            "max_epochs": 30,
            "patience": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
            "cpu_threads": 4,
            "global_budget_seconds": args.global_budget_seconds,
            "per_run_budget_seconds": args.per_run_budget_seconds,
            "final_two_seed_reserve_seconds": args.final_two_reserve_seconds,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume request differs from the frozen run configuration.")
        inflight_started = state.pop("inflight_started_epoch_seconds", None)
        if inflight_started is not None:
            state["cumulative_wall_seconds"] = (
                float(state.get("cumulative_wall_seconds", 0.0))
                + max(0.0, time.time() - float(inflight_started))
            )
            _write_state(state_path, state)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": "ao_confirmation_orchestration_v1",
            "manifest_sha256": _sha256(manifest_path),
            "cache_sha256": manifest["cache_sha256"],
            "seeds": list(SEEDS),
            "models": list(MODELS),
            "max_epochs": 30,
            "patience": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
            "cpu_threads": 4,
            "global_budget_seconds": args.global_budget_seconds,
            "per_run_budget_seconds": args.per_run_budget_seconds,
            "final_two_seed_reserve_seconds": args.final_two_reserve_seconds,
            "cumulative_wall_seconds": 0.0,
            "completed_runs": [],
            "incomplete_runs": [],
            "status": "running",
        }
        _write_state(state_path, state)

    remaining_global = (
        args.global_budget_seconds
        - float(state.get("cumulative_wall_seconds", 0.0))
    )
    invocation_deadline = time.monotonic() + max(0.0, remaining_global)
    log_path = output_root / "orchestration.log"
    runner_script = ROOT / "MTN" / "run_ao_v1.py"

    for seed_index, seed in enumerate(SEEDS):
        baseline_run = output_root / "AO-0" / f"seed_{seed}"
        if _status_for(baseline_run) != "completed":
            command = [
                sys.executable, str(runner_script),
                "--cache-dir", str(cache_dir),
                "--output-root", str(output_root),
                "--models", "AO-0",
                "--seed", str(seed),
            ]
            if not _run_command(
                command, log_path, invocation_deadline,
                args.per_run_budget_seconds, state, state_path,
            ):
                state["status"] = "incomplete_global_budget"
                _write_state(state_path, state)
                return 0

        if seed_index == 0:
            for model_id in MODELS:
                _fork_seed_20260928(model_id, output_root, manifest)

        for model_id in MODELS:
            run_dir = output_root / model_id / f"seed_{seed}"
            if _status_for(run_dir) == "completed":
                continue
            if time.monotonic() >= invocation_deadline:
                state["status"] = "incomplete_global_budget"
                _write_state(state_path, state)
                return 0
            resume = (run_dir / "run_config.json").is_file()
            while True:
                command = [
                    sys.executable, str(runner_script),
                    "--cache-dir", str(cache_dir),
                    "--output-root", str(output_root),
                    "--models", model_id,
                    "--seed", str(seed),
                ]
                if resume:
                    command.append("--resume")
                if not _run_command(
                    command, log_path, invocation_deadline,
                    args.per_run_budget_seconds, state, state_path,
                ):
                    state["status"] = "incomplete_global_budget"
                    _write_state(state_path, state)
                    return 0
                status = _status_for(run_dir)
                if status == "completed":
                    break
                if status != "incomplete":
                    raise RuntimeError(
                        f"Unexpected state {status!r} for {model_id}, seed {seed}."
                    )
                resume = True
                if time.monotonic() >= invocation_deadline:
                    state["status"] = "incomplete_global_budget"
                    _write_state(state_path, state)
                    return 0
            entry = f"{model_id}/seed_{seed}"
            if entry not in state["completed_runs"]:
                state["completed_runs"].append(entry)
            _write_state(state_path, state)

        finalize = [
            sys.executable, str(runner_script),
            "--finalize-only", "--output-root", str(output_root),
        ]
        if not _run_command(
            finalize, log_path, invocation_deadline,
            args.per_run_budget_seconds, state, state_path,
        ):
            state["status"] = "incomplete_global_budget"
            _write_state(state_path, state)
            return 0

        if seed_index == 2:
            remaining = invocation_deadline - time.monotonic()
            if remaining < args.final_two_reserve_seconds:
                state["status"] = "incomplete_budget_before_final_two"
                state["remaining_seconds_at_gate"] = max(0, remaining)
                _write_state(state_path, state)
                return 0

    state["status"] = "completed"
    state["completed_at_epoch_seconds"] = time.time()
    _write_state(state_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--global-budget-seconds",
        type=int,
        default=DEFAULT_GLOBAL_BUDGET_SECONDS,
    )
    parser.add_argument(
        "--per-run-budget-seconds",
        type=int,
        default=DEFAULT_PER_RUN_BUDGET_SECONDS,
    )
    parser.add_argument(
        "--final-two-reserve-seconds",
        type=int,
        default=DEFAULT_FINAL_TWO_RESERVE_SECONDS,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
        args.global_budget_seconds,
        args.per_run_budget_seconds,
        args.final_two_reserve_seconds,
    ) <= 0:
        parser.error("All wall-clock budgets must be positive.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
