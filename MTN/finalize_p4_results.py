"""Validation-only Phase 4 paired comparisons and reports."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .p4_models_v1 import P4_CONFIGURATIONS, P4_EVALUATION_POLICIES, P4_SELECTION_POLICY
    from .run_e8_v2 import atomic_write_bytes, atomic_write_csv, atomic_write_json
    from .run_from_scratch_e1 import RESULTS
except ImportError:
    from p4_models_v1 import P4_CONFIGURATIONS, P4_EVALUATION_POLICIES, P4_SELECTION_POLICY
    from run_e8_v2 import atomic_write_bytes, atomic_write_csv, atomic_write_json
    from run_from_scratch_e1 import RESULTS


DEFAULT_SEEDS = (20260928, 20260929, 20260930)
DEFAULT_OUTPUT_ROOT = RESULTS / "P4_initial_screen_v1"
METRICS = (
    "normalized_mse",
    "raw_pixel_mse",
    "cmse",
    "cfmse",
)
WINDOW_METRICS = ("raw_pixel_mse", "cmse", "cfmse")
T_CRITICAL_95 = {1: 12.7062047364, 2: 4.30265272991}
PAIR_DEFINITIONS = (
    ("p4_a", "cv_baseline", "MTN vs deterministic CV"),
    ("p4_b", "cv_baseline", "SATIE MTN vs deterministic CV"),
    ("p4_c", "cv_baseline", "Always-open context without SATIE vs deterministic CV"),
    ("p4_d", "cv_baseline", "Always-open context with SATIE vs deterministic CV"),
    ("p4_e", "cv_baseline", "Original hard SAGE vs deterministic CV"),
    ("p4_f", "cv_baseline", "Differentiable context-only SAGE soft vs deterministic CV"),
    ("p4_b", "p4_a", "SATIE effect on MTN"),
    ("p4_d", "p4_c", "SATIE effect on always-open context"),
    ("p4_c", "p4_a", "Always-open context without SATIE vs MTN"),
    ("p4_d", "p4_b", "Always-open context with SATIE vs MTN with SATIE"),
    ("p4_e", "p4_d", "Original hard SAGE vs independently trained always-open"),
    ("p4_f", "p4_d", "Differentiable context-only SAGE soft vs always-open"),
    ("p4_f", "p4_e", "Differentiable context-only SAGE soft vs original hard SAGE"),
)
DESCRIPTIONS = {
    "p4_a": "MTN with constant-velocity residual anchor; no context; no SATIE",
    "p4_b": "MTN with constant-velocity residual anchor and SATIE; no context",
    "p4_c": "Always-open context with MASC and constant-velocity residual anchor; no SATIE",
    "p4_d": "Always-open context with MASC, SATIE, and constant-velocity residual anchor",
    "p4_e": "Original hard SAGE with MASC, SATIE, and constant-velocity residual anchor",
    "p4_f": "Differentiable context-only SAGE V2 with MASC, SATIE, and constant-velocity residual anchor",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_complete_runs(root: Path, seeds: tuple[int, ...]) -> dict[tuple[str, int], dict[str, Any]]:
    runs = {}
    failures = []
    reference_role_hashes = None
    for configuration in P4_CONFIGURATIONS:
        for seed in seeds:
            path = root / configuration / str(seed)
            status_path = path / "status.json"
            metrics_path = path / "validation_metrics.json"
            if not status_path.is_file() or not metrics_path.is_file():
                failures.append(f"{configuration}/{seed}: missing status or validation metrics")
                continue
            status = _read_json(status_path)
            metrics = _read_json(metrics_path)
            config = _read_json(path / "run_config.json")
            role_hashes = config.get("role_cache_hashes")
            if not isinstance(role_hashes, dict) or set(role_hashes) != {
                "train", "validation"
            }:
                failures.append(f"{configuration}/{seed}: invalid role-cache provenance")
                continue
            if reference_role_hashes is None:
                reference_role_hashes = role_hashes
            elif role_hashes != reference_role_hashes:
                failures.append(f"{configuration}/{seed}: role-cache hashes differ across runs")
                continue
            if config.get("configuration") != configuration or config.get("seed") != seed:
                failures.append(f"{configuration}/{seed}: run configuration identity mismatch")
                continue
            if status.get("state") != "completed" or metrics.get("status") != "completed":
                failures.append(
                    f"{configuration}/{seed}: status={status.get('state')!r}, "
                    f"metrics={metrics.get('status')!r}"
                )
                continue
            if metrics.get("cache_roles_loaded") != ["train", "validation"]:
                failures.append(f"{configuration}/{seed}: invalid loaded-role record")
                continue
            if (
                config.get("max_train_batches") is not None
                or config.get("max_validation_batches") is not None
                or config.get("max_epochs") != 30
                or config.get("patience") != 10
                or config.get("batch_size") != 8
                or config.get("learning_rate") != 1e-3
                or config.get("cpu_threads") != 4
                or config.get("validation_only") is not True
            ):
                failures.append(f"{configuration}/{seed}: run settings are not the full P4 screen")
                continue
            if metrics.get("train", {}).get("sample_count") != 212:
                failures.append(f"{configuration}/{seed}: training metrics are not full-split")
                continue
            if any(row.get("sample_count") != 43 for row in metrics.get("policies", [])):
                failures.append(f"{configuration}/{seed}: validation metrics are not full-split")
                continue
            runs[(configuration, seed)] = metrics
    if failures:
        raise RuntimeError(
            "Finalization requires every requested run to be completed and validation-only:\n"
            + "\n".join(failures)
        )
    return runs


def _policy_metrics(
    runs: dict[tuple[str, int], dict[str, Any]],
    configuration: str,
    seed: int,
    policy: str,
) -> dict[str, Any]:
    run = runs[(configuration, seed)]
    for row in run["policies"]:
        if row["policy"] == policy:
            return row
    raise KeyError(f"Missing policy {policy} for {configuration}, seed {seed}")


def _seed_summary(
    root: Path,
    runs: dict[tuple[str, int], dict[str, Any]],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for configuration in P4_CONFIGURATIONS:
        policy = P4_SELECTION_POLICY[configuration]
        for seed in seeds:
            run = runs[(configuration, seed)]
            metrics = _policy_metrics(runs, configuration, seed, policy)
            samples = _load_sample_rows(root, configuration, seed, policy)
            correction_values = [
                float(row["correction_rms_pixels"]) for row in samples.values()
            ]
            rows.append({
                "configuration": configuration,
                "description": DESCRIPTIONS[configuration],
                "seed": seed,
                "policy": policy,
                "epochs_completed": run["epochs_completed"],
                "selected_epoch": run["selected_epoch"],
                "unique_parameters": run["unique_parameters"],
                "training_seconds": run["training_seconds"],
                "peak_sampled_rss_bytes": run["peak_sampled_rss_bytes"],
                "checkpoint_bytes": run["checkpoint_bytes"],
                "train_normalized_mse": run["train"]["normalized_mse"],
                "train_raw_pixel_mse": run["train"]["raw_pixel_mse"],
                **{
                    f"validation_{metric}": metrics.get(metric)
                    for metric in METRICS
                },
                "mean_correction_rms_pixels": metrics.get("correction_rms_pixels"),
                "correction_rms_sd_pixels": float(np.std(correction_values, ddof=0)),
                "mean_gate_gradient_l2": run["gate_gradient_l2_mean_present_batches"],
                "mean_context_encoder_gradient_l2": (
                    run["context_gradient_l2_mean_present_batches"]
                ),
                "mean_admission_probability": metrics.get("mean_admission_probability"),
                "admission_probability_sd": metrics.get("admission_probability_sd"),
                "hard_admission_rate": metrics.get("hard_admission_rate"),
                "effective_admission_rate": metrics.get("effective_admission_rate"),
                "gate_entropy_nats": metrics.get("gate_entropy_nats"),
                "context_attention_weight": metrics.get("context_attention_weight"),
            })
    return rows


def _validation_summary(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for configuration in P4_CONFIGURATIONS:
        subset = [row for row in seed_rows if row["configuration"] == configuration]
        row: dict[str, Any] = {
            "configuration": configuration,
            "description": DESCRIPTIONS[configuration],
            "policy": P4_SELECTION_POLICY[configuration],
            "n_seeds": len(subset),
            "unique_parameters": subset[0]["unique_parameters"],
        }
        for field in (
            "epochs_completed", "selected_epoch", "training_seconds",
            "peak_sampled_rss_bytes", "checkpoint_bytes",
            "train_normalized_mse", "train_raw_pixel_mse",
            *(f"validation_{metric}" for metric in METRICS),
            "mean_gate_gradient_l2", "mean_context_encoder_gradient_l2",
            "mean_admission_probability", "hard_admission_rate",
            "effective_admission_rate", "gate_entropy_nats",
            "context_attention_weight", "mean_correction_rms_pixels",
            "correction_rms_sd_pixels",
        ):
            values = np.asarray([
                float(item[field]) for item in subset if item.get(field) not in (None, "")
            ], dtype=np.float64)
            row[f"{field}_mean"] = float(np.mean(values)) if values.size else None
            row[f"{field}_sd"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else None
            )
        rows.append(row)
    return rows


def _paired_stats(differences: list[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    n = len(values)
    mean = float(np.mean(values)) if n else None
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    critical = T_CRITICAL_95.get(n - 1)
    half_width = critical * sd / math.sqrt(n) if critical is not None and sd is not None else None
    return {
        "n_seeds": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "ci95_t_low": mean - half_width if half_width is not None else None,
        "ci95_t_high": mean + half_width if half_width is not None else None,
        "dz": mean / sd if mean is not None and sd not in (None, 0.0) else None,
        "first_model_wins": int(np.sum(values < 0.0)),
    }


def _load_sample_rows(
    root: Path,
    configuration: str,
    seed: int,
    policy: str,
) -> dict[str, dict[str, str]]:
    if configuration == "cv_baseline":
        rows = _read_csv(root / "P4_CV_BASELINE_PER_WINDOW.csv")
    else:
        rows = _read_csv(root / configuration / str(seed) / "validation_per_sample.csv")
    selected = {
        row["sample_id"]: row for row in rows
        if row["policy"] == policy
    }
    if not selected:
        raise RuntimeError(f"No per-sample rows for {configuration}/{seed}/{policy}")
    return selected


def _cluster_bootstrap(
    root: Path,
    seeds: tuple[int, ...],
    first: str,
    first_policy: str,
    second: str,
    second_policy: str,
    metric: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    by_track: dict[str, list[float]] = {}
    sample_count = 0
    for seed in seeds:
        left = _load_sample_rows(root, first, seed, first_policy)
        right = _load_sample_rows(root, second, seed, second_policy)
        if left.keys() != right.keys():
            raise RuntimeError(
                f"Per-sample IDs differ for paired models {first} and {second}, seed {seed}."
            )
        for sample_id, left_row in left.items():
            right_row = right[sample_id]
            if left_row["pid"] != right_row["pid"]:
                raise RuntimeError(f"Pedestrian ID mismatch for {sample_id}.")
            difference = float(left_row[metric]) - float(right_row[metric])
            by_track.setdefault(left_row["pid"], []).append(difference)
            sample_count += 1
    track_ids = sorted(by_track)
    if len(track_ids) < 2:
        raise RuntimeError("Track-cluster bootstrap requires at least two pedestrians.")
    boot_means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.choice(track_ids, size=len(track_ids), replace=True)
        values = [value for track_id in draw for value in by_track[str(track_id)]]
        boot_means[index] = float(np.mean(values))
    return {
        "n_validation_tracks": len(track_ids),
        "n_seed_window_pairs": sample_count,
        "bootstrap_repetitions": repetitions,
        "bootstrap_ci95_low": float(np.quantile(boot_means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot_means, 0.975)),
        "bootstrap_caveat": "43 overlapping validation windows from one video; exploratory only",
    }


def _paired_comparisons(
    root: Path,
    runs: dict[tuple[str, int], dict[str, Any]],
    seeds: tuple[int, ...],
    repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    baseline = _read_json(root / "P4_CV_BASELINE.json")
    for first, second, label in PAIR_DEFINITIONS:
        first_policy = P4_SELECTION_POLICY[first]
        second_policy = "deterministic" if second == "cv_baseline" else P4_SELECTION_POLICY[second]
        for metric in METRICS:
            differences = [
                float(_policy_metrics(runs, first, seed, first_policy)[metric])
                - float(
                    baseline["validation"][metric]
                    if second == "cv_baseline"
                    else _policy_metrics(runs, second, seed, second_policy)[metric]
                )
                for seed in seeds
            ]
            row = {
                "comparison": label,
                "first_configuration": first,
                "first_policy": first_policy,
                "second_configuration": second,
                "second_policy": second_policy,
                "metric": metric,
                **_paired_stats(differences),
                "bootstrap_ci95_low": None,
                "bootstrap_ci95_high": None,
                "n_validation_tracks": None,
                "n_seed_window_pairs": None,
                "bootstrap_repetitions": repetitions,
                "bootstrap_caveat": "not applicable to normalized MSE",
            }
            if metric in WINDOW_METRICS:
                row.update(_cluster_bootstrap(
                    root, seeds, first, first_policy, second, second_policy,
                    metric, repetitions, rng,
                ))
            output.append(row)

    for metric in WINDOW_METRICS:
        differences = [
            float(_policy_metrics(runs, "p4_f", seed, "hard")[metric])
            - float(_policy_metrics(runs, "p4_f", seed, "soft")[metric])
            for seed in seeds
        ]
        row = {
            "comparison": "P4-F hard inference minus soft inference",
            "first_configuration": "p4_f",
            "first_policy": "hard",
            "second_configuration": "p4_f",
            "second_policy": "soft",
            "metric": metric,
            **_paired_stats(differences),
            **_cluster_bootstrap(
                root, seeds, "p4_f", "hard", "p4_f", "soft", metric,
                repetitions, rng,
            ),
        }
        output.append(row)
    return output


def _pearson(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _gate_utility(root: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    result = []
    for configuration, natural_policy in (("p4_e", "hard"), ("p4_f", "soft")):
        decision_policy = "hard"
        for seed in seeds:
            rows = {
                policy: _load_sample_rows(root, configuration, seed, policy)
                for policy in (natural_policy, decision_policy, "forced_open", "forced_closed")
            }
            sample_ids = rows[natural_policy].keys()
            if any(row.keys() != sample_ids for row in rows.values()):
                raise RuntimeError(f"Gate intervention sample IDs differ for {configuration}/{seed}.")
            natural_errors, open_errors, closed_errors = [], [], []
            probabilities, hard_decisions = [], []
            for sample_id in sample_ids:
                natural = rows[natural_policy][sample_id]
                open_row = rows["forced_open"][sample_id]
                closed_row = rows["forced_closed"][sample_id]
                if natural["admission_probability"] == "":
                    raise RuntimeError(f"Missing gate probability for {configuration}/{seed}.")
                natural_errors.append(float(natural["raw_pixel_mse"]))
                open_errors.append(float(open_row["raw_pixel_mse"]))
                closed_errors.append(float(closed_row["raw_pixel_mse"]))
                probabilities.append(float(natural["admission_probability"]))
                decision = rows[decision_policy][sample_id]["hard_gate_decision"]
                hard_decisions.append(decision.lower() == "true")
            benefit = np.asarray(closed_errors) - np.asarray(open_errors)
            run = _read_json(root / configuration / str(seed) / "validation_metrics.json")
            result.append({
                "configuration": configuration,
                "seed": seed,
                "natural_policy": natural_policy,
                "n_windows": len(sample_ids),
                "mean_admission_probability": float(np.mean(probabilities)),
                "admission_probability_sd": float(np.std(probabilities, ddof=0)),
                "hard_admission_rate": float(np.mean(hard_decisions)),
                "mean_natural_raw_mse": float(np.mean(natural_errors)),
                "mean_forced_open_raw_mse": float(np.mean(open_errors)),
                "mean_forced_closed_raw_mse": float(np.mean(closed_errors)),
                "open_beneficial_windows": int(np.sum(benefit > 0.0)),
                "closed_beneficial_windows": int(np.sum(benefit < 0.0)),
                "hard_decision_matches_better_intervention": int(np.sum(
                    np.asarray(hard_decisions) == (benefit > 0.0)
                )),
                "probability_vs_open_benefit_pearson_r": _pearson(
                    probabilities, benefit.tolist()
                ),
                "mean_gate_gradient_l2": run["gate_gradient_l2_mean_present_batches"],
                "mean_context_encoder_gradient_l2": (
                    run["context_gradient_l2_mean_present_batches"]
                ),
                "gate_gradient_batches_all_missing": run["gate_gradient_batches_all_missing"],
                "context_gradient_batches_all_missing": (
                    run["context_gradient_batches_all_missing"]
                ),
                "interpretation": "exploratory post-hoc validation intervention; not a causal estimate",
            })
    return result


def _parameter_efficiency(
    summary: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    model_rows = []
    for row in summary:
        model_rows.append({
            "configuration": row["configuration"],
            "description": row["description"],
            "unique_parameters": row["unique_parameters"],
            "validation_raw_pixel_mse_mean": row["validation_raw_pixel_mse_mean"],
            "validation_raw_pixel_mse_sd": row["validation_raw_pixel_mse_sd"],
            "delta_raw_mse_vs_cv": (
                row["validation_raw_pixel_mse_mean"]
                - baseline["validation"]["raw_pixel_mse"]
            ),
            "is_pareto_efficient": True,
            "n_seeds": row["n_seeds"],
        })
    for candidate in model_rows:
        for other in model_rows + [{
            "configuration": "cv_baseline",
            "unique_parameters": 0,
            "validation_raw_pixel_mse_mean": baseline["validation"]["raw_pixel_mse"],
        }]:
            if candidate is other:
                continue
            no_worse = (
                other["unique_parameters"] <= candidate["unique_parameters"]
                and other["validation_raw_pixel_mse_mean"]
                <= candidate["validation_raw_pixel_mse_mean"]
            )
            strictly_better = (
                other["unique_parameters"] < candidate["unique_parameters"]
                or other["validation_raw_pixel_mse_mean"]
                < candidate["validation_raw_pixel_mse_mean"]
            )
            if no_worse and strictly_better:
                candidate["is_pareto_efficient"] = False
                break
    model_rows.append({
        "configuration": "cv_baseline",
        "description": "Deterministic constant-velocity baseline",
        "unique_parameters": 0,
        "validation_raw_pixel_mse_mean": baseline["validation"]["raw_pixel_mse"],
        "validation_raw_pixel_mse_sd": 0.0,
        "delta_raw_mse_vs_cv": 0.0,
        "is_pareto_efficient": True,
        "n_seeds": 0,
    })
    return model_rows


def _runtime_rows(
    root: Path,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for configuration in P4_CONFIGURATIONS:
        for seed in seeds:
            run = root / configuration / str(seed)
            metrics = _read_json(run / "validation_metrics.json")
            config = _read_json(run / "run_config.json")
            history = _read_csv(run / "history.csv")
            epoch_times = [float(row["epoch_seconds"]) for row in history]
            rows.append({
                "configuration": configuration,
                "seed": seed,
                "status": metrics["status"],
                "epochs_completed": metrics["epochs_completed"],
                "selected_epoch": metrics["selected_epoch"],
                "training_seconds": metrics["training_seconds"],
                "mean_epoch_seconds": float(np.mean(epoch_times)),
                "min_epoch_seconds": float(np.min(epoch_times)),
                "max_epoch_seconds": float(np.max(epoch_times)),
                "peak_sampled_rss_bytes": metrics["peak_sampled_rss_bytes"],
                "checkpoint_bytes": metrics["checkpoint_bytes"],
                "cpu_threads": config["cpu_threads"],
                "batch_size": config["batch_size"],
                "wall_clock_budget_seconds": config["wall_clock_budget_seconds"],
            })
    return rows


def _write_runtime_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 4 Runtime Report",
        "",
        "All rows are completed train/validation-only runs. `training_seconds` is",
        "the runner's accumulated training-loop time; per-epoch duration includes",
        "the validation pass. Final diagnostic evaluations and report writing are",
        "not included. RSS is sampled process RSS, not a hardware-counter peak.",
        "",
        "| Configuration | Seeds | Mean epochs | Mean seconds | Mean seconds/epoch | Max sampled RSS (bytes) | Max checkpoint (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for configuration in P4_CONFIGURATIONS:
        group = [row for row in rows if row["configuration"] == configuration]
        lines.append(
            f"| {configuration.upper()} | {len(group)} | "
            f"{np.mean([row['epochs_completed'] for row in group]):.2f} | "
            f"{np.mean([row['training_seconds'] for row in group]):.2f} | "
            f"{np.mean([row['mean_epoch_seconds'] for row in group]):.2f} | "
            f"{max(row['peak_sampled_rss_bytes'] for row in group)} | "
            f"{max(row['checkpoint_bytes'] for row in group)} |"
        )
    lines.extend([
        "",
        "The three-seed validation set is one video with 43 overlapping windows.",
        "No test split was loaded or evaluated. Runs used the exact settings saved",
        "in each `run_config.json`; seeds and epoch counts are recorded in",
        "`P4_RUNTIME_SUMMARY.csv`.",
        "",
        "Reproduction command (run from the repository root in PowerShell):",
        "",
        "```powershell",
        "python -m MTN.run_p4_screen --configurations p4_a p4_b p4_c p4_d p4_e p4_f --seeds 20260928 20260929 20260930 --max-epochs 30 --patience 10 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --cache-root results_paper\\splits\\gp6_from_scratch_v1\\roles_v1 --output-root results_paper\\P4_initial_screen_v1",
        "```",
        "",
    ])
    atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def _write_reports(
    root: Path,
    seeds: tuple[int, ...],
    seed_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    gate: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
) -> None:
    atomic_write_csv(root / "P4_VALIDATION_SUMMARY.csv", summary, list(summary[0]))
    atomic_write_csv(root / "P4_PAIRED_ABLATIONS.csv", paired, list(paired[0]))
    atomic_write_csv(root / "P4_GATE_UTILITY.csv", gate, list(gate[0]))
    atomic_write_csv(root / "P4_PARAMETER_EFFICIENCY.csv", efficiency, list(efficiency[0]))
    atomic_write_csv(root / "P4_RUNTIME_SUMMARY.csv", runtime, list(runtime[0]))

    baseline = _read_json(root / "P4_CV_BASELINE.json")
    _write_runtime_report(root / "P4_RUNTIME_REPORT.md", runtime)

    paired_lines = [
        "# Phase 4 Paired Ablation Report",
        "",
        "Primary paired contrasts use seed-level validation raw-pixel MSE and",
        "normalized MSE. Negative deltas favor the first-named model. The",
        "t-intervals and standardized paired effect sizes are descriptive with",
        "three seeds; no confirmatory significance claim is made. Track-cluster",
        "bootstrap intervals are secondary and based on one validation video.",
        "",
        "| Comparison | Metric | Mean paired delta | t 95% CI | d_z | First wins | Track bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        ci_t = (
            f"[{row['ci95_t_low']:.4g}, {row['ci95_t_high']:.4g}]"
            if row["ci95_t_low"] is not None else "n/a"
        )
        ci_b = (
            f"[{row['bootstrap_ci95_low']:.4g}, {row['bootstrap_ci95_high']:.4g}]"
            if row.get("bootstrap_ci95_low") is not None else "n/a"
        )
        paired_lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:.5g} | "
            f"{ci_t} | {row['dz'] if row['dz'] is not None else 'n/a'} | "
            f"{row['first_model_wins']}/{row['n_seeds']} | {ci_b} |"
        )
    paired_lines.extend([
        "",
        "The three-seed t intervals are highly uncertain and are not a substitute",
        "for more independent videos. The 43 validation windows overlap in time;",
        "pedestrian-track resampling does not resolve the single-video limitation.",
        "All results are validation-only and exploratory.",
        "",
    ])
    atomic_write_bytes(
        root / "P4_PAIRED_ABLATIONS.md", "\n".join(paired_lines).encode("utf-8")
    )

    gate_lines = [
        "# Phase 4 Gate Utility Report",
        "",
        "For each learned-gate checkpoint, forced-open and forced-closed",
        "predictions are post-hoc validation interventions. `open_beneficial`",
        "means forced-open raw MSE is lower than forced-closed raw MSE for that",
        "window. These interventions are not independent models and do not",
        "establish causal utility or generalization.",
        "",
        "| Model | Seed | Mean p | Hard-open | Gate grad L2 | Context grad L2 | Open beneficial | Closed beneficial | Decision matches | Corr(p, open benefit) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate:
        gate_gradient = row["mean_gate_gradient_l2"]
        context_gradient = row["mean_context_encoder_gradient_l2"]
        gate_lines.append(
            f"| {row['configuration']} | {row['seed']} | "
            f"{row['mean_admission_probability']:.4f} | "
            f"{row['hard_admission_rate']:.3f} | "
            f"{gate_gradient if gate_gradient is None else format(gate_gradient, '.6g')} | "
            f"{context_gradient if context_gradient is None else format(context_gradient, '.6g')} | "
            f"{row['open_beneficial_windows']} | "
            f"{row['closed_beneficial_windows']} | "
            f"{row['hard_decision_matches_better_intervention']} | "
            f"{row['probability_vs_open_benefit_pearson_r'] if row['probability_vs_open_benefit_pearson_r'] is not None else 'n/a'} |"
        )
    gate_lines.extend([
        "",
        "Only 43 overlapping validation windows from a single video are available",
        "per seed. Treat all gate-utility results as exploratory.",
        "",
    ])
    atomic_write_bytes(
        root / "P4_GATE_UTILITY.md", "\n".join(gate_lines).encode("utf-8")
    )

    efficiency_lines = [
        "# Phase 4 Parameter / Accuracy Efficiency",
        "",
        "Parameter counts are unique trainable parameters audited by module",
        "identity, not repeated state-dictionary entries. Accuracy is the mean",
        "selected-policy raw validation MSE across completed seeds. The table",
        "does not infer statistical equivalence or generalization.",
        "",
        "| Model | Parameters | Mean validation raw MSE | SD across seeds | Delta vs CV | Pareto efficient |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in efficiency:
        sd = row["validation_raw_pixel_mse_sd"]
        efficiency_lines.append(
            f"| {row['configuration']} | {row['unique_parameters']} | "
            f"{row['validation_raw_pixel_mse_mean']:.3f} | "
            f"{sd if sd is not None else 'n/a'} | "
            f"{row['delta_raw_mse_vs_cv']:.3f} | "
            f"{row['is_pareto_efficient']} |"
        )
    efficiency_lines.extend([
        "",
        "The CV point has zero learned parameters and is included as a baseline",
        "anchor. The validation split is one video; this efficiency frontier is",
        "descriptive only.",
        "",
    ])
    atomic_write_bytes(
        root / "P4_PARAMETER_EFFICIENCY.md",
        "\n".join(efficiency_lines).encode("utf-8"),
    )

    primary = {row["configuration"]: row for row in summary}
    report_lines = [
        "# Phase 4 Validation Results",
        "",
        "All runs use the role-separated cache and only train/validation data.",
        "The 43 validation windows come from one video and overlap in time.",
        "Three-seed comparisons are exploratory; no test metrics, test-based",
        "selection, or confirmatory significance claims are reported.",
        "",
        "| Configuration | Policy | Parameters | Epochs (mean) | Validation normalized MSE | Raw pixel MSE | C-MSE | CF-MSE | Correction RMS (px) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for configuration in P4_CONFIGURATIONS:
        row = primary[configuration]
        report_lines.append(
            f"| {configuration.upper()} | {row['policy']} | {row['unique_parameters']} | "
            f"{row['epochs_completed_mean']:.2f} | "
            f"{row['validation_normalized_mse_mean']:.6f} | "
            f"{row['validation_raw_pixel_mse_mean']:.3f} | "
            f"{row['validation_cmse_mean']:.3f} | {row['validation_cfmse_mean']:.3f} | "
            f"{row['mean_correction_rms_pixels_mean']:.3f} |"
        )
    report_lines.extend([
        "",
        f"Deterministic CV validation raw MSE: {baseline['validation']['raw_pixel_mse']:.3f}.",
        "",
        "Read `P4_PAIRED_ABLATIONS.md`, `P4_GATE_UTILITY.md`, and",
        "`P4_PARAMETER_EFFICIENCY.md` for paired and diagnostic analyses. No",
        "model-selection decision was based on test data. These limited-video",
        "results cannot establish robustness to video or dataset shift.",
        "",
    ])
    atomic_write_bytes(
        root / "P4_RESEARCH_REPORT.md", "\n".join(report_lines).encode("utf-8")
    )

    first_run = _read_json(root / P4_CONFIGURATIONS[0] / str(seeds[0]) / "run_config.json")
    metadata = {
        "report_version": "P4_finalization_v1",
        "validation_only": True,
        "test_split_loaded": False,
        "loaded_roles": ["train", "validation"],
        "normalization_fit_role": "train",
        "seeds": list(seeds),
        "configurations": list(P4_CONFIGURATIONS),
        "validation_sample_count_per_seed": 43,
        "validation_video_caveat": "single video with overlapping windows",
        "cluster_bootstrap_seed": 20260928,
        "cluster_bootstrap_repetitions": paired[0].get("bootstrap_repetitions"),
        "role_cache_hashes": first_run["role_cache_hashes"],
        "source_sha256": first_run["source_sha256"],
        "run_count_completed": len(seed_rows),
        "training_settings": {
            key: first_run[key]
            for key in (
                "max_epochs", "patience", "batch_size", "learning_rate",
                "optimizer", "loss", "cpu_threads", "wall_clock_budget_seconds",
            )
        },
    }
    atomic_write_json(root / "P4_METADATA.json", metadata)


def finalize(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260928,
) -> None:
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"Final Phase 4 summary requires the frozen seeds {DEFAULT_SEEDS}.")
    if bootstrap_repetitions < 1000:
        raise ValueError("Use at least 1,000 track-bootstrap repetitions.")
    runs = _require_complete_runs(output_root, seeds)
    seed_rows = _seed_summary(output_root, runs, seeds)
    summary = _validation_summary(seed_rows)
    paired = _paired_comparisons(
        output_root, runs, seeds, bootstrap_repetitions, bootstrap_seed
    )
    gate = _gate_utility(output_root, seeds)
    baseline = _read_json(output_root / "P4_CV_BASELINE.json")
    efficiency = _parameter_efficiency(summary, baseline)
    runtime = _runtime_rows(output_root, seeds)
    _write_reports(
        output_root, seeds, seed_rows, summary, paired, gate, efficiency, runtime
    )
    print(json.dumps({
        "status": "completed",
        "validation_only": True,
        "completed_runs": len(seed_rows),
        "output_root": str(output_root.resolve()),
        "bootstrap_repetitions": bootstrap_repetitions,
    }, indent=2), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed list contains duplicates.")
    finalize(
        args.output_root,
        tuple(args.seeds),
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
