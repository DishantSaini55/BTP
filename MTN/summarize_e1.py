"""Create auditable E1 summaries after all 50 from-scratch runs complete."""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy import stats

try:
    from .run_from_scratch_e1 import NAMES, RESULTS, SEEDS, slug
except ImportError:
    from run_from_scratch_e1 import NAMES, RESULTS, SEEDS, slug


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired(left: np.ndarray, right: np.ndarray) -> dict:
    """Seed-level paired statistics; positive means left has higher error."""
    difference = left - right
    n = len(difference)
    mean = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(n)
    critical = float(stats.t.ppf(.975, n - 1))
    test = stats.ttest_rel(left, right)
    return {
        "n_seeds": n,
        "mean_difference_left_minus_right": mean,
        "ci95": [mean - critical * standard_error, mean + critical * standard_error],
        "paired_t_p": float(test.pvalue),
        "cohen_dz": mean / standard_deviation if standard_deviation else None,
        "left_lower_error_seeds": int((difference < 0).sum()),
        "wilcoxon_p": float(stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue)
        if np.any(difference) else 1.0,
    }


def differences_by_track(per_sample: dict[str, dict[str, float]], left: str, right: str) -> dict[str, list[float]]:
    """Seed-averaged MSE differences grouped by pedestrian identifier."""
    records = []
    for sample_id, values in per_sample.items():
        _, _, pid, frame_start, _ = sample_id.split("/")
        records.append((pid, int(frame_start), values[left] - values[right]))
    records.sort()
    tracks: dict[str, list[float]] = {}
    for pid, _frame_start, difference in records:
        tracks.setdefault(pid, []).append(difference)
    return tracks


def track_cluster_bootstrap(per_sample: dict[str, dict[str, float]], left: str, right: str) -> dict:
    """Secondary bootstrap resampling whole tracks, retaining every selected window."""
    tracks = differences_by_track(per_sample, left, right)
    clusters = list(tracks.values())
    observed = float(np.mean([value for cluster in clusters for value in cluster]))
    rng = np.random.default_rng(20260928)
    estimates = np.empty(10_000)
    for index in range(len(estimates)):
        sampled = [value for cluster_index in rng.choice(len(clusters), len(clusters), replace=True)
                   for value in clusters[cluster_index]]
        estimates[index] = np.mean(sampled)
    return {
        "comparison": f"{left} minus {right}",
        "observed_mean_difference": observed,
        "ci95": np.quantile(estimates, [.025, .975]).tolist(),
        "resamples": len(estimates),
        "resampling_unit": "whole pedestrian track (pid); all sampled-track windows retained",
        "clusters": len(clusters),
        "caveat": "All test tracks are from one video; this is exploratory and is not cross-video inference.",
    }


def two_window_block_sensitivity(per_sample: dict[str, dict[str, float]], left: str, right: str) -> dict:
    """Sensitivity-only moving-block bootstrap, not an inferential primary result."""
    tracks = differences_by_track(per_sample, left, right)
    blocks = [cluster[index:index + 2] for cluster in tracks.values() for index in range(len(cluster))]
    count = sum(map(len, tracks.values()))
    observed = float(np.mean([value for cluster in tracks.values() for value in cluster]))
    rng = np.random.default_rng(20260928)
    estimates = np.empty(10_000)
    for index in range(len(estimates)):
        sampled = [value for block_index in rng.choice(len(blocks), int(np.ceil(count / 2)), replace=True)
                   for value in blocks[block_index]][:count]
        estimates[index] = np.mean(sampled)
    return {
        "comparison": f"{left} minus {right}",
        "observed_mean_difference": observed,
        "ci95": np.quantile(estimates, [.025, .975]).tolist(),
        "resamples": len(estimates),
        "block_length_selected_windows": 2,
        "role": "sensitivity check only",
        "caveat": "All windows are from one video; this is exploratory and is not cross-video inference.",
    }


def values_for(records: list[dict], model: str) -> np.ndarray:
    return np.array([
        next(record["mse"] for record in records if record["model"] == model and record["seed"] == seed)
        for seed in SEEDS
    ])


def main() -> None:
    summary = RESULTS / "summary"
    if summary.exists():
        raise FileExistsError(f"Refusing to overwrite {summary}")
    records, per_video, per_sample = [], [], {}
    for name in NAMES:
        for seed in SEEDS:
            run = RESULTS / "runs" / "E1" / slug(name) / str(seed)
            metrics = json.loads((run / "metrics.json").read_text())
            if metrics["test_sample_id_sha256"] != "6e19d53af0552d271ec1ec7dfd63c0a9aaa803d52250b018fdcbdbf655a71dbe":
                raise ValueError(f"Test-ID hash mismatch: {run}")
            records.append({"model": name, "seed": seed, **metrics})
            with (run / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for video in sorted({row["video"] for row in rows}):
                subset = [row for row in rows if row["video"] == video]
                per_video.append({
                    "model": name, "seed": seed, "video": video, "windows": len(subset),
                    **{metric: float(np.mean([float(row[metric]) for row in subset]))
                       for metric in ("mse", "cmse", "cfmse")},
                })
            for row in rows:
                per_sample.setdefault(row["sample_id"], {}).setdefault(name, []).append(float(row["mse"]))
    if len(records) != len(NAMES) * len(SEEDS):
        raise ValueError("Incomplete E1 runs")
    per_sample_mean = {sample: {name: float(np.mean(values)) for name, values in models.items()}
                       for sample, models in per_sample.items()}
    if any(set(values) != set(NAMES) for values in per_sample_mean.values()):
        raise ValueError("Per-sample model coverage mismatch")

    summary.mkdir(parents=True)
    with (summary / "all_runs.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (summary / "per_video_E1.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0])); writer.writeheader(); writer.writerows(per_video)

    group = []
    for name in NAMES:
        subset = [record for record in records if record["model"] == name]
        group.append({
            "model": name, "seeds": len(subset),
            **{metric + "_mean": float(np.mean([record[metric] for record in subset]))
               for metric in ("mse", "cmse", "cfmse", "epochs_completed")},
            **{metric + "_sample_std": float(np.std([record[metric] for record in subset], ddof=1))
               for metric in ("mse", "cmse", "cfmse", "epochs_completed")},
        })
    with (summary / "E1_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group[0])); writer.writeheader(); writer.writerows(group)

    # Only H1 and H2 are confirmatory and receive a two-test Holm correction.
    confirmatory = [("H1", NAMES[1], NAMES[0]), ("H2", NAMES[2], NAMES[1])]
    confirmatory_pairs = {frozenset((left, right)) for _, left, right in confirmatory}
    comparison_specs = list(confirmatory)
    for left_index, left in enumerate(NAMES):
        for right in NAMES[left_index + 1:]:
            if frozenset((left, right)) not in confirmatory_pairs:
                comparison_specs.append(("exploratory", left, right))
    comparisons = [
        {"family": family, "left": left, "right": right, **paired(values_for(records, left), values_for(records, right))}
        for family, left, right in comparison_specs
    ]
    adjusted = holm([item["paired_t_p"] for item in comparisons if item["family"] != "exploratory"])
    for item, adjusted_p in zip((item for item in comparisons if item["family"] != "exploratory"), adjusted):
        item["holm_adjusted_p"] = adjusted_p

    cluster_bootstraps = [track_cluster_bootstrap(per_sample_mean, left, right) for _, left, right in confirmatory]
    block_sensitivities = [two_window_block_sensitivity(per_sample_mean, left, right) for _, left, right in confirmatory]
    (summary / "E1_statistics.json").write_text(json.dumps({
        "primary": {
            "unit": "seed",
            "confirmatory_family": "H1 and H2 only; two-sided paired t tests with Holm correction over two tests",
            "comparisons": comparisons,
        },
        "secondary_track_cluster_bootstrap": cluster_bootstraps,
        "sensitivity_two_window_moving_block_bootstrap": block_sensitivities,
    }, indent=2), encoding="utf-8")

    lines = [
        "# E1 from-scratch summary", "",
        "All values below are generated from the 50 run directories. Confirmatory inference is seed-level for H1 and H2 only, with Holm correction over those two tests. The remaining eight pairs are exploratory and report unadjusted p values. Track-cluster and moving-block bootstraps are secondary because all test tracks come from one video.", "",
        "| Model | MSE mean ± sample SD | C-MSE mean ± sample SD | CF-MSE mean ± sample SD | Epochs mean ± sample SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in group:
        lines.append(
            f"| {item['model']} | {item['mse_mean']:.6f} ± {item['mse_sample_std']:.6f} | "
            f"{item['cmse_mean']:.6f} ± {item['cmse_sample_std']:.6f} | "
            f"{item['cfmse_mean']:.6f} ± {item['cfmse_sample_std']:.6f} | "
            f"{item['epochs_completed_mean']:.2f} ± {item['epochs_completed_sample_std']:.2f} |"
        )
    (summary / "E1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
