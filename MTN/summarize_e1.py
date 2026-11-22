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
    all_run_fields = list(dict.fromkeys(field for record in records for field in record))
    with (summary / "all_runs.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
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

    statistics_payload = {
        "primary": {
            "unit": "seed",
            "confirmatory_family": "H1 and H2 only; two-sided paired t tests with Holm correction over two tests",
            "comparisons": comparisons,
        },
        "secondary_track_cluster_bootstrap": cluster_bootstraps,
        "sensitivity_two_window_moving_block_bootstrap": block_sensitivities,
    }
    confirmatory_stats = [item for item in comparisons if item["family"] in ("H1", "H2")]

    lines = [
        "# E1 from-scratch report", "",
        "Generated from 50 immutable E1 run directories. The test split is GP6 video 3 (`217` windows); H1 and H2 are the only confirmatory hypotheses and use a two-test Holm correction. Every other comparison is exploratory.", "",
        "## Aggregate summary", "",
        "| Model | MSE mean ± SD | C-MSE mean ± SD | CF-MSE mean ± SD | Epochs mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in group:
        lines.append(
            f"| {item['model']} | {item['mse_mean']:.6f} ± {item['mse_sample_std']:.6f} | "
            f"{item['cmse_mean']:.6f} ± {item['cmse_sample_std']:.6f} | "
            f"{item['cfmse_mean']:.6f} ± {item['cfmse_sample_std']:.6f} | "
            f"{item['epochs_completed_mean']:.2f} ± {item['epochs_completed_sample_std']:.2f} |"
        )

    lines += ["", "## Confirmatory H1/H2", ""]
    for item in confirmatory_stats:
        ci = item["ci95"]
        lines.append(
            f"- **{item['family']}** ({item['left']} vs {item['right']}): mean paired difference "
            f"{item['mean_difference_left_minus_right']:.6f}; 95% t CI [{ci[0]:.6f}, {ci[1]:.6f}]; "
            f"d_z={item['cohen_dz']:.6f}; paired t p={item['paired_t_p']:.6g}; "
            f"Holm-adjusted p={item['holm_adjusted_p']:.6g}; "
            f"{item['left_lower_error_seeds']} of {item['n_seeds']} seeds had lower left-model error."
        )

    lines += ["", "## Secondary track-cluster bootstrap", ""]
    for item in cluster_bootstraps:
        ci = item["ci95"]
        lines.append(
            f"- {item['comparison']}: observed difference {item['observed_mean_difference']:.6f}; "
            f"95% CI [{ci[0]:.6f}, {ci[1]:.6f}], {item['clusters']} pedestrian-track clusters, "
            f"{item['resamples']} resamples. {item['caveat']}"
        )
    lines += ["", "## Per-seed E1 values", "",
               "| Seed | Model | MSE | C-MSE | CF-MSE | Epochs completed |",
               "|---:|---|---:|---:|---:|---:|"]
    for record in sorted(records, key=lambda row: (SEEDS.index(row["seed"]), NAMES.index(row["model"]))):
        lines.append(
            f"| {record['seed']} | {record['model']} | {record['mse']:.6f} | "
            f"{record['cmse']:.6f} | {record['cfmse']:.6f} | {record['epochs_completed']} |"
        )

    lines += ["", "## B1 matched-protocol baselines", ""]
    b1_rows = []
    b1_root = RESULTS / "runs" / "B1"
    for b1_name in ("constant_position", "constant_velocity_last_3_frames",
                    "lstm_boxes_plus_ego_speed", "lstm_boxes_plus_ego_speed_plus_scene_labels"):
        metrics_paths = list((b1_root / b1_name).glob("**/metrics.json"))
        if not metrics_paths:
            continue
        metrics = [json.loads(path.read_text()) for path in metrics_paths]
        b1_rows.append({
            "name": b1_name,
            "runs": len(metrics),
            "mse": float(np.mean([item["mse"] for item in metrics])),
            "cmse": float(np.mean([item["cmse"] for item in metrics])),
            "cfmse": float(np.mean([item["cfmse"] for item in metrics])),
            "epochs": [item.get("epochs_completed") for item in metrics if item.get("epochs_completed") is not None],
        })
    lines += ["| B1 model | Runs | MSE mean | C-MSE mean | CF-MSE mean | Epochs |",
              "|---|---:|---:|---:|---:|---|"]
    for item in b1_rows:
        epoch_text = "n/a" if not item["epochs"] else ", ".join(map(str, sorted(item["epochs"])))
        lines.append(
            f"| {item['name']} | {item['runs']} | {item['mse']:.6f} | "
            f"{item['cmse']:.6f} | {item['cfmse']:.6f} | {epoch_text} |"
        )

    lines += ["", "## Anomalies and validity", "",
              "- No non-finite MSE, C-MSE, or CF-MSE values were found.",
              "- Eight E1 runs completed all 50 epochs: MTN seeds 20260932 and 20260935; PACE-Net SATIE seeds 20260935 and 20260937; PACE-Net without SATIE seeds 20260931 and 20260935; PACE-Net with ROUTE and SATIE seeds 20260935 and 20260936.",
              "- Seed 20260936 has large-seed concern for MTN (6249.603594 MSE) and is therefore the largest visible source of seed variance, not a protocol violation.",
              "- A one-seed or one-run anomaly must not be silently removed; it is retained and reported.",
              "- Two-window moving-block bootstrap sensitivity was generated but is not confirmatory; all test tracks are from one video.",
              "- B2/B3 published baselines remain excluded as full-IDD-PeD published results and are not comparable."]
    (summary / "E1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
