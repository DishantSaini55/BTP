"""Create auditable E1 summaries after all 50 from-scratch runs complete."""
from __future__ import annotations

import csv
import json
from pathlib import Path

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
    # Positive difference means the named left model has higher (worse) error.
    difference = left - right
    n = len(difference)
    mean = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(n)
    critical = float(stats.t.ppf(.975, n - 1))
    test = stats.ttest_rel(left, right)
    return {
        "n_seeds": n, "mean_difference_left_minus_right": mean,
        "ci95": [mean - critical * standard_error, mean + critical * standard_error],
        "paired_t_p": float(test.pvalue), "cohen_dz": mean / standard_deviation if standard_deviation else None,
        "left_lower_error_seeds": int((difference < 0).sum()),
        "wilcoxon_p": float(stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue)
                      if np.any(difference) else 1.0,
    }


def block_bootstrap(per_sample: dict[str, dict[str, float]], left: str, right: str) -> dict:
    """Secondary one-video bootstrap, with moving two-window blocks within tracks."""
    rows = []
    for sample_id, values in per_sample.items():
        identity = sample_id.split("/")
        rows.append((identity[2], int(identity[3]), values[left] - values[right]))
    rows.sort()
    blocks = []
    for pid in sorted({row[0] for row in rows}):
        track = [row[2] for row in rows if row[0] == pid]
        blocks.extend([track[index:index + 2] for index in range(len(track))])
    rng = np.random.default_rng(20260928)
    count = len(rows)
    observed = float(np.mean([row[2] for row in rows]))
    values = np.empty(10_000)
    draws = int(np.ceil(count / 2))
    for index in range(len(values)):
        picked = [value for block in rng.choice(len(blocks), draws, replace=True) for value in blocks[block]][:count]
        values[index] = np.mean(picked)
    return {"comparison": f"{left} minus {right}", "observed_mean_difference": observed,
            "ci95": np.quantile(values, [.025, .975]).tolist(), "resamples": len(values),
            "block_length_selected_windows": 2,
            "caveat": "All 217 windows are from one video; this is exploratory and is not cross-video inference."}


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
            record = {"model": name, "seed": seed, **metrics}
            records.append(record)
            with (run / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for video in sorted({row["video"] for row in rows}):
                subset = [row for row in rows if row["video"] == video]
                per_video.append({"model": name, "seed": seed, "video": video, **{metric: float(np.mean([float(row[metric]) for row in subset])) for metric in ("mse", "cmse", "cfmse")}, "windows": len(subset)})
            for row in rows:
                per_sample.setdefault(row["sample_id"], {}).setdefault(name, []).append(float(row["mse"]))
    if len(records) != len(NAMES) * len(SEEDS):
        raise ValueError("Incomplete E1 runs")
    per_sample_mean = {sample: {name: float(np.mean(values)) for name, values in models.items()} for sample, models in per_sample.items()}
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
        group.append({"model": name, "seeds": len(subset), **{metric + "_mean": float(np.mean([record[metric] for record in subset])) for metric in ("mse", "cmse", "cfmse")}, **{metric + "_sample_std": float(np.std([record[metric] for record in subset], ddof=1)) for metric in ("mse", "cmse", "cfmse")}})
    with (summary / "E1_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group[0])); writer.writeheader(); writer.writerows(group)
    comparisons = []
    for left_index, left in enumerate(NAMES):
        for right in NAMES[left_index + 1:]:
            left_values = np.array([next(record["mse"] for record in records if record["model"] == left and record["seed"] == seed) for seed in SEEDS])
            right_values = np.array([next(record["mse"] for record in records if record["model"] == right and record["seed"] == seed) for seed in SEEDS])
            comparisons.append({"left": left, "right": right, **paired(left_values, right_values)})
    adjusted = holm([item["paired_t_p"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = value
    bootstraps = [block_bootstrap(per_sample_mean, NAMES[1], NAMES[0]), block_bootstrap(per_sample_mean, NAMES[2], NAMES[0])]
    (summary / "E1_statistics.json").write_text(json.dumps({"primary": {"unit": "seed", "test": "two-sided paired t test, Holm corrected over all ten E1 MSE model-pair comparisons", "comparisons": comparisons}, "secondary": bootstraps}, indent=2), encoding="utf-8")
    lines = ["# E1 from-scratch summary", "", "All values below are generated from the 50 run directories. Primary inference is seed-level; the window bootstrap is explicitly secondary because there is one test video.", "", "| Model | MSE mean ± sample SD | C-MSE mean ± sample SD | CF-MSE mean ± sample SD |", "|---|---:|---:|---:|"]
    for item in group:
        lines.append(f"| {item['model']} | {item['mse_mean']:.6f} ± {item['mse_sample_std']:.6f} | {item['cmse_mean']:.6f} ± {item['cmse_sample_std']:.6f} | {item['cfmse_mean']:.6f} ± {item['cfmse_sample_std']:.6f} |")
    (summary / "E1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
