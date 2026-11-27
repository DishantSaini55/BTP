"""Aggregate the frozen Phase 6B official-test evaluation."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)
T_CRITICAL_95_DF4 = 2.7764451051977987


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_paired_row(
    first: str,
    second: str,
    mse_by: dict[tuple[str, int], float],
) -> dict[str, object]:
    """Summarize one predeclared comparison at the appropriate seed level."""
    if first in ("AO-0", "AO-CP", "AO-RIDGE"):
        first_seeds = (20260928,)
    else:
        first_seeds = SEEDS
    if second in ("AO-0", "AO-CP", "AO-RIDGE"):
        second_seeds = (20260928,)
    else:
        second_seeds = SEEDS
    paired_seeds = first_seeds if len(first_seeds) == len(second_seeds) == 5 else (20260928,)
    differences = np.asarray([
        mse_by[(first, seed if first not in ("AO-0", "AO-CP", "AO-RIDGE") else 20260928)]
        - mse_by[(second, seed if second not in ("AO-0", "AO-CP", "AO-RIDGE") else 20260928)]
        for seed in paired_seeds
    ], dtype=np.float64)
    n_seeds = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1)) if n_seeds > 1 else 0.0
    half_width = (
        T_CRITICAL_95_DF4 * sd / math.sqrt(n_seeds)
        if n_seeds > 1 else 0.0
    )
    return {
        "comparison": f"{first} vs {second}",
        "n_seeds": n_seeds,
        "mean_delta_first_minus_second": mean,
        "sd_delta": sd,
        "ci95_t_low": mean - half_width,
        "ci95_t_high": mean + half_width,
        "dz": mean / sd if sd else None,
        "first_model_wins": int(np.sum(differences < 0)),
    }


def main() -> int:
    root = Path("results_paper/AO_confirmation_v1")
    result_dir = root / "OFFICIAL_TEST_RESULTS_v2"
    seed_rows = read_csv(result_dir / "OFFICIAL_TEST_SEED_RESULTS.csv")
    window_rows = read_csv(result_dir / "OFFICIAL_TEST_PER_WINDOW.csv")

    parameters = {
        "AO-0": 0,
        "AO-1": 76804,
        "AO-2": 76804,
        "AO-3": 68356,
        "AO-CP": 0,
        "AO-RIDGE": 11700,
    }
    model_ids = ("AO-0", "AO-1", "AO-2", "AO-3", "AO-CP", "AO-RIDGE")
    model_summary: list[dict[str, object]] = []
    for model_id in model_ids:
        rows = [row for row in seed_rows if row["model_id"] == model_id]
        values = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)
        model_summary.append({
            "model_id": model_id,
            "n_completed_runs": len(rows),
            "parameter_count": parameters[model_id],
            "mse_mean": float(values.mean()),
            "mse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "normalized_mse_mean": float(
                np.mean([float(row["normalized_mse"]) for row in rows])
            ),
            "cmse_mean": float(
                np.mean([float(row["cmse"]) for row in rows])
            ),
            "cfmse_mean": float(
                np.mean([float(row["cfmse"]) for row in rows])
            ),
            "fraction_improved_vs_cv_mean": float(
                np.mean([float(row["fraction_improved_vs_cv"]) for row in rows])
            ),
            "horizon_1_mse_mean": float(
                np.mean([float(row["horizon_1_mse"]) for row in rows])
            ),
            "horizon_15_mse_mean": float(
                np.mean([float(row["horizon_15_mse"]) for row in rows])
            ),
            "horizon_30_mse_mean": float(
                np.mean([float(row["horizon_30_mse"]) for row in rows])
            ),
            "horizon_45_mse_mean": float(
                np.mean([float(row["horizon_45_mse"]) for row in rows])
            ),
        })
    write_csv(root / "OFFICIAL_TEST_MODEL_SUMMARY_v3.csv", model_summary)

    mse_by = {
        (row["model_id"], int(row["seed"])): float(row["mse"])
        for row in seed_rows
    }
    paired: list[dict[str, object]] = []
    neural_pairs = (
        ("AO-2", "AO-1"),
        ("AO-2", "AO-0"),
        ("AO-3", "AO-0"),
        ("AO-3", "AO-2"),
    )
    for first, second in neural_pairs:
        paired.append(_seed_paired_row(first, second, mse_by))
    for first, second in (
        ("AO-RIDGE", "AO-0"),
        ("AO-RIDGE", "AO-2"),
        ("AO-RIDGE", "AO-3"),
    ):
        paired.append(_seed_paired_row(first, second, mse_by))
    write_csv(root / "OFFICIAL_TEST_PAIRED_COMPARISONS_v3.csv", paired)

    clusters = sorted(set(
        (row["video_id"], row["track_id"]) for row in window_rows
    ))
    bootstrap: list[dict[str, object]] = []
    rng = np.random.default_rng(20260928)
    cluster_mse: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in window_rows:
        cluster_mse[
            (
                row["model_id"],
                int(row["seed"]),
                row["video_id"],
                row["track_id"],
            )
        ].append(float(row["mse"]))
    pairs = (
        ("AO-2", "AO-1"),
        ("AO-2", "AO-0"),
        ("AO-3", "AO-0"),
        ("AO-3", "AO-2"),
        ("AO-RIDGE", "AO-0"),
        ("AO-RIDGE", "AO-2"),
        ("AO-RIDGE", "AO-3"),
    )
    for first, second in pairs:
        by_track: dict[tuple[str, str], list[float]] = defaultdict(list)
        if first == "AO-RIDGE":
            first_seeds = (20260928,)
        elif first in ("AO-0", "AO-CP"):
            first_seeds = (20260928,)
        else:
            first_seeds = SEEDS
        if second == "AO-RIDGE":
            second_seeds = (20260928,)
        elif second in ("AO-0", "AO-CP"):
            second_seeds = (20260928,)
        else:
            second_seeds = SEEDS
        paired_seeds = (
            first_seeds if len(first_seeds) == len(second_seeds) == 5
            else (20260928,)
        )
        for seed in paired_seeds:
            for cluster in clusters:
                first_values = cluster_mse[
                    (first, seed, cluster[0], cluster[1])
                ]
                second_seed = seed if second not in ("AO-0", "AO-CP", "AO-RIDGE") else 20260928
                second_values = cluster_mse[
                    (second, second_seed, cluster[0], cluster[1])
                ]
                if first_values and second_values:
                    by_track[cluster].append(
                        float(np.mean(first_values) - np.mean(second_values))
                    )
        estimates = []
        for _ in range(10000):
            selected = rng.integers(0, len(clusters), size=len(clusters))
            values = []
            for index in selected:
                values.extend(by_track[clusters[int(index)]])
            estimates.append(float(np.mean(values)))
        low, high = np.quantile(estimates, [0.025, 0.975])
        bootstrap.append({
            "comparison": f"{first} vs {second}",
            "n_track_clusters": len(clusters),
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": 20260928,
            "ci95_low": float(low),
            "ci95_high": float(high),
        })
    write_csv(root / "OFFICIAL_TEST_TRACK_BOOTSTRAP_v3.csv", bootstrap)

    cv_mean = mse_by[("AO-0", 20260928)]
    fresh: list[dict[str, object]] = []
    for model_id in ("AO-1", "AO-2", "AO-3"):
        values = np.asarray([
            mse_by[(model_id, seed)] for seed in SEEDS if seed != 20260928
        ], dtype=np.float64)
        fresh.append({
            "model_id": model_id,
            "seeds": "20260929,20260930,20260931,20260932",
            "individual_mse": ";".join(f"{value:.6f}" for value in values),
            "mean_mse": float(values.mean()),
            "sd_mse": float(values.std(ddof=1)),
            "cv_mean_mse": cv_mean,
            "relative_change_vs_cv_pct": 100.0 * (values.mean() - cv_mean) / cv_mean,
        })
    write_csv(root / "OFFICIAL_TEST_FRESH_SEED_SENSITIVITY_v3.csv", fresh)

    summary = {row["model_id"]: row for row in model_summary}
    lines = [
        "# Phase 6B Official-Test Evaluation Report",
        "",
        "**Status:** complete.",
        "",
        "- Eligible windows: 2,615",
        "- Eligible videos: 13",
        "- Track clusters: 861",
        "- Official test sets: GP3, GP5, GP8, GP9",
        "- Protocol: annotation-only 15 observed / 45 predicted frames",
        "- No training, test-set tuning or architecture update was performed.",
        "",
        "## Model means",
        "",
        "| Model | Runs | Parameters | MSE | SD | Normalized MSE | C-MSE | CF-MSE | Improved vs CV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in model_ids:
        row = summary[model_id]
        lines.append(
            f"| {model_id} | {row['n_completed_runs']} | {row['parameter_count']} | "
            f"{row['mse_mean']:.3f} | {row['mse_sd']:.3f} | "
            f"{row['normalized_mse_mean']:.6f} | {row['cmse_mean']:.3f} | "
            f"{row['cfmse_mean']:.3f} | "
            f"{100.0 * row['fraction_improved_vs_cv_mean']:.3f}% |"
        )
    cv = summary["AO-0"]["mse_mean"]
    lines.extend([
        "",
        "## Headline findings",
        "",
        f"- AO-RIDGE: MSE {summary['AO-RIDGE']['mse_mean']:.3f}; "
        f"{100.0 * (cv - summary['AO-RIDGE']['mse_mean']) / cv:.3f}% better than CV.",
        f"- AO-2: MSE {summary['AO-2']['mse_mean']:.3f}; "
        f"{100.0 * (cv - summary['AO-2']['mse_mean']) / cv:.3f}% better than CV.",
        f"- AO-3: MSE {summary['AO-3']['mse_mean']:.3f}; "
        f"{100.0 * (cv - summary['AO-3']['mse_mean']) / cv:.3f}% better than CV.",
        f"- AO-1: MSE {summary['AO-1']['mse_mean']:.3f}; "
        f"{100.0 * (summary['AO-1']['mse_mean'] - cv) / cv:.3f}% worse than CV.",
        f"- AO-CP: MSE {summary['AO-CP']['mse_mean']:.3f}.",
        "",
        "## Predeclared paired comparisons",
        "",
    ])
    for row in paired:
        if row["ci95_t_low"] is None:
            interval = "one-deterministic-result comparison"
        else:
            interval = f"95% t interval [{row['ci95_t_low']:.3f}, {row['ci95_t_high']:.3f}]"
        lines.append(
            f"- {row['comparison']}: mean first-minus-second "
            f"{row['mean_delta_first_minus_second']:.3f}; {interval}."
        )
    lines.extend([
        "",
        "## Track-cluster bootstrap",
        "",
        "The resampling unit is `(video_id, track_id)`; overlapping windows "
        "belonging to the same track are aggregated before resampling.",
        "",
    ])
    for row in bootstrap:
        lines.append(
            f"- {row['comparison']}: 95% "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] "
            "from 10,000 repetitions."
        )
    lines.extend([
        "",
        "## Fresh-seed sensitivity",
        "",
    ])
    for row in fresh:
        lines.append(
            f"- {row['model_id']}: mean {row['mean_mse']:.3f} +/- "
            f"{row['sd_mse']:.3f}; individual MSE {row['individual_mse']}; "
            f"change vs CV {row['relative_change_vs_cv_pct']:.3f}%."
        )
    lines.extend([
        "",
        "## Runtime",
        "",
        "The existing Phase 6A benchmark measured 99.049/114.574 ms for AO-2 "
        "and 49.395/102.235 ms for AO-3 at batches 1/8 on the Intel i5-8365U "
        "with four threads, 50 warm-ups and 500 timed iterations.",
        "",
        "The official-test wall-clock inference time was not recorded separately "
        "and is not reported as a measured value.",
        "",
        "## Limitations",
        "",
        "- All 13 official-test videos with locally available pedestrian XML "
        "were included; GP3/GP5/GP8/GP9 membership is complete for the downloaded annotation set.",
        "- Official raw videos were not needed and not opened.",
        "- The two fixed GP7 validation videos influenced ridge alpha selection.",
        "- Five seeds are stochastic realizations on the same test sample set.",
        "- Track bootstrap intervals are conditional on these 861 test clusters.",
        "- No equivalence claim is made.",
    ])
    (root / "OFFICIAL_TEST_FINAL_REPORT_v3.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "model_summary": model_summary,
        "paired": paired,
        "fresh": fresh,
        "bootstrap": bootstrap,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
