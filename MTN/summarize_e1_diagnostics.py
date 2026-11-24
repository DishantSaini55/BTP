"""Summarize E1 epoch-level diagnostics without using test curves for selection."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"


def main() -> None:
    paths = sorted(glob.glob(str(RESULTS / "runs" / "E1_DIAGNOSTICS" / "*" / "*" / "diagnostics.json")))
    rows = [json.loads(Path(path).read_text()) for path in paths]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 diagnostics, found {len(rows)}")

    summary = RESULTS / "summary"
    summary.mkdir(exist_ok=True)
    fields = list(rows[0])
    with (summary / "E1_DIAGNOSTICS_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    group = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        gaps = [abs(row["epoch_gap"]) for row in subset]
        group.append({
            "model": model,
            "runs": len(subset),
            "mean_selected_epoch": float(np.mean([row["selected_epoch_validation_only"] for row in subset])),
            "mean_test_optimal_epoch": float(np.mean([row["test_optimal_epoch_not_used_for_selection"] for row in subset])),
            "mean_abs_epoch_gap": float(np.mean(gaps)),
            "max_abs_epoch_gap": int(max(gaps)),
            "far_gt_5_epochs": int(sum(gap > 5 for gap in gaps)),
        })
    with (summary / "E1_DIAGNOSTICS_aggregate.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group[0]))
        writer.writeheader()
        writer.writerows(group)

    e3 = json.loads((summary / "E3_gate_behaviour.json").read_text())
    x_ranges = e3["x_t_normalized_ranges"]
    seed32 = next(row for row in rows if row["model"] == "MTN (baseline)" and row["seed"] == 20260932)

    lines = [
        "# E1 epoch and split diagnostics",
        "",
        "These reruns are diagnostics only. Test curves were saved but were never used to select a checkpoint.",
        "",
        "## Aggregate checkpoint gap",
        "",
        "| Model | Runs | Mean selected epoch | Mean test-optimal epoch | Mean absolute gap | Max absolute gap | Seeds with |gap| > 5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in group:
        lines.append(
            f"| {item['model']} | {item['runs']} | {item['mean_selected_epoch']:.2f} | "
            f"{item['mean_test_optimal_epoch']:.2f} | {item['mean_abs_epoch_gap']:.2f} | "
            f"{item['max_abs_epoch_gap']} | {item['far_gt_5_epochs']} |"
        )
    lines += [
        "",
        "## MTN seed 20260932",
        "",
        f"- Validation-selected epoch: {seed32['selected_epoch_validation_only']}.",
        f"- Test-optimal epoch: {seed32['test_optimal_epoch_not_used_for_selection']}.",
        f"- Epoch gap: {seed32['epoch_gap']}.",
        f"- Selected test normalized MSE: {seed32['selected_test_normalized_mse']:.6f}; test raw MSE: {seed32['selected_test_raw_mse']:.6f}.",
        f"- Test-optimal normalized MSE: {seed32['test_optimal_normalized_mse']:.6f}; test raw MSE: {seed32['test_optimal_raw_mse']:.6f}.",
        "- This diagnostic run stopped early at 44 epochs, not at 50. The original seed 20260932 run stopped at 50 epochs. The shift from GP6 validation video 1 to test video 3 makes the validation-selected checkpoint diagnostic evidence, but it is not by itself proof that seed 20260932 failed solely because of early stopping; the two reruns also differ in completed-epoch behavior.",
        "",
        "## Normalized x_T ranges",
        "",
        f"- Train: {x_ranges['train']}",
        f"- Validation: {x_ranges['validation']}",
        f"- Test: {x_ranges['test']}",
        "",
        "## Baseline errors by split",
        "",
        "| Baseline | Train MSE / C-MSE / CF-MSE | Validation MSE / C-MSE / CF-MSE | Test MSE / C-MSE / CF-MSE |",
        "|---|---|---|---|",
        "| Constant position | 4923.146615 / 4801.150281 / 19888.327424 | 16953.994406 / 15852.451038 / 60252.517385 | 4696.976911 / 4597.240027 / 13682.455598 |",
        "| Constant velocity (last 3) | 1513.182154 / 1390.725872 / 7250.118652 | 3630.315279 / 3074.487717 / 15769.099758 | 1496.018040 / 655.570860 / 1948.649039 |",
    ]
    (summary / "E1_DIAGNOSTICS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
