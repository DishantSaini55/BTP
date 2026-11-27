"""Create validation-only trajectory plots using the predeclared P3 selection rule."""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _array_index(archive: dict[str, np.ndarray], sample_ids: np.ndarray) -> dict[str, int]:
    return {str(sample_id): index for index, sample_id in enumerate(sample_ids)}


def _bbox_center(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2


def _svg_case(
    parts: list[str],
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    sample_id: str,
    observed: np.ndarray,
    target: np.ndarray,
    constant_position: np.ndarray,
    cv_prediction: np.ndarray,
    mtn_cv: np.ndarray,
    pace_cv: np.ndarray,
    raw_mse: float,
    cv_mse: float,
    improvement: float,
) -> None:
    anchor = observed[-1]
    paths = (
        ("Constant position", constant_position, "#8C8C8C", "6 4"),
        ("Constant velocity", cv_prediction, "#F58518", "2 4"),
        ("MTN + CV residual", mtn_cv, "#54A24B", "8 3 2 3"),
        ("PACE-Net + CV residual", pace_cv, "#B279A2", ""),
        ("Ground truth", target, "#E45756", "10 3"),
    )
    all_boxes = np.concatenate((observed, target, constant_position, cv_prediction, mtn_cv, pace_cv))
    x_min, y_min = all_boxes[:, :2].min(axis=0)
    x_max, y_max = all_boxes[:, 2:].max(axis=0)
    x_span, y_span = max(float(x_max - x_min), 1.0), max(float(y_max - y_min), 1.0)
    x_margin, y_margin = max(8.0, 0.12 * x_span), max(8.0, 0.12 * y_span)
    x_min, x_max = float(x_min - x_margin), float(x_max + x_margin)
    y_min, y_max = float(y_min - y_margin), float(y_max + y_margin)
    pad_left, pad_right, pad_top, pad_bottom = 48.0, 14.0, 42.0, 36.0
    plot_width, plot_height = width - pad_left - pad_right, height - pad_top - pad_bottom
    scale = min(plot_width / (x_max - x_min), plot_height / (y_max - y_min))
    drawn_width, drawn_height = scale * (x_max - x_min), scale * (y_max - y_min)
    left = origin_x + pad_left + (plot_width - drawn_width) / 2
    top = origin_y + pad_top + (plot_height - drawn_height) / 2

    def point(x: float, y: float) -> tuple[float, float]:
        return left + (x - x_min) * scale, top + (y_max - y) * scale

    def path_data(boxes: np.ndarray) -> str:
        cx, cy = _bbox_center(boxes)
        coordinates = [point(float(x), float(y)) for x, y in zip(cx, cy)]
        return " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(coordinates)
        )

    title = escape(sample_id.rsplit("/", 2)[-2] + "/" + sample_id.rsplit("/", 1)[-1])
    parts.append(
        f'<text x="{origin_x + 8}" y="{origin_y + 17}" class="title">{title}</text>'
        f'<text x="{origin_x + 8}" y="{origin_y + 32}" class="subtitle">'
        f'PACE MSE={raw_mse:.1f}; CV MSE={cv_mse:.1f}; CV-PACE={improvement:.1f}</text>'
    )
    parts.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{drawn_width:.2f}" '
        f'height="{drawn_height:.2f}" class="plot"/>'
    )
    for tick in range(1, 4):
        x = left + drawn_width * tick / 4
        y = top + drawn_height * tick / 4
        parts.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" '
            f'y2="{top + drawn_height:.2f}" class="grid"/>'
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + drawn_width:.2f}" '
            f'y2="{y:.2f}" class="grid"/>'
        )
    for index, box in enumerate(observed):
        x1, y1 = point(float(box[0]), float(box[1]))
        x2, y2 = point(float(box[2]), float(box[3]))
        alpha = 0.18 + 0.7 * (index + 1) / len(observed)
        parts.append(
            f'<rect x="{min(x1, x2):.2f}" y="{min(y1, y2):.2f}" '
            f'width="{abs(x2 - x1):.2f}" height="{abs(y2 - y1):.2f}" '
            f'stroke="#4C78A8" stroke-opacity="{alpha:.3f}" class="bbox"/>'
        )
    observed_centers = _bbox_center(observed)
    points = " ".join(
        f"{point(float(x), float(y))[0]:.2f},{point(float(x), float(y))[1]:.2f}"
        for x, y in zip(*observed_centers)
    )
    parts.append(f'<polyline points="{points}" stroke="#4C78A8" class="observed"/>')
    start = _bbox_center(anchor)
    for label, sequence, color, dash in paths:
        start_point = point(float(start[0]), float(start[1]))
        data = path_data(np.concatenate((anchor[None, :], sequence), axis=0))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<path d="{data}" fill="none" stroke="{color}"{dash_attr} class="forecast"/>'
        )
        final_box = sequence[-1]
        x1, y1 = point(float(final_box[0]), float(final_box[1]))
        x2, y2 = point(float(final_box[2]), float(final_box[3]))
        parts.append(
            f'<rect x="{min(x1, x2):.2f}" y="{min(y1, y2):.2f}" '
            f'width="{abs(x2 - x1):.2f}" height="{abs(y2 - y1):.2f}" '
            f'stroke="{color}" class="final-box"/>'
        )
    parts.append(
        f'<text x="{origin_x + width / 2:.2f}" y="{origin_y + height - 7:.2f}" '
        f'class="axis" text-anchor="middle">x (pixels); y increases downward</text>'
    )


def _save_grid(
    output: Path,
    selected_ids: list[str],
    cv_rows: dict[str, dict[str, str]],
    pace_rows: dict[str, dict[str, str]],
    arrays: dict[str, dict[str, np.ndarray]],
    columns: int = 3,
) -> None:
    rows = int(np.ceil(len(selected_ids) / columns))
    cell_width, cell_height = 500, 390
    legend_height = 72
    width, height = columns * cell_width, rows * cell_height + legend_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>.title{font: bold 12px Arial,sans-serif;fill:#172B4D}'
        '.subtitle{font: 10px Arial,sans-serif;fill:#42526E}'
        '.plot{fill:#fff;stroke:#AAB7C4;stroke-width:1}'
        '.grid{stroke:#E6EAF0;stroke-width:1}'
        '.bbox{fill:none;stroke-width:1}'
        '.observed{fill:none;stroke-width:2}'
        '.forecast{stroke-width:2}'
        '.final-box{fill:none;stroke-width:2.4}'
        '.axis{font:10px Arial,sans-serif;fill:#42526E}'
        '.legend{font:11px Arial,sans-serif;fill:#172B4D}</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for index, sample_id in enumerate(selected_ids):
        cv_index = arrays["p3_a"]["index"][sample_id]
        pace_index = arrays["p3_e"]["index"][sample_id]
        observed = arrays["p3_a"]["observed"][cv_index]
        target = arrays["p3_a"]["target"][cv_index]
        constant_position = arrays["p3_b"]["prediction"][
            arrays["p3_b"]["index"][sample_id]
        ]
        cv_prediction = arrays["p3_a"]["prediction"][cv_index]
        mtn_cv = arrays["p3_c"]["prediction"][
            arrays["p3_c"]["index"][sample_id]
        ]
        pace_cv = arrays["p3_e"]["prediction"][pace_index]
        pace_row, cv_row = pace_rows[sample_id], cv_rows[sample_id]
        _svg_case(
            parts,
            (index % columns) * cell_width,
            (index // columns) * cell_height,
            cell_width,
            cell_height,
            sample_id,
            observed,
            target,
            constant_position,
            cv_prediction,
            mtn_cv,
            pace_cv,
            float(pace_row["raw_pixel_mse"]),
            float(cv_row["raw_pixel_mse"]),
            float(cv_row["raw_pixel_mse"]) - float(pace_row["raw_pixel_mse"]),
        )
    legend = (
        ("Observed", "#4C78A8"),
        ("Constant position", "#8C8C8C"),
        ("Constant velocity", "#F58518"),
        ("MTN + CV residual", "#54A24B"),
        ("PACE-Net + CV residual", "#B279A2"),
        ("Ground truth", "#E45756"),
    )
    legend_y = height - legend_height + 24
    slot = width / len(legend)
    for index, (label, color) in enumerate(legend):
        x = 14 + index * slot
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 25}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<text x="{x + 31}" y="{legend_y + 4}" class="legend">{escape(label)}</text>'
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def make_figures(output_root: Path) -> dict[str, list[str]]:
    comparison_path = output_root / "P3_PER_WINDOW_COMPARISON.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing completed validation comparison: {comparison_path}")
    rows = _read_rows(comparison_path)
    by_config_policy = {
        "p3_a": {}, "p3_b": {}, "p3_c": {}, "p3_e": {},
    }
    for row in rows:
        configuration, policy = row["configuration"], row["policy"]
        if configuration in by_config_policy and (
            (configuration == "p3_a" and policy == "exact_b1_formula")
            or (configuration == "p3_b" and policy == "base")
            or (configuration == "p3_c" and policy == "base")
            or (configuration == "p3_e" and policy == "hard")
        ):
            by_config_policy[configuration][row["sample_id"]] = row
    needed = ("p3_a", "p3_b", "p3_c", "p3_e")
    if any(not by_config_policy[key] for key in needed):
        raise RuntimeError("Qualitative plots require completed P3-A/B/C/E validation outputs.")
    common = set.intersection(*(set(by_config_policy[key]) for key in needed))
    candidates = sorted(common)
    if len(candidates) < 6:
        raise RuntimeError("Fewer than six matching validation windows are available.")

    pids: dict[str, list[str]] = {}
    for sample_id in candidates:
        pid = by_config_policy["p3_a"][sample_id]["pid"]
        pids.setdefault(pid, []).append(sample_id)
    rng = np.random.default_rng(20260928)
    distinct_pids = sorted(pids)
    rng.shuffle(distinct_pids)
    random_ids = [pids[pid][int(rng.integers(len(pids[pid])))] for pid in distinct_pids[:6]]
    remaining = [sample_id for sample_id in candidates if sample_id not in set(random_ids)]
    improvement = {
        sample_id: (
            float(by_config_policy["p3_a"][sample_id]["raw_pixel_mse"])
            - float(by_config_policy["p3_e"][sample_id]["raw_pixel_mse"])
        )
        for sample_id in remaining
    }
    best_ids = sorted(remaining, key=lambda value: (-improvement[value], value))[:2]
    remaining = [sample_id for sample_id in remaining if sample_id not in set(best_ids)]
    worst_ids = sorted(remaining, key=lambda value: (improvement[value], value))[:2]
    selection = {
        "random_distinct_pedestrians": random_ids,
        "largest_validation_improvements_over_cv": best_ids,
        "largest_validation_degradations_vs_cv": worst_ids,
    }
    selection_path = output_root / "P3_QUALITATIVE_SELECTION.md"
    if selection_path.exists():
        old = selection_path.read_text(encoding="utf-8")
        if "Selection (frozen before plotting)" not in old:
            raise RuntimeError(f"Unexpected pre-existing qualitative selection file: {selection_path}")
        old_json = json.loads(old.split("```json\n", 1)[1].split("\n```", 1)[0])
        if old_json != selection:
            raise RuntimeError("Frozen plot selection differs from the saved selection.")
    else:
        selection_path.write_text(
            "# Phase 3 validation plot selection\n\n"
            "## Selection rule\n\n"
            "Selection (frozen before plotting): use only the fixed validation split and seed "
            "20260928. Select six windows by a fixed RNG seed (20260928), choosing at most one "
            "window per pedestrian; then select the two largest P3-E improvements and two "
            "largest degradations versus P3-A constant velocity by per-window raw-pixel MSE. "
            "Ties are resolved by lexicographic sample ID. No image or plot is viewed before "
            "this rule and exact IDs are written. P3-E uses hard inference. This is descriptive "
            "selection, not model selection.\n\n"
            "## Selection (frozen before plotting)\n\n```json\n"
            + json.dumps(selection, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )

    arrays: dict[str, dict[str, np.ndarray]] = {}
    run_paths = {
        "p3_a": output_root / "p3_a" / "20260928" / "validation_predictions.npz",
        "p3_b": output_root / "p3_b" / "20260928" / "validation_predictions.npz",
        "p3_c": output_root / "p3_c" / "20260928" / "validation_predictions.npz",
        "p3_e": output_root / "p3_e" / "20260928" / "validation_predictions.npz",
    }
    for configuration, path in run_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        archive = np.load(path, allow_pickle=False)
        if configuration == "p3_a":
            prefix = "validation_"
            ids = archive[f"{prefix}sample_id"]
            prediction = archive[f"{prefix}predictions_raw"]
            target = archive[f"{prefix}targets_raw"]
            observed = archive[f"{prefix}observed_boxes_raw"]
        else:
            policy = "base" if configuration in {"p3_b", "p3_c"} else "hard"
            prefix = policy + "__"
            ids = archive[f"{prefix}sample_id"]
            prediction = archive[f"{prefix}predictions_raw"]
            target = archive[f"{prefix}targets_raw"]
            observed = archive[f"{prefix}observed_boxes_raw"]
        arrays[configuration] = {
            "index": _array_index(archive, ids),
            "prediction": prediction,
            "target": target,
            "observed": observed,
        }
        archive.close()

    output_dir = output_root / "figures"
    _save_grid(
        output_dir / "p3_random_validation_windows.svg",
        random_ids, by_config_policy["p3_a"], by_config_policy["p3_e"], arrays,
    )
    _save_grid(
        output_dir / "p3_best_worst_validation_windows.svg",
        best_ids + worst_ids, by_config_policy["p3_a"], by_config_policy["p3_e"],
        arrays, columns=2,
    )
    return selection


if __name__ == "__main__":
    root = Path("results_paper/P3_initial_screen_v1")
    print(json.dumps(make_figures(root), indent=2))
