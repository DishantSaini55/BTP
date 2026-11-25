"""Render zoomed panels for the frozen seed-20260928 qualitative windows."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import render_qualitative as base


OUTPUT_PATH = base.QUALITATIVE / "ZOOMED_OVERLAYS_seed20260928.pdf"
MARGIN_FRACTION = 0.30
MODEL_ORDER = ("MTN", "PACE-Net without SATIE", "PACE-Net", "Constant velocity")
PAGE_GROUPS = ("Random", "Largest improvements", "Largest degradations")


def select_windows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected = base.load_selection()
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_group.setdefault(row["group"], []).append(row)

    expected = {
        "Random": 6,
        "Largest improvements": 2,
        "Largest degradations": 2,
        "Gate closed": 2,
    }
    for group, count in expected.items():
        if len(by_group.get(group, [])) != count:
            raise ValueError(
                f"Expected {count} frozen selections in {group!r}; "
                f"found {len(by_group.get(group, []))}"
            )

    grid = by_group["Random"][:2]
    for group in PAGE_GROUPS[1:]:
        grid.extend(by_group[group])
    closed = by_group["Gate closed"]
    if len({row["sample_id"] for row in grid + closed}) != 8:
        raise ValueError("The paper panels must use eight distinct frozen sample IDs.")
    return grid, closed


def crop_bounds(box_groups: list[np.ndarray], width: int, height: int) -> tuple[float, ...]:
    boxes = np.concatenate([np.asarray(group).reshape(-1, 4) for group in box_groups])
    if boxes.size == 0 or not np.isfinite(boxes).all():
        raise ValueError("Cannot crop a sample with empty or non-finite box coordinates.")

    left = float(np.min(boxes[:, 0]))
    top = float(np.min(boxes[:, 1]))
    right = float(np.max(boxes[:, 2]))
    bottom = float(np.max(boxes[:, 3]))
    box_width = max(right - left, 1.0)
    box_height = max(bottom - top, 1.0)
    crop_width = box_width * (1 + 2 * MARGIN_FRACTION)
    crop_height = box_height * (1 + 2 * MARGIN_FRACTION)

    aspect = width / height
    crop_width = max(crop_width, crop_height * aspect)
    crop_height = crop_width / aspect
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    x0 = center_x - crop_width / 2
    y0 = center_y - crop_height / 2
    bounds = (x0, x0 + crop_width, y0, y0 + crop_height)

    if not (bounds[0] <= left and bounds[1] >= right and bounds[2] <= top and bounds[3] >= bottom):
        raise AssertionError("Computed zoom crop does not contain every drawn box.")
    return bounds


def box_groups_for_sample(
    sample: dict,
    outputs: dict,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    observed = np.asarray(sample["bbox"][:15], dtype=np.float32)
    target = np.asarray(outputs["MTN"]["targets"][sample_index], dtype=np.float32)
    predictions = {
        label: np.asarray(outputs[label]["predictions"][sample_index], dtype=np.float32)
        for label in MODEL_ORDER
    }
    return observed, target, predictions


def draw_box(ax, box: np.ndarray, color: str, linewidth: float, alpha: float) -> None:
    x1, y1, x2, y2 = (float(value) for value in box)
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def draw_sequence(
    ax,
    boxes: np.ndarray,
    start_center: np.ndarray,
    color: str,
    *,
    draw_all_boxes: bool,
    alpha: float,
) -> None:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    points = np.concatenate((start_center[None, :], centers), axis=0)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.15, alpha=alpha)
    if draw_all_boxes:
        for box in boxes:
            draw_box(ax, box, color, 0.38, alpha * 0.34)
    draw_box(ax, boxes[-1], color, 1.0, alpha)


def draw_overlay(
    ax,
    frame: np.ndarray,
    observed: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    crop: tuple[float, ...] | None,
    *,
    inset: bool,
) -> None:
    ax.imshow(frame, origin="upper")
    for box in observed:
        draw_box(ax, box, base.COLORS["Observed"], 0.48 if inset else 0.62, 0.72)

    observed_centers = (observed[:, :2] + observed[:, 2:]) / 2
    ax.plot(
        observed_centers[:, 0],
        observed_centers[:, 1],
        color=base.COLORS["Observed"],
        linewidth=0.85 if inset else 1.2,
        linestyle="--",
        alpha=0.9,
    )
    start_center = observed_centers[-1]
    draw_sequence(
        ax,
        target,
        start_center,
        base.COLORS["Ground truth"],
        draw_all_boxes=True,
        alpha=0.75 if inset else 1.0,
    )
    for label in MODEL_ORDER:
        draw_sequence(
            ax,
            predictions[label],
            start_center,
            base.COLORS[label],
            draw_all_boxes=True,
            alpha=0.66 if inset else 0.95,
        )

    height, width = frame.shape[:2]
    if crop is None:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    else:
        x0, x1, y0, y1 = crop
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    if inset:
        if crop is not None:
            x0, x1, y0, y1 = crop
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#00ffff",
                    linewidth=1.0,
                )
            )
        ax.set_axis_off()


def draw_panel(
    ax,
    row: dict[str, str],
    sample: dict,
    sample_index: int,
    outputs: dict,
    rows_by_model: dict,
    capture: cv2.VideoCapture,
    crop_manifest: dict[str, tuple[float, ...]],
) -> None:
    sample_id = row["sample_id"]
    frame_number = int(sample["frames"][14])
    frame = base.decode_frame(capture, frame_number)
    observed, target, predictions = box_groups_for_sample(sample, outputs, sample_index)
    crop = crop_bounds(
        [observed, target, *predictions.values()],
        frame.shape[1],
        frame.shape[0],
    )
    crop_manifest[sample_id] = crop

    draw_overlay(ax, frame, observed, target, predictions, crop, inset=False)
    ax.set_xlabel("Image x (pixels)", fontsize=6, labelpad=1)
    ax.set_ylabel("Image y (pixels)", fontsize=6, labelpad=1)
    ax.tick_params(axis="both", labelsize=5, length=2, pad=1)
    mse_line = "MSE: " + " | ".join(
        f"{short}={base.figure_mse(rows_by_model, label, sample_id)}"
        for label, short in (
            ("MTN", "MTN"),
            ("PACE-Net without SATIE", "no-SATIE"),
            ("PACE-Net", "PACE"),
            ("Constant velocity", "CV"),
        )
    )
    ax.set_title(
        f"{row['group']} | {sample['pid']} | frames "
        f"{sample['frame_start']}–{sample['frame_end']} | g(no/full)=0/0\n{mse_line}",
        fontsize=6.8,
        pad=3,
    )

    inset_ax = ax.inset_axes([0.69, 0.63, 0.29, 0.30])
    draw_overlay(inset_ax, frame, observed, target, predictions, None, inset=True)
    inset_ax.add_patch(
        Rectangle(
            (crop[0], crop[2]),
            crop[1] - crop[0],
            crop[3] - crop[2],
            fill=False,
            edgecolor="#00ffff",
            linewidth=1.0,
        )
    )
    inset_ax.set_facecolor("white")
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)


def add_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color="#777777", linestyle="--", label="Observed (15)"),
        Line2D([0], [0], color=base.COLORS["Ground truth"], label="Ground truth"),
        *[
            Line2D([0], [0], color=base.COLORS[label], label=label)
            for label in MODEL_ORDER
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )


def render() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {OUTPUT_PATH}")
    if not base.VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Source video is unavailable: {base.VIDEO_PATH}")

    grid_rows, closed_rows = select_windows()
    selected = grid_rows + closed_rows
    outputs, rows_by_model, _ = base.load_model_outputs()
    samples, index_by_id = base.load_test_samples()
    for label, rows in rows_by_model.items():
        if set(rows) != {sample["id"] for sample in samples}:
            raise ValueError(f"Saved per-sample IDs do not match test split for {label}")
    for row in selected:
        if row["sample_id"] not in index_by_id:
            raise ValueError(f"Frozen sample is missing from test split: {row['sample_id']}")

    full_gate = outputs["PACE-Net"]["gate"]
    no_satie_gate = outputs["PACE-Net without SATIE"]["gate"]
    if np.any(full_gate != 0) or np.any(no_satie_gate != 0):
        raise AssertionError("Frozen selection gate states no longer match saved outputs.")

    capture = cv2.VideoCapture(str(base.VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {base.VIDEO_PATH}")

    crop_manifest: dict[str, tuple[float, ...]] = {}
    try:
        with PdfPages(OUTPUT_PATH) as pdf:
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            fig.suptitle(
                "Zoomed E1 forecasts — frozen seed 20260928 windows",
                fontsize=14,
                y=0.98,
            )
            for ax, row in zip(axes.flat, grid_rows):
                sample_index = index_by_id[row["sample_id"]]
                draw_panel(
                    ax,
                    row,
                    samples[sample_index],
                    sample_index,
                    outputs,
                    rows_by_model,
                    capture,
                    crop_manifest,
                )
            add_legend(fig)
            fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.12, wspace=0.16, hspace=0.28)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))
            fig.suptitle(
                "Zoomed E1 forecasts — frozen closed-gate windows (g=0)",
                fontsize=14,
                y=0.98,
            )
            for ax, row in zip(axes.flat, closed_rows):
                sample_index = index_by_id[row["sample_id"]]
                draw_panel(
                    ax,
                    row,
                    samples[sample_index],
                    sample_index,
                    outputs,
                    rows_by_model,
                    capture,
                    crop_manifest,
                )
            add_legend(fig)
            fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.16, wspace=0.14)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    finally:
        capture.release()

    print(
        json.dumps(
            {
                "pdf": str(OUTPUT_PATH),
                "pages": 2,
                "grid_sample_ids": [row["sample_id"] for row in grid_rows],
                "closed_gate_sample_ids": [row["sample_id"] for row in closed_rows],
                "crop_bounds_original_pixels": crop_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    render()
