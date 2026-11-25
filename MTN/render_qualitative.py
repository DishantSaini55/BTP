"""Render frozen seed-20260928 qualitative windows from saved outputs."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"
QUALITATIVE = RESULTS / "qualitative"
SEED = "20260928"
TEST_VIDEO = "gp_set_0006_vid_0003"
VIDEO_PATH = Path(
    r"D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0003.MP4"
)

MODEL_SPECS = {
    "MTN": ("mtn_baseline", "results_paper/runs/E1/mtn_baseline/20260928"),
    "PACE-Net without SATIE": (
        "pace_net_without_satie_sage_plus_masc",
        "results_paper/runs/E1/pace_net_without_satie_sage_plus_masc/20260928",
    ),
    "PACE-Net": (
        "pace_net_sage_plus_masc_plus_satie",
        "results_paper/runs/E1/pace_net_sage_plus_masc_plus_satie/20260928",
    ),
    "Constant velocity": (
        "constant_velocity_last_3_frames",
        "results_paper/runs/B1/constant_velocity_last_3_frames/deterministic",
    ),
}
COLORS = {
    "Observed": "#eeeeee",
    "Ground truth": "#ffd400",
    "MTN": "#e63946",
    "PACE-Net without SATIE": "#00a6a6",
    "PACE-Net": "#7b2cbf",
    "Constant velocity": "#ff7f0e",
}
SCENE_LABELS = {
    "intersection_type": ("NI", "U-turn", "T-right", "T-left", "four-way", "Y-intersection", "UNKNOWN"),
    "signalized_type": ("N/A", "C", "S", "CS", "UNKNOWN"),
    "road_type": ("main", "secondary", "street", "lane", "UNKNOWN"),
    "location_type": ("urban", "rural", "commercial", "residential", "UNKNOWN"),
    "motion_direction": ("OW", "TW", "UNKNOWN"),
}
SCENE_FIELDS = tuple(SCENE_LABELS)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_selection() -> list[dict[str, str]]:
    path = QUALITATIVE / "SELECTION.md"
    selected = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Group | Sample ID |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "---" or cells[1] in {"None available", "—"}:
            continue
        selected.append({"group": cells[0], "sample_id": cells[1].strip("`")})
    if not selected:
        raise ValueError(f"No selected sample IDs found in {path}")
    return selected


def load_model_outputs() -> tuple[dict, dict, dict]:
    outputs = {}
    rows_by_model = {}
    for label, (slug, relative) in MODEL_SPECS.items():
        directory = ROOT / relative
        outputs[label] = np.load(directory / "predictions.npz", allow_pickle=False)
        rows_by_model[label] = {
            row["sample_id"]: row for row in read_rows(directory / "per_sample.csv")
        }
    return outputs, rows_by_model, {}


def load_test_samples() -> tuple[list[dict], dict[str, int]]:
    split_dir = RESULTS / "splits" / "gp6_from_scratch_v1"
    with (split_dir / "raw_splits.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    samples = raw["test"]
    return samples, {sample["id"]: index for index, sample in enumerate(samples)}


def decode_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_number} from {VIDEO_PATH}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def scene_description(values: np.ndarray) -> str:
    parts = []
    for name, value in zip(SCENE_FIELDS, values):
        labels = SCENE_LABELS[name]
        index = int(value)
        label = labels[index] if 0 <= index < len(labels) else f"code-{index}"
        parts.append(f"{name}={label}")
    return ", ".join(parts)


def draw_box(ax, box: np.ndarray, color: str, linewidth: float, alpha: float = 1.0) -> None:
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


def draw_path(ax, boxes: np.ndarray, start: np.ndarray, label: str, color: str) -> None:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    points = np.concatenate((start[None, :], centers), axis=0)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.6, label=label)
    draw_box(ax, boxes[-1], color, 1.5)


def figure_mse(rows_by_model: dict, label: str, sample_id: str) -> str:
    return f"{float(rows_by_model[label][sample_id]['mse']):.2f}"


def render() -> None:
    pdf_path = QUALITATIVE / "OVERLAYS_seed20260928.pdf"
    csv_path = QUALITATIVE / "OVERLAYS_seed20260928.csv"
    if pdf_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing qualitative artifact.")
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Source video is unavailable: {VIDEO_PATH}")

    selected = load_selection()
    outputs, rows_by_model, _ = load_model_outputs()
    samples, index_by_id = load_test_samples()
    test_ids = [sample["id"] for sample in samples]
    for label, rows in rows_by_model.items():
        if set(rows) != set(test_ids):
            raise ValueError(f"Saved per-sample IDs do not match test split for {label}")
    for selected_row in selected:
        if selected_row["sample_id"] not in index_by_id:
            raise ValueError(f"Selected ID is not in the frozen test split: {selected_row}")

    full_gate = outputs["PACE-Net"]["gate"]
    no_satie_gate = outputs["PACE-Net without SATIE"]["gate"]
    attention_lengths = outputs["PACE-Net"]["attention_memory_tokens"]
    if np.any(full_gate != 0) or np.any(no_satie_gate != 0):
        raise AssertionError("The frozen selection document's gate availability is stale.")
    if np.any(attention_lengths != 15):
        raise AssertionError("Expected the saved closed-gate PACE-Net to use 15 memory keys.")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    rows_out = []
    try:
        with PdfPages(pdf_path) as pdf:
            for page_number, selection in enumerate(selected, start=1):
                sample_id = selection["sample_id"]
                sample_index = index_by_id[sample_id]
                sample = samples[sample_index]
                if sample["video"] != TEST_VIDEO:
                    raise ValueError(f"Unexpected video in qualitative selection: {sample['video']}")

                last_observed_frame = int(sample["frames"][14])
                background = decode_frame(capture, last_observed_frame)
                height, width = background.shape[:2]
                observed = np.asarray(sample["bbox"][:15], dtype=np.float32)
                target = outputs["MTN"]["targets"][sample_index]
                start_center = (observed[-1, :2] + observed[-1, 2:]) / 2

                fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
                ax.imshow(background)
                for box in observed:
                    draw_box(ax, box, COLORS["Observed"], 0.7, alpha=0.58)
                observed_centers = (observed[:, :2] + observed[:, 2:]) / 2
                ax.plot(
                    observed_centers[:, 0],
                    observed_centers[:, 1],
                    color=COLORS["Observed"],
                    linewidth=1.3,
                    linestyle="--",
                    label="Observed (15 boxes)",
                )
                draw_path(ax, target, start_center, "Ground truth", COLORS["Ground truth"])
                for model_label in ("MTN", "PACE-Net without SATIE", "PACE-Net", "Constant velocity"):
                    prediction = outputs[model_label]["predictions"][sample_index]
                    draw_path(
                        ax,
                        prediction,
                        start_center,
                        f"{model_label} (MSE {figure_mse(rows_by_model, model_label, sample_id)})",
                        COLORS[model_label],
                    )

                gate_full_value = int(full_gate[sample_index])
                gate_no_satie_value = int(no_satie_gate[sample_index])
                scene = scene_description(np.asarray(sample["scene_context"][0]))
                title = (
                    f"{selection['group']} | {sample['pid']} | frames "
                    f"{sample['frame_start']}–{sample['frame_end']} | "
                    f"last observed frame {last_observed_frame}\n"
                    f"g (without SATIE/full PACE-Net)={gate_no_satie_value}/{gate_full_value} | {scene}"
                )
                ax.set_title(title, fontsize=9)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_xlabel("Image x (pixels)")
                ax.set_ylabel("Image y (pixels)")
                ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
                ax.grid(False)
                pdf.savefig(fig, dpi=140)
                plt.close(fig)

                row = {
                    "page": page_number,
                    "selection_group": selection["group"],
                    "sample_id": sample_id,
                    "pid": sample["pid"],
                    "last_observed_frame": last_observed_frame,
                    "g_without_satie": gate_no_satie_value,
                    "g_pace_net": gate_full_value,
                    "scene_labels": scene,
                }
                for label in rows_by_model:
                    row[f"{label}_mse"] = rows_by_model[label][sample_id]["mse"]
                rows_out.append(row)
    finally:
        capture.release()

    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps({"pdf": str(pdf_path), "csv": str(csv_path), "pages": len(rows_out)}, indent=2))


if __name__ == "__main__":
    render()
