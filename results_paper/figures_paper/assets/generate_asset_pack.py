"""Export real-data visual assets for the fixed E1 architecture example."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


ASSET_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
MTN_DIR = ROOT / "MTN"
SPLIT_DIR = ROOT / "results_paper" / "splits" / "gp6_from_scratch_v1"
RUNS_DIR = ROOT / "results_paper" / "runs" / "E1"
SAMPLE_ID = "gp_set_0006/gp_set_0006_vid_0003/gp_6933/01415/01474"
SEED = 20260928
VIDEO_PATH = Path(
    r"D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0003.MP4"
)

PACE_SLUG = "pace_net_sage_plus_masc_plus_satie"
ROUTE_SLUG = "pace_net_with_route"
COLORS = {
    "observed": (230, 57, 70),
    "ground_truth": (255, 212, 0),
    "pace": (123, 44, 191),
}
OBSERVED_FRAME_INDICES = (0, 3, 7, 11, 14)
PREDICTED_STEP_INDICES = (0, 11, 22, 33, 44)
FLOW_STEP_INDICES = (11, 12, 13)
GENERATED_FILES = (
    "observed_frames.png",
    "observed_strip.png",
    "predicted_frames.png",
    "predicted_strip.png",
    "ego_flow.png",
    "ped_flow.png",
    "flow_strip.png",
    "speed_bars.png",
    "frame_with_labels.png",
    "scene_labels.json",
    "memory_tokens_15.png",
    "memory_tokens_16.png",
    "decoder_attention_strip.png",
    "route_operators.png",
    "params_bar.png",
)

sys.path.insert(0, str(MTN_DIR))
import render_qualitative as qualitative
from run_from_scratch_e1 import NAMES, load_data, model_for


def refuse_overwrite() -> None:
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Source video is unavailable: {VIDEO_PATH}")
    if not (SPLIT_DIR / "flow_splits.pkl").is_file():
        raise FileNotFoundError(f"Cached flow split is unavailable: {SPLIT_DIR}")
    existing = [name for name in GENERATED_FILES if (ASSET_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite asset files: {existing}")


def read_prediction_index(directory: Path, sample_id: str) -> int:
    with (directory / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [index for index, row in enumerate(rows) if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved output row for {sample_id}; found {len(matches)}")
    return matches[0]


def crop_rectangle(
    frame_shape: tuple[int, ...],
    boxes: np.ndarray,
    margin: float = 0.30,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("A real, finite box is required to crop the pedestrian.")

    left = max(0.0, float(np.minimum(values[:, 0], values[:, 2]).min()))
    top = max(0.0, float(np.minimum(values[:, 1], values[:, 3]).min()))
    right = min(float(width), float(np.maximum(values[:, 0], values[:, 2]).max()))
    bottom = min(float(height), float(np.maximum(values[:, 1], values[:, 3]).max()))
    if right <= left or bottom <= top:
        raise ValueError("The requested boxes do not intersect the source frame.")

    box_width = max(right - left, 1.0)
    box_height = max(bottom - top, 1.0)
    aspect = width / height
    crop_width = max(box_width * (1 + 2 * margin), box_height * (1 + 2 * margin) * aspect)
    crop_height = crop_width / aspect
    if crop_width >= width or crop_height >= height:
        return 0, 0, width, height

    crop_width = max(2, int(math.ceil(crop_width)))
    crop_height = max(2, int(round(crop_width / aspect)))
    if crop_width > width or crop_height > height:
        return 0, 0, width, height
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    x0 = min(max(int(round(center_x - crop_width / 2)), 0), width - crop_width)
    y0 = min(max(int(round(center_y - crop_height / 2)), 0), height - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def draw_box(image: Image.Image, box: np.ndarray, crop: tuple[int, int, int, int],
             color: tuple[int, int, int], line_width: int = 3) -> None:
    x0, y0, _, _ = crop
    x1, y1, x2, y2 = (float(value) for value in box)
    shifted = (
        int(round(min(x1, x2) - x0)),
        int(round(min(y1, y2) - y0)),
        int(round(max(x1, x2) - x0)),
        int(round(max(y1, y2) - y0)),
    )
    ImageDraw.Draw(image).rectangle(shifted, outline=color, width=line_width)


def photo_tile(
    frame: np.ndarray,
    boxes_and_colors: list[tuple[np.ndarray, tuple[int, int, int]]],
    size: tuple[int, int],
) -> Image.Image:
    all_boxes = np.concatenate([np.asarray(boxes).reshape(-1, 4) for boxes, _ in boxes_and_colors])
    crop = crop_rectangle(frame.shape, all_boxes)
    x0, y0, crop_right, crop_bottom = crop
    image = Image.fromarray(frame).crop(crop).resize(
        size, resample=Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(image)
    for boxes, color in boxes_and_colors:
        for box in np.asarray(boxes).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in box)
            bx1, bx2 = sorted((x1, x2))
            by1, by2 = sorted((y1, y2))
            sx = size[0] / (crop_right - x0)
            sy = size[1] / (crop_bottom - y0)
            coords = (
                int(round((bx1 - x0) * sx)),
                int(round((by1 - y0) * sy)),
                int(round((bx2 - x0) * sx)),
                int(round((by2 - y0) * sy)),
            )
            draw.rectangle(coords, outline=color, width=2)
    return image


def save_filmstrip(tiles: list[Image.Image], path: Path) -> None:
    if not tiles:
        raise ValueError("Cannot save an empty filmstrip.")
    width, height = tiles[0].size
    if any(tile.size != (width, height) for tile in tiles):
        raise ValueError("Filmstrip tiles must have a common size.")
    strip = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        strip.paste(tile, (index * width, 0))
    strip.save(path, format="PNG", dpi=(300, 300), optimize=True)


def save_contact_sheet(tiles: list[Image.Image], path: Path) -> None:
    save_filmstrip(tiles, path)


def save_flow_map(flow_grid: np.ndarray, rows: int, columns: int, vmax: float,
                  size: int = 240) -> np.ndarray:
    grid = np.asarray(flow_grid, dtype=np.float32).reshape(rows, columns, 2)
    upsampled = cv2.resize(grid, (size, size), interpolation=cv2.INTER_NEAREST)
    u, v = upsampled[..., 0], upsampled[..., 1]
    magnitude = np.hypot(u, v)
    angle = np.mod(np.arctan2(v, u), 2 * np.pi)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = np.round(angle * (180.0 / (2 * np.pi))).astype(np.uint8)
    hsv[..., 1] = 255
    if vmax > 0:
        hsv[..., 2] = np.clip(np.round(magnitude * 255 / vmax), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_flow_images(ego: np.ndarray, pedestrian: np.ndarray) -> dict[str, float]:
    ego_grids = ego[list(FLOW_STEP_INDICES)]
    ped_grids = pedestrian[list(FLOW_STEP_INDICES)]
    ego_max = float(np.max(np.linalg.norm(ego_grids, axis=-1)))
    ped_max = float(np.max(np.linalg.norm(ped_grids, axis=-1)))
    ego_images = [save_flow_map(grid, 8, 8, ego_max) for grid in ego_grids]
    ped_images = [save_flow_map(grid, 3, 3, ped_max) for grid in ped_grids]

    def save_rgb_strip(images: list[np.ndarray], path: Path) -> None:
        Image.fromarray(np.concatenate(images, axis=1)).save(
            path, format="PNG", dpi=(300, 300), optimize=True
        )

    save_rgb_strip(ego_images, ASSET_DIR / "ego_flow.png")
    save_rgb_strip(ped_images, ASSET_DIR / "ped_flow.png")
    flow_strip = np.concatenate(
        (np.concatenate(ego_images, axis=1), np.concatenate(ped_images, axis=1)),
        axis=0,
    )
    Image.fromarray(flow_strip).save(
        ASSET_DIR / "flow_strip.png", format="PNG", dpi=(300, 300), optimize=True
    )
    return {"ego_vmax": ego_max, "pedestrian_vmax": ped_max}


def save_colormap_strip(values: np.ndarray, path: Path, cmap: str,
                        vmax: float, *, colorbar: bool = True) -> None:
    fig = plt.figure(figsize=(8, 1.35), facecolor="none")
    ax = fig.add_axes([0.04, 0.34 if colorbar else 0.08, 0.92, 0.62 if colorbar else 0.84])
    ax.imshow(np.asarray(values), aspect="auto", interpolation="nearest",
              cmap=cmap, vmin=0.0, vmax=max(vmax, np.finfo(float).eps))
    ax.set_axis_off()
    if colorbar:
        cax = fig.add_axes([0.08, 0.13, 0.84, 0.07])
        bar = fig.colorbar(ax.images[0], cax=cax, orientation="horizontal")
        bar.set_ticks([])
        bar.outline.set_visible(False)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_params_bar() -> None:
    independent = 3 * 4 * (32 * 32 + 32)
    satie = 4 * (32 * 32 + 32)
    if (independent, satie) != (12672, 4224):
        raise AssertionError("Encoder self-attention parameter calculation changed.")
    fig, ax = plt.subplots(figsize=(3.0, 1.8), facecolor="none")
    ax.bar((0, 1), (independent, satie), width=0.58,
           color=("#00a6a6", "#7b2cbf"))
    ax.set_ylim(0, independent * 1.08)
    ax.set_xlim(-0.6, 1.6)
    ax.set_axis_off()
    fig.savefig(ASSET_DIR / "params_bar.png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_model_batch(dataset, index: int) -> dict[str, torch.Tensor]:
    item = dataset[index]
    return {
        key: value.unsqueeze(0)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }


def load_saved_model(name: str, checkpoint: Path) -> torch.nn.Module:
    model = model_for(name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def capture_model_tokens(dataset, index: int, saved_gate: float) -> tuple[np.ndarray, np.ndarray, float]:
    model = load_saved_model(
        NAMES[2], RUNS_DIR / PACE_SLUG / str(SEED) / "checkpoint.pth"
    )
    batch = make_model_batch(dataset, index)
    captured: dict[str, list] = {"encoder": [], "decoder": []}

    def encoder_hook(_module, _inputs, output):
        captured["encoder"].append(output)

    def decoder_pre_hook(_module, inputs):
        captured["decoder"].append(inputs[1].detach().cpu())

    handles = [
        model.base.model.encoder.register_forward_hook(encoder_hook),
        model.base.model.decoder.register_forward_pre_hook(decoder_pre_hook),
    ]
    try:
        with torch.no_grad():
            model.forward_batch(batch)
            context_token = model.context_encoder(
                batch["scene_context"], batch["observed_occlusion"]
            ).detach().cpu()
        computed_gate = float(model.last_context_gate.item())
        if computed_gate != saved_gate:
            raise AssertionError(f"Checkpoint gate {computed_gate} != saved gate {saved_gate}.")
        if not captured["encoder"] or not captured["decoder"]:
            raise RuntimeError("Could not capture saved-model encoder/decoder tokens.")
        encoder_output, speed_memory = captured["encoder"][-1]
        actual_memory = captured["decoder"][-1]
        if actual_memory.shape != (1, 15, 32):
            raise AssertionError(f"Expected [1,15,32] decoder memory, got {actual_memory.shape}.")
        if encoder_output.shape != (1, 15, 32) or speed_memory.shape != (1, 1, 32):
            raise AssertionError("Unexpected saved-model encoder token shapes.")
        # If the saved gate admitted context, the decoder would receive the
        # context token plus the encoder's speed memory at position 16.
        candidate_context = context_token + speed_memory.detach().cpu()
        candidate_16 = torch.cat((actual_memory, candidate_context), dim=1)
        return (
            actual_memory[0].numpy(),
            candidate_16[0].numpy(),
            computed_gate,
        )
    finally:
        for handle in handles:
            handle.remove()


def capture_route_operators(dataset, index: int, saved_gate: np.ndarray) -> dict[str, np.ndarray]:
    model = load_saved_model(
        NAMES[3], RUNS_DIR / ROUTE_SLUG / str(SEED) / "checkpoint.pth"
    )
    batch = make_model_batch(dataset, index)
    captured: dict[str, list] = {
        "fusion_input": [],
        "additive": [],
        "cross_attention": [],
    }

    def fusion_pre_hook(_module, inputs):
        captured["fusion_input"].append(inputs[0].detach().cpu())

    def module_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name].append(output.detach().cpu())
        return hook

    handles = [
        model.fusion.register_forward_pre_hook(fusion_pre_hook),
        model.fusion.add_norm.register_forward_hook(module_hook("additive")),
        model.fusion.cross_ff_norm.register_forward_hook(module_hook("cross_attention")),
    ]
    try:
        with torch.no_grad():
            model.forward_batch(batch)
        computed_gate = model.fusion.last_gate.detach().cpu().numpy()[0]
        if not np.allclose(computed_gate, saved_gate, atol=1e-6, rtol=0):
            raise AssertionError("Saved ROUTE gate differs from checkpoint inference.")
        if not all(captured[name] for name in captured):
            raise RuntimeError("Could not capture all three ROUTE operator outputs.")
        return {
            "identity": captured["fusion_input"][-1][0].numpy(),
            "additive": captured["additive"][-1][0].numpy(),
            "cross_attention": captured["cross_attention"][-1][0].numpy(),
        }
    finally:
        for handle in handles:
            handle.remove()


def save_frame_assets(sample: dict, targets: np.ndarray, predictions: np.ndarray) -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    try:
        frame_numbers = [int(value) for value in sample["frames"]]
        if frame_numbers != list(range(frame_numbers[0], frame_numbers[0] + len(frame_numbers))):
            raise ValueError("The fixed example window is not a contiguous video sequence.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
        frame_cache: dict[int, np.ndarray] = {}
        for frame_number in frame_numbers:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode source video frame {frame_number}.")
            frame_cache[frame_number] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        observed_boxes = np.asarray(sample["bbox"][:15])
        observed_tiles = []
        for step in range(15):
            observed_tiles.append(
                photo_tile(
                    frame_cache[int(sample["frames"][step])],
                    [(observed_boxes[step:step + 1], COLORS["observed"])],
                    (128, 72),
                )
            )
        save_contact_sheet(observed_tiles, ASSET_DIR / "observed_strip.png")

        observed_frames = []
        for step in OBSERVED_FRAME_INDICES:
            observed_frames.append(
                photo_tile(
                    frame_cache[int(sample["frames"][step])],
                    [(observed_boxes[step:step + 1], COLORS["observed"])],
                    (480, 270),
                )
            )
        save_contact_sheet(observed_frames, ASSET_DIR / "observed_frames.png")

        predicted_tiles = []
        for step in range(45):
            frame = frame_cache[int(sample["frames"][15 + step])]
            predicted_tiles.append(
                photo_tile(
                    frame,
                    [
                        (targets[step:step + 1], COLORS["ground_truth"]),
                        (predictions[step:step + 1], COLORS["pace"]),
                    ],
                    (128, 72),
                )
            )
        save_contact_sheet(predicted_tiles, ASSET_DIR / "predicted_strip.png")

        predicted_frames = []
        for step in PREDICTED_STEP_INDICES:
            frame = frame_cache[int(sample["frames"][15 + step])]
            predicted_frames.append(
                photo_tile(
                    frame,
                    [
                        (targets[step:step + 1], COLORS["ground_truth"]),
                        (predictions[step:step + 1], COLORS["pace"]),
                    ],
                    (480, 270),
                )
            )
        save_contact_sheet(predicted_frames, ASSET_DIR / "predicted_frames.png")

        last_frame = Image.fromarray(frame_cache[int(sample["frames"][14])])
        draw_box(
            last_frame,
            observed_boxes[14],
            (0, 0, last_frame.width, last_frame.height),
            COLORS["observed"],
        )
        last_frame.save(
            ASSET_DIR / "frame_with_labels.png",
            format="PNG",
            dpi=(300, 300),
            optimize=True,
        )
    finally:
        capture.release()


def save_scene_labels(sample: dict, saved_gate: float, route_gate: np.ndarray,
                      speeds: np.ndarray, predictions: np.ndarray) -> dict:
    scene_codes = np.asarray(sample["scene_context"][0], dtype=np.int64)
    scene_fields = qualitative.SCENE_FIELDS
    scene_names = {
        name: qualitative.SCENE_LABELS[name][int(value)]
        for name, value in zip(scene_fields, scene_codes)
    }
    occlusion = np.asarray(sample["occlusion"], dtype=np.int64)
    predicted_boxes = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    inverted_x_steps = np.flatnonzero(predicted_boxes[:, 2] < predicted_boxes[:, 0])
    inverted_y_steps = np.flatnonzero(predicted_boxes[:, 3] < predicted_boxes[:, 1])
    route_names = ("identity", "additive", "cross_attention")
    route_chosen = route_names[int(np.argmax(route_gate))]
    result = {
        "sample_id": SAMPLE_ID,
        "scene_context_field_order": list(scene_fields),
        "scene_context_codes": {
            name: int(value) for name, value in zip(scene_fields, scene_codes)
        },
        "scene_context_labels": scene_names,
        "occlusion_per_window_frame_60": occlusion.tolist(),
        "occlusion_observed_15": occlusion[:15].tolist(),
        "occlusion_window_max": int(occlusion.max()),
        "occlusion_observed_15_max": int(occlusion[:15].max()),
        "obd_speed_observed_15_raw": np.asarray(speeds).reshape(-1).tolist(),
        "pace_prediction_inverted_x_steps": inverted_x_steps.tolist(),
        "pace_prediction_inverted_y_steps": inverted_y_steps.tolist(),
        "pace_net_saved_gate_g": float(saved_gate),
        "route_saved_gate": {
            name: float(value) for name, value in zip(route_names, route_gate)
        },
        "route_selected_operator": route_chosen,
    }
    return result


def main() -> None:
    refuse_overwrite()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    datasets, _stats = load_data()
    test_samples = datasets["test"].samples
    matches = [i for i, sample in enumerate(test_samples) if sample["id"] == SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed window in the cached test split; found {len(matches)}.")
    sample_index = matches[0]
    sample = test_samples[sample_index]

    pace_dir = RUNS_DIR / PACE_SLUG / str(SEED)
    pace_npz = np.load(pace_dir / "predictions.npz", allow_pickle=False)
    saved_index = read_prediction_index(pace_dir, SAMPLE_ID)
    if saved_index != sample_index:
        raise AssertionError("Saved prediction order differs from the frozen flow split.")
    targets = np.asarray(pace_npz["targets"][saved_index])
    predictions = np.asarray(pace_npz["predictions"][saved_index])
    saved_gate = float(pace_npz["gate"][saved_index])
    if saved_gate != 0.0:
        raise AssertionError(f"Expected this frozen PACE-Net gate to be closed; found {saved_gate}.")

    route_dir = RUNS_DIR / ROUTE_SLUG / str(SEED)
    route_npz = np.load(route_dir / "predictions.npz", allow_pickle=False)
    route_index = read_prediction_index(route_dir, SAMPLE_ID)
    if route_index != sample_index:
        raise AssertionError("Saved ROUTE prediction order differs from the frozen flow split.")
    route_gate = np.asarray(route_npz["gate"][route_index], dtype=np.float32)
    if route_gate.shape != (3,):
        raise AssertionError(f"Expected three saved ROUTE operator decisions; got {route_gate.shape}.")

    save_frame_assets(sample, targets, predictions)
    flow_ranges = save_flow_images(
        np.asarray(sample["ego_op_flow"]),
        np.asarray(sample["ped_op_flow"]),
    )

    speeds = np.asarray(sample["obd_speed"][:15], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.0, 0.9), facecolor="none")
    positions = np.arange(15)
    ax.bar(positions, speeds.reshape(-1), width=0.62, color="#00a6a6", alpha=0.9)
    ax.scatter(positions, speeds.reshape(-1), s=11, color="#7b2cbf", zorder=3)
    ax.set_xlim(-0.5, 14.5)
    low, high = float(speeds.min()), float(speeds.max())
    if low == high:
        ax.set_ylim(low - 1.0, high + 1.0)
    else:
        ax.set_ylim(low, high)
    ax.set_axis_off()
    fig.savefig(ASSET_DIR / "speed_bars.png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    memory_15, memory_16, computed_gate = capture_model_tokens(
        datasets["test"], sample_index, saved_gate
    )
    if computed_gate != saved_gate or memory_15.shape != (15, 32) or memory_16.shape != (16, 32):
        raise AssertionError("Unexpected saved-model memory-token values or gate.")
    token_norms_15 = np.linalg.norm(memory_15, axis=1)[None, :]
    token_norms_16 = np.linalg.norm(memory_16, axis=1)[None, :]
    token_vmax = float(max(token_norms_15.max(), token_norms_16.max()))
    save_colormap_strip(
        token_norms_15, ASSET_DIR / "memory_tokens_15.png", "viridis", token_vmax
    )
    save_colormap_strip(
        token_norms_16, ASSET_DIR / "memory_tokens_16.png", "viridis", token_vmax
    )

    attention = [
        np.asarray(pace_npz[f"attention_layer_{layer}"][saved_index], dtype=np.float32)
        for layer in range(3)
    ]
    if any(values.shape != (4, 45, 15) for values in attention):
        raise AssertionError("Saved decoder attention does not have [4 heads,45 queries,15 keys].")
    attention_strip = np.concatenate([values.mean(axis=0) for values in attention], axis=0)
    if np.max(np.abs(attention_strip.sum(axis=1) - 1.0)) > 1e-5:
        raise AssertionError("Mean-head saved attention rows do not sum to one.")
    attention_vmax = float(attention_strip.max())
    fig = plt.figure(figsize=(3.2, 8.0), facecolor="none")
    ax = fig.add_axes([0.05, 0.04, 0.79, 0.92])
    image = ax.imshow(
        attention_strip,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=attention_vmax,
    )
    ax.axhline(44.5, color="white", linewidth=0.6, alpha=0.8)
    ax.axhline(89.5, color="white", linewidth=0.6, alpha=0.8)
    ax.set_axis_off()
    cax = fig.add_axes([0.88, 0.08, 0.04, 0.84])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([])
    colorbar.outline.set_visible(False)
    fig.savefig(
        ASSET_DIR / "decoder_attention_strip.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    route_outputs = capture_route_operators(datasets["test"], sample_index, route_gate)
    route_norms = np.stack(
        [np.linalg.norm(route_outputs[name], axis=1) for name in
         ("identity", "additive", "cross_attention")]
    )
    route_vmax = float(route_norms.max())
    save_colormap_strip(
        route_norms, ASSET_DIR / "route_operators.png", "viridis", route_vmax
    )
    save_params_bar()

    labels = save_scene_labels(sample, saved_gate, route_gate, speeds, predictions)
    labels["flow_map_range"] = flow_ranges
    labels["token_norm_color_range"] = [0.0, token_vmax]
    labels["decoder_attention_color_range"] = [0.0, attention_vmax]
    labels["route_operator_norm_color_range"] = [0.0, route_vmax]
    (ASSET_DIR / "scene_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "asset_directory": str(ASSET_DIR),
        "sample_id": SAMPLE_ID,
        "saved_gate_g": saved_gate,
        "route_gate": route_gate.tolist(),
        "flow_steps": list(FLOW_STEP_INDICES),
        "flow_ranges": flow_ranges,
        "token_norm_color_max": token_vmax,
        "attention_color_max": attention_vmax,
        "route_norm_color_max": route_vmax,
        "files": list(GENERATED_FILES),
    }, indent=2))


if __name__ == "__main__":
    main()
