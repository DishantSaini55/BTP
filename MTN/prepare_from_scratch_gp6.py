"""Materialise the frozen GP6 from-scratch split and cached Farneback features.

This intentionally reads the annotation cache directly so split construction does
not depend on an official checkpoint or the legacy POC split files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from .context_schema import SCENE_UNKNOWN_VALUES
    from .cpu_farneback import FarnebackConfig, _mean_grid, _resize_bgr
except ImportError:
    from context_schema import SCENE_UNKNOWN_VALUES
    from cpu_farneback import FarnebackConfig, _mean_grid, _resize_bgr


WORKSPACE = Path(__file__).resolve().parents[1]
RESULT_ROOT = WORKSPACE / "results_paper" / "splits" / "gp6_from_scratch_v1"
DATA_ROOT = Path(r"D:\IDD-PeD-Evaluation")
CACHE = DATA_ROOT / "annotations" / "IDDPedestrian" / "data_cache" / "iddp_database.pkl"
WINDOW = 60
STRIDE = 30
VIDEOS = {
    "train": "gp_set_0006_vid_0002",
    "validation": "gp_set_0006_vid_0001",
    "test": "gp_set_0006_vid_0003",
}


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def frame_number(path: str) -> int:
    match = re.search(r"(\d+)\.png$", path)
    if not match:
        raise ValueError(f"Unexpected generated image path: {path}")
    return int(match.group(1))


def image_path(video: str, frame: int) -> str:
    return str(DATA_ROOT / "annotations" / "IDDPedestrian" / "images" / "gopro" /
               "gp_set_0006" / video / f"{frame:05d}.png")


def sample_id(video: str, pid: str, frames: list[int]) -> str:
    return f"gp_set_0006/{video}/{pid}/{frames[0]:05d}/{frames[-1]:05d}"


def build_samples(database: dict, role: str) -> list[dict]:
    video = VIDEOS[role]
    source = database["gp_set_0006"][video]
    samples: list[dict] = []
    for pid, annotation in sorted(source["pedestrian_annotations"].items()):
        attributes = annotation.get("attributes", {})
        if "crossing_motive" not in attributes:
            continue
        frames = annotation["frames"]
        if len(frames) < WINDOW:
            continue
        context = [
            attributes.get("intersection_type", SCENE_UNKNOWN_VALUES[0]),
            attributes.get("signalized_type", SCENE_UNKNOWN_VALUES[1]),
            attributes.get("road_type", SCENE_UNKNOWN_VALUES[2]),
            attributes.get("location_type", SCENE_UNKNOWN_VALUES[3]),
            attributes.get("motion_direction", SCENE_UNKNOWN_VALUES[4]),
        ]
        for start in range(0, len(frames) - WINDOW + 1, STRIDE):
            window_frames = frames[start:start + WINDOW]
            if any(right - left != 1 for left, right in zip(window_frames, window_frames[1:])):
                continue
            if role == "validation" and window_frames[-1] >= 4000:
                continue
            vehicle = source["vehicle_annotations"]
            try:
                speed = [[float(vehicle[frame]["OBD_speed"])] for frame in window_frames]
            except KeyError as error:
                raise RuntimeError(f"Missing OBD speed for {video}/{pid} at {error}") from error
            samples.append({
                "id": sample_id(video, pid, window_frames),
                "video": video,
                "pid": pid,
                "frame_start": int(window_frames[0]),
                "frame_end": int(window_frames[-1]),
                "frames": [int(frame) for frame in window_frames],
                "image": [image_path(video, frame) for frame in window_frames],
                "bbox": np.asarray(annotation["bbox"][start:start + WINDOW], dtype=np.float32),
                "occlusion": np.asarray(annotation["occlusion"][start:start + WINDOW], dtype=np.int64),
                "scene_context": np.asarray([context] * WINDOW, dtype=np.int64),
                "obd_speed": np.asarray(speed, dtype=np.float32),
            })
    samples.sort(key=lambda sample: (sample["video"], sample["pid"], sample["frame_start"]))
    return samples


def manifest(samples: list[dict]) -> list[dict]:
    return [{
        "id": sample["id"], "video": sample["video"], "pid": sample["pid"],
        "frame_start": sample["frame_start"], "frame_end": sample["frame_end"],
        "scene_context": sample["scene_context"][0].tolist(),
        "max_occlusion": int(sample["occlusion"].max()),
    } for sample in samples]


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def materialise() -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite split directory {RESULT_ROOT}")
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)
    with CACHE.open("rb") as handle:
        database = pickle.load(handle)
    split = {role: build_samples(database, role) for role in VIDEOS}
    RESULT_ROOT.mkdir(parents=True)
    for role, samples in split.items():
        write_json(RESULT_ROOT / f"{role}_manifest.json", manifest(samples))
    test_ids = [sample["id"] for sample in split["test"]]
    info = {
        "version": "gp6_from_scratch_v1",
        "window_frames": WINDOW,
        "stride_frames": STRIDE,
        "selection_seed": 20260928,
        "videos": VIDEOS,
        "counts": {role: len(samples) for role, samples in split.items()},
        "test_sample_id_sha256": sha256_lines(test_ids),
        "farneback_config": FarnebackConfig().as_dict(),
    }
    write_json(RESULT_ROOT / "split_metadata.json", info)
    with (RESULT_ROOT / "raw_splits.pkl").open("xb") as handle:
        pickle.dump(split, handle, pickle.HIGHEST_PROTOCOL)
    print(json.dumps(info, indent=2))


def features_from_dense_flow(flow: np.ndarray, bbox: np.ndarray, original_shape: tuple[int, int], config: FarnebackConfig) -> tuple[np.ndarray, np.ndarray]:
    ego = _mean_grid(flow, 8, 8)
    original_height, original_width = original_shape
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    x_scale, y_scale = config.width / original_width, config.height / original_height
    left, right = sorted((int(np.floor(x1 * x_scale)), int(np.ceil(x2 * x_scale))))
    top, bottom = sorted((int(np.floor(y1 * y_scale)), int(np.ceil(y2 * y_scale))))
    left, right = np.clip((left, right), 0, config.width)
    top, bottom = np.clip((top, bottom), 0, config.height)
    crop = flow[top:bottom, left:right]
    pedestrian = _mean_grid(crop, 3, 3) if crop.shape[0] and crop.shape[1] else np.zeros((9, 2), dtype=np.float32)
    return ego, pedestrian


def generate_video_flow(samples: list[dict], video: str, config: FarnebackConfig) -> None:
    references: dict[tuple[int, int], list[tuple[dict, int]]] = defaultdict(list)
    for sample in samples:
        for step, (previous, current) in enumerate(zip(sample["frames"], sample["frames"][1:])):
            references[(previous, current)].append((sample, step))
        sample["ego_op_flow"] = np.zeros((WINDOW, 64, 2), dtype=np.float32)
        sample["ped_op_flow"] = np.zeros((WINDOW, 9, 2), dtype=np.float32)
    if not references:
        return
    path = DATA_ROOT / "videos" / "gopro" / "gp_set_0006" / f"{video}.MP4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    maximum = max(current for _, current in references)
    previous_frame = None
    previous_index = None
    try:
        for current_index in range(maximum + 1):
            ok, current_frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {current_index} from {path}")
            key = (current_index - 1, current_index)
            if previous_frame is not None and key in references:
                previous = _resize_bgr(previous_frame, config)
                current = _resize_bgr(current_frame, config)
                dense = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None,
                    config.pyr_scale, config.levels, config.winsize, config.iterations,
                    config.poly_n, config.poly_sigma, config.flags,
                )
                for sample, step in references[key]:
                    ego, pedestrian = features_from_dense_flow(
                        dense, sample["bbox"][step], previous_frame.shape[:2], config
                    )
                    sample["ego_op_flow"][step] = ego
                    sample["ped_op_flow"][step] = pedestrian
            previous_frame, previous_index = current_frame, current_index
    finally:
        capture.release()


def generate_flow() -> None:
    raw_path = RESULT_ROOT / "raw_splits.pkl"
    flow_path = RESULT_ROOT / "flow_splits.pkl"
    if not raw_path.exists():
        raise FileNotFoundError("Run materialisation before flow generation.")
    if flow_path.exists() or (RESULT_ROOT / "flow_metadata.json").exists():
        raise FileExistsError("Refusing to overwrite existing flow artifacts.")
    with raw_path.open("rb") as handle:
        split = pickle.load(handle)
    config = FarnebackConfig()
    for role, video in VIDEOS.items():
        generate_video_flow(split[role], video, config)
    with flow_path.open("xb") as handle:
        pickle.dump(split, handle, pickle.HIGHEST_PROTOCOL)
    metadata = json.loads((RESULT_ROOT / "split_metadata.json").read_text(encoding="utf-8"))
    metadata["flow_feature_shapes"] = {"ego_op_flow": [WINDOW, 64, 2], "ped_op_flow": [WINDOW, 9, 2]}
    metadata["flow_config_sha256"] = hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    write_json(RESULT_ROOT / "flow_metadata.json", metadata)
    print(json.dumps({"flow_output": str(flow_path), "counts": metadata["counts"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialise", "flow"))
    args = parser.parse_args()
    if args.action == "materialise":
        materialise()
    else:
        generate_flow()


if __name__ == "__main__":
    main()
