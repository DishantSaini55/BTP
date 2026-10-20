"""Create the fixed small CPU POC manifest and Farneback feature files.

This utility records selections before model inference.  It intentionally
selects by native context/occlusion strata, never by prediction quality.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cpu_farneback import FarnebackConfig, farneback_mtn_features
from iddped_interface_traj import IDDPedestrian


ROOT = Path(r"D:\IDD-PeD-Evaluation")
DATA_ROOT = ROOT / "annotations" / "IDDPedestrian"
OUTPUT = ROOT / "flow" / "cpu_farneback_poc"
CONFIG = FarnebackConfig()
# MTN consumes 15 observed and 45 predicted positions.  Keeping exactly one
# deterministic 60-frame window per selected trajectory is sufficient for this
# minimal POC and avoids generating unused CPU flow for the rest of a track.
WINDOW_FRAMES = 60


def _frame_number(path: str) -> int:
    return int(re.search(r"(\d+)\.png$", path).group(1))


def _video_path(image_path: str) -> Path:
    parts = Path(image_path).parts
    video_index = parts.index("gopro")
    set_id, video_id = parts[video_index + 1], parts[video_index + 2]
    return ROOT / "videos" / "gopro" / set_id / f"{video_id}.MP4"


def _load_set(set_id: str, split: str) -> dict:
    dataset = IDDPedestrian(data_path=str(DATA_ROOT))
    dataset._get_image_set_ids = lambda _split: [set_id]
    return dataset.generate_data_trajectory_sequence(
        split, fstride=1, sample_type="all", height_rng=[0, float("inf")],
        squarify_ratio=0, data_split_type="default", seq_type="trajectory",
        min_track_size=60,
    )


def _stratified_indices(raw: dict, candidates: list[int], limit: int) -> list[int]:
    """Round-robin static-context/max-observed-occlusion groups deterministically."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index in candidates:
        context = tuple(int(value) for value in raw["scene_context"][index][0])
        max_occlusion = int(max(raw["occlusion"][index]))
        groups[context + (max_occlusion,)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (str(raw["pid"][i][0]), str(raw["image"][i][0])))
    chosen = []
    while len(chosen) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop(0))
    return chosen


def _subset(raw: dict, indices: list[int]) -> dict:
    subset = {}
    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        selected = []
        for index in indices:
            item = values[index]
            # Per-frame annotations are lists as long as the trajectory. Static
            # fields (for example a one-element PID) remain untouched.
            if isinstance(item, list) and len(item) >= WINDOW_FRAMES:
                item = item[:WINDOW_FRAMES]
            selected.append(item)
        subset[key] = selected
    return subset


def _add_flows(data: dict) -> None:
    """Decode requested frames monotonically per video.

    OpenCV/FFmpeg random seeking was observed to lose H.264 reference frames in
    the verified GP6 archive.  This does not alter sampling or flow; it merely
    decodes the same requested source frame indices in presentation order.
    """
    ego_tracks, pedestrian_tracks = [None] * len(data["image"]), [None] * len(data["image"])
    by_video: dict[Path, list[int]] = defaultdict(list)
    for index, image_names in enumerate(data["image"]):
        by_video[_video_path(image_names[0])].append(index)
    for video, indices in by_video.items():
        indices.sort(key=lambda index: _frame_number(data["image"][index][0]))
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video}.")
        cursor = -1
        try:
            for index in indices:
                image_names, boxes = data["image"][index], data["bbox"][index]
                frames = []
                for number in [_frame_number(name) for name in image_names]:
                    while cursor < number:
                        ok, frame = cap.read()
                        cursor += 1
                        if not ok:
                            raise RuntimeError(f"Could not sequentially decode frame {number} from {video}.")
                    frames.append(frame.copy())
                ego, pedestrian = [], []
                for previous, current, bbox in zip(frames[:-1], frames[1:], boxes[:-1]):
                    ego_grid, pedestrian_grid = farneback_mtn_features(previous, current, np.asarray(bbox), CONFIG)
                    ego.append(ego_grid); pedestrian.append(pedestrian_grid)
                ego.append(np.zeros((64, 2), dtype=np.float32))
                pedestrian.append(np.zeros((9, 2), dtype=np.float32))
                ego_tracks[index] = np.asarray(ego, dtype=np.float32)
                pedestrian_tracks[index] = np.asarray(pedestrian, dtype=np.float32)
        finally:
            cap.release()
    data["ego_op_flow"] = ego_tracks
    data["ped_op_flow"] = pedestrian_tracks


def _write(partition: str, data: dict, indices: list[int]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / f"{partition}.pkl").open("wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
    manifest = []
    for local_index, source_index in enumerate(indices):
        manifest.append({
            "local_index": local_index, "source_index": source_index,
            "pid": str(data["pid"][local_index][0]),
            "first_frame": str(data["image"][local_index][0]),
            "scene_context": [int(value) for value in data["scene_context"][local_index][0]],
            "max_occlusion": int(max(data["occlusion"][local_index])),
            "track_frames": len(data["image"][local_index]),
        })
    (OUTPUT / f"{partition}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    train_raw = _load_set("gp_set_0006", "train")
    # Strict video-level holdout: no trajectory from either validation or test
    # video may enter the training partition.
    # Video 1 has a confirmed H.264 corruption at frame 4139.  Use the fully
    # decoded video 2 for training; reserve only video-1 windows finishing
    # before that damaged region for validation.  Test stays video 3.
    train_candidates = [i for i, images in enumerate(train_raw["image"])
                        if "gp_set_0006_vid_0002" in str(images[0])]
    validation_candidates = [i for i, images in enumerate(train_raw["image"])
                             if "gp_set_0006_vid_0001" in str(images[0])
                             and _frame_number(images[-1]) < 4000]
    test_candidates = [i for i, images in enumerate(train_raw["image"])
                       if "gp_set_0006_vid_0003" in str(images[0])]
    selections = {
        "train": (train_raw, _stratified_indices(train_raw, train_candidates, 32)),
        "validation": (train_raw, _stratified_indices(train_raw, validation_candidates, 8)),
        "test": (train_raw, _stratified_indices(train_raw, test_candidates, 10)),
    }
    for partition, (raw, indices) in selections.items():
        data = _subset(raw, indices)
        _add_flows(data)
        _write(partition, data, indices)
        print(partition, "tracks", len(indices), "frames", sum(len(track) for track in data["image"]))
    (OUTPUT / "farneback_config.json").write_text(json.dumps(CONFIG.as_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
