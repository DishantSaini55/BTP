"""One-video CPU Farneback -> MTN compatibility test for the POC."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cpu_farneback import FarnebackConfig, farneback_mtn_features
from context_TF import ContextIndividualTF
from iddped_interface_traj import IDDPedestrian
from individual_TF import IndividualTF


ROOT = Path(r"D:\IDD-PeD-Evaluation")
DATA_ROOT = ROOT / "annotations" / "IDDPedestrian"
CHECKPOINT = ROOT / "checkpoints" / "mtn" / "mtn" / "epoch_latest.pth"
VIDEO = ROOT / "videos" / "gopro" / "gp_set_0006" / "gp_set_0006_vid_0001.MP4"
OPTIONS = dict(N=3, d_model=32, d_ff=128, h=4, dropout=0.1)


def frame_number(image_path: str) -> int:
    return int(re.search(r"(\d+)\.png$", image_path).group(1))


def read_contiguous_frames(cap: cv2.VideoCapture, image_names) -> list[np.ndarray]:
    """Decode one trajectory sequentially; random seeks are prohibitively slow on CPU."""
    numbers = [frame_number(name) for name in image_names]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise RuntimeError("The POC sanity track is not frame-contiguous.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, numbers[0])
    frames = []
    for number in numbers:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {number} from {VIDEO}.")
        frames.append(frame)
    return frames


def main() -> None:
    if not CHECKPOINT.is_file() or not VIDEO.is_file():
        raise FileNotFoundError("Expected official MTN checkpoint and set-0006 video on D:.")
    dataset = IDDPedestrian(data_path=str(DATA_ROOT))
    dataset._get_image_set_ids = lambda _split: ["gp_set_0006"]
    raw = dataset.generate_data_trajectory_sequence(
        "train", fstride=1, sample_type="all", height_rng=[0, float("inf")],
        squarify_ratio=0, data_split_type="default", seq_type="trajectory",
        min_track_size=60,
    )
    indices = [i for i, track in enumerate(raw["image"])
               if "gp_set_0006_vid_0001" in str(track[0])][:2]
    if len(indices) != 2:
        raise RuntimeError("Expected two usable trajectories in video 0001.")

    config = FarnebackConfig()
    ego_batches, ped_batches, boxes, speeds, contexts, occlusions = [], [], [], [], [], []
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        for index in indices:
            images, track_boxes = raw["image"][index][:15], raw["bbox"][index][:15]
            frames = read_contiguous_frames(cap, images)
            ego, pedestrian = [], []
            for previous, current, bbox in zip(frames[:-1], frames[1:], track_boxes[:-1]):
                ego_feature, pedestrian_feature = farneback_mtn_features(previous, current, np.asarray(bbox), config)
                ego.append(ego_feature)
                pedestrian.append(pedestrian_feature)
            ego.append(np.zeros((64, 2), dtype=np.float32))
            pedestrian.append(np.zeros((9, 2), dtype=np.float32))
            ego_batches.append(ego)
            ped_batches.append(pedestrian)
            boxes.append(track_boxes)
            speeds.append(raw["obd_speed"][index][:15])
            contexts.append(raw["scene_context"][index][0])
            occlusions.append(raw["occlusion"][index][:15])
    finally:
        cap.release()
    flow_seconds = time.perf_counter() - started
    batch = {
        "boxes": torch.tensor(np.asarray(boxes), dtype=torch.float32),
        "speed": torch.tensor(np.asarray(speeds), dtype=torch.float32),
        "ego_flow": torch.tensor(np.asarray(ego_batches), dtype=torch.float32),
        "ped_flow": torch.tensor(np.asarray(ped_batches), dtype=torch.float32),
        "scene_context": torch.tensor(np.asarray(contexts), dtype=torch.long),
        "observed_occlusion": torch.tensor(np.asarray(occlusions), dtype=torch.long),
    }
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    baseline = IndividualTF(15, 4, 4, 4, **OPTIONS).eval()
    baseline.load_state_dict(checkpoint)
    context = ContextIndividualTF(15, 4, 4, 4, **OPTIONS).eval()
    context_result = context.load_state_dict({f"base.{key}": value for key, value in checkpoint.items()}, strict=False)
    with torch.no_grad():
        start = time.perf_counter(); baseline_output = baseline.forward_batch(batch); baseline_seconds = time.perf_counter() - start
        start = time.perf_counter(); context_output = context.forward_batch(batch); context_seconds = time.perf_counter() - start
    values = {
        "samples": len(indices), "flow_shape_ego": tuple(batch["ego_flow"].shape),
        "flow_shape_pedestrian": tuple(batch["ped_flow"].shape), "flow_seconds": round(flow_seconds, 4),
        "flow_seconds_per_pair": round(flow_seconds / (len(indices) * 14), 4),
        "baseline_shape": tuple(baseline_output.shape), "context_shape": tuple(context_output.shape),
        "baseline_seconds": round(baseline_seconds, 4), "context_seconds": round(context_seconds, 4),
        "finite_flow": bool(torch.isfinite(batch["ego_flow"]).all() and torch.isfinite(batch["ped_flow"]).all()),
        "finite_outputs": bool(torch.isfinite(baseline_output).all() and torch.isfinite(context_output).all()),
        "context_untrained_keys": len(context_result.missing_keys),
    }
    print(values)


if __name__ == "__main__":
    main()
