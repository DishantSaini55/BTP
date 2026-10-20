"""CPU-only optical-flow features for the MTN context ablation POC.

This is deliberately *not* a RAFT implementation.  It converts consecutive
RGB video frames to the two feature tensors consumed by MTN: an 8x8 ego-flow
grid and a 3x3 pedestrian-crop grid.  Both architecture variants must use the
same saved outputs from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FarnebackConfig:
    """Fixed, recorded CPU flow configuration for the proof-of-concept."""

    width: int = 320
    height: int = 180
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def as_dict(self):
        return asdict(self)


def _resize_bgr(frame: np.ndarray, config: FarnebackConfig) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected one BGR video frame with shape [height, width, 3].")
    return cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA)


def _mean_grid(flow: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """Mean-pool an HxWx2 dense flow field to [rows*columns, 2]."""
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape [height, width, 2].")
    # A valid, but tiny, pedestrian box can be smaller than the required 3x3
    # output grid after CPU resizing.  Upsample its own flow field (never a
    # neighbouring region) so every requested grid cell is defined.
    if flow.shape[0] < rows or flow.shape[1] < columns:
        flow = cv2.resize(flow, (max(columns, flow.shape[1]), max(rows, flow.shape[0])),
                          interpolation=cv2.INTER_LINEAR)
    row_edges = np.linspace(0, flow.shape[0], rows + 1, dtype=int)
    column_edges = np.linspace(0, flow.shape[1], columns + 1, dtype=int)
    cells = []
    for row in range(rows):
        for column in range(columns):
            cell = flow[row_edges[row]:row_edges[row + 1],
                        column_edges[column]:column_edges[column + 1]]
            cells.append(cell.mean(axis=(0, 1), dtype=np.float64))
    result = np.asarray(cells, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Farneback pooling produced NaN or Inf.")
    return result


def farneback_mtn_features(previous_bgr: np.ndarray, current_bgr: np.ndarray,
                            observed_bbox_xyxy: np.ndarray,
                            config: FarnebackConfig = FarnebackConfig()
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Return MTN-compatible ego [64,2] and pedestrian [9,2] float32 flow.

    ``observed_bbox_xyxy`` uses the original-frame coordinate system.  It is
    scaled to the fixed CPU processing resolution before extracting the 3x3
    pedestrian crop.  An invalid/empty crop receives zero flow rather than
    leaking a neighbouring region.
    """
    original_height, original_width = previous_bgr.shape[:2]
    if current_bgr.shape[:2] != (original_height, original_width):
        raise ValueError("Consecutive frames must have the same original dimensions.")
    if np.asarray(observed_bbox_xyxy).shape != (4,):
        raise ValueError("observed_bbox_xyxy must contain [x1, y1, x2, y2].")

    previous = _resize_bgr(previous_bgr, config)
    current = _resize_bgr(current_bgr, config)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        None, config.pyr_scale, config.levels, config.winsize, config.iterations,
        config.poly_n, config.poly_sigma, config.flags,
    )
    ego = _mean_grid(flow, 8, 8)

    x1, y1, x2, y2 = np.asarray(observed_bbox_xyxy, dtype=np.float64)
    x_scale, y_scale = config.width / original_width, config.height / original_height
    left, right = sorted((int(np.floor(x1 * x_scale)), int(np.ceil(x2 * x_scale))))
    top, bottom = sorted((int(np.floor(y1 * y_scale)), int(np.ceil(y2 * y_scale))))
    left, right = np.clip((left, right), 0, config.width)
    top, bottom = np.clip((top, bottom), 0, config.height)
    crop = flow[top:bottom, left:right]
    pedestrian = _mean_grid(crop, 3, 3) if crop.shape[0] and crop.shape[1] else np.zeros((9, 2), dtype=np.float32)

    if ego.shape != (64, 2) or pedestrian.shape != (9, 2):
        raise AssertionError("MTN flow contract violated.")
    return ego, pedestrian
