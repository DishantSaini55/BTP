"""Annotation-only IDD-PeD development data for Phase 5B.

Only the explicit official development videos below are read. This module
never opens caches, images, videos, checkpoints, or official-test annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "reference_IDD-PeD" / "data" / "IDDPedestrian"
RESULTS = ROOT / "results_paper"
SPLIT_ROOT = RESULTS / "splits" / "ao_annotation_only_v1"
WINDOW_FRAMES = 60
OBSERVED_FRAMES = 15
FUTURE_FRAMES = 45
STRIDE_FRAMES = 30
SCENE_FIELDS = (
    "intersection_type",
    "signalized_type",
    "road_type",
    "location_type",
    "motion_direction",
)

TRAIN_VIDEOS = (
    ("gp_set_0001", "gp_set_0001_vid_0001"),
    ("gp_set_0002", "gp_set_0002_vid_0001"),
    ("gp_set_0004", "gp_set_0004_vid_0001"),
    ("gp_set_0006", "gp_set_0006_vid_0002"),
)
VALIDATION_VIDEOS = (
    ("gp_set_0007", "gp_set_0007_vid_0004"),
    ("gp_set_0007", "gp_set_0007_vid_0005"),
)
INVENTORY_VIDEOS = {
    "gp_set_0001": tuple(
        f"gp_set_0001_vid_{video:04d}" for video in (1, 2, 3, 5, 6, 7, 8)
    ),
    "gp_set_0002": tuple(
        f"gp_set_0002_vid_{video:04d}" for video in (1, 2, 4, 5)
    ),
    "gp_set_0004": tuple(
        f"gp_set_0004_vid_{video:04d}" for video in (1, 2, 3, 5)
    ),
    "gp_set_0006": tuple(
        f"gp_set_0006_vid_{video:04d}" for video in (1, 2, 3)
    ),
    "gp_set_0007": tuple(
        f"gp_set_0007_vid_{video:04d}" for video in (4, 5)
    ),
}
HISTORICAL_GP6_TEST_VIDEO = ("gp_set_0006", "gp_set_0006_vid_0003")


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_video_path(
    annotation_root: Path,
    set_id: str,
    video_id: str,
    kind: str,
    metadata_only: bool = False,
) -> Path:
    if set_id not in {
        "gp_set_0001", "gp_set_0002", "gp_set_0004",
        "gp_set_0006", "gp_set_0007",
        "gp_set_0003", "gp_set_0005", "gp_set_0008", "gp_set_0009",
    }:
        raise ValueError(f"Unsupported set: {set_id}")
    if not video_id.startswith(f"{set_id}_vid_"):
        raise ValueError(f"Video {video_id!r} does not belong to {set_id!r}.")
    if (
        (set_id, video_id) == HISTORICAL_GP6_TEST_VIDEO
        and not metadata_only
    ):
        raise ValueError("Historical GP6 test-role annotation contents are excluded.")
    if kind not in {"annotations", "annotations_vehicle"}:
        raise ValueError(f"Unsupported annotation kind: {kind}")
    root = annotation_root.resolve()
    filename = (
        f"{video_id}_obd.xml"
        if kind == "annotations_vehicle"
        else f"{video_id}.xml"
    )
    path = (root / kind / "gopro" / set_id / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Annotation path escapes the configured root.") from error
    return path


def _attribute(box: ET.Element, name: str) -> str | None:
    item = box.find(f"attribute[@name='{name}']")
    if item is None or item.text is None:
        return None
    return item.text.strip()


def _contiguous_runs(
    boxes_by_frame: dict[int, tuple[float, float, float, float]]
) -> list[list[tuple[int, tuple[float, float, float, float]]]]:
    runs: list[list[tuple[int, tuple[float, float, float, float]]]] = []
    for frame in sorted(boxes_by_frame):
        item = (frame, boxes_by_frame[frame])
        if not runs or frame != runs[-1][-1][0] + 1:
            runs.append([item])
        else:
            runs[-1].append(item)
    return runs


def parse_video(
    annotation_root: Path,
    set_id: str,
    video_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _safe_video_path(annotation_root, set_id, video_id, "annotations")
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    task = root.find("./meta/task")
    if task is None:
        raise ValueError(f"Missing CVAT task metadata in {path}")
    task_name = task.findtext("name")
    if not task_name or Path(task_name).stem != video_id:
        raise ValueError(f"CVAT video name mismatch in {path}")
    frame_count = int(task.findtext("size", "0"))
    original_size = task.find("./original_size")
    if original_size is None:
        raise ValueError(f"Missing original image size in {path}")
    width = int(original_size.findtext("width", "0"))
    height = int(original_size.findtext("height", "0"))
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame count or dimensions in {path}")

    report: dict[str, Any] = {
        "set_id": set_id,
        "video_id": video_id,
        "cvat_task_name": task_name,
        "pedestrian_xml": str(path),
        "pedestrian_xml_present": True,
        "pedestrian_xml_bytes": path.stat().st_size,
        "pedestrian_xml_sha256": _sha256_file(path),
        "task_frame_count": frame_count,
        "frame_width": width,
        "frame_height": height,
        "coordinate_fields": "xtl,ytl,xbr,ybr; pixel coordinates",
        "pedestrian_tracks": 0,
        "valid_bbox_records": 0,
        "outside_bbox_records": 0,
        "invalid_bbox_records": 0,
        "out_of_bounds_bbox_records": 0,
        "duplicate_frame_records": 0,
        "missing_manual_pedestrian_ids": 0,
        "duplicate_manual_pedestrian_ids": 0,
        "tracks_with_60_frame_windows": 0,
        "eligible_windows_stride30": 0,
    }
    samples: list[dict[str, Any]] = []
    seen_track_ids: set[str] = set()
    manual_ids: list[str] = []
    cvat_track_ids: set[str] = set()
    for track in root.findall("track"):
        if track.get("label") != "pedestrian":
            continue
        report["pedestrian_tracks"] += 1
        cvat_track_id = track.get("id")
        if cvat_track_id is None or cvat_track_id in cvat_track_ids:
            raise ValueError(f"Missing or duplicate CVAT track id in {path}")
        cvat_track_ids.add(cvat_track_id)
        boxes = track.findall("box")
        manual_id = _attribute(boxes[0], "id") if boxes else None
        if not manual_id:
            report["missing_manual_pedestrian_ids"] += 1
        else:
            manual_ids.append(manual_id)
        valid_by_frame: dict[int, tuple[float, float, float, float]] = {}
        duplicate_frames: set[int] = set()
        for box in boxes:
            try:
                frame = int(box.attrib["frame"])
                outside = int(box.get("outside", "0")) == 1
            except (KeyError, ValueError):
                report["invalid_bbox_records"] += 1
                continue
            if outside:
                report["outside_bbox_records"] += 1
                continue
            try:
                coords = tuple(
                    float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")
                )
            except (KeyError, ValueError):
                report["invalid_bbox_records"] += 1
                continue
            if (
                not all(math.isfinite(value) for value in coords)
                or coords[0] >= coords[2]
                or coords[1] >= coords[3]
                or frame < 0
                or frame >= frame_count
            ):
                report["invalid_bbox_records"] += 1
                continue
            if (
                coords[0] < 0
                or coords[1] < 0
                or coords[2] > width
                or coords[3] > height
            ):
                report["out_of_bounds_bbox_records"] += 1
                continue
            if frame in duplicate_frames:
                report["duplicate_frame_records"] += 1
                continue
            if frame in valid_by_frame:
                report["duplicate_frame_records"] += 1
                valid_by_frame.pop(frame)
                report["valid_bbox_records"] -= 1
                duplicate_frames.add(frame)
                continue
            valid_by_frame[frame] = coords
            report["valid_bbox_records"] += 1

        track_window_count = 0
        for run in _contiguous_runs(valid_by_frame):
            for offset in range(0, len(run) - WINDOW_FRAMES + 1, STRIDE_FRAMES):
                selected = run[offset:offset + WINDOW_FRAMES]
                if len(selected) != WINDOW_FRAMES:
                    continue
                frames = [item[0] for item in selected]
                if any(right != left + 1 for left, right in zip(frames, frames[1:])):
                    raise AssertionError("Window contains a non-contiguous frame.")
                sample_id = (
                    f"{set_id}/{video_id}/track_{cvat_track_id}/"
                    f"{frames[0]:05d}/{frames[-1]:05d}"
                )
                if sample_id in seen_track_ids:
                    raise ValueError(f"Duplicate sample id: {sample_id}")
                seen_track_ids.add(sample_id)
                samples.append({
                    "sample_id": sample_id,
                    "set_id": set_id,
                    "video_id": video_id,
                    "track_id": cvat_track_id,
                    "manual_pedestrian_id": manual_id or "",
                    "frame_start": frames[0],
                    "frame_end": frames[-1],
                    "boxes": np.asarray(
                        [item[1] for item in selected], dtype=np.float32
                    ),
                })
                track_window_count += 1
        if track_window_count:
            report["tracks_with_60_frame_windows"] += 1
            report["eligible_windows_stride30"] += track_window_count

    report["duplicate_manual_pedestrian_ids"] = (
        len(manual_ids) - len(set(manual_ids))
    )
    samples.sort(key=lambda item: (
        item["video_id"], int(item["track_id"]), item["frame_start"]
    ))
    return report, samples


def _audit_obd(
    annotation_root: Path, set_id: str, video_id: str, expected_frames: int
) -> dict[str, Any]:
    path = _safe_video_path(annotation_root, set_id, video_id, "annotations_vehicle")
    result: dict[str, Any] = {
        "obd_xml_present": path.is_file(),
        "obd_xml_bytes": path.stat().st_size if path.is_file() else 0,
        "obd_xml_sha256": "",
        "obd_frame_count": 0,
        "obd_ids_complete_zero_based": False,
        "obd_bad_speed_records": 0,
        "obd_speed_min": "",
        "obd_speed_max": "",
        "obd_speed_unit_verified": False,
    }
    if not path.is_file():
        return result
    root = ET.parse(path).getroot()
    ids: list[int] = []
    speeds: list[float] = []
    for frame in root.findall(".//frame"):
        try:
            frame_id = int(frame.attrib["id"])
            speed = float(frame.attrib["OBD_speed"])
            if not math.isfinite(speed):
                raise ValueError("non-finite speed")
            ids.append(frame_id)
            speeds.append(speed)
        except (KeyError, ValueError):
            result["obd_bad_speed_records"] += 1
    result.update({
        "obd_xml_sha256": _sha256_file(path),
        "obd_frame_count": len(root.findall(".//frame")),
        "obd_ids_complete_zero_based": (
            len(ids) == expected_frames
            and len(set(ids)) == expected_frames
            and set(ids) == set(range(expected_frames))
            and result["obd_bad_speed_records"] == 0
        ),
        "obd_speed_min": min(speeds) if speeds else "",
        "obd_speed_max": max(speeds) if speeds else "",
    })
    return result


def _audit_scene(
    annotation_root: Path, set_id: str, video_id: str
) -> dict[str, Any]:
    path = _safe_video_path(annotation_root, set_id, video_id, "annotations")
    root = ET.parse(path).getroot()
    pedestrian_ids = {
        _attribute(boxes[0], "id")
        for track in root.findall("track")
        if track.get("label") == "pedestrian"
        for boxes in [track.findall("box")]
        if boxes and _attribute(boxes[0], "id")
    }
    scene_by_id: dict[str, dict[str, str]] = {}
    varying = {field: 0 for field in SCENE_FIELDS}
    for track in root.findall("track"):
        if track.get("label") != "POI":
            continue
        boxes = track.findall("box")
        if not boxes:
            continue
        manual_id = _attribute(boxes[0], "id")
        if not manual_id:
            continue
        attrs_by_box = [
            {item.get("name", ""): (item.text or "").strip()
             for item in box.findall("attribute")}
            for box in boxes
        ]
        scene_by_id[manual_id] = {
            field: attrs_by_box[0].get(field, "") for field in SCENE_FIELDS
        }
        for field in SCENE_FIELDS:
            values = {
                attrs.get(field, "") for attrs in attrs_by_box
                if attrs.get(field, "")
            }
            if len(values) > 1:
                varying[field] += 1
    output: dict[str, Any] = {
        "poi_tracks": sum(
            1 for track in root.findall("track") if track.get("label") == "POI"
        ),
        "scene_varying_poi_tracks": sum(varying.values()),
    }
    for field in SCENE_FIELDS:
        output[f"scene_{field}_matched_manual_ids"] = sum(
            1 for manual_id in pedestrian_ids
            if scene_by_id.get(manual_id, {}).get(field)
        )
        output[f"scene_{field}_varying_poi_tracks"] = varying[field]
    return output


def audit_video(
    annotation_root: Path, set_id: str, video_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, samples = parse_video(annotation_root, set_id, video_id)
    report.update(_audit_obd(
        annotation_root, set_id, video_id, report["task_frame_count"]
    ))
    report.update(_audit_scene(annotation_root, set_id, video_id))
    return report, samples


INVENTORY_FIELDS = (
    "set_id", "video_id", "inventory_status", "pedestrian_xml_present",
    "pedestrian_xml_bytes", "pedestrian_xml_sha256", "cvat_task_name",
    "task_frame_count",
    "frame_width", "frame_height", "coordinate_fields", "pedestrian_tracks",
    "valid_bbox_records", "outside_bbox_records", "invalid_bbox_records",
    "out_of_bounds_bbox_records", "duplicate_frame_records",
    "missing_manual_pedestrian_ids", "duplicate_manual_pedestrian_ids",
    "tracks_with_60_frame_windows", "eligible_windows_stride30",
    "obd_xml_present", "obd_xml_bytes", "obd_xml_sha256", "obd_frame_count",
    "obd_ids_complete_zero_based", "obd_bad_speed_records", "obd_speed_min",
    "obd_speed_max", "obd_speed_unit_verified", "poi_tracks",
    "scene_intersection_type_matched_manual_ids",
    "scene_intersection_type_varying_poi_tracks",
    "scene_signalized_type_matched_manual_ids",
    "scene_signalized_type_varying_poi_tracks",
    "scene_road_type_matched_manual_ids", "scene_road_type_varying_poi_tracks",
    "scene_location_type_matched_manual_ids",
    "scene_location_type_varying_poi_tracks",
    "scene_motion_direction_matched_manual_ids",
    "scene_motion_direction_varying_poi_tracks",
)


def build_inventory(annotation_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for set_id, videos in INVENTORY_VIDEOS.items():
        for video_id in videos:
            pedestrian_path = _safe_video_path(
                annotation_root, set_id, video_id, "annotations",
                metadata_only=(set_id, video_id) == HISTORICAL_GP6_TEST_VIDEO,
            )
            obd_path = _safe_video_path(
                annotation_root, set_id, video_id, "annotations_vehicle",
                metadata_only=(set_id, video_id) == HISTORICAL_GP6_TEST_VIDEO,
            )
            if (set_id, video_id) == HISTORICAL_GP6_TEST_VIDEO:
                rows.append({
                    "set_id": set_id,
                    "video_id": video_id,
                    "inventory_status": "metadata_only_excluded_historical_test_role",
                    "pedestrian_xml_present": pedestrian_path.is_file(),
                    "pedestrian_xml_bytes": (
                        pedestrian_path.stat().st_size if pedestrian_path.is_file() else 0
                    ),
                    "obd_xml_present": obd_path.is_file(),
                    "obd_xml_bytes": obd_path.stat().st_size if obd_path.is_file() else 0,
                })
                continue
            row, _ = audit_video(annotation_root, set_id, video_id)
            row["inventory_status"] = "parsed_development_annotation"
            rows.append(row)
    return rows


def write_inventory(annotation_root: Path, output_path: Path) -> list[dict[str, Any]]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    rows = build_inventory(annotation_root)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(output_path, buffer.getvalue().encode("utf-8"))
    return rows


def _split_arrays(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("A role cannot have zero eligible annotation windows.")
    return {
        "observed_boxes": np.stack(
            [sample["boxes"][:OBSERVED_FRAMES] for sample in samples]
        ).astype(np.float32, copy=False),
        "future_boxes": np.stack(
            [sample["boxes"][OBSERVED_FRAMES:] for sample in samples]
        ).astype(np.float32, copy=False),
        "sample_id": np.asarray(
            [sample["sample_id"] for sample in samples], dtype=np.str_
        ),
        "video_id": np.asarray(
            [sample["video_id"] for sample in samples], dtype=np.str_
        ),
        "set_id": np.asarray(
            [sample["set_id"] for sample in samples], dtype=np.str_
        ),
        "track_id": np.asarray(
            [sample["track_id"] for sample in samples], dtype=np.str_
        ),
        "manual_pedestrian_id": np.asarray(
            [sample["manual_pedestrian_id"] for sample in samples], dtype=np.str_
        ),
        "frame_start": np.asarray(
            [sample["frame_start"] for sample in samples], dtype=np.int32
        ),
        "frame_end": np.asarray(
            [sample["frame_end"] for sample in samples], dtype=np.int32
        ),
    }


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _sample_digest(sample_ids: np.ndarray) -> str:
    return hashlib.sha256(
        ("\n".join(sample_ids.tolist()) + "\n").encode("utf-8")
    ).hexdigest()


def prepare_split(
    annotation_root: Path = ANNOTATION_ROOT,
    output_root: Path = SPLIT_ROOT,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite split directory {output_root}")
    sources: dict[str, dict[str, Any]] = {}
    role_samples: dict[str, list[dict[str, Any]]] = {}
    for role, videos in (
        ("train", TRAIN_VIDEOS),
        ("validation", VALIDATION_VIDEOS),
    ):
        samples: list[dict[str, Any]] = []
        for set_id, video_id in videos:
            report, video_samples = audit_video(annotation_root, set_id, video_id)
            if not report["obd_ids_complete_zero_based"]:
                raise ValueError(f"OBD frame coverage is incomplete for {video_id}.")
            sources[video_id] = {
                "set_id": set_id,
                "pedestrian_xml_sha256": report["pedestrian_xml_sha256"],
                "obd_xml_sha256": report["obd_xml_sha256"],
                "task_frame_count": report["task_frame_count"],
                "frame_width": report["frame_width"],
                "frame_height": report["frame_height"],
                "eligible_tracks": report["tracks_with_60_frame_windows"],
                "eligible_windows": report["eligible_windows_stride30"],
            }
            samples.extend(video_samples)
        role_samples[role] = samples

    train_videos = {sample["video_id"] for sample in role_samples["train"]}
    validation_videos = {
        sample["video_id"] for sample in role_samples["validation"]
    }
    if train_videos.intersection(validation_videos):
        raise ValueError("Training and validation share a source video.")
    if any(sample["set_id"] not in {
        "gp_set_0001", "gp_set_0002", "gp_set_0004", "gp_set_0006"
    } for sample in role_samples["train"]):
        raise ValueError("Training split contains a non-training set.")
    if any(sample["set_id"] != "gp_set_0007"
           for sample in role_samples["validation"]):
        raise ValueError("Validation split contains a non-GP7 set.")

    arrays = {
        role: _split_arrays(samples) for role, samples in role_samples.items()
    }
    if set(arrays["train"]["sample_id"]).intersection(
        set(arrays["validation"]["sample_id"])
    ):
        raise ValueError("Training and validation share sample IDs.")
    manifest = {
        "schema": "ao_annotation_only_v1",
        "observation_frames": OBSERVED_FRAMES,
        "prediction_frames": FUTURE_FRAMES,
        "window_frames": WINDOW_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "frames_per_second": 30,
        "fps_basis": (
            "IDD-PeD paper reports GoPro Hero 8 at 30 FPS; all selected sets "
            "are GoPro annotations."
        ),
        "coordinate_convention": "absolute pixel xyxy (xtl,ytl,xbr,ybr)",
        "eligibility": (
            "60 consecutive annotated frames; outside boxes, malformed boxes, "
            "non-positive boxes, and out-of-frame boxes break a track segment; "
            "stride 30 within each contiguous valid segment."
        ),
        "training_videos": [list(item) for item in TRAIN_VIDEOS],
        "validation_videos": [list(item) for item in VALIDATION_VIDEOS],
        "official_test_sets_read": [],
        "historical_gp6_test_video_used": False,
        "normalization_fit_role": "train only, unique training observations/targets",
        "model_input_fields": ["observed_boxes[0:15]"],
        "future_fields_are_supervision_only": ["future_boxes[15:60]"],
        "ego_speed_used": False,
        "scene_context_used": False,
        "flow_used": False,
        "sources": sources,
        "roles": {},
    }
    for role, role_arrays in arrays.items():
        samples = role_samples[role]
        manifest["roles"][role] = {
            "sample_count": len(samples),
            "video_count": len({
                sample["video_id"] for sample in samples
            }),
            "distinct_track_count": len({
                (sample["video_id"], sample["track_id"]) for sample in samples
            }),
            "sample_ids_sha256": _sample_digest(role_arrays["sample_id"]),
        }

    output_root.mkdir(parents=True, exist_ok=False)
    for role, role_arrays in arrays.items():
        _atomic_write(output_root / f"{role}.npz", _npz_bytes(role_arrays))
    _atomic_write(
        output_root / "split_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--annotation-root", type=Path, default=ANNOTATION_ROOT)
    audit_parser.add_argument(
        "--output", type=Path,
        default=RESULTS / "ANNOTATION_INVENTORY.csv",
    )
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--annotation-root", type=Path, default=ANNOTATION_ROOT)
    prepare_parser.add_argument("--output-root", type=Path, default=SPLIT_ROOT)
    args = parser.parse_args()
    if args.command == "audit":
        rows = write_inventory(args.annotation_root, args.output)
        print(json.dumps({
            "inventory": str(args.output),
            "videos": len(rows),
            "parsed": sum(
                row.get("inventory_status") == "parsed_development_annotation"
                for row in rows
            ),
        }, indent=2))
    else:
        print(json.dumps(prepare_split(args.annotation_root, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
