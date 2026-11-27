"""Strict role-separated loading for validation-only Phase 4 experiments."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_from_scratch_e1 import RESULTS, SPLIT, WindowDataset
except ImportError:
    from run_from_scratch_e1 import RESULTS, SPLIT, WindowDataset


ROLE_CACHE_ROOT = SPLIT / "roles_v1"
ROLE_CACHE_SCHEMA = "gp6_independent_roles_v1"
SCREENING_ROLES = ("train", "validation")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_role(cache_root: Path, role: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if role not in SCREENING_ROLES:
        raise ValueError(f"Screening loader does not support role {role!r}.")
    cache_path = cache_root / f"{role}.pkl"
    manifest_path = cache_root / f"{role}_role_manifest.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Required {role} role cache not found: {cache_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required {role} role manifest not found: {manifest_path}")

    with cache_path.open("rb") as handle:
        payload = handle.read()
    if _sha256(payload) != _manifest_hash(manifest_path):
        raise ValueError(f"{role} role cache checksum does not match its manifest.")
    samples = pickle.loads(payload)
    if not isinstance(samples, list):
        raise TypeError(f"{role} role cache must contain a list of samples.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROLE_CACHE_SCHEMA or manifest.get("role") != role:
        raise ValueError(f"Invalid {role} role manifest schema or role label.")
    sample_ids = [str(sample["id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{role} role cache contains duplicate sample IDs.")
    if sample_ids != manifest.get("sample_ids"):
        raise ValueError(f"{role} role cache sample IDs differ from its manifest.")
    videos = sorted({str(sample["video"]) for sample in samples})
    if videos != manifest.get("video_ids"):
        raise ValueError(f"{role} role cache video IDs differ from its manifest.")
    if len(samples) != manifest.get("sample_count"):
        raise ValueError(f"{role} role cache count differs from its manifest.")
    return samples, manifest


def _manifest_hash(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    value = manifest.get("pickle_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid pickle checksum in role manifest: {path}")
    return value


def load_train_validation(
    cache_root: Path | str | None = None,
) -> tuple[dict[str, WindowDataset], tuple[np.ndarray, ...], dict[str, Any]]:
    """Load only the train and validation files; the test path is never formed."""
    root = Path(cache_root) if cache_root is not None else ROLE_CACHE_ROOT
    train_samples, train_manifest = _load_role(root, "train")
    validation_samples, validation_manifest = _load_role(root, "validation")
    if not train_samples or not validation_samples:
        raise ValueError("Train and validation role caches must both be non-empty.")

    train_videos = set(train_manifest["video_ids"])
    validation_videos = set(validation_manifest["video_ids"])
    if train_videos.intersection(validation_videos):
        raise ValueError("Train and validation roles share a video.")

    train_boxes = np.concatenate([sample["bbox"] for sample in train_samples])
    train_speed = np.concatenate([sample["obd_speed"] for sample in train_samples])
    stats = (
        train_boxes.mean(0),
        np.maximum(train_boxes.std(0), 1e-6),
        train_speed.mean(0),
        np.maximum(train_speed.std(0), 1e-6),
    )
    datasets = {
        "train": WindowDataset(train_samples, stats),
        "validation": WindowDataset(validation_samples, stats),
    }
    metadata = {
        "schema": ROLE_CACHE_SCHEMA,
        "cache_root": str(root.resolve()),
        "roles_loaded": {
            "train": {
                "sample_count": train_manifest["sample_count"],
                "sample_ids_sha256": train_manifest["sample_ids_sha256"],
                "pickle_sha256": train_manifest["pickle_sha256"],
            },
            "validation": {
                "sample_count": validation_manifest["sample_count"],
                "sample_ids_sha256": validation_manifest["sample_ids_sha256"],
                "pickle_sha256": validation_manifest["pickle_sha256"],
            },
        },
        "normalization_fit_role": "train",
    }
    return datasets, stats, metadata
