"""One-time trusted migration from the monolithic flow pickle to role files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .dataset_roles_v1 import ROLE_CACHE_SCHEMA
    from .run_from_scratch_e1 import SPLIT
except ImportError:
    from dataset_roles_v1 import ROLE_CACHE_SCHEMA
    from run_from_scratch_e1 import SPLIT


SOURCE_CACHE = SPLIT / "flow_splits.pkl"
DESTINATION = SPLIT / "roles_v1"
ROLE_NAMES = ("train", "validation", "test")
SOURCE_METADATA_NAMES = (
    "flow_metadata.json",
    "split_metadata.json",
    "train_manifest.json",
    "validation_manifest.json",
    "test_manifest.json",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sample_digest(sample: dict[str, Any]) -> str:
    return _sha256(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> bytes:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes(path, payload)
    return payload


def _role_manifest(role: str, samples: list[dict[str, Any]], payload: bytes) -> dict[str, Any]:
    ids = [str(sample["id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in source role {role}.")
    videos = sorted({str(sample["video"]) for sample in samples})
    content_hash = _sha256(pickle.dumps(samples, protocol=pickle.HIGHEST_PROTOCOL))
    return {
        "schema": ROLE_CACHE_SCHEMA,
        "role": role,
        "sample_count": len(samples),
        "sample_ids": ids,
        "sample_ids_sha256": _sha256("\n".join(ids).encode("utf-8")),
        "video_ids": videos,
        "pickle_sha256": _sha256(payload),
        "sample_content_sha256": content_hash,
        "per_sample_sha256": [_sample_digest(sample) for sample in samples],
    }


def migrate(source: Path, destination: Path, trusted_local_cache: bool) -> dict[str, Any]:
    if not trusted_local_cache:
        raise PermissionError(
            "Pass --trusted-local-cache only after confirming the source pickle is trusted. "
            "Migration necessarily deserializes every role, including test, exactly once."
        )
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing role cache or partial migration: {destination}"
        )
    if not source.is_file():
        raise FileNotFoundError(f"Original flow cache not found: {source}")

    source_payload = source.read_bytes()
    source_hash = _sha256(source_payload)
    split = pickle.loads(source_payload)
    if not isinstance(split, dict) or any(role not in split for role in ROLE_NAMES):
        raise ValueError("Original cache must be a mapping with train, validation, and test roles.")

    source_metadata: dict[str, Any] = {}
    source_metadata_hashes: dict[str, str] = {}
    for name in SOURCE_METADATA_NAMES:
        metadata_path = source.parent / name
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Required original split metadata is missing: {metadata_path}")
        payload = metadata_path.read_bytes()
        source_metadata_hashes[name] = _sha256(payload)
        if name in {"flow_metadata.json", "split_metadata.json"}:
            source_metadata[name] = json.loads(payload.decode("utf-8"))

    expected_counts = source_metadata["flow_metadata.json"].get("counts", {})
    expected_videos = source_metadata["flow_metadata.json"].get("videos", {})
    for role in ROLE_NAMES:
        samples = split[role]
        if not isinstance(samples, list):
            raise TypeError(f"Original {role} role is not a list.")
        if expected_counts.get(role) != len(samples):
            raise ValueError(f"Original {role} count disagrees with flow_metadata.json.")
        if any(str(sample["video"]) != expected_videos.get(role) for sample in samples):
            raise ValueError(f"Original {role} includes an unexpected video.")
        manifest_path = source.parent / f"{role}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ids = [str(item["id"]) for item in manifest]
        sample_ids = [str(sample["id"]) for sample in samples]
        if len(manifest_ids) != len(sample_ids) or set(manifest_ids) != set(sample_ids):
            raise ValueError(f"Original {role} samples differ from its saved manifest.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        role_manifests: dict[str, dict[str, Any]] = {}
        role_payloads: dict[str, bytes] = {}
        for role in ROLE_NAMES:
            samples = split[role]
            payload = pickle.dumps(samples, protocol=pickle.HIGHEST_PROTOCOL)
            _write_bytes(staging / f"{role}.pkl", payload)
            restored = pickle.loads((staging / f"{role}.pkl").read_bytes())
            source_hashes = [_sample_digest(sample) for sample in samples]
            restored_hashes = [_sample_digest(sample) for sample in restored]
            if source_hashes != restored_hashes:
                raise RuntimeError(f"Migrated {role} samples failed content verification.")
            manifest = _role_manifest(role, samples, payload)
            role_manifests[role] = manifest
            role_payloads[role] = payload
            _write_json(staging / f"{role}_role_manifest.json", manifest)

        source_metadata_dir = staging / "source_metadata"
        source_metadata_dir.mkdir()
        for name in SOURCE_METADATA_NAMES:
            shutil.copyfile(source.parent / name, source_metadata_dir / name)
        for name in SOURCE_METADATA_NAMES:
            original = (source.parent / name).read_bytes()
            copied = (source_metadata_dir / name).read_bytes()
            if original != copied:
                raise RuntimeError(f"Copied source metadata did not verify: {name}")

        train_validation_metadata = {
            "schema": ROLE_CACHE_SCHEMA,
            "source_dataset_version": source_metadata["flow_metadata.json"].get("version"),
            "normalization_role": "train",
            "roles": {
                role: {
                    "sample_count": role_manifests[role]["sample_count"],
                    "sample_ids_sha256": role_manifests[role]["sample_ids_sha256"],
                    "pickle_sha256": role_manifests[role]["pickle_sha256"],
                }
                for role in ("train", "validation")
            },
        }
        _write_json(staging / "screening_manifest.json", train_validation_metadata)
        migration_record = {
            "schema": ROLE_CACHE_SCHEMA,
            "migrated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trusted_one_time_full_source_deserialization": True,
            "source_archive": str(source.resolve()),
            "source_archive_bytes": len(source_payload),
            "source_archive_sha256": source_hash,
            "source_metadata_sha256": source_metadata_hashes,
            "roles": role_manifests,
            "verification": {
                "sample_ids_and_per_sample_hashes_match": True,
                "copied_source_metadata_byte_identical": True,
                "flow_cache_recomputed": False,
            },
        }
        _write_json(staging / "migration_manifest.json", migration_record)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing cache: {destination}")
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return migration_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument(
        "--trusted-local-cache",
        action="store_true",
        help="Acknowledge this one-time migration unpickles train, validation, and test.",
    )
    args = parser.parse_args()
    result = migrate(args.source, args.destination, args.trusted_local_cache)
    print(json.dumps({
        "destination": str(args.destination.resolve()),
        "source_sha256": result["source_archive_sha256"],
        "role_counts": {
            role: result["roles"][role]["sample_count"] for role in ROLE_NAMES
        },
        "test_deserialized_during_trusted_migration": True,
        "flow_recomputed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
