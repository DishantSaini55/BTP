import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from MTN.dataset_roles_v1 import ROLE_CACHE_SCHEMA, load_train_validation
from MTN.migrate_flow_cache_v1 import migrate


def _sample(role, index, video):
    sample_id = f"{role}/{video}/{index}"
    return {
        "bbox": np.full((60, 4), index, dtype=np.float32),
        "obd_speed": np.full((60, 1), index + 1, dtype=np.float32),
        "ego_op_flow": np.zeros((60, 64, 2), dtype=np.float32),
        "ped_op_flow": np.zeros((60, 9, 2), dtype=np.float32),
        "scene_context": np.zeros((1, 5), dtype=np.int64),
        "occlusion": np.zeros(60, dtype=np.int64),
        "id": sample_id,
        "video": video,
        "pid": f"ped-{index}",
        "frame_start": 0,
        "frame_end": 59,
    }


def _write_test_role(root, role, samples):
    payload = pickle.dumps(samples, protocol=pickle.HIGHEST_PROTOCOL)
    (root / f"{role}.pkl").write_bytes(payload)
    ids = [sample["id"] for sample in samples]
    manifest = {
        "schema": ROLE_CACHE_SCHEMA,
        "role": role,
        "sample_count": len(samples),
        "sample_ids": ids,
        "sample_ids_sha256": "fixture",
        "video_ids": sorted({sample["video"] for sample in samples}),
        "pickle_sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }
    (root / f"{role}_role_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class RoleSeparatedCacheTests(unittest.TestCase):
    def test_validation_loader_succeeds_when_test_cache_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_test_role(root, "train", [_sample("train", 0, "video-a")])
            _write_test_role(root, "validation", [_sample("validation", 1, "video-b")])
            self.assertFalse((root / "test.pkl").exists())

            opened = []
            original_open = Path.open

            def tracked_open(path, *args, **kwargs):
                opened.append(path.name)
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", tracked_open):
                datasets, stats, metadata = load_train_validation(root)

            self.assertEqual(set(datasets), {"train", "validation"})
            self.assertEqual(len(datasets["train"]), 1)
            self.assertEqual(len(datasets["validation"]), 1)
            self.assertEqual(metadata["normalization_fit_role"], "train")
            self.assertEqual(float(stats[0][0]), 0.0)
            self.assertNotIn("test.pkl", opened)

    def test_loader_rejects_cross_video_role_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_test_role(root, "train", [_sample("train", 0, "shared-video")])
            _write_test_role(root, "validation", [_sample("validation", 1, "shared-video")])
            with self.assertRaisesRegex(ValueError, "share a video"):
                load_train_validation(root)

    def test_loader_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_test_role(root, "train", [_sample("train", 0, "video-a")])
            _write_test_role(root, "validation", [_sample("validation", 1, "video-b")])
            manifest_path = root / "train_role_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sample_ids"] = ["not-the-saved-id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample IDs differ"):
                load_train_validation(root)

    def test_migration_requires_explicit_trust_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pkl"
            destination = root / "roles"
            source.write_bytes(b"not opened without explicit trust")
            with self.assertRaises(PermissionError):
                migrate(source, destination, trusted_local_cache=False)
            self.assertFalse(destination.exists())

            destination.mkdir()
            with self.assertRaises(FileExistsError):
                migrate(source, destination, trusted_local_cache=True)


if __name__ == "__main__":
    unittest.main()
