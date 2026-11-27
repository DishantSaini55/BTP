import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from MTN.run_ao_v1 import run


def _role_arrays(set_id, video_ids, sample_prefix):
    count = len(video_ids)
    observed = np.zeros((count, 15, 4), dtype=np.float32)
    future = np.zeros((count, 45, 4), dtype=np.float32)
    for index in range(count):
        x = float(index * 3)
        observed[index, :, :] = np.asarray([x, 2, x + 10, 20], np.float32)
        future[index, :, :] = np.asarray([x + 1, 2, x + 11, 20], np.float32)
    return {
        "observed_boxes": observed,
        "future_boxes": future,
        "sample_id": np.asarray(
            [f"{sample_prefix}/track_{index}/00000" for index in range(count)],
            dtype=np.str_,
        ),
        "video_id": np.asarray(video_ids, dtype=np.str_),
        "set_id": np.asarray([set_id] * count, dtype=np.str_),
        "track_id": np.asarray([str(index) for index in range(count)], dtype=np.str_),
        "frame_start": np.asarray([index * 60 for index in range(count)], dtype=np.int32),
    }


class AnnotationOnlyRunnerTests(unittest.TestCase):
    def test_training_outputs_and_validation_only_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            train = _role_arrays(
                "gp_set_0001", ["gp_set_0001_vid_0001"] * 4, "train"
            )
            validation = _role_arrays(
                "gp_set_0007", ["gp_set_0007_vid_0004"] * 2, "validation"
            )
            np.savez_compressed(cache / "train.npz", **train)
            np.savez_compressed(cache / "validation.npz", **validation)
            (cache / "split_manifest.json").write_text(json.dumps({
                "schema": "ao_annotation_only_v1",
                "official_test_sets_read": [],
                "historical_gp6_test_video_used": False,
                "model_input_fields": ["observed_boxes[0:15]"],
                "roles": {
                    "train": {"sample_count": len(train["sample_id"])},
                    "validation": {"sample_count": len(validation["sample_id"])},
                },
            }), encoding="utf-8")

            output = root / "screen"
            args = SimpleNamespace(
                seed=20260928,
                max_epochs=1,
                cpu_threads=1,
                batch_size=2,
                patience=1,
                learning_rate=1e-3,
                wall_clock_budget_seconds=120,
                cache_dir=cache,
                output_root=output,
                models="AO-0,AO-2",
                validation_only=False,
                benchmark_only=False,
                resume=False,
                checkpoint=None,
                finalize_only=False,
            )
            run(args)

            ao2_run = output / "AO-2" / "seed_20260928"
            metrics = json.loads(
                (ao2_run / "metrics.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (ao2_run / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "completed")
            self.assertTrue(metrics["validation_complete"])
            self.assertEqual(metrics["epochs_completed"], 1)
            self.assertEqual(metrics["sample_count"], 2)
            self.assertTrue((ao2_run / "best_model.pt").is_file())
            self.assertTrue((ao2_run / "validation_predictions.npz").is_file())
            self.assertTrue((output / "AO_VALIDATION_SUMMARY.csv").is_file())

            validation_args = SimpleNamespace(
                seed=20260928,
                max_epochs=1,
                cpu_threads=1,
                batch_size=2,
                patience=1,
                learning_rate=1e-3,
                wall_clock_budget_seconds=120,
                cache_dir=cache,
                output_root=root / "validation_only",
                models="AO-2",
                validation_only=True,
                benchmark_only=False,
                resume=False,
                checkpoint=ao2_run / "best_model.pt",
                finalize_only=False,
            )
            run(validation_args)
            validation_metrics = json.loads(
                (root / "validation_only" / "AO-2"
                 / "validation_only_seed_20260928" / "metrics.json")
                .read_text(encoding="utf-8")
            )
            self.assertTrue(validation_metrics["validation_only"])
            self.assertEqual(
                validation_metrics["mse"], metrics["mse"]
            )

            resume_args = SimpleNamespace(
                seed=20260928,
                max_epochs=1,
                cpu_threads=1,
                batch_size=2,
                patience=1,
                learning_rate=1e-3,
                wall_clock_budget_seconds=0.001,
                cache_dir=cache,
                output_root=root / "resumable",
                models="AO-2",
                validation_only=False,
                benchmark_only=False,
                resume=False,
                checkpoint=None,
                finalize_only=False,
            )
            run(resume_args)
            resumable_run = (
                resume_args.output_root / "AO-2" / "seed_20260928"
            )
            incomplete = json.loads(
                (resumable_run / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(incomplete["state"], "incomplete")
            self.assertTrue((resumable_run / "latest_checkpoint.pt").is_file())

            resume_args.wall_clock_budget_seconds = 120
            resume_args.resume = True
            run(resume_args)
            completed = json.loads(
                (resumable_run / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
