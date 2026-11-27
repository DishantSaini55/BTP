import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MTN.finalize_ao_confirmation_v1 import (
    _correction_rms,
    _paired_stats,
    _track_bootstrap,
)


class AOConfirmationFinalizationTests(unittest.TestCase):
    def test_paired_statistics_use_seed_level_t_interval(self):
        result = _paired_stats([-1.0, -2.0, -3.0, -4.0, -5.0])
        self.assertEqual(result["n_seeds"], 5)
        self.assertEqual(result["degrees_of_freedom"], 4)
        self.assertEqual(result["first_model_wins"], 5)
        self.assertAlmostEqual(result["mean_delta_first_minus_second"], -3.0)
        self.assertIsNotNone(result["ci95_t_low"])
        self.assertIsNotNone(result["ci95_t_high"])

    def test_track_bootstrap_is_paired_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in range(5):
                for model_id in ("AO-1", "AO-0"):
                    run = root / model_id / f"seed_{seed}"
                    run.mkdir(parents=True)
                    offset = 1.0 if model_id == "AO-1" else 0.0
                    rows = [
                        {
                            "sample_id": f"{seed}-{track}",
                            "video_id": "gp_set_0007_vid_0004",
                            "track_id": str(track),
                            "mse": offset + float(track),
                        }
                        for track in (0, 1)
                    ]
                    with (run / "validation_per_sample.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            first = _track_bootstrap(
                root, "AO-1", "AO-0", tuple(range(5)), "mse", 500, 7
            )
            second = _track_bootstrap(
                root, "AO-1", "AO-0", tuple(range(5)), "mse", 500, 7
            )
            self.assertEqual(first, second)
            self.assertEqual(first["n_track_clusters"], 2)
            self.assertAlmostEqual(first["bootstrap_ci95_low"], 1.0)
            self.assertAlmostEqual(first["bootstrap_ci95_high"], 1.0)

    def test_correction_rms_uses_declared_anchor(self):
        observed = np.zeros((2, 15, 4), dtype=np.float32)
        observed[:, :, 0] = np.arange(15, dtype=np.float32)
        observed[:, :, 2] = np.arange(15, dtype=np.float32) + 2
        velocity = (observed[:, -1, :] - observed[:, -3, :]) / 2.0
        steps = np.arange(1, 46, dtype=np.float32).reshape(1, 45, 1)
        cv = observed[:, -1:, :] + steps * velocity[:, None, :]
        predicted = cv.copy()
        predicted[:, :, 1] += 3.0
        correction = _correction_rms(predicted, observed, "AO-2")
        self.assertTrue(np.all(np.isfinite(correction)))
        self.assertTrue(np.allclose(correction, 3.0))


if __name__ == "__main__":
    unittest.main()
