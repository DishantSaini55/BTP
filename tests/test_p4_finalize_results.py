import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MTN.finalize_p4_results import _cluster_bootstrap, _paired_stats


class P4FinalizationTests(unittest.TestCase):
    def test_seed_paired_statistics_use_sample_sd_and_t_interval(self):
        result = _paired_stats([1.0, 2.0, 3.0])
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(result["mean_delta"], 2.0)
        self.assertEqual(result["sd_delta"], 1.0)
        self.assertAlmostEqual(result["ci95_t_low"], -0.48413771184)
        self.assertAlmostEqual(result["ci95_t_high"], 4.48413771184)
        self.assertEqual(result["dz"], 2.0)
        self.assertEqual(result["first_model_wins"], 0)

    def test_track_bootstrap_resamples_whole_pedestrians_reproducibly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = (1, 2, 3)
            for configuration, errors in (
                ("p4_a", {"track-a": 1.0, "track-b": 3.0}),
                ("p4_b", {"track-a": 2.0, "track-b": 5.0}),
            ):
                for seed in seeds:
                    run = root / configuration / str(seed)
                    run.mkdir(parents=True)
                    path = run / "validation_per_sample.csv"
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=("sample_id", "pid", "policy", "raw_pixel_mse"),
                        )
                        writer.writeheader()
                        for sample_id, (track_id, error) in enumerate(errors.items()):
                            writer.writerow({
                                "sample_id": f"{track_id}-{seed}-{sample_id}",
                                "pid": track_id,
                                "policy": "base",
                                "raw_pixel_mse": error,
                            })
            first = _cluster_bootstrap(
                root, seeds, "p4_a", "base", "p4_b", "base",
                "raw_pixel_mse", 1000, np.random.default_rng(42),
            )
            second = _cluster_bootstrap(
                root, seeds, "p4_a", "base", "p4_b", "base",
                "raw_pixel_mse", 1000, np.random.default_rng(42),
            )
        self.assertEqual(first, second)
        self.assertEqual(first["n_validation_tracks"], 2)
        self.assertEqual(first["n_seed_window_pairs"], 6)
        self.assertEqual(first["bootstrap_ci95_low"], -2.0)
        self.assertEqual(first["bootstrap_ci95_high"], -1.0)


if __name__ == "__main__":
    unittest.main()
