import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from MTN.finalize_official_test_macros_v1 import build_macro_rows
from MTN.finalize_official_test_v1 import _seed_paired_row


ROOT = Path("results_paper/AO_confirmation_v1")
SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


class OfficialFinalizerRegressionTests(unittest.TestCase):
    def test_neural_comparison_has_five_seed_interval(self):
        mse_by = {
            ("AO-2", seed): 100.0 + index
            for index, seed in enumerate(SEEDS)
        }
        mse_by.update({
            ("AO-0", 20260928): 105.0,
            **{("AO-1", seed): 110.0 + 2 * index for index, seed in enumerate(SEEDS)},
        })
        result = _seed_paired_row("AO-2", "AO-1", mse_by)
        self.assertEqual(result["n_seeds"], 5)
        self.assertEqual(result["first_model_wins"], 5)
        self.assertAlmostEqual(result["mean_delta_first_minus_second"], -12.0)
        self.assertLess(result["ci95_t_low"], result["ci95_t_high"])

    def test_deterministic_comparison_has_one_seed_and_no_t_interval_width(self):
        mse_by = {
            ("AO-RIDGE", 20260928): 10.0,
            ("AO-0", 20260928): 20.0,
        }
        result = _seed_paired_row("AO-RIDGE", "AO-0", mse_by)
        self.assertEqual(result["n_seeds"], 1)
        self.assertEqual(result["mean_delta_first_minus_second"], -10.0)
        self.assertEqual(result["ci95_t_low"], -10.0)
        self.assertEqual(result["ci95_t_high"], -10.0)

    def test_macro_aggregation_uses_equal_video_and_track_weighting(self):
        video_rows = [
            {"model_id": "AO-2", "seed": "7", "mse": "2.0"},
            {"model_id": "AO-2", "seed": "7", "mse": "6.0"},
        ]
        track_rows = [
            {"model_id": "AO-2", "seed": "7", "mse": "1.0"},
            {"model_id": "AO-2", "seed": "7", "mse": "2.0"},
            {"model_id": "AO-2", "seed": "7", "mse": "9.0"},
        ]
        self.assertEqual(build_macro_rows(video_rows, track_rows), [{
            "model_id": "AO-2",
            "seed": 7,
            "video_count": 2,
            "macro_video_mse": 4.0,
            "track_cluster_count": 3,
            "macro_track_mse": 4.0,
        }])


class Phase6BArtifactChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = json.loads((ROOT / "OFFICIAL_TEST_RESULTS.json").read_text(encoding="utf-8"))
        cls.v2 = json.loads((ROOT / "OFFICIAL_TEST_RESULTS_v2.json").read_text(encoding="utf-8"))
        cls.seed_rows = read_csv(ROOT / "OFFICIAL_TEST_RESULTS_v2" / "OFFICIAL_TEST_SEED_RESULTS.csv")
        cls.video_rows = read_csv(ROOT / "OFFICIAL_TEST_RESULTS_v2" / "OFFICIAL_TEST_PER_VIDEO.csv")
        cls.track_rows = read_csv(ROOT / "OFFICIAL_TEST_RESULTS_v2" / "OFFICIAL_TEST_PER_TRACK.csv")

    def test_v2_preserves_membership_and_corrects_only_ridge_seed_result(self):
        self.assertEqual(self.v1["source_video_membership"], self.v2["source_video_membership"])
        self.assertEqual(self.v1["source_xml_sha256"], self.v2["source_xml_sha256"])
        self.assertEqual(self.v2["sample_count"], 2615)
        self.assertEqual(self.v2["video_count"], 13)
        self.assertEqual(self.v2["track_count"], 861)
        before = {(row["model_id"], row["seed"]): row["mse"] for row in self.v1["seed_rows"]}
        after = {(row["model_id"], row["seed"]): row["mse"] for row in self.v2["seed_rows"]}
        changed = {key for key in before if before[key] != after[key]}
        self.assertEqual(changed, {("AO-RIDGE", 20260928)})
        self.assertAlmostEqual(after[("AO-RIDGE", 20260928)], 2889.71826171875)

    def test_v2_csv_and_v3_summary_are_consistent(self):
        self.assertEqual(len(self.seed_rows), 18)
        json_rows = {(row["model_id"], row["seed"]): row for row in self.v2["seed_rows"]}
        for row in self.seed_rows:
            expected = json_rows[(row["model_id"], int(row["seed"]))]
            self.assertAlmostEqual(float(row["mse"]), expected["mse"])
        summary = {row["model_id"]: row for row in read_csv(ROOT / "OFFICIAL_TEST_MODEL_SUMMARY_v3.csv")}
        for model_id, run_count in (("AO-0", 1), ("AO-CP", 1), ("AO-RIDGE", 1), ("AO-1", 5), ("AO-2", 5), ("AO-3", 5)):
            values = [float(row["mse"]) for row in self.seed_rows if row["model_id"] == model_id]
            self.assertEqual(int(summary[model_id]["n_completed_runs"]), run_count)
            self.assertAlmostEqual(float(summary[model_id]["mse_mean"]), float(np.mean(values)))

    def test_v3_macro_rows_match_corrected_per_video_and_per_track_data(self):
        expected = build_macro_rows(self.video_rows, self.track_rows)
        actual = read_csv(ROOT / "OFFICIAL_TEST_MACRO_RESULTS.csv")
        self.assertEqual(len(actual), 18)
        for expected_row, actual_row in zip(expected, actual):
            self.assertEqual(expected_row["model_id"], actual_row["model_id"])
            self.assertEqual(expected_row["seed"], int(actual_row["seed"]))
            self.assertEqual(expected_row["video_count"], int(actual_row["video_count"]))
            self.assertEqual(expected_row["track_cluster_count"], int(actual_row["track_cluster_count"]))
            self.assertAlmostEqual(expected_row["macro_video_mse"], float(actual_row["macro_video_mse"]))
            self.assertAlmostEqual(expected_row["macro_track_mse"], float(actual_row["macro_track_mse"]))


if __name__ == "__main__":
    unittest.main()
