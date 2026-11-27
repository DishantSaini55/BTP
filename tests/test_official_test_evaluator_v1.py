import tempfile
import unittest
from pathlib import Path

import numpy as np

from MTN.official_test_evaluator_v1 import pooled_raw_pixel_mse


class OfficialTestEvaluatorPreparationTests(unittest.TestCase):
    def test_pooled_mse_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            pooled_raw_pixel_mse(
                np.zeros((2, 45, 4), dtype=np.float32),
                np.zeros((2, 44, 4), dtype=np.float32),
            )

    def test_pooled_mse_is_exact_for_known_arrays(self):
        predicted = np.ones((2, 45, 4), dtype=np.float32)
        target = np.zeros((2, 45, 4), dtype=np.float32)
        self.assertAlmostEqual(pooled_raw_pixel_mse(predicted, target), 1.0)


if __name__ == "__main__":
    unittest.main()
