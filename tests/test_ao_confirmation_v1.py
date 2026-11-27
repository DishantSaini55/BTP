import unittest

import numpy as np

from MTN.ao_confirmation_v1 import (
    fit_ridge_cv_residual,
    observation_features,
    predict_ridge_cv_residual,
)


class AOConfirmationBaselineTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(4, dtype=np.float32)
        self.std = np.ones(4, dtype=np.float32)
        rng = np.random.default_rng(42)
        positions = rng.normal(size=(8, 1, 4)).astype(np.float32)
        increments = rng.normal(scale=0.1, size=(8, 15, 4)).astype(np.float32)
        self.observed = positions + np.cumsum(increments, axis=1)
        final = self.observed[:, -1:, :]
        velocity = (self.observed[:, -1:, :] - self.observed[:, -3:-2, :]) / 2
        steps = np.arange(1, 46, dtype=np.float32).reshape(1, 45, 1)
        cv = final + steps * velocity
        residual = rng.normal(scale=0.01, size=(8, 45, 4)).astype(np.float32)
        self.future = cv + residual

    def test_features_use_only_observed_boxes_and_have_fixed_shape(self):
        features = observation_features(self.observed, self.mean, self.std)
        self.assertEqual(features.shape, (8, 64))
        np.testing.assert_array_equal(features[:, -4:], self.observed[:, -1, :])

    def test_ridge_direct_multioutput_prediction_is_finite(self):
        features = observation_features(self.observed, self.mean, self.std)
        model = fit_ridge_cv_residual(
            features,
            self.observed,
            self.future,
            self.mean,
            self.std,
            alpha=1.0,
        )
        prediction = predict_ridge_cv_residual(
            model, self.observed, self.mean, self.std
        )
        self.assertEqual(prediction.shape, (8, 45, 4))
        self.assertTrue(np.isfinite(prediction).all())

    def test_ridge_rejects_nonpositive_regularization(self):
        features = observation_features(self.observed, self.mean, self.std)
        with self.assertRaises(ValueError):
            fit_ridge_cv_residual(
                features,
                self.observed,
                self.future,
                self.mean,
                self.std,
                alpha=0.0,
            )


if __name__ == "__main__":
    unittest.main()
