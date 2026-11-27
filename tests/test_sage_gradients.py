import unittest

import torch

from MTN.sage_gradient_diagnostics import _case, make_inputs, make_model, run_all


class SageGradientAuditTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260928)
        torch.set_num_threads(1)

    def test_all_open_gate_has_prediction_loss_gradients(self):
        result = _case("all_open")
        self.assertEqual(result["gate_values"], [1.0, 1.0, 1.0])
        self.assertTrue(result["predictions_finite"])
        self.assertTrue(result["gradients_finite"])
        self.assertGreater(result["gate_gradient"]["l2_norm"], 0.0)
        self.assertGreater(result["context_encoder_gradient"]["l2_norm"], 0.0)

    def test_all_closed_prediction_loss_disconnects_sage_parameters(self):
        result = _case("all_closed")
        self.assertEqual(result["gate_values"], [0.0, 0.0, 0.0])
        self.assertEqual(
            result["gate_gradient"]["grad_absent_count"],
            result["gate_gradient"]["parameter_count"],
        )
        self.assertEqual(
            result["context_encoder_gradient"]["grad_absent_count"],
            result["context_encoder_gradient"]["parameter_count"],
        )
        self.assertIsNone(result["gate_gradient"]["l2_norm"])
        self.assertIsNone(result["context_encoder_gradient"]["l2_norm"])

    def test_mixed_batch_only_open_rows_contribute_to_sage_gradient(self):
        result = _case("mixed")
        self.assertEqual(result["gate_values"], [1.0, 0.0, 1.0])
        self.assertGreater(result["gate_gradient"]["l2_norm"], 0.0)
        self.assertGreater(result["context_encoder_gradient"]["l2_norm"], 0.0)
        closed = result["closed_sample_gradient"]
        self.assertEqual(closed["grad_nonzero_count"], 0)
        self.assertEqual(closed["grad_absent_count"], 0)
        self.assertEqual(closed["grad_zero_count"], 16)

    def test_continuous_soft_admission_retains_gradients(self):
        result = _case("soft")
        self.assertTrue(all(0.0 < value < 1.0 for value in result["gate_values"]))
        self.assertGreater(result["gate_gradient"]["l2_norm"], 0.0)
        self.assertGreater(result["context_encoder_gradient"]["l2_norm"], 0.0)
        self.assertTrue(result["gradients_finite"])

    def test_fallback_and_satie_sharing_invariance(self):
        result = run_all()
        self.assertTrue(result["all_closed_fallback_bitwise_equal"])
        self.assertTrue(result["all_closed_expected_grad_absent"])
        self.assertTrue(result["satie_sharing"]["sharing_evidence"]["verified"])
        self.assertTrue(result["satie_sharing"]["bitwise_equal"])
        self.assertEqual(result["satie_sharing"]["max_abs_prediction_difference"], 0.0)

    def test_model_predictions_are_finite_for_deterministic_inputs(self):
        model = make_model("soft").eval()
        batch = make_inputs()
        boxes = batch["boxes"]
        residual = boxes[:, -1:, :].expand(-1, batch["pred_target"].shape[1], -1)
        batch_size = boxes.shape[0]
        with torch.no_grad():
            prediction = model(
                residual,
                boxes,
                batch["speed"],
                batch["ego_flow"],
                batch["ped_flow"],
                torch.zeros_like(batch["pred_target"]),
                torch.ones(batch_size, 1, boxes.shape[1]),
                torch.ones(batch_size, 1, 1),
                torch.ones(batch_size, 1, 64),
                torch.ones(batch_size, 1, 9),
                batch["scene_context"],
                batch["observed_occlusion"],
            )
        self.assertEqual(tuple(prediction.shape), (batch_size, 2, 4))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
