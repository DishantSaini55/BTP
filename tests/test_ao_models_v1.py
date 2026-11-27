import unittest

import torch

from MTN.ao_models_v1 import (
    AnnotationOnlyTransformer,
    build_ao_model,
    constant_velocity_forecast,
    unique_trainable_parameter_count,
    verify_satie_identity,
)
from MTN.cv_residual_v1 import constant_velocity_forecast as historical_cv


class AnnotationOnlyModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(20260928)
        self.observed = torch.randn(2, 15, 4, generator=generator)

    def test_constant_velocity_matches_historical_formula(self):
        self.assertTrue(torch.equal(
            constant_velocity_forecast(self.observed),
            historical_cv(self.observed),
        ))

    def test_all_trainable_models_return_finite_45_step_predictions(self):
        for model_id in ("AO-1", "AO-2", "AO-3"):
            torch.manual_seed(20260928)
            model = build_ao_model(model_id).eval()
            prediction = model(self.observed)
            self.assertEqual(tuple(prediction.shape), (2, 45, 4))
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            self.assertGreater(unique_trainable_parameter_count(model), 0)
            self.assertFalse(any(
                "flow" in name.lower() or "speed" in name.lower()
                for name, _ in model.named_modules()
            ))

    def test_prediction_loss_has_finite_nonzero_model_gradients(self):
        torch.manual_seed(19)
        model = build_ao_model("AO-2").train()
        prediction = model(self.observed)
        target = torch.zeros_like(prediction)
        torch.nn.functional.mse_loss(prediction, target).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))
        total_norm = torch.sqrt(sum(grad.square().sum() for grad in gradients))
        self.assertGreater(float(total_norm), 0.0)

    def test_zero_correction_is_exactly_the_selected_anchor(self):
        for anchor in ("constant_position", "constant_velocity"):
            torch.manual_seed(13)
            model = AnnotationOnlyTransformer(anchor).eval()
            with torch.no_grad():
                model.generator.weight.zero_()
                model.generator.bias.zero_()
                actual = model(self.observed)
            if anchor == "constant_velocity":
                expected = constant_velocity_forecast(self.observed)
            else:
                expected = self.observed[:, -1:, :].expand(-1, 45, -1)
            self.assertTrue(torch.equal(actual, expected))

    def test_residual_is_added_once(self):
        torch.manual_seed(14)
        model = build_ao_model("AO-2").eval()
        with torch.no_grad():
            model.generator.weight.zero_()
            model.generator.bias.fill_(1.0)
            prediction = model(self.observed)
        expected = constant_velocity_forecast(self.observed) + 1.0
        self.assertTrue(torch.equal(prediction, expected))

    def test_satie_shares_attention_module_and_counts_unique_parameters(self):
        torch.manual_seed(20260928)
        model = build_ao_model("AO-3")
        self.assertTrue(verify_satie_identity(model))
        attention = model.encoder_layers[0].self_attn
        self.assertTrue(all(
            layer.self_attn is attention for layer in model.encoder_layers
        ))
        self.assertEqual(
            unique_trainable_parameter_count(model),
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertLess(
            unique_trainable_parameter_count(model),
            unique_trainable_parameter_count(build_ao_model("AO-2")),
        )

    def test_position_and_velocity_models_have_matched_initialization(self):
        torch.manual_seed(20260928)
        position = build_ao_model("AO-1")
        torch.manual_seed(20260928)
        velocity = build_ao_model("AO-2")
        self.assertEqual(position.state_dict().keys(), velocity.state_dict().keys())
        self.assertTrue(all(
            torch.equal(position.state_dict()[key], value)
            for key, value in velocity.state_dict().items()
        ))

    def test_bad_input_shape_is_rejected(self):
        model = build_ao_model("AO-2")
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 14, 4))


if __name__ == "__main__":
    unittest.main()
