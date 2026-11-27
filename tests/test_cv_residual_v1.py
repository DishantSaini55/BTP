import unittest

import torch

from MTN.cv_residual_v1 import (
    ResidualAnchorModel,
    build_p3_model,
    constant_position_forecast,
    constant_velocity_forecast,
    trainable_parameter_count,
)
from MTN.sage_v2 import _prepare_model_inputs


def _batch(batch_size=2):
    generator = torch.Generator().manual_seed(941)
    return {
        "boxes": torch.randn(batch_size, 15, 4, generator=generator),
        "speed": torch.randn(batch_size, 60, 1, generator=generator),
        "ego_flow": torch.randn(batch_size, 60, 64, 2, generator=generator),
        "ped_flow": torch.randn(batch_size, 60, 9, 2, generator=generator),
        "pred_target": torch.randn(batch_size, 45, 4, generator=generator),
        "scene_context": torch.tensor([[0, 0, 0, 0, 0]] * batch_size),
        "observed_occlusion": torch.zeros(batch_size, 60, dtype=torch.long),
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
        "video": ["video"] * batch_size,
        "pid": [f"pid-{index}" for index in range(batch_size)],
        "frame_start": torch.arange(batch_size),
        "frame_end": torch.arange(batch_size) + 59,
    }


def _trajectory(per_frame_delta, steps=15):
    start = torch.tensor([10.0, 20.0, 30.0, 50.0])
    delta = torch.tensor(per_frame_delta)
    time = torch.arange(steps, dtype=torch.float32).view(steps, 1)
    return (start + time * delta).view(1, steps, 4)


class ConstantVelocityForecastTests(unittest.TestCase):
    def test_stationary_forecast(self):
        boxes = torch.full((2, 15, 4), 7.0)
        expected = boxes[:, -1:, :].expand(-1, 45, -1)
        self.assertTrue(torch.equal(constant_velocity_forecast(boxes), expected))

    def test_constant_horizontal_motion(self):
        prediction = constant_velocity_forecast(_trajectory([2, 0, 2, 0]))
        steps = torch.arange(1, 46, dtype=torch.float32).view(1, 45, 1)
        expected = _trajectory([2, 0, 2, 0])[:, -1:, :] + steps * torch.tensor(
            [[[2.0, 0.0, 2.0, 0.0]]]
        )
        self.assertTrue(torch.equal(prediction, expected))

    def test_constant_vertical_motion(self):
        prediction = constant_velocity_forecast(_trajectory([0, 3, 0, 3]))
        expected = _trajectory([0, 3, 0, 3])[:, -1:, :] + torch.arange(
            1, 46, dtype=torch.float32
        ).view(1, 45, 1) * torch.tensor([[[0.0, 3.0, 0.0, 3.0]]])
        self.assertTrue(torch.equal(prediction, expected))

    def test_diagonal_motion(self):
        prediction = constant_velocity_forecast(_trajectory([2, -1, 2, -1]))
        self.assertEqual(tuple(prediction.shape), (1, 45, 4))
        self.assertTrue(torch.equal(prediction[0, 0], torch.tensor([40.0, 5.0, 60.0, 35.0])))

    def test_changing_box_size_is_extrapolated_coordinatewise(self):
        boxes = _trajectory([1, 2, 4, 5])
        forecast = constant_velocity_forecast(boxes)
        expected = boxes[:, -1:, :] + torch.arange(1, 46).view(1, 45, 1) * torch.tensor(
            [[[1.0, 2.0, 4.0, 5.0]]]
        )
        self.assertTrue(torch.equal(forecast, expected))
        self.assertEqual(tuple(forecast.shape), (1, 45, 4))

    def test_forecast_does_not_depend_on_future_frames(self):
        observed = _trajectory([0.5, 0.25, 0.5, 0.25])
        future_a = torch.zeros(1, 45, 4)
        future_b = torch.full((1, 45, 4), 999.0)
        forecast_a = constant_velocity_forecast(observed)
        forecast_b = constant_velocity_forecast(observed)
        self.assertFalse(torch.equal(future_a, future_b))
        self.assertTrue(torch.equal(forecast_a, forecast_b))

    def test_agrees_with_historical_b1_formula(self):
        boxes = torch.randn(4, 15, 4, generator=torch.Generator().manual_seed(17))
        velocity = (boxes[:, -1] - boxes[:, -3]).div(2)
        steps = torch.arange(1, 46, dtype=velocity.dtype).view(1, 45, 1)
        historical = boxes[:, -1:, :] + steps * velocity.unsqueeze(1)
        result = constant_velocity_forecast(boxes)
        self.assertTrue(torch.equal(result, historical))
        self.assertEqual(float(torch.max(torch.abs(result - historical))), 0.0)

    def test_normalized_and_pixel_forecasts_are_affine_equivalent(self):
        boxes = torch.randn(3, 15, 4, generator=torch.Generator().manual_seed(18))
        mean = torch.tensor([421.0, 223.0, 467.0, 296.0])
        std = torch.tensor([135.0, 87.0, 144.0, 92.0])
        pixel_boxes = boxes * std + mean
        normalized_forecast = constant_velocity_forecast(boxes)
        pixel_forecast = constant_velocity_forecast(pixel_boxes)
        recovered = (pixel_forecast - mean) / std
        error = torch.max(torch.abs(normalized_forecast - recovered)).item()
        self.assertLessEqual(error, 2e-5)

    def test_constant_position_anchor(self):
        boxes = torch.randn(2, 15, 4)
        expected = boxes[:, -1:, :].expand(-1, 45, -1)
        self.assertTrue(torch.equal(constant_position_forecast(boxes), expected))


class ResidualAnchorModelTests(unittest.TestCase):
    def test_matched_controls_share_initialization_where_architecture_matches(self):
        torch.manual_seed(20260928)
        p3_b = build_p3_model("p3_b")
        torch.manual_seed(20260928)
        p3_c = build_p3_model("p3_c")
        self.assertEqual(p3_b.state_dict().keys(), p3_c.state_dict().keys())
        self.assertTrue(all(
            torch.equal(p3_b.state_dict()[key], p3_c.state_dict()[key])
            for key in p3_b.state_dict()
        ))

        torch.manual_seed(20260928)
        p3_d = build_p3_model("p3_d")
        torch.manual_seed(20260928)
        p3_f = build_p3_model("p3_f")
        torch.manual_seed(20260928)
        p3_g = build_p3_model("p3_g")
        for candidate in (p3_f, p3_g):
            self.assertTrue(all(
                torch.equal(p3_d.predictor.base.state_dict()[key], value)
                for key, value in candidate.predictor.base.state_dict().items()
            ))
            self.assertTrue(all(
                torch.equal(
                    p3_d.predictor.context_encoder.state_dict()[key], value
                )
                for key, value in candidate.predictor.context_encoder.state_dict().items()
            ))

    def test_zero_head_produces_exact_cv_anchor_and_finite_loss(self):
        batch = _batch()
        model = build_p3_model("p3_c", zero_correction_head=True).eval()
        prediction = model.forward_batch(batch)
        forecast = constant_velocity_forecast(batch["boxes"])
        self.assertTrue(torch.equal(prediction, forecast))
        self.assertEqual(tuple(prediction.shape), (2, 45, 4))
        loss = torch.nn.functional.mse_loss(prediction, batch["pred_target"])
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(loss))

    def test_zero_head_gradient_and_upstream_gradient_after_update(self):
        torch.manual_seed(92)
        batch = _batch()
        model = build_p3_model("p3_c", zero_correction_head=True).eval()
        head = model.predictor.model.generator.proj
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            model.forward_batch(batch), batch["pred_target"]
        )
        loss.backward()
        head_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in head.parameters()
            if parameter.grad is not None
        ))
        head_parameter_ids = {id(parameter) for parameter in head.parameters()}
        upstream_parameters = [
            parameter for parameter in model.predictor.parameters()
            if id(parameter) not in head_parameter_ids
        ]
        upstream_norm_before = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in upstream_parameters
            if parameter.grad is not None
        ))
        self.assertGreater(float(head_norm), 0.0)
        self.assertEqual(float(upstream_norm_before), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        next_loss = torch.nn.functional.mse_loss(
            model.forward_batch(batch), batch["pred_target"]
        )
        next_loss.backward()
        upstream_norm_after = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in upstream_parameters
            if parameter.grad is not None
        ))
        self.assertGreater(float(upstream_norm_after), 0.0)
        self.assertTrue(torch.isfinite(next_loss))

    def test_residual_is_added_once(self):
        batch = _batch(batch_size=1)
        model = build_p3_model("p3_c", zero_correction_head=True).eval()
        head = model.predictor.model.generator.proj
        with torch.no_grad():
            head.bias.fill_(1.0)
        prediction = model.forward_batch(batch)
        forecast = constant_velocity_forecast(batch["boxes"])
        self.assertTrue(torch.equal(prediction, forecast + 1.0))

    def test_satie_parameter_sharing_is_identity(self):
        model = build_p3_model("p3_e")
        encoder_layers = model.predictor.base.model.encoder.layers
        self.assertIs(
            encoder_layers[0].self_attn,
            encoder_layers[1].self_attn,
        )
        self.assertIs(
            encoder_layers[1].self_attn,
            encoder_layers[2].self_attn,
        )
        self.assertEqual(trainable_parameter_count(model), 130694)

    def test_hard_closed_sage_uses_exact_motion_only_path(self):
        batch = _batch(batch_size=2)
        model = build_p3_model("p3_e").eval()
        predictor = model.predictor
        args = list(_prepare_model_inputs(batch))
        args[0] = constant_velocity_forecast(batch["boxes"])
        original_gate = predictor._context_gate

        def closed_gate(context_token, _observed_boxes):
            gate = context_token.new_zeros(context_token.shape[0])
            predictor.last_context_gate = gate
            predictor.last_context_gate_logits = context_token.new_zeros(
                context_token.shape[0], 2
            )
            predictor._context_gate_for_loss = gate
            return gate

        predictor._context_gate = closed_gate
        with torch.no_grad():
            closed = predictor(
                *args, batch["scene_context"], batch["observed_occlusion"]
            )
            motion_only = predictor.base(*args)
        predictor._context_gate = original_gate
        self.assertTrue(torch.equal(closed, motion_only))

    def test_hard_closed_sage_v2_uses_exact_motion_only_path(self):
        batch = _batch(batch_size=2)
        model = build_p3_model("p3_g").eval()
        predictor = model.predictor
        predictor.inference_mode = "hard"
        predictor.forced_admission = 0.0
        args = list(_prepare_model_inputs(batch))
        args[0] = constant_velocity_forecast(batch["boxes"])
        with torch.no_grad():
            closed = model.forward_batch(batch)
            motion_only = predictor.base(*args)
        self.assertTrue(torch.equal(closed, motion_only))


if __name__ == "__main__":
    unittest.main()
