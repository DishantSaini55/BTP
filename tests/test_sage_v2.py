import unittest

import torch
import torch.nn as nn

from MTN.c4_encoder_shared import verify_c4_sharing
from MTN.sage_v2 import (
    ContextPriorMultiHeadAttention,
    DifferentiableSAGEV2,
    _prepare_model_inputs,
    build_satie_model,
    unique_trainable_parameter_count,
)
from MTN.transformer.multihead_attention import MultiHeadAttention


def make_batch(batch_size=2):
    torch.manual_seed(20260928)
    return {
        "boxes": torch.randn(batch_size, 15, 4),
        "speed": torch.randn(batch_size, 15, 1),
        "ego_flow": torch.randn(batch_size, 14, 64, 2),
        "ped_flow": torch.randn(batch_size, 14, 9, 2),
        "pred_target": torch.randn(batch_size, 45, 4),
        "scene_context": torch.randint(0, 3, (batch_size, 5)),
        "observed_occlusion": torch.randint(0, 4, (batch_size, 15)),
    }


def make_model(gate_input_mode="original"):
    torch.manual_seed(20260928)
    return DifferentiableSAGEV2(
        15,
        4,
        4,
        4,
        gate_input_mode=gate_input_mode,
        N=3,
        d_model=32,
        d_ff=128,
        h=4,
        dropout=0.0,
    )


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits.expand(inputs.shape[0], -1)


class SageV2Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_attention_prior_normalizes_and_preserves_unrelated_mask(self):
        torch.manual_seed(7)
        source = MultiHeadAttention(h=2, d_model=8, dropout=0.0).eval()
        attention = ContextPriorMultiHeadAttention(
            source
        ).eval()
        query = torch.randn(2, 3, 8)
        memory = torch.randn(2, 4, 8)
        mask = torch.tensor([[[1, 1, 0, 1]], [[1, 0, 1, 1]]], dtype=torch.float32)
        probability = torch.tensor([0.2, 0.8], requires_grad=True)
        attention.set_context_probability(probability)

        result = attention(query, memory, memory, mask)
        weights = attention.attn
        self.assertEqual(tuple(result.shape), (2, 3, 8))
        attention.set_context_probability(None)
        self.assertTrue(torch.equal(
            source(query, memory, memory, mask),
            attention(query, memory, memory, mask),
        ))
        attention.set_context_probability(probability)
        attention(query, memory, memory, mask)
        weights = attention.attn
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))))
        self.assertEqual(float(weights[0, :, :, 2].detach().abs().sum()), 0.0)
        self.assertEqual(float(weights[1, :, :, 1].detach().abs().sum()), 0.0)
        gradient = torch.autograd.grad(weights[..., -1].sum(), probability)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

        def context_mass(value):
            attention.set_context_probability(torch.tensor([value, 0.8]))
            attention(query, memory, memory, mask)
            return attention.attn[0, ..., -1].sum()

        step = 1e-3
        finite_difference = (
            context_mass(0.2 + step) - context_mass(0.2 - step)
        ) / (2 * step)
        self.assertAlmostEqual(
            float(gradient[0].detach()),
            float(finite_difference.detach()),
            delta=2e-3,
        )

    def test_low_probability_admission_has_measured_nonzero_gradients(self):
        model = make_model().train()
        model.gate = FixedLogits([[0.0, -4.0]])
        batch = make_batch(1)
        prediction = model.forward_batch(batch)
        self.assertAlmostEqual(
            float(model.last_admission_probability[0].detach()), 0.017986, places=5
        )
        prediction.square().mean().backward()
        gate_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in model.gate.parameters()
            if parameter.grad is not None
        ))
        context_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in model.context_encoder.parameters()
            if parameter.grad is not None
        ))
        self.assertGreater(float(gate_norm), 0.0)
        self.assertGreater(float(context_norm), 0.0)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(gate_norm))
        self.assertTrue(torch.isfinite(context_norm))

    def test_learned_gate_mlp_and_context_encoder_receive_low_admission_gradients(self):
        model = make_model().train()
        with torch.no_grad():
            model.gate[-1].bias.copy_(torch.tensor([0.0, -6.0]))
        batch = make_batch(1)
        prediction = model.forward_batch(batch)
        self.assertLess(float(model.last_admission_probability[0].detach()), 0.1)
        prediction.square().mean().backward()
        gate_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in model.gate.parameters()
            if parameter.grad is not None
        ))
        context_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in model.context_encoder.parameters()
            if parameter.grad is not None
        ))
        self.assertGreater(float(gate_norm), 0.0)
        self.assertGreater(float(context_norm), 0.0)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_mixed_low_and_high_admission_probabilities(self):
        model = make_model().train()
        model.gate = FixedLogits([[0.0, -4.0], [0.0, 4.0]])
        prediction = model.forward_batch(make_batch(2))
        probability = model.last_admission_probability
        self.assertLess(float(probability[0].detach()), 0.1)
        self.assertGreater(float(probability[1].detach()), 0.9)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(model.gate.logits.grad).all())
        self.assertGreater(float(model.gate.logits.grad.abs().sum()), 0.0)

    def test_admission_boundaries_remain_finite(self):
        model = make_model().train()
        model.gate = FixedLogits([[0.0, -100.0], [0.0, 100.0]])
        prediction = model.forward_batch(make_batch(2))
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(model.gate.logits.grad).all())
        for layer in model.base.model.decoder.layers:
            self.assertTrue(torch.isfinite(layer.src_attn.attn).all())

    def test_always_open_matches_unmasked_context_path(self):
        model = make_model().eval()
        model.gate = FixedLogits([[0.0, 100.0]])
        batch = make_batch(1)
        with torch.no_grad():
            actual = model.forward_batch(batch)
            arguments = _prepare_model_inputs(batch)
            residual, inp, speed, ego, ped, target, src_mask, obd_mask, speed_mask, ped_mask = arguments
            context = model.context_encoder(batch["scene_context"], batch["observed_occlusion"])
            decoded = model.base.model(
                inp, speed, ego, ped, target, src_mask, obd_mask,
                speed_mask, ped_mask,
                context=context,
                context_mask=torch.ones(1, 1, 1),
            )
            expected = model.base.model.generator(decoded) + residual
        self.assertEqual(float(model.last_admission_probability[0]), 1.0)
        self.assertTrue(torch.equal(actual, expected))

    def test_p2_always_open_baseline_equivalence_and_parameter_counts(self):
        torch.manual_seed(19)
        always_open = build_satie_model("p2_b").eval()
        torch.manual_seed(19)
        learned = build_satie_model("p2_c").eval()
        learned.forced_admission = 1.0
        batch = make_batch(1)
        with torch.no_grad():
            baseline_prediction = always_open.forward_batch(batch)
            learned_prediction = learned.forward_batch(batch)
        self.assertTrue(torch.equal(baseline_prediction, learned_prediction))
        expected = {
            "p2_a": 130694,
            "p2_b": 130068,
            "p2_c": 130694,
            "p2_d": 130694,
            "p2_e": 130630,
        }
        for configuration, parameter_count in expected.items():
            torch.manual_seed(20260928)
            model = build_satie_model(configuration)
            self.assertEqual(
                unique_trainable_parameter_count(model), parameter_count
            )

    def test_hard_closed_is_exact_motion_only_and_low_soft_is_not_rejection(self):
        model = make_model().eval()
        model.gate = FixedLogits([[0.0, -4.0], [0.0, -100.0]])
        batch = make_batch(2)
        model.inference_mode = "soft"
        with torch.no_grad():
            soft = model.forward_batch(batch)
            low_probability = model.last_admission_probability.clone()
            context_weight = model.decoder_context_attention()
        self.assertTrue(torch.all(low_probability > 0.0))
        self.assertIsNone(model.last_gate_decision)
        self.assertTrue(torch.all(context_weight > 0.0))

        model.inference_mode = "hard"
        with torch.no_grad():
            hard = model.forward_batch(batch)
            base_arguments = _prepare_model_inputs(batch)
            closed_reference = model.base(*base_arguments)
        self.assertFalse(bool(model.last_gate_decision[0]))
        self.assertFalse(bool(model.last_gate_decision[1]))
        self.assertTrue(torch.equal(hard, closed_reference))
        self.assertTrue(torch.isfinite(soft).all())

    def test_mixed_hard_gate_has_exact_closed_row_fallback(self):
        model = make_model().eval()
        model.gate = FixedLogits([[0.0, -4.0], [0.0, 4.0]])
        batch = make_batch(2)
        model.inference_mode = "hard"
        with torch.no_grad():
            actual = model.forward_batch(batch)
            arguments = _prepare_model_inputs(batch)
            closed_reference = model.base(*tuple(value[:1] for value in arguments))
        self.assertFalse(bool(model.last_gate_decision[0]))
        self.assertTrue(bool(model.last_gate_decision[1]))
        self.assertTrue(torch.equal(actual[:1], closed_reference))

    def test_output_shape_and_satie_parameter_identity(self):
        model = make_model().eval()
        batch = make_batch(1)
        with torch.no_grad():
            prediction = model.forward_batch(batch)
        self.assertEqual(tuple(prediction.shape), (1, 45, 4))
        evidence = verify_c4_sharing(model)
        self.assertTrue(evidence["verified"])
        parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            unique_trainable_parameter_count(model),
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )
        self.assertGreater(len(parameter_ids), 0)


if __name__ == "__main__":
    unittest.main()
