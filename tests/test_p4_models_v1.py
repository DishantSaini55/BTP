import unittest

import torch

from MTN.p4_models_v1 import (
    P4_CONFIGURATIONS,
    build_p4_model,
    parameter_sharing_audit,
    unique_parameter_count,
)


EXPECTED_PARAMETERS = {
    "p4_a": 133700,
    "p4_b": 125252,
    "p4_c": 138516,
    "p4_d": 130068,
    "p4_e": 130694,
    "p4_f": 130630,
}


class P4ModelConfigurationTests(unittest.TestCase):
    def test_exact_unique_parameter_counts_and_attention_sharing(self):
        torch.set_num_threads(1)
        for configuration in P4_CONFIGURATIONS:
            with self.subTest(configuration=configuration):
                torch.manual_seed(20260928)
                model = build_p4_model(configuration)
                audit = parameter_sharing_audit(model)
                self.assertEqual(
                    unique_parameter_count(model),
                    EXPECTED_PARAMETERS[configuration],
                )
                self.assertTrue(audit["verified"])
                self.assertEqual(
                    audit["satie_enabled"],
                    configuration in {"p4_b", "p4_d", "p4_e", "p4_f"},
                )

    def test_satie_changes_only_encoder_self_attention_aliases(self):
        torch.set_num_threads(1)
        torch.manual_seed(20260928)
        model = build_p4_model("p4_b")
        base = model.predictor
        layers = base.model.encoder.layers
        self.assertIs(layers[0].self_attn, layers[1].self_attn)
        self.assertIs(layers[1].self_attn, layers[2].self_attn)
        for name in ("src_attn", "cross_attn"):
            self.assertIsNot(getattr(layers[0], name), getattr(layers[1], name))
            self.assertIsNot(getattr(layers[1], name), getattr(layers[2], name))
        decoder = base.model.decoder.layers
        for name in ("src_attn", "spd_attn", "ped_attn"):
            self.assertIsNot(getattr(decoder[0], name), getattr(decoder[1], name))
            self.assertIsNot(getattr(decoder[1], name), getattr(decoder[2], name))

    def test_anchor_is_constant_velocity_for_every_configuration(self):
        for configuration in P4_CONFIGURATIONS:
            with self.subTest(configuration=configuration):
                model = build_p4_model(configuration)
                self.assertEqual(model.anchor, "constant_velocity")


if __name__ == "__main__":
    unittest.main()
