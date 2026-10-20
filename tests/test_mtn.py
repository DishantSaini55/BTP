import unittest

import torch

from MTN.config import DEFAULT_CONFIG
from MTN.context_TF import ContextIndividualTF
from MTN.individual_TF import IndividualTF
from MTN.preprocessing import prepare_model_inputs


class MTNForwardTest(unittest.TestCase):
    def test_shared_batch_contract(self):
        config = DEFAULT_CONFIG
        batch = {
            "boxes": torch.zeros(2, config.observe_length, config.box_size),
            "speed": torch.zeros(2, config.sequence_length, config.speed_size),
            "ego_flow": torch.zeros(2, config.sequence_length, config.ego_patches, 2),
            "ped_flow": torch.zeros(2, config.sequence_length, config.ped_patches, 2),
        }
        model_inputs = prepare_model_inputs(batch, config, torch.device("cpu"))
        model = IndividualTF(
            config.observe_length,
            config.box_size,
            config.box_size,
            config.box_size,
            N=config.layers,
            d_model=config.embedding_size,
            d_ff=config.feed_forward_size,
            h=config.heads,
            dropout=config.dropout,
        )
        with torch.no_grad():
            predictions = model.forward_batch(model_inputs)
        self.assertEqual(
            tuple(predictions.shape),
            (2, config.predict_length, config.box_size),
        )

    def test_context_branch_contract(self):
        config = DEFAULT_CONFIG
        batch = {
            "boxes": torch.zeros(2, config.observe_length, config.box_size),
            "speed": torch.zeros(2, config.sequence_length, config.speed_size),
            "ego_flow": torch.zeros(2, config.sequence_length, config.ego_patches, 2),
            "ped_flow": torch.zeros(2, config.sequence_length, config.ped_patches, 2),
            "scene_context": torch.tensor([[0, 0, 0, 0, 0], [6, 4, 4, 4, 2]]),
            "observed_occlusion": torch.tensor([[0] * config.observe_length, [3] * config.observe_length]),
        }
        model = ContextIndividualTF(
            config.observe_length, config.box_size, config.box_size, config.box_size,
            N=1, d_model=config.embedding_size, d_ff=config.feed_forward_size,
            h=config.heads, dropout=0.0, gate_mode="soft",
        ).eval()
        with torch.no_grad():
            predictions = model.forward_batch(batch)
        self.assertEqual(tuple(predictions.shape), (2, config.predict_length, config.box_size))
        self.assertTrue(torch.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
