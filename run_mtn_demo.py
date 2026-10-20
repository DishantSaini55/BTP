"""Run the MTN trajectory model locally with synthetic input data."""

import argparse
from pathlib import Path

import numpy as np
import torch

from MTN.config import DEFAULT_CONFIG
from MTN.individual_TF import IndividualTF
from MTN.preprocessing import prepare_model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local MTN trajectory demo")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/mtn_demo_predictions.npy"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    config = DEFAULT_CONFIG

    model = IndividualTF(
        config.observe_length,
        4,
        4,
        4,
        N=config.layers,
        d_model=config.embedding_size,
        d_ff=config.feed_forward_size,
        h=config.heads,
        dropout=config.dropout,
    ).to(device)
    model.eval()

    with torch.no_grad():
        batch = {
            "boxes": torch.randn(args.samples, config.observe_length, config.box_size),
            "speed": torch.randn(args.samples, config.sequence_length, config.speed_size),
            "ego_flow": torch.randn(args.samples, config.sequence_length, config.ego_patches, 2),
            "ped_flow": torch.randn(args.samples, config.sequence_length, config.ped_patches, 2),
        }
        model_inputs = prepare_model_inputs(batch, config, device)
        predictions = model.forward_batch(model_inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions.numpy())
    print(f"MTN demo completed on {device}")
    print(f"Predictions shape: {tuple(predictions.shape)}")
    print(f"First predicted bounding box: {predictions[0, 0].tolist()}")
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()