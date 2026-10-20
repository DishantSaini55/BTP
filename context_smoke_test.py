"""Small-sample architectural test for baseline MTN and context-aware MTN.

This is intentionally a forward-pass test, not a quality claim: meaningful
MSE comparison requires the IDD-PeD data and optical-flow files.
"""

import torch

from MTN.context_TF import ContextIndividualTF
from MTN.individual_TF import IndividualTF


def make_samples(batch_size=4):
    return {
        "boxes": torch.randn(batch_size, 15, 4),
        "speed": torch.randn(batch_size, 60, 1),
        "ego_flow": torch.randn(batch_size, 60, 64, 2),
        "ped_flow": torch.randn(batch_size, 60, 9, 2),
        # intersection, signal, road, location, direction, day/night
        "scene_context": torch.tensor([
            [0, 0, 0, 0, 0, 0], [4, 3, 3, 2, 1, 1],
            [1, 1, 2, 3, 0, 0], [6, 4, 4, 4, 2, 2],
        ])[:batch_size],
        # Observation-period visibility only: none, partial, full, unknown.
        "observed_occlusion": torch.tensor([
            [0] * 15, [1] * 15, [2] * 15, [3] * 15,
        ])[:batch_size],
    }


def main():
    torch.manual_seed(7)
    samples = make_samples()
    options = {"N": 1, "d_model": 32, "d_ff": 64, "h": 4, "dropout": 0.0}
    baseline = IndividualTF(15, 4, 4, 4, **options).eval()
    context_model = ContextIndividualTF(
        15, 4, 4, 4, gate_mode="soft", **options
    ).eval()
    with torch.no_grad():
        baseline_output = baseline.forward_batch(samples)
        context_output = context_model.forward_batch(samples)

    assert baseline_output.shape == context_output.shape == (4, 45, 4)
    assert torch.isfinite(baseline_output).all() and torch.isfinite(context_output).all()
    assert context_model.last_context_gate.shape == (4,)
    print("Samples:", baseline_output.shape[0])
    print("Baseline output:", tuple(baseline_output.shape))
    print("Context output:", tuple(context_output.shape))
    print("Context gates:", context_model.last_context_gate.tolist())
    print("Context architecture smoke test: OK")


if __name__ == "__main__":
    main()
