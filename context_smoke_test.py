"""Small-sample forward tests for baseline MTN, context MTN, and C4.

These tests validate tensor contracts and module sharing, not trajectory
accuracy. Meaningful MSE comparison requires the IDD-PeD data and flow files.
"""

import torch

from MTN.context_TF import ContextIndividualTF
from MTN.individual_TF import IndividualTF
from MTN.c4_encoder_shared import (
    make_c4_encoder_self_attention_shared,
    verify_c4_sharing,
)


def make_samples(batch_size=4):
    """Build synthetic samples using the paper-aligned native context schema."""
    if not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between 1 and 4")

    # Exact order from MTN.context_schema.SCENE_FIELDS:
    # intersection_type, signalized_type, road_type, location_type,
    # motion_direction. Occlusion is a separate observed 15-step sequence.
    scene_rows = [
        [0, 0, 0, 0, 0],
        [4, 3, 3, 2, 1],
        [1, 1, 2, 3, 0],
        [6, 4, 4, 4, 2],
    ]
    occlusion_rows = [
        [0] * 15,
        [1] * 15,
        [2] * 15,
        [3] * 15,
    ]

    return {
        "boxes": torch.randn(batch_size, 15, 4),
        # The model consumes the first 15 observation frames.
        "speed": torch.randn(batch_size, 60, 1),
        "ego_flow": torch.randn(batch_size, 60, 64, 2),
        "ped_flow": torch.randn(batch_size, 60, 9, 2),
        "scene_context": torch.tensor(scene_rows[:batch_size], dtype=torch.long),
        "observed_occlusion": torch.tensor(
            occlusion_rows[:batch_size], dtype=torch.long
        ),
    }


def assert_prediction(name, output, batch_size):
    expected = (batch_size, 45, 4)
    assert tuple(output.shape) == expected, (
        f"{name} output shape {tuple(output.shape)} != {expected}"
    )
    assert torch.isfinite(output).all(), f"{name} output contains NaN/Inf"


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    samples = make_samples(batch_size=4)
    # Match the documented three-layer POC configuration.
    options = {
        "N": 3,
        "d_model": 32,
        "d_ff": 128,
        "h": 4,
        "dropout": 0.0,
    }

    baseline = IndividualTF(15, 4, 4, 4, **options).eval()
    context_model = ContextIndividualTF(
        15, 4, 4, 4, gate_mode="hard", **options
    ).eval()
    c4_model = ContextIndividualTF(
        15, 4, 4, 4, gate_mode="hard", **options
    ).eval()
    make_c4_encoder_self_attention_shared(c4_model)

    c4_evidence = verify_c4_sharing(c4_model)
    assert c4_evidence["verified"], c4_evidence
    assert len(c4_model.base.model.encoder.layers) == 3

    with torch.no_grad():
        baseline_output = baseline.forward_batch(samples)
        context_output = context_model.forward_batch(samples)
        c4_output = c4_model.forward_batch(samples)

    batch_size = samples["boxes"].shape[0]
    assert_prediction("Baseline MTN", baseline_output, batch_size)
    assert_prediction("Context MTN", context_output, batch_size)
    assert_prediction("C4 Context MTN", c4_output, batch_size)

    assert context_model.last_context_gate.shape == (batch_size,)
    assert c4_model.last_context_gate.shape == (batch_size,)
    assert set(c4_model.last_context_gate.unique().tolist()).issubset({0.0, 1.0})

    print("Baseline output:", tuple(baseline_output.shape))
    print("Context output:", tuple(context_output.shape))
    print("C4 output:", tuple(c4_output.shape))
    print("C4 Q/K/V/O object and parameter sharing:", c4_evidence["verified"])
    print("Context gate values:", c4_model.last_context_gate.tolist())
    print("Context + C4 architecture smoke test: OK")


if __name__ == "__main__":
    main()
