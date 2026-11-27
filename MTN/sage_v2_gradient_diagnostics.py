"""Measure SAGE V2 gradients on fixed synthetic CPU inputs."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from .run_from_scratch_e1 import RESULTS
    from .sage_v2 import DifferentiableSAGEV2
except ImportError:
    from run_from_scratch_e1 import RESULTS
    from sage_v2 import DifferentiableSAGEV2


class FixedLogits(nn.Module):
    def __init__(self, values: list[list[float]]):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] != self.logits.shape[0]:
            raise ValueError("Fixed diagnostic logits require a matching batch size.")
        return self.logits


def _batch(size: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.randn(size, 15, 4, generator=generator),
        "speed": torch.randn(size, 15, 1, generator=generator),
        "ego_flow": torch.randn(size, 14, 64, 2, generator=generator),
        "ped_flow": torch.randn(size, 14, 9, 2, generator=generator),
        "pred_target": torch.randn(size, 45, 4, generator=generator),
        "scene_context": torch.randint(0, 3, (size, 5), generator=generator),
        "observed_occlusion": torch.randint(0, 4, (size, 15), generator=generator),
    }


def _gradient_summary(module: nn.Module) -> dict[str, Any]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    missing = sum(parameter.grad is None for parameter in parameters)
    zero = 0
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            return {
                "finite": False,
                "l2_norm": None,
                "parameter_count": len(parameters),
                "absent_gradient_parameters": missing,
                "zero_gradient_parameters": zero,
            }
        norm_squared = float(gradient.square().sum())
        zero += int(norm_squared == 0.0)
        squared_norm += norm_squared
    return {
        "finite": True,
        "l2_norm": float(np.sqrt(squared_norm)) if missing < len(parameters) else None,
        "parameter_count": len(parameters),
        "absent_gradient_parameters": missing,
        "zero_gradient_parameters": zero,
        "nonzero_gradient_parameters": len(parameters) - missing - zero,
    }


def _case(
    name: str,
    batch: dict[str, torch.Tensor],
    logits: list[list[float]] | None = None,
    gate_bias: list[float] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(20260928)
    model = DifferentiableSAGEV2(
        15, 4, 4, 4,
        N=3, d_model=32, d_ff=128, h=4, dropout=0.0,
    ).train()
    if logits is not None:
        model.gate = FixedLogits(logits)
        parameterization = "fixed_logits_probe"
    else:
        if gate_bias is not None:
            with torch.no_grad():
                model.gate[-1].bias.copy_(torch.tensor(gate_bias))
        parameterization = "learned_gate_mlp"
    prediction = model.forward_batch(batch)
    loss = torch.nn.functional.mse_loss(prediction, batch["pred_target"])
    loss.backward()
    return {
        "case": name,
        "gate_parameterization": parameterization,
        "admission_probabilities": [
            float(value) for value in model.last_admission_probability.detach()
        ],
        "normalized_mse": float(loss.detach()),
        "predictions_finite": bool(torch.isfinite(prediction).all()),
        "gate_gradient": _gradient_summary(model.gate),
        "context_encoder_gradient": _gradient_summary(model.context_encoder),
        "context_attention_mean": float(model.last_context_attention.mean()),
    }


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator().manual_seed(20260928)
    low = _case(
        "low_probability",
        _batch(1, generator),
        gate_bias=[0.0, -6.0],
    )
    mixed = _case(
        "mixed_low_high",
        _batch(2, generator),
        logits=[[0.0, -4.0], [0.0, 4.0]],
    )
    boundary = _case(
        "near_probability_boundaries",
        _batch(2, generator),
        logits=[[0.0, -100.0], [0.0, 100.0]],
    )
    result = {
        "diagnostic_version": "SAGE_V2_gradient_measurements_v2",
        "seed": 20260928,
        "device": "CPU",
        "cpu_threads": torch.get_num_threads(),
        "gradient_convention": "Absent gradients are counted separately from present exact-zero gradients.",
        "cases": [low, mixed, boundary],
    }
    return result


def main() -> None:
    output = RESULTS / "diagnostics" / "P2" / "sage_v2_gradient_diagnostics_v2.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), **result}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
