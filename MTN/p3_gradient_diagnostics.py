"""Measure zero-correction-head gradient flow on a deterministic synthetic batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .cv_residual_v1 import build_p3_model, constant_velocity_forecast
except ImportError:
    from cv_residual_v1 import build_p3_model, constant_velocity_forecast


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(941)
    return {
        "boxes": torch.randn(2, 15, 4, generator=generator),
        "speed": torch.randn(2, 60, 1, generator=generator),
        "ego_flow": torch.randn(2, 60, 64, 2, generator=generator),
        "ped_flow": torch.randn(2, 60, 9, 2, generator=generator),
        "pred_target": torch.randn(2, 45, 4, generator=generator),
        "scene_context": torch.zeros(2, 5, dtype=torch.long),
        "observed_occlusion": torch.zeros(2, 60, dtype=torch.long),
    }


def _gradient_l2(parameters) -> tuple[float, int, int]:
    norm_squared = 0.0
    present = 0
    missing = 0
    for parameter in parameters:
        if parameter.grad is None:
            missing += 1
            continue
        present += 1
        norm_squared += float(parameter.grad.detach().double().square().sum())
    return norm_squared ** 0.5, present, missing


def run_diagnostic() -> dict:
    torch.manual_seed(20260928)
    torch.set_num_threads(1)
    batch = _batch()
    model = build_p3_model("p3_c", zero_correction_head=True).eval()
    head = model.predictor.model.generator.proj
    head_ids = {id(parameter) for parameter in head.parameters()}
    upstream = [
        parameter for parameter in model.predictor.parameters()
        if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    initial = model.forward_batch(batch)
    cv = constant_velocity_forecast(batch["boxes"])
    initial_error = float(torch.max(torch.abs(initial.detach() - cv)))
    loss_before = torch.nn.functional.mse_loss(initial, batch["pred_target"])
    loss_before.backward()
    head_norm_before, head_present, head_missing = _gradient_l2(head.parameters())
    upstream_norm_before, upstream_present_before, upstream_missing_before = _gradient_l2(upstream)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    after_update = model.forward_batch(batch)
    loss_after = torch.nn.functional.mse_loss(after_update, batch["pred_target"])
    loss_after.backward()
    upstream_norm_after, upstream_present_after, upstream_missing_after = _gradient_l2(upstream)
    head_norm_after, _, _ = _gradient_l2(head.parameters())
    return {
        "protocol": "synthetic deterministic CPU diagnostic; not an accuracy experiment",
        "seed": 20260928,
        "initial_prediction_max_abs_difference_from_cv": initial_error,
        "initial_prediction_finite": bool(torch.isfinite(initial).all()),
        "initial_loss": float(loss_before.detach()),
        "initial_loss_finite": bool(torch.isfinite(loss_before)),
        "head_gradient_l2_before_update": head_norm_before,
        "head_gradient_present_parameter_tensors": head_present,
        "head_gradient_missing_parameter_tensors": head_missing,
        "upstream_gradient_l2_before_update": upstream_norm_before,
        "upstream_gradient_present_parameter_tensors_before_update": upstream_present_before,
        "upstream_gradient_missing_parameter_tensors_before_update": upstream_missing_before,
        "post_update_prediction_finite": bool(torch.isfinite(after_update).all()),
        "post_update_loss": float(loss_after.detach()),
        "post_update_loss_finite": bool(torch.isfinite(loss_after)),
        "upstream_gradient_l2_after_update": upstream_norm_after,
        "upstream_gradient_present_parameter_tensors_after_update": upstream_present_after,
        "upstream_gradient_missing_parameter_tensors_after_update": upstream_missing_after,
        "head_gradient_l2_after_update": head_norm_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_paper/P3_diagnostics/gradient_diagnostics.json"),
    )
    args = parser.parse_args()
    result = run_diagnostic()
    if not all((
        result["initial_prediction_finite"],
        result["initial_loss_finite"],
        result["post_update_prediction_finite"],
        result["post_update_loss_finite"],
    )):
        raise FloatingPointError("Non-finite value in zero-head gradient diagnostic.")
    if result["initial_prediction_max_abs_difference_from_cv"] != 0.0:
        raise AssertionError("Zero correction head did not reproduce the CV anchor exactly.")
    if result["head_gradient_l2_before_update"] <= 0.0:
        raise AssertionError("The zero-initialized prediction head received no gradient.")
    if result["upstream_gradient_l2_before_update"] != 0.0:
        raise AssertionError("Expected zero upstream gradient before the zero head updates.")
    if result["upstream_gradient_l2_after_update"] <= 0.0:
        raise AssertionError("Upstream gradients did not appear after the head update.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
