"""Deterministic SAGE gate/context gradient probes for Phase 1."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from .context_TF import ContextIndividualTF
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from context_TF import ContextIndividualTF


def make_model(gate_mode: str = "hard") -> ContextIndividualTF:
    return ContextIndividualTF(
        4, 4, 4, 4, N=3, d_model=8, d_ff=16, h=2, dropout=0.0, gate_mode=gate_mode
    )


def make_inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7401)
    boxes = torch.randn(batch_size, 4, 4, generator=generator) * 0.2
    speed = torch.randn(batch_size, 4, 1, generator=generator) * 0.2
    ego_flow = torch.randn(batch_size, 6, 64, generator=generator) * 0.2
    ped_flow = torch.randn(batch_size, 6, 9, generator=generator) * 0.2
    target = torch.randn(batch_size, 2, 4, generator=generator) * 0.2
    scene = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 0]], dtype=torch.long)[:batch_size]
    occlusion = torch.tensor([[0, 1, 0, 0], [1, 1, 2, 2], [0, 0, 1, 2]], dtype=torch.long)[:batch_size]
    batch = {
        "boxes": boxes,
        "speed": speed,
        "ego_flow": ego_flow,
        "ped_flow": ped_flow,
        "pred_target": target,
        "scene_context": scene,
        "observed_occlusion": occlusion,
        "predict_length": 2,
    }
    return batch


def force_straight_through_decisions(model: ContextIndividualTF, decisions: list[float]) -> None:
    """Fix the forward gate decisions while retaining the ST softmax derivative."""
    hard = torch.tensor(decisions, dtype=torch.float32)

    def forced_gate(context_token: torch.Tensor, observed_boxes: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((context_token, observed_boxes[:, -1]), dim=-1)
        logits = model.gate(inputs)
        probability = logits.softmax(dim=-1)[:, 1]
        selected = hard.to(device=context_token.device, dtype=context_token.dtype)
        if selected.numel() != context_token.shape[0]:
            raise ValueError("Decision vector length must equal batch size.")
        gate = selected + probability - probability.detach()
        model.last_context_gate = gate.detach()
        model.last_context_gate_logits = logits.detach()
        model._context_gate_for_loss = gate
        return gate

    model._context_gate = forced_gate


def forward_model(model: ContextIndividualTF, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = batch["boxes"]
    batch_size = boxes.shape[0]
    target = batch["pred_target"]
    residual = boxes[:, -1:, :].expand(-1, target.shape[1], -1)
    src_mask = torch.ones(batch_size, 1, boxes.shape[1])
    obd_mask = torch.ones(batch_size, 1, 1)
    speed_mask = torch.ones(batch_size, 1, 64)
    ped_mask = torch.ones(batch_size, 1, 9)
    return model(
        residual,
        boxes,
        batch["speed"],
        batch["ego_flow"],
        batch["ped_flow"],
        torch.zeros_like(target),
        src_mask,
        obd_mask,
        speed_mask,
        ped_mask,
        batch["scene_context"],
        batch["observed_occlusion"],
    )


def grad_state(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    present = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    missing = len(parameters) - len(present)
    zero = sum(float(torch.sum(grad * grad).item()) == 0.0 for grad in present)
    squared = sum(float(torch.sum(grad * grad).item()) for grad in present)
    return {
        "l2_norm": float(math.sqrt(squared)) if present else None,
        "parameter_count": len(parameters),
        "grad_present_count": len(present),
        "grad_absent_count": missing,
        "grad_zero_count": int(zero),
        "grad_nonzero_count": len(present) - int(zero),
        "all_finite": all(bool(torch.isfinite(grad).all()) for grad in present),
    }


def _case(mode: str, seed: int = 20260928) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    batch = make_inputs()
    model = make_model("soft" if mode == "soft" else "hard")
    model.train()
    if mode == "all_open":
        force_straight_through_decisions(model, [1.0, 1.0, 1.0])
    elif mode == "all_closed":
        force_straight_through_decisions(model, [0.0, 0.0, 0.0])
    elif mode == "mixed":
        force_straight_through_decisions(model, [1.0, 0.0, 1.0])
    elif mode != "soft":
        raise ValueError(mode)

    prediction = forward_model(model, batch)
    loss = torch.nn.functional.mse_loss(prediction, batch["pred_target"])
    closed_sample_gradients = None
    if mode == "mixed":
        sample_loss = torch.mean((prediction - batch["pred_target"]) ** 2, dim=(1, 2))
        params = list(model.gate.parameters()) + list(model.context_encoder.parameters())
        gradients = torch.autograd.grad(sample_loss[1], params, retain_graph=True, allow_unused=True)
        closed_sample_gradients = {
            "parameter_count": len(params),
            "grad_absent_count": sum(gradient is None for gradient in gradients),
            "grad_zero_count": sum(
                gradient is not None and float(torch.sum(gradient.detach().float() ** 2).item()) == 0.0
                for gradient in gradients
            ),
            "grad_nonzero_count": sum(
                gradient is not None and float(torch.sum(gradient.detach().float() ** 2).item()) > 0.0
                for gradient in gradients
            ),
        }
    loss.backward()
    gate = model.last_context_gate.detach().cpu().numpy()
    return {
        "mode": mode,
        "seed": seed,
        "loss": float(loss.detach().item()),
        "prediction_shape": list(prediction.shape),
        "predictions_finite": bool(torch.isfinite(prediction).all()),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "gate_values": [float(value) for value in gate],
        "gate_mean": float(np.mean(gate)),
        "gate_gradient": grad_state(list(model.gate.parameters())),
        "context_encoder_gradient": grad_state(list(model.context_encoder.parameters())),
        "closed_sample_gradient": closed_sample_gradients,
    }


def run_all(seed: int = 20260928) -> dict[str, Any]:
    cases = {mode: _case(mode, seed) for mode in ("all_open", "all_closed", "mixed", "soft")}
    all_closed = cases["all_closed"]
    batch = make_inputs()
    model = make_model("hard").eval()
    force_straight_through_decisions(model, [0.0, 0.0, 0.0])
    with torch.no_grad():
        gated = forward_model(model, batch)
        boxes = batch["boxes"]
        residual = boxes[:, -1:, :].expand(-1, batch["pred_target"].shape[1], -1)
        base = model.base(
            residual,
            boxes,
            batch["speed"],
            batch["ego_flow"],
            batch["ped_flow"],
            torch.zeros_like(batch["pred_target"]),
            torch.ones(boxes.shape[0], 1, boxes.shape[1]),
            torch.ones(boxes.shape[0], 1, 1),
            torch.ones(boxes.shape[0], 1, 64),
            torch.ones(boxes.shape[0], 1, 9),
        )
    exact_fallback = bool(torch.equal(gated, base))

    # Equal self-attention weights at each depth make tied/unshared execution comparable.
    torch.manual_seed(seed + 1)
    unshared = make_model("hard").eval()
    layers = unshared.base.model.encoder.layers
    for layer in layers[1:]:
        layer.self_attn.load_state_dict(layers[0].self_attn.state_dict())
    shared = copy.deepcopy(unshared)
    make_c4_encoder_self_attention_shared(shared)
    with torch.no_grad():
        unshared_prediction = forward_model(unshared, batch)
        shared_prediction = forward_model(shared, batch)
    max_abs_difference = float(torch.max(torch.abs(unshared_prediction - shared_prediction)).item())
    return {
        "seed": seed,
        "torch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "cases": cases,
        "all_closed_fallback_bitwise_equal": exact_fallback,
        "all_closed_expected_grad_absent": (
            all_closed["gate_gradient"]["grad_absent_count"] == all_closed["gate_gradient"]["parameter_count"]
            and all_closed["context_encoder_gradient"]["grad_absent_count"]
            == all_closed["context_encoder_gradient"]["parameter_count"]
        ),
        "satie_sharing": {
            "max_abs_prediction_difference": max_abs_difference,
            "bitwise_equal": bool(torch.equal(unshared_prediction, shared_prediction)),
            "sharing_evidence": verify_c4_sharing(shared),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
