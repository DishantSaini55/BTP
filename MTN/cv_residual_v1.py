"""Versioned constant-velocity residual-anchor models for Phase 3."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from .context_TF import ContextIndividualTF
    from .individual_TF import IndividualTF
    from .sage_v2 import (
        AlwaysOpenContextIndividualTF,
        DifferentiableSAGEV2,
        _prepare_model_inputs,
        unique_trainable_parameter_count,
    )
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
except ImportError:
    from context_TF import ContextIndividualTF
    from individual_TF import IndividualTF
    from sage_v2 import (
        AlwaysOpenContextIndividualTF,
        DifferentiableSAGEV2,
        _prepare_model_inputs,
        unique_trainable_parameter_count,
    )
    from c4_encoder_shared import make_c4_encoder_self_attention_shared


PREDICTION_LENGTH = 45
VELOCITY_FRAMES = 3
P3_CONFIGURATIONS = ("p3_b", "p3_c", "p3_d", "p3_e", "p3_f", "p3_g")
ANCHOR_BY_CONFIGURATION = {
    "p3_b": "constant_position",
    "p3_c": "constant_velocity",
    "p3_d": "constant_position",
    "p3_e": "constant_velocity",
    "p3_f": "constant_velocity",
    "p3_g": "constant_velocity",
}


def constant_velocity_forecast(
    observed_boxes: torch.Tensor,
    prediction_length: int = PREDICTION_LENGTH,
    velocity_frames: int = VELOCITY_FRAMES,
) -> torch.Tensor:
    """Reproduce B1's last-three-observed-box extrapolation in input coordinates."""
    if observed_boxes.ndim != 3 or observed_boxes.shape[-1] != 4:
        raise ValueError("observed_boxes must have shape [batch, observed_steps, 4].")
    if velocity_frames != 3:
        raise ValueError("The historical B1 baseline uses exactly the last three frames.")
    if observed_boxes.shape[1] < velocity_frames:
        raise ValueError("At least three observed boxes are required.")
    if prediction_length < 1:
        raise ValueError("prediction_length must be positive.")
    velocity = (
        observed_boxes[:, -1] - observed_boxes[:, -velocity_frames]
    ) / float(velocity_frames - 1)
    steps = torch.arange(
        1, prediction_length + 1,
        dtype=observed_boxes.dtype,
        device=observed_boxes.device,
    ).view(1, prediction_length, 1)
    return observed_boxes[:, -1:, :] + steps * velocity.unsqueeze(1)


def constant_position_forecast(
    observed_boxes: torch.Tensor, prediction_length: int = PREDICTION_LENGTH
) -> torch.Tensor:
    if observed_boxes.ndim != 3 or observed_boxes.shape[-1] != 4:
        raise ValueError("observed_boxes must have shape [batch, observed_steps, 4].")
    if prediction_length < 1:
        raise ValueError("prediction_length must be positive.")
    return observed_boxes[:, -1:, :].expand(-1, prediction_length, -1)


class ResidualAnchorModel(nn.Module):
    """Apply exactly one selected anchor to an otherwise unchanged MTN predictor."""

    def __init__(
        self,
        predictor: nn.Module,
        anchor: str,
        zero_correction_head: bool = False,
    ) -> None:
        super().__init__()
        if anchor not in {"constant_position", "constant_velocity"}:
            raise ValueError("anchor must be 'constant_position' or 'constant_velocity'.")
        self.predictor = predictor
        self.anchor = anchor
        self.zero_correction_head = zero_correction_head
        if zero_correction_head:
            head = self._prediction_head()
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _prediction_head(self) -> nn.Linear:
        predictor = self.predictor
        base = predictor.base if hasattr(predictor, "base") else predictor
        head = base.model.generator.proj
        if not isinstance(head, nn.Linear) or head.out_features != 4:
            raise TypeError("Expected the existing Linear(32, 4) prediction head.")
        return head

    def anchor_forecast(self, observed_boxes: torch.Tensor) -> torch.Tensor:
        if self.anchor == "constant_position":
            return constant_position_forecast(observed_boxes)
        return constant_velocity_forecast(observed_boxes)

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        boxes = batch.get("boxes", batch.get("enc_input")).float()
        residual = self.anchor_forecast(boxes)
        arguments = list(_prepare_model_inputs(batch))
        arguments[0] = residual
        if isinstance(self.predictor, IndividualTF):
            (
                residual, inp, obd_spd, ego_flow, ped_flow, tgt,
                src_mask, obd_mask, spd_mask, ped_mask,
            ) = arguments
            decoded = self.predictor.model(
                inp, obd_spd, ego_flow, ped_flow, tgt, src_mask, obd_mask,
                spd_mask, ped_mask,
            )
            return self.predictor.model.generator(decoded) + residual
        return self.predictor(
            *arguments,
            batch["scene_context"].to(boxes.device),
            batch["observed_occlusion"].to(boxes.device),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.forward_batch(batch)


def build_p3_model(
    configuration: str,
    zero_correction_head: bool = False,
) -> ResidualAnchorModel:
    if configuration not in ANCHOR_BY_CONFIGURATION:
        raise ValueError(f"Unknown trainable Phase 3 configuration: {configuration}")
    options = dict(N=3, d_model=32, d_ff=128, h=4, dropout=0.1)
    if configuration in {"p3_b", "p3_c"}:
        predictor: nn.Module = IndividualTF(15, 4, 4, 4, **options)
    elif configuration in {"p3_d", "p3_e"}:
        predictor = ContextIndividualTF(15, 4, 4, 4, **options)
        make_c4_encoder_self_attention_shared(predictor)
    elif configuration == "p3_f":
        predictor = AlwaysOpenContextIndividualTF(
            15, 4, 4, 4, share_satie=True, **options
        )
    else:
        predictor = DifferentiableSAGEV2(
            15, 4, 4, 4, gate_input_mode="context_only",
            share_satie=True, **options,
        )
    return ResidualAnchorModel(
        predictor,
        ANCHOR_BY_CONFIGURATION[configuration],
        zero_correction_head=zero_correction_head,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return unique_trainable_parameter_count(model)


def unwrap_predictor(model: ResidualAnchorModel) -> nn.Module:
    return model.predictor


def configure_validation_policy(model: ResidualAnchorModel, policy: str):
    """Temporarily apply a validation-only gate policy and restore it afterwards."""
    predictor = model.predictor

    class Policy:
        def __enter__(self):
            self.old_gate_mode = None
            self.old_gate_fn = None
            self.old_inference_mode = None
            self.old_forced_admission = None
            if isinstance(predictor, ContextIndividualTF):
                if policy in {"hard", "soft"}:
                    self.old_gate_mode = predictor.gate_mode
                    predictor.gate_mode = policy
                elif policy in {"forced_open", "forced_closed"}:
                    self.old_gate_fn = predictor._context_gate
                    admission = 1.0 if policy == "forced_open" else 0.0

                    def forced_gate(context_token, _observed_boxes):
                        gate = context_token.new_full(
                            (context_token.shape[0],), admission
                        )
                        logits = context_token.new_zeros(context_token.shape[0], 2)
                        predictor.last_context_gate = gate.detach()
                        predictor.last_context_gate_logits = logits
                        predictor._context_gate_for_loss = gate
                        return gate

                    predictor._context_gate = forced_gate
                else:
                    raise ValueError(f"Invalid SAGE validation policy: {policy}")
            elif isinstance(predictor, DifferentiableSAGEV2):
                self.old_inference_mode = predictor.inference_mode
                self.old_forced_admission = predictor.forced_admission
                if policy in {"soft", "hard"}:
                    predictor.inference_mode = policy
                    predictor.forced_admission = None
                elif policy in {"forced_open", "forced_closed"}:
                    predictor.inference_mode = "hard"
                    predictor.forced_admission = (
                        1.0 if policy == "forced_open" else 0.0
                    )
                else:
                    raise ValueError(f"Invalid SAGE V2 validation policy: {policy}")
            elif isinstance(predictor, AlwaysOpenContextIndividualTF):
                if policy != "always_open":
                    raise ValueError(f"Invalid always-open policy: {policy}")
            elif policy != "base":
                raise ValueError(f"Invalid MTN validation policy: {policy}")
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            if self.old_gate_mode is not None:
                predictor.gate_mode = self.old_gate_mode
            if self.old_gate_fn is not None:
                predictor._context_gate = self.old_gate_fn
            if self.old_inference_mode is not None:
                predictor.inference_mode = self.old_inference_mode
                predictor.forced_admission = self.old_forced_admission
            return False

    return Policy()
