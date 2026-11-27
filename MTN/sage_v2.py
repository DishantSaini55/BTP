"""Experimental differentiable SAGE admission without changing historical SAGE."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
    from .context_TF import SceneContextEncoder
    from .individual_TF import IndividualTF
    from .transformer.multihead_attention import MultiHeadAttention
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared
    from context_TF import SceneContextEncoder
    from individual_TF import IndividualTF
    from transformer.multihead_attention import MultiHeadAttention


ADMISSION_EPSILON = 1e-8
GATE_INPUT_MODES = ("original", "clipped", "context_only")


def _install_satie(base: IndividualTF, enabled: bool) -> None:
    if enabled:
        make_c4_encoder_self_attention_shared(SimpleNamespace(base=base))


def _prepare_model_inputs(batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    boxes = batch.get("boxes", batch.get("enc_input")).float()
    speed = batch.get("speed", batch.get("obd_speed")).float()
    ego_flow = batch["ego_flow"].float()
    ped_flow = batch["ped_flow"].float()
    batch_size, observe_length = boxes.shape[:2]
    if ego_flow.ndim == 4:
        ego_flow = ego_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
        ego_flow = ego_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
    if ped_flow.ndim == 4:
        ped_flow = ped_flow[:, :observe_length - 1].permute(0, 1, 3, 2)
        ped_flow = ped_flow.reshape(batch_size, 2 * (observe_length - 1), -1)
    prediction_length = batch.get("predict_length", 45)
    residual = boxes[:, -1:, :].expand(-1, prediction_length, -1)
    target_decoder = torch.zeros_like(residual)
    device = boxes.device
    src_mask = torch.ones(batch_size, 1, observe_length, device=device)
    speed_mask = torch.ones(batch_size, 1, 1, device=device)
    ego_mask = torch.ones(batch_size, 1, ego_flow.shape[-1], device=device)
    ped_mask = torch.ones(batch_size, 1, ped_flow.shape[-1], device=device)
    return (
        residual,
        boxes,
        speed[:, :observe_length],
        ego_flow,
        ped_flow,
        target_decoder,
        src_mask,
        speed_mask,
        ego_mask,
        ped_mask,
    )


class ContextPriorMultiHeadAttention(nn.Module):
    """Multi-head attention with a differentiable prior on the final memory key."""

    def __init__(self, source: MultiHeadAttention):
        super().__init__()
        self.d_k = source.d_k
        self.h = source.h
        self.linears = source.linears
        self.dropout = source.dropout
        self.attn: torch.Tensor | None = None
        self._context_probability: torch.Tensor | None = None

    def set_context_probability(self, probability: torch.Tensor | None) -> None:
        self._context_probability = probability

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        cross_attention: bool = False,
    ) -> torch.Tensor:
        del cross_attention
        batch_size = query.shape[0]
        query, key, value = [
            linear(tensor)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, tensor in zip(self.linears[:3], (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if self._context_probability is not None:
            if key.shape[-2] < 1:
                raise ValueError("Context-prior attention requires at least one memory key.")
            probability = self._context_probability.to(device=scores.device, dtype=scores.dtype)
            if probability.ndim != 1 or probability.shape[0] != batch_size:
                raise ValueError("Context probability must have shape [batch].")
            probability = probability.clamp(0.0, 1.0)
            log_prior = (
                torch.log(probability.to(torch.float32) + ADMISSION_EPSILON)
                .to(scores.dtype)
                .view(batch_size, 1, 1, 1)
            )
            trajectory_prior = scores.new_zeros(batch_size, 1, 1, key.shape[-2] - 1)
            scores = scores + torch.cat((trajectory_prior, log_prior), dim=-1)
        if mask is not None:
            scores = scores.masked_fill_(mask.unsqueeze(1) == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        self.attn = attention
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](attended)


def install_context_prior_attention(base: IndividualTF) -> None:
    """Replace each decoder source-attention implementation, preserving weights."""
    for layer in base.model.decoder.layers:
        if not isinstance(layer.src_attn, ContextPriorMultiHeadAttention):
            layer.src_attn = ContextPriorMultiHeadAttention(layer.src_attn)


def set_decoder_context_probability(
    base: IndividualTF, probability: torch.Tensor | None
) -> None:
    for layer in base.model.decoder.layers:
        attention = layer.src_attn
        if not isinstance(attention, ContextPriorMultiHeadAttention):
            raise TypeError("Decoder source attention is missing its SAGE V2 prior wrapper.")
        attention.set_context_probability(probability)


def _slice_model_args(arguments: tuple[torch.Tensor, ...], indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(value.index_select(0, indices) for value in arguments)


class AlwaysOpenContextIndividualTF(nn.Module):
    """Independent always-open context baseline with no unused gate parameters."""

    def __init__(
        self,
        inp_l: int,
        enc_inp_size: int,
        dec_inp_size: int,
        dec_out_size: int,
        context_dim: int = 16,
        share_satie: bool = True,
        **model_options: Any,
    ):
        super().__init__()
        self.base = IndividualTF(
            inp_l, enc_inp_size, dec_inp_size, dec_out_size, **model_options
        )
        _install_satie(self.base, share_satie)
        d_model = model_options.get("d_model", 512)
        self.context_encoder = SceneContextEncoder(
            d_model, context_dim, model_options.get("dropout", 0.1)
        )

    def forward(
        self,
        residual: torch.Tensor,
        inp: torch.Tensor,
        obd_spd: torch.Tensor,
        ego_flow: torch.Tensor,
        ped_flow: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        obd_enc_mask: torch.Tensor,
        spd_mask: torch.Tensor,
        ped_mask: torch.Tensor,
        scene_context: torch.Tensor,
        observed_occlusion: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_encoder(scene_context, observed_occlusion)
        mask = torch.ones(
            src_mask.shape[0], 1, 1, dtype=src_mask.dtype, device=src_mask.device
        )
        decoded = self.base.model(
            inp, obd_spd, ego_flow, ped_flow, tgt, src_mask, obd_enc_mask,
            spd_mask, ped_mask, context=context, context_mask=mask,
        )
        return self.base.model.generator(decoded) + residual

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        args = _prepare_model_inputs(batch)
        boxes = batch.get("boxes", batch.get("enc_input")).float()
        return self(
            *args,
            batch["scene_context"].to(boxes.device),
            batch["observed_occlusion"].to(boxes.device),
        )


class DifferentiableSAGEV2(nn.Module):
    """SAGE V2 with a continuous log-prior on decoder context attention."""

    def __init__(
        self,
        inp_l: int,
        enc_inp_size: int,
        dec_inp_size: int,
        dec_out_size: int,
        context_dim: int = 16,
        gate_input_mode: str = "original",
        share_satie: bool = True,
        **model_options: Any,
    ):
        super().__init__()
        if gate_input_mode not in GATE_INPUT_MODES:
            raise ValueError(f"gate_input_mode must be one of {GATE_INPUT_MODES}")
        self.base = IndividualTF(
            inp_l, enc_inp_size, dec_inp_size, dec_out_size, **model_options
        )
        _install_satie(self.base, share_satie)
        d_model = model_options.get("d_model", 512)
        self.context_encoder = SceneContextEncoder(
            d_model, context_dim, model_options.get("dropout", 0.1)
        )
        gate_input_dim = d_model if gate_input_mode == "context_only" else d_model + enc_inp_size
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        self.gate_input_mode = gate_input_mode
        self.inference_mode = "soft"
        self.last_gate_logits: torch.Tensor | None = None
        self.last_admission_probability: torch.Tensor | None = None
        self.last_gate_decision: torch.Tensor | None = None
        self.last_context_attention: torch.Tensor | None = None
        self.forced_admission: float | None = None
        install_context_prior_attention(self.base)

    def _gate_inputs(
        self, context_token: torch.Tensor, observed_boxes: torch.Tensor
    ) -> torch.Tensor:
        if self.gate_input_mode == "context_only":
            return context_token
        x_t = observed_boxes[:, -1]
        if self.gate_input_mode == "clipped":
            x_t = x_t.clamp(-5.0, 5.0)
        return torch.cat((context_token, x_t), dim=-1)

    def _admission_probability(
        self, context_token: torch.Tensor, observed_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(self._gate_inputs(context_token, observed_boxes))
        soft_probability = logits.softmax(dim=-1)[:, 1]
        self.last_gate_logits = logits
        self.last_admission_probability = soft_probability
        if self.forced_admission is not None:
            if self.training:
                raise RuntimeError("Forced admission is an inference-only diagnostic.")
            if self.forced_admission not in (0.0, 1.0):
                raise ValueError("forced_admission must be None, 0.0, or 1.0.")
            probability = torch.full_like(soft_probability, self.forced_admission)
            self.last_admission_probability = probability
            self.last_gate_decision = probability.bool() if not self.training else None
            return probability, logits
        if self.training or self.inference_mode == "soft":
            probability = soft_probability
            self.last_gate_decision = None
        elif self.inference_mode == "hard":
            decision = logits.argmax(dim=-1).eq(1)
            probability = decision.to(context_token.dtype)
            self.last_gate_decision = decision
        else:
            raise ValueError("inference_mode must be 'soft' or 'hard'.")
        return probability, logits

    def _context_path(
        self,
        arguments: tuple[torch.Tensor, ...],
        context_token: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        residual, inp, obd_spd, ego_flow, ped_flow, tgt, src_mask, obd_mask, spd_mask, ped_mask = arguments
        context_mask = torch.ones(
            src_mask.shape[0], 1, 1, dtype=src_mask.dtype, device=src_mask.device
        )
        set_decoder_context_probability(self.base, probability)
        try:
            decoded = self.base.model(
                inp, obd_spd, ego_flow, ped_flow, tgt, src_mask, obd_mask,
                spd_mask, ped_mask, context=context_token, context_mask=context_mask,
            )
        finally:
            set_decoder_context_probability(self.base, None)
        layer_weights = [
            layer.src_attn.attn[..., -1].mean(dim=(1, 2))
            for layer in self.base.model.decoder.layers
            if layer.src_attn.attn is not None and layer.src_attn.attn.shape[-1] == 16
        ]
        self.last_context_attention = (
            torch.stack(layer_weights).mean(dim=0).detach()
            if layer_weights
            else torch.zeros(
                residual.shape[0], dtype=residual.dtype, device=residual.device
            )
        )
        return self.base.model.generator(decoded) + residual

    def forward(
        self,
        residual: torch.Tensor,
        inp: torch.Tensor,
        obd_spd: torch.Tensor,
        ego_flow: torch.Tensor,
        ped_flow: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        obd_enc_mask: torch.Tensor,
        spd_mask: torch.Tensor,
        ped_mask: torch.Tensor,
        scene_context: torch.Tensor,
        observed_occlusion: torch.Tensor,
    ) -> torch.Tensor:
        arguments = (
            residual, inp, obd_spd, ego_flow, ped_flow, tgt,
            src_mask, obd_enc_mask, spd_mask, ped_mask,
        )
        context_token = self.context_encoder(scene_context, observed_occlusion)
        self.last_context_attention = None
        probability, _ = self._admission_probability(context_token, inp)

        if not self.training and self.inference_mode == "hard":
            admitted = probability.bool()
            self.last_gate_decision = admitted.detach()
            if not bool(admitted.any()):
                set_decoder_context_probability(self.base, None)
                self.last_context_attention = torch.zeros(
                    residual.shape[0], dtype=residual.dtype, device=residual.device
                )
                return self.base(*arguments)
            if not bool(admitted.all()):
                open_indices = admitted.nonzero(as_tuple=False).squeeze(1)
                closed_indices = (~admitted).nonzero(as_tuple=False).squeeze(1)
                open_arguments = _slice_model_args(arguments, open_indices)
                closed_arguments = _slice_model_args(arguments, closed_indices)
                open_context = context_token.index_select(0, open_indices)
                open_probability = probability.index_select(0, open_indices)
                opened = self._context_path(open_arguments, open_context, open_probability)
                open_attention = self.last_context_attention
                closed = self.base(*closed_arguments)
                output = residual.new_empty(residual.shape)
                output.index_copy_(0, open_indices, opened)
                output.index_copy_(0, closed_indices, closed)
                context_attention = residual.new_zeros(residual.shape[0])
                if open_attention is not None:
                    context_attention.index_copy_(0, open_indices, open_attention)
                self.last_context_attention = context_attention
                return output

        return self._context_path(arguments, context_token, probability)

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        arguments = _prepare_model_inputs(batch)
        boxes = batch.get("boxes", batch.get("enc_input")).float()
        return self(
            *arguments,
            batch["scene_context"].to(boxes.device),
            batch["observed_occlusion"].to(boxes.device),
        )

    def decoder_context_attention(self) -> torch.Tensor | None:
        return self.last_context_attention


def unique_trainable_parameter_count(model: nn.Module) -> int:
    unique = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    return sum(parameter.numel() for parameter in unique.values())


def build_satie_model(
    configuration: str,
    gate_input_mode: str = "original",
    model_options: dict[str, Any] | None = None,
) -> nn.Module:
    options = dict(model_options or {})
    options.setdefault("N", 3)
    options.setdefault("d_model", 32)
    options.setdefault("d_ff", 128)
    options.setdefault("h", 4)
    options.setdefault("dropout", 0.1)
    if configuration == "p2_b":
        return AlwaysOpenContextIndividualTF(
            15, 4, 4, 4, share_satie=True, **options
        )
    if configuration == "p2_a":
        try:
            from .context_TF import ContextIndividualTF
        except ImportError:
            from context_TF import ContextIndividualTF
        model = ContextIndividualTF(15, 4, 4, 4, **options)
        _install_satie(model.base, True)
        return model
    if configuration in {"p2_c", "p2_d", "p2_e"}:
        mode = {
            "p2_c": "original",
            "p2_d": "clipped",
            "p2_e": "context_only",
        }[configuration]
        if gate_input_mode != "original" and gate_input_mode != mode:
            raise ValueError(f"{configuration} defines gate input mode {mode}, not {gate_input_mode}.")
        return DifferentiableSAGEV2(
            15, 4, 4, 4,
            gate_input_mode=mode,
            share_satie=True,
            **options,
        )
    raise ValueError(f"Unknown P2 configuration: {configuration}")
