"""Versioned bbox-only transformer models for the Phase 5B pilot."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from .transformer.layer_norm import LayerNorm
    from .transformer.multihead_attention import MultiHeadAttention
except ImportError:
    from transformer.layer_norm import LayerNorm
    from transformer.multihead_attention import MultiHeadAttention


OBSERVED_FRAMES = 15
PREDICTION_FRAMES = 45
MODEL_CONFIG = {
    "layers": 3,
    "d_model": 32,
    "d_ff": 128,
    "heads": 4,
    "dropout": 0.1,
}


def constant_velocity_forecast(
    observed_boxes: torch.Tensor,
    prediction_length: int = PREDICTION_FRAMES,
) -> torch.Tensor:
    if observed_boxes.ndim != 3 or observed_boxes.shape[-1] != 4:
        raise ValueError("observed_boxes must have shape [batch, steps, 4].")
    if observed_boxes.shape[1] < 3 or prediction_length < 1:
        raise ValueError("At least three observed frames and one prediction are required.")
    velocity = (observed_boxes[:, -1] - observed_boxes[:, -3]) / 2.0
    steps = torch.arange(
        1, prediction_length + 1,
        dtype=observed_boxes.dtype,
        device=observed_boxes.device,
    ).view(1, prediction_length, 1)
    return observed_boxes[:, -1:, :] + steps * velocity.unsqueeze(1)


def unique_trainable_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    count = 0
    for parameter in module.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            count += parameter.numel()
            seen.add(id(parameter))
    return count


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_length: int = 5000):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence exceeds the configured positional encoding.")
        return self.dropout(values + self.encoding[:, :values.shape[1]])


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.norm_attention = LayerNorm(d_model)
        self.norm_feedforward = LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(values)
        values = values + self.dropout(
            self.self_attn(normalized, normalized, normalized)
        )
        return values + self.dropout(
            self.feedforward(self.norm_feedforward(values))
        )


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.source_attn = MultiHeadAttention(heads, d_model, dropout)
        self.norm_source = LayerNorm(d_model)
        self.norm_feedforward = LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm_source(queries)
        queries = queries + self.dropout(
            self.source_attn(normalized, memory, memory)
        )
        return queries + self.dropout(
            self.feedforward(self.norm_feedforward(queries))
        )


class AnnotationOnlyTransformer(nn.Module):
    """BBox-only encoder/decoder with an explicit, single residual anchor."""

    def __init__(
        self,
        anchor: str,
        share_encoder_attention: bool = False,
        layers: int = MODEL_CONFIG["layers"],
        d_model: int = MODEL_CONFIG["d_model"],
        d_ff: int = MODEL_CONFIG["d_ff"],
        heads: int = MODEL_CONFIG["heads"],
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super().__init__()
        if anchor not in {"constant_position", "constant_velocity"}:
            raise ValueError("Unsupported residual anchor.")
        if layers < 1 or d_model % heads:
            raise ValueError("Invalid transformer dimensions.")
        self.anchor = anchor
        self.share_encoder_attention = share_encoder_attention
        self.input_projection = nn.Linear(4, d_model)
        self.target_projection = nn.Linear(4, d_model)
        self.input_position = SinusoidalPositionEncoding(d_model, dropout)
        self.target_position = SinusoidalPositionEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, heads, dropout)
            for _ in range(layers)
        ])
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, heads, dropout)
            for _ in range(layers)
        ])
        self.decoder_norm = LayerNorm(d_model)
        self.generator = nn.Linear(d_model, 4)

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        if share_encoder_attention:
            shared_attention = self.encoder_layers[0].self_attn
            for layer in self.encoder_layers[1:]:
                layer.self_attn = shared_attention

    def anchor_forecast(self, observed_boxes: torch.Tensor) -> torch.Tensor:
        if self.anchor == "constant_position":
            return observed_boxes[:, -1:, :].expand(
                -1, PREDICTION_FRAMES, -1
            )
        return constant_velocity_forecast(observed_boxes)

    def forward(self, observed_boxes: torch.Tensor) -> torch.Tensor:
        if (
            observed_boxes.ndim != 3
            or observed_boxes.shape[1:] != (OBSERVED_FRAMES, 4)
        ):
            raise ValueError("Expected observed boxes with shape [batch, 15, 4].")
        if not bool(torch.isfinite(observed_boxes).all()):
            raise ValueError("Observed boxes must be finite.")

        memory = self.input_position(
            self.input_projection(observed_boxes) * math.sqrt(
                self.input_projection.out_features
            )
        )
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.encoder_norm(memory)

        target = observed_boxes.new_zeros(
            observed_boxes.shape[0], PREDICTION_FRAMES, 4
        )
        queries = self.target_position(
            self.target_projection(target) * math.sqrt(
                self.target_projection.out_features
            )
        )
        for layer in self.decoder_layers:
            queries = layer(queries, memory)
        correction = self.generator(self.decoder_norm(queries))
        return self.anchor_forecast(observed_boxes) + correction


def build_ao_model(model_id: str) -> AnnotationOnlyTransformer:
    options = dict(MODEL_CONFIG)
    if model_id == "AO-1":
        return AnnotationOnlyTransformer("constant_position", **options)
    if model_id == "AO-2":
        return AnnotationOnlyTransformer("constant_velocity", **options)
    if model_id == "AO-3":
        return AnnotationOnlyTransformer(
            "constant_velocity", share_encoder_attention=True, **options
        )
    raise ValueError(f"{model_id!r} is not a trainable AO model.")


def verify_satie_identity(model: AnnotationOnlyTransformer) -> bool:
    if len(model.encoder_layers) < 2:
        return True
    first = model.encoder_layers[0].self_attn
    return all(layer.self_attn is first for layer in model.encoder_layers[1:])


__all__ = [
    "AnnotationOnlyTransformer",
    "MODEL_CONFIG",
    "build_ao_model",
    "constant_velocity_forecast",
    "unique_trainable_parameter_count",
    "verify_satie_identity",
]
