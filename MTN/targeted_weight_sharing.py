"""Targeted attention-module tying for Version B; no FFN or context sharing."""
from __future__ import annotations


def apply_targeted_sharing(model, strategy: str) -> None:
    """Alias selected existing modules after independent official initialization.

    Strategies retain every layer execution and all context components.  They
    share only same-role attention components across depth.
    """
    encoder_layers = model.base.model.encoder.layers
    decoder_layers = model.base.model.decoder.layers
    if strategy == "none":
        return
    if strategy == "decoder_src_attn":
        shared = decoder_layers[0].src_attn
        for layer in decoder_layers[1:]:
            layer.src_attn = shared
    elif strategy == "decoder_src_qkv":
        reference = decoder_layers[0].src_attn.linears
        for layer in decoder_layers[1:]:
            for index in (0, 1, 2):
                layer.src_attn.linears[index] = reference[index]
    elif strategy == "decoder_src_kv":
        reference = decoder_layers[0].src_attn.linears
        for layer in decoder_layers[1:]:
            for index in (1, 2):
                layer.src_attn.linears[index] = reference[index]
    elif strategy == "encoder_self_attn":
        shared = encoder_layers[0].self_attn
        for layer in encoder_layers[1:]:
            layer.self_attn = shared
    else:
        raise ValueError(f"Unknown targeted sharing strategy: {strategy}")


STRATEGY_LABELS = {
    "none": "No sharing (Version B)",
    "decoder_src_attn": "Decoder source cross-attention: complete Q/K/V/O module",
    "decoder_src_qkv": "Decoder source cross-attention: Q/K/V projections only",
    "decoder_src_kv": "Decoder source cross-attention: K/V projections only",
    "encoder_self_attn": "Encoder self-attention: complete Q/K/V/O module",
}
