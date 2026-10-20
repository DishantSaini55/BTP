"""C4: Version B with encoder self-attention Q/K/V/O tied across depth.

The function intentionally aliases existing ``MultiHeadAttention`` modules.
It does not tie any decoder attention, feed-forward layer, or context module.
"""
from __future__ import annotations

from typing import Any


def make_c4_encoder_self_attention_shared(model: Any) -> Any:
    """Tie all encoder ``self_attn`` modules to layer 0 by object identity."""
    layers = model.base.model.encoder.layers
    if len(layers) != 3:
        raise ValueError(f"C4 was specified for three encoder layers, got {len(layers)}")
    shared = layers[0].self_attn
    for layer in layers[1:]:
        layer.self_attn = shared
    return model


def verify_c4_sharing(model: Any) -> dict[str, Any]:
    """Return explicit identity/data-pointer/optimizer-registration evidence."""
    layers = model.base.model.encoder.layers
    attentions = [layer.self_attn for layer in layers]
    projection_names = ("Q", "K", "V", "O")
    # MTN's implementation stores Q/K/V/O in ``linears[0..3]``.
    evidence: dict[str, Any] = {"encoder_layers": len(layers), "projections": {}}
    for name, index in zip(projection_names, range(4)):
        modules = [attention.linears[index] for attention in attentions]
        weights = [module.weight for module in modules]
        biases = [module.bias for module in modules]
        evidence["projections"][name] = {
            "same_module_object": all(module is modules[0] for module in modules[1:]),
            "same_weight_parameter_object": all(weight is weights[0] for weight in weights[1:]),
            "same_weight_data_ptr": all(weight.data_ptr() == weights[0].data_ptr() for weight in weights[1:]),
            "same_bias_parameter_object": all(bias is biases[0] for bias in biases[1:]),
            "same_bias_data_ptr": all(bias.data_ptr() == biases[0].data_ptr() for bias in biases[1:]),
        }
    named = list(model.named_parameters())
    parameter_ids = [id(parameter) for _, parameter in named]
    evidence["optimizer_registration"] = {
        "named_parameter_entries": len(named),
        "unique_parameter_objects": len(set(parameter_ids)),
        "has_duplicate_parameter_objects": len(parameter_ids) != len(set(parameter_ids)),
    }
    evidence["verified"] = all(
        all(values[key] for key in ("same_module_object", "same_weight_parameter_object", "same_weight_data_ptr", "same_bias_parameter_object", "same_bias_data_ptr"))
        for values in evidence["projections"].values()
    ) and not evidence["optimizer_registration"]["has_duplicate_parameter_objects"]
    return evidence
