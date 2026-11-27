"""Controlled Phase 4 model configurations with a constant-velocity anchor."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch.nn as nn

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from .context_TF import ContextIndividualTF
    from .cv_residual_v1 import ResidualAnchorModel, trainable_parameter_count
    from .individual_TF import IndividualTF
    from .sage_v2 import AlwaysOpenContextIndividualTF, DifferentiableSAGEV2
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared, verify_c4_sharing
    from context_TF import ContextIndividualTF
    from cv_residual_v1 import ResidualAnchorModel, trainable_parameter_count
    from individual_TF import IndividualTF
    from sage_v2 import AlwaysOpenContextIndividualTF, DifferentiableSAGEV2


P4_CONFIGURATIONS = ("p4_a", "p4_b", "p4_c", "p4_d", "p4_e", "p4_f")
P4_SELECTION_POLICY = {
    "p4_a": "base",
    "p4_b": "base",
    "p4_c": "always_open",
    "p4_d": "always_open",
    "p4_e": "hard",
    "p4_f": "soft",
}
P4_EVALUATION_POLICIES = {
    "p4_a": ("base",),
    "p4_b": ("base",),
    "p4_c": ("always_open",),
    "p4_d": ("always_open",),
    "p4_e": ("hard", "forced_open", "forced_closed"),
    "p4_f": ("soft", "hard", "forced_open", "forced_closed"),
}
MODEL_OPTIONS = {"N": 3, "d_model": 32, "d_ff": 128, "h": 4, "dropout": 0.1}


def build_p4_model(configuration: str) -> ResidualAnchorModel:
    if configuration not in P4_CONFIGURATIONS:
        raise ValueError(f"Unknown Phase 4 configuration: {configuration}")

    if configuration in {"p4_a", "p4_b"}:
        predictor: nn.Module = IndividualTF(15, 4, 4, 4, **MODEL_OPTIONS)
        if configuration == "p4_b":
            make_c4_encoder_self_attention_shared(SimpleNamespace(base=predictor))
    elif configuration in {"p4_c", "p4_d"}:
        predictor = AlwaysOpenContextIndividualTF(
            15, 4, 4, 4,
            share_satie=configuration == "p4_d",
            **MODEL_OPTIONS,
        )
    elif configuration == "p4_e":
        predictor = ContextIndividualTF(15, 4, 4, 4, **MODEL_OPTIONS)
        make_c4_encoder_self_attention_shared(predictor)
    else:
        predictor = DifferentiableSAGEV2(
            15,
            4,
            4,
            4,
            gate_input_mode="context_only",
            share_satie=True,
            **MODEL_OPTIONS,
        )
    return ResidualAnchorModel(predictor, anchor="constant_velocity")


def parameter_sharing_audit(model: ResidualAnchorModel) -> dict[str, Any]:
    predictor = model.predictor
    base = predictor if isinstance(predictor, IndividualTF) else getattr(predictor, "base", None)
    if not isinstance(base, IndividualTF):
        return {
            "satie_enabled": False,
            "encoder_self_attention": None,
            "other_attention_sharing_checked": False,
        }

    encoder_layers = base.model.encoder.layers
    decoder_layers = base.model.decoder.layers
    encoder_self_attn_shared = all(
        layer.self_attn is encoder_layers[0].self_attn
        for layer in encoder_layers[1:]
    )
    sharing_view = nn.Module()
    sharing_view.add_module("base", base)
    decoder_attention_names = ("src_attn", "spd_attn", "ped_attn")
    audit = {
        "satie_enabled": encoder_self_attn_shared,
        "encoder_self_attention": verify_c4_sharing(sharing_view)
        if encoder_self_attn_shared
        else None,
        "encoder_other_attention_is_independent": all(
            getattr(encoder_layers[index], name)
            is not getattr(encoder_layers[0], name)
            for name in ("src_attn", "cross_attn")
            for index in range(1, len(encoder_layers))
        ),
        "decoder_attention_is_independent": {
            name: all(
                getattr(decoder_layers[index], name)
                is not getattr(decoder_layers[0], name)
                for index in range(1, len(decoder_layers))
            )
            for name in decoder_attention_names
        },
        "encoder_decoder_attention_is_independent": all(
            encoder_layer.self_attn is not getattr(decoder_layer, name)
            for encoder_layer in encoder_layers
            for decoder_layer in decoder_layers
            for name in decoder_attention_names
        ),
        "unique_trainable_parameters": trainable_parameter_count(model),
    }
    audit["other_attention_sharing_checked"] = True
    audit["verified"] = (
        (not encoder_self_attn_shared or audit["encoder_self_attention"]["verified"])
        and audit["encoder_other_attention_is_independent"]
        and all(audit["decoder_attention_is_independent"].values())
        and audit["encoder_decoder_attention_is_independent"]
    )
    return audit


def unique_parameter_count(model: nn.Module) -> int:
    return trainable_parameter_count(model)
