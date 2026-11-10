"""Regression tests for selective C4 encoder self-attention sharing."""

import torch

from MTN.context_TF import ContextIndividualTF
from MTN.c4_encoder_shared import (
    make_c4_encoder_self_attention_shared,
    verify_c4_sharing,
)


def test_c4_shares_only_encoder_self_attention():
    torch.manual_seed(11)
    model = ContextIndividualTF(
        15,
        4,
        4,
        4,
        N=3,
        d_model=32,
        d_ff=128,
        h=4,
        dropout=0.0,
        gate_mode="hard",
    )

    # C4 is an explicit model transformation; ContextIndividualTF alone
    # intentionally remains the non-C4 context baseline.
    layers = model.base.model.encoder.layers
    assert len(layers) == 3
    assert len({id(layer.self_attn) for layer in layers}) == 3

    make_c4_encoder_self_attention_shared(model)
    evidence = verify_c4_sharing(model)
    assert evidence["verified"], evidence

    shared = layers[0].self_attn
    assert all(layer.self_attn is shared for layer in layers)
    for projection_index in range(4):  # Q, K, V, output projection
        owner = shared.linears[projection_index]
        for layer in layers[1:]:
            candidate = layer.self_attn.linears[projection_index]
            assert candidate is owner
            assert candidate.weight is owner.weight
            assert candidate.bias is owner.bias

    # C4 must not tie the other encoder submodules.
    for attribute in ("src_attn", "cross_attn", "feed_forward", "sublayer"):
        modules = [getattr(layer, attribute) for layer in layers]
        assert len({id(module) for module in modules}) == 3, attribute

    # Nor should it tie decoder layers.
    decoder_layers = model.base.model.decoder.layers
    assert len(decoder_layers) == 3
    assert len({id(layer.src_attn) for layer in decoder_layers}) == 3
    assert len({id(layer.spd_attn) for layer in decoder_layers}) == 3
    assert len({id(layer.ped_attn) for layer in decoder_layers}) == 3
