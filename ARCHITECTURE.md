# MTN scene-context ablation — implementation-aligned architecture

This document describes the source-code path in `MTN/context_TF.py`,
`MTN/context_schema.py`, `MTN/individual_TF.py`, and the Transformer
modules under `MTN/transformer/`.

## Model variants

| Variant | Inputs | Context / parameter sharing |
| --- | --- | --- |
| `IndividualTF` | Observed boxes, OBD speed, ego optical flow, pedestrian optical flow | Original MTN encoder and multi-source decoder |
| `ContextIndividualTF` | Baseline inputs plus native scene labels and observed occlusion history | A gated context token is appended to encoded trajectory memory |
| `ContextIndividualTF` + C4 | Same as ContextIndividualTF | `make_c4_encoder_self_attention_shared(model)` ties only encoder self-attention across the three encoder layers |

**Important:** constructing `ContextIndividualTF` alone does not activate C4.
For C4, instantiate the context model with `N=3`, then call
`make_c4_encoder_self_attention_shared(model)`. The smoke test now exercises
this exact helper and verifies the shared module/parameter identities.

## Native context schema (C1)

The context encoder consumes exactly five categorical scene fields in this
order, as defined by `MTN/context_schema.py`:

`intersection_type, signalized_type, road_type, location_type, motion_direction`

Observed occlusion is a separate sequence over the 15 observed frames. It is
not a sixth static scene field.

- Each of the five categorical embeddings has 16 dimensions.
- The 15 occlusion embeddings (16 dimensions each) are mean-pooled into one
  16-dimensional history summary.
- The five static embeddings plus pooled occlusion summary concatenate to
  96 dimensions.
- The context projection applies LayerNorm(96), Linear(96 → d_model), GELU,
  Dropout, then Linear(d_model → d_model). For the POC configuration
  `d_model=32`, the emitted context vector is 32-D.
- The gate concatenates that context vector with the last observed box
  (4 dimensions), then applies Linear(36 → 16), GELU, Linear(16 → 2) in the
  `d_model=32` POC configuration.
- Training with `gate_mode="hard"` uses hard straight-through
  Gumbel-softmax (temperature 1.0); evaluation uses argmax. The selected gate
  multiplies the context vector and is also used as the appended-token mask.

`time_of_day` is not in `SCENE_FIELDS` and is not passed to
`SceneContextEncoder` by `ContextIndividualTF`. Any day/night labels used for
subgroup evaluation must be treated as evaluation metadata, not as a sixth
model input in this implementation.

## Encoder and C4

The three-layer POC configuration is `N=3`, `d_model=32`, `d_ff=128`,
`h=4`. These are configuration values, not hard-coded architectural limits.

Each `EncoderLayer` contains:
1. `self_attn`: trajectory tokens attend to trajectory tokens.
2. `src_attn`: trajectory tokens attend to the encoded speed stream.
3. `cross_attn`: the speed stream attends to the pre-source-attention
   trajectory representation.
4. Separate feed-forward processing for trajectory and speed streams.

C4 aliases the complete `MultiHeadAttention` module stored in
`EncoderLayer.self_attn` across `encoder.layers[0]`, `[1]`, and `[2]`.
That module contains the Q, K, V and output (O) linear projections in
`linears[0..3]`. The encoder `src_attn`, `cross_attn`, feed-forward blocks,
normalization/sublayer modules, decoder attention, and context modules are
not shared by the C4 helper. All three encoder layers still execute; C4 shares
parameters rather than skipping layer computation.

## Memory augmentation and decoder

The trajectory encoder produces memory with shape `[B, 15, 32]` for the
POC setting. The selected context vector is appended after the encoder,
producing `[B, 16, 32]`; the source mask is extended with the gate value.

The encoded speed stream has shape `[B, 1, 32]`. In
`EncoderDecoder.decode`, it is broadcast-added to the augmented memory
`[B, 16, 32]`. This 1-to-16 addition is intentional and shape-compatible.

The decoder has three layers in the POC configuration. There is no active
decoder self-attention call in `DecoderLayer.forward`. Each decoder layer
combines three parallel cross-source attention branches:

- `src_attn`: decoder query attends to augmented memory;
- `spd_attn`: decoder query attends to ego-flow embeddings;
- `ped_attn`: decoder query attends to pedestrian-flow embeddings.

The three attention outputs are added through the MTN sublayer residual path,
then passed through the feed-forward block (32 → 128 → 32 in the POC
configuration).

The generator predicts 45 sets of four box values. The last observed box is
repeated over the 45-step horizon and added as the residual, yielding
`[B, 45, 4]`.

## Synthetic smoke-test input contract

| Input | Shape supplied to `forward_batch` |
| --- | --- |
| `boxes` | `[B, 15, 4]` |
| `speed` | `[B, 60, 1]` (the model slices the first 15 observation frames) |
| `ego_flow` | `[B, 60, 64, 2]` |
| `ped_flow` | `[B, 60, 9, 2]` |
| `scene_context` | `[B, 5]` |
| `observed_occlusion` | `[B, 15]` |

This smoke test validates tensor contracts and C4 sharing; it is not a
trajectory-quality benchmark. Report MSE only from the documented video-level
evaluation split and clearly distinguish CPU/Farneback proof-of-concept runs
from full-dataset benchmark results. Never use future occlusion or future
trajectory-derived labels as model inputs.
