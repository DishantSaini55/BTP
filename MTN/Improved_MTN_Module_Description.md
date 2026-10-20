# Context-Aware MTN: Module-wise Description

This document describes the current **Version B** implementation in `context_TF.py`, `individual_TF.py`, and the MTN transformer modules. It does not describe the separate experimental weight-sharing Version C.

## 1. Trajectory input

- **Input:** observed pedestrian bounding-box sequence, 15 steps × 4 coordinates.
- **Operation:** the source linear embedding maps each 4-D box to `d_model = 32`; positional encoding is then applied.
- **Output:** a 15 × 32-D trajectory representation.
- **Purpose:** supplies the original MTN with observed pedestrian motion/position information.
- **Why retained:** this is the unchanged trajectory path of the MTN backbone.

The existing MTN also receives observed OBD speed in its encoder pathway and ego/pedestrian flow inputs in the decoder pathway. These remain unchanged from the baseline MTN.

## 2. MTN encoder

- **Input:** projected trajectory tokens and the existing observed-speed representation.
- **Operation:** three encoder-layer executions. Each layer applies trajectory self-attention, source/speed attention, cross-attention back to the speed representation, and feed-forward networks. The configured dimensions are `d_model = 32`, 4 heads, and `d_ff = 128`.
- **Output:** encoder memory with shape 15 × 32-D, plus the internal speed representation.
- **Purpose:** encodes the observed trajectory with the existing MTN auxiliary speed path.
- **Why retained:** Version B does not replace or alter the baseline MTN encoder.

## 3. Scene context embeddings

- **Input:** five static native IDD-PeD labels: `intersection_type`, `signalized_type`, `road_type`, `location_type`, and `motion_direction`; plus observed pedestrian occlusion history.
- **Operation:** each static label uses its own 16-D embedding lookup. Occlusion values use a 16-D embedding and are mean-aggregated over observed steps.
- **Output:** five 16-D static embeddings and one 16-D occlusion representation.
- **Purpose:** turns categorical scene/state values into learned continuous representations.
- **Why introduced:** similar short-term pedestrian motion can occur in different road/visibility conditions.

## 4. Context feature concatenation

- **Input:** five 16-D static embeddings and the 16-D aggregated occlusion embedding.
- **Operation:** concatenation.
- **Output:** a 96-D context feature (6 × 16-D).
- **Purpose:** preserves each context source before learned projection.
- **Why introduced:** supplies one unified context feature without adding unsupported day/night or illumination fields.

## 5. Context projection

- **Input:** 96-D context feature.
- **Operation:** `LayerNorm(96) → Linear(96, 32) → GELU → Dropout → Linear(32, 32)`.
- **Output:** one 32-D context token.
- **Purpose:** aligns context with MTN’s `d_model = 32` representation space.
- **Why introduced:** the token can be appended to decoder memory without changing MTN tensor width.

## 6. Dynamic context gate

- **Input:** 32-D context token concatenated with the final observed bounding box (4-D).
- **Operation:** `Linear(36, 16) → GELU → Linear(16, 2)`. In training, the implementation selects with hard Gumbel-softmax; in evaluation, it uses the argmax branch.
- **Output:** a binary context-admission value and a gated 32-D context token.
- **Purpose:** controls whether the context token is made available to decoder memory for each sample.
- **Why introduced:** context is supplemental rather than a replacement for trajectory evidence.

## 7. Context-memory fusion

- **Input:** 15 × 32-D MTN encoder memory and the gated 32-D context token.
- **Operation:** append the token along the memory-sequence dimension; append its gate-derived mask entry to the source mask.
- **Output:** augmented decoder memory with 16 × 32-D tokens.
- **Purpose:** makes context available through the existing decoder source-attention mechanism.
- **Why introduced:** preserves the baseline trajectory memory and provides context as one additional memory token.

## 8. MTN decoder

- **Input:** zero target sequence, augmented decoder memory, and the unchanged ego-flow and pedestrian-flow representations.
- **Operation:** three decoder-layer executions. The decoder combines its existing source-memory, ego-flow, and pedestrian-flow attention paths, then applies the feed-forward network.
- **Output:** 45 × 32-D decoder representation.
- **Purpose:** produces future-step features conditioned on trajectory memory, auxiliary flows, and the optional context token.
- **Why retained:** the baseline decoder operations are unchanged; only its memory gains one gated context token.

## 9. Prediction head

- **Input:** 45 × 32-D decoder representation and the final observed bounding box.
- **Operation:** `Linear(32, 4)` prediction FC; add the final observed box repeated for 45 prediction steps as a residual.
- **Output:** 45 future 4-D bounding boxes.
- **Purpose:** predicts future pedestrian trajectory boxes in the original MTN output format.
- **Why retained:** keeps the target and output interface identical to baseline MTN.

## Scope note

The final Version B architecture contains no day/night or illumination feature. These were not used because reliable native IDD-PeD fields were not available in the implemented context schema.
