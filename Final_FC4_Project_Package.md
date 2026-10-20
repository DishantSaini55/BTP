# Final Project Package — F+C4 Candidate

## Status

**Candidate architecture only; not a final proven architecture.** All reported values are CPU Farneback proof-of-concept (POC) results, not authors' RAFT-pipeline or full IDD-PeD benchmark results.

## 1. Final architecture specification

**Name:** DynMM-style Fusion MTN with optional C4 encoder self-attention sharing (F+C4).

Input boxes `[B,15,4]` are embedded to `[B,15,32]` and processed by the explicit MTN `Encoder` (`base.model.encoder`), containing three `EncoderLayer` blocks. The encoder produces trajectory memory `[B,15,32]`. Five static context labels plus observed occlusion form six separate 16-D embeddings. Each is projected to 32-D, forming six context tokens `[B,6,32]`; their flattened 96-D representation is separately projected to a global context vector `[B,32]`.

Dynamic Multimodal Fusion is inserted after the encoder and before the MTN decoder. It selects one per-sample candidate output: identity trajectory memory, normalized additive fusion with broadcast global context, or trajectory-query/context-key-value cross-attention followed by residual/norm/FFN. The selected memory remains `[B,15,32]`. The original decoder and output head then produce `[B,45,4]` future boxes.

C4 ties only `EncoderLayer.self_attn` across encoder depths 0–2: Q/K/V/O are `self_attn.linears[0..3]`. Encoder `src_attn`, `cross_attn`, FFNs/norms, all decoder modules, the context branch, and output head remain independent.

## 2. Module-wise explanation

| Module | Simple language | Technical function |
|---|---|---|
| MTN input path | Converts observed boxes into motion features. | `[B,15,4] → Linear(4,32)+PE → [B,15,32]`. |
| MTN encoder ×3 | Learns observed trajectory and speed relationships. | `Encoder.layers[0..2]`; self-attn, source-speed attention, reverse cross-attn, FFN. |
| Six context inputs | Supplies road and visibility clues available at inference. | Five scene labels plus observed occlusion. |
| Context embeddings | Turns categories into learnable vectors. | Six 16-D vectors; occlusion is mean-pooled over observed steps. |
| Context tokens | Keeps each context source separate for attention. | `6×16 → token projection → [B,6,32]`. |
| Global context | Supplies one summary feature to gate/additive fusion. | Flattened `[B,96] → 96→32→32 → [B,32]`. |
| Dynamic fusion | Lets the model choose how much/context type to use. | Identity, additive, or cross-attention output, each `[B,15,32]`. |
| Dynamic gate | Selects one fusion operation for each sample. | pooled trajectory `[B,32]` + global context `[B,32]`: `64→32→3`; hard Gumbel-softmax in training, argmax at evaluation. |
| C4 sharing | Reuses one encoder self-attention parameter set across depth. | aliases Q/K/V/O in `encoder.layers[0..2].self_attn`. |
| MTN decoder ×3 | Converts fused memory and flow features into future motion features. | Original decoder with source-memory, ego-flow, pedestrian-flow attention and FFN. |
| Output head | Predicts future boxes. | `Linear(32,4)` + final observed-box residual → `[B,45,4]`. |

## 3. Evolution and modification table

| Variant | Modification | Location | Research question | Status |
|---|---|---|---|---|
| Original MTN / A | No context, no sharing. | `IndividualTF`. | Baseline trajectory prediction. | POC baseline. |
| B | Context encoder + gated context-token admission. | After encoder, before decoder. | Does native context help? | Context ablation. |
| C4 | Q/K/V/O sharing only. | `EncoderLayer.self_attn`, depths 0–2. | Can parameters be reduced selectively? | Sharing ablation. |
| F | Genuine three-operation DynMM-style fusion. | Encoder→decoder interface. | Does trajectory/context interaction help? | Fusion candidate. |
| F+C4 | F plus C4 sharing. | Fusion interface + encoder self-attn aliases. | Can fusion and sharing coexist? | Candidate; not finalized. |

## 4. Key architecture clarifications

- **Encoder:** `base.model.encoder`, class `Encoder`, file `MTN/transformer/encoder.py`; three `EncoderLayer` blocks.
- **Decoder:** `base.model.decoder`, class `Decoder`, file `MTN/transformer/decoder.py`; three `DecoderLayer` blocks.
- **Context entry:** F receives context after encoder output and before decoder input.
- **Six context inputs:** `intersection_type`, `signalized_type`, `road_type`, `location_type`, `motion_direction`, observed `occlusion`.
- **DynMM fusion:** operates on trajectory `[B,15,32]`, six context tokens `[B,6,32]`, and global context `[B,32]`.
- **C4:** shares layer-0 `self_attn` with layers 1 and 2; `linears[0]=Q`, `[1]=K`, `[2]=V`, `[3]=O`, each `Linear(32,32)`.
- **No day/night or illumination:** neither is implemented.
- **No branch-skipping saving in current F:** all three fusion branches are computed before selection; no resource-aware loss is used.

## 5. Measured POC ablation evidence

| Direct POC comparison | Parameters | MSE | C-MSE | CF-MSE | Latency | Seeds / test / epochs |
|---|---:|---:|---:|---:|---:|---|
| A | 133,700 | 1242.98 | 1225.17 | 4093.85 | 69.24 ms | 1 / 10 / 12 |
| B | 139,142 | 1113.21 | 1094.57 | 3646.51 | 61.41 ms | 1 / 10 / 12 |
| B, larger POC | 139,142 | 1137.11 ± 20.16 | 1107.40 ± 19.63 | 2971.36 ± 103.13 | 74.09 ± 2.48 ms | 3 / 24 / 1 |
| C4, larger POC | 130,694 | 1080.16 ± 93.33 | 1053.83 ± 95.13 | 2887.51 ± 237.14 | 81.60 ± 13.34 ms | 3 / 24 / 1 |
| F | 154,007 | 1154.78 | 1124.20 | 3098.16 | 45.82 ms | 3 / 24 / 1 |
| F+C4 | 145,559 | 1126.67 | 1098.43 | 3111.62 | 49.14 ms | 3 / 24 / 1 |

Do not rank all rows as one benchmark: A/B is a separate 10-trajectory, 12-epoch POC; B/C4 and F/F+C4 are separate 24-trajectory, one-epoch POCs. All use CPU Farneback flow and one held-out GP6 video.

## 6. Paper-ready proposed-architecture section

We propose a candidate extension of MTN that integrates native scene context through dynamic multimodal fusion. The original MTN encoder maps observed pedestrian bounding boxes to trajectory memory of shape `[B,15,32]`. Five static scene attributes and observed occlusion are separately embedded, yielding six context representations. The representations are projected into six context tokens `[B,6,32]` for cross-attention and concatenated into a 96-D global context feature, which is projected to `[B,32]` for gating and additive fusion.

At the encoder–decoder interface, a per-sample gate selects identity mapping, additive fusion, or trajectory-query/context-key-value cross-attention. Cross-attention uses trajectory memory as queries and six context tokens as keys/values; its output is refined with residual connections, normalization, and an FFN. The selected fused representation remains `[B,15,32]` and is processed by the unchanged MTN decoder with the original ego-flow and pedestrian-flow paths. The prediction head preserves the original `[B,45,4]` output format.

An optional C4 parameter-sharing variant aliases only encoder self-attention Q/K/V/O projections across all three encoder layers. This reduces parameter storage but does not remove attention executions; it must not be described as a FLOP reduction.

## 7. One-page supervisor summary

1. Original MTN predicts future boxes from observed boxes, speed, and optical flow.
2. B adds six native scene/visibility sources as a context branch.
3. C4 shares only encoder self-attention Q/K/V/O across the three encoder depths.
4. F replaces simple context-token admission with genuine trajectory/context fusion.
5. F uses identity, additive, and cross-attention fusion choices, selected per sample.
6. F+C4 combines genuine fusion with selective sharing; it is the proposed candidate, not a proven final architecture.
7. Current POC evidence supports more testing, not final superiority or computation-saving claims.

# FINAL DRAW.IO STRUCTURE

Draw left-to-right in this exact order. Use **solid arrows** for tensors/data, **dashed teal arrows** for C4 shared parameter references, and a **dashed orange annotation** for the gate’s operation-selection relation.

1. **Observed pedestrian bounding boxes**
   - Label: `Observed boxes`
   - Input/output: `[B,15,4]`
   - Solid arrow to source embedding.

2. **MTN source embedding + positional encoding**
   - Label: `Linear(4,32) + Positional Encoding`
   - Output: `[B,15,32]`
   - Solid arrow to MTN encoder.

3. **MTN Encoder ×3 container**
   - Sub-boxes: `EncoderLayer 1`, `EncoderLayer 2`, `EncoderLayer 3`.
   - Each sub-box label: `self_attn + src_attn + cross_attn + FFN`.
   - Input/output: trajectory `[B,15,32]`; speed state `[B,1,32]`.
   - Solid arrows sequentially between layers.

4. **C4 shared self-attention box**
   - Label exactly: `C4 shared EncoderLayer.self_attn Q/K/V/O` and `Shared across Encoder.layers[0..2]`.
   - Input/output: parameter reference, not a tensor flow.
   - Dashed teal arrows to each encoder layer’s `self_attn` sub-box.
   - Note: `All 3 attention operations still execute.`

5. **Trajectory memory**
   - Label: `MTN trajectory memory`
   - Output: `[B,15,32]`
   - Solid arrows to all three dynamic-fusion operation boxes.

6. **Six native context-input boxes**
   - Labels: `intersection_type`, `signalized_type`, `road_type`, `location_type`, `motion_direction`, `observed occlusion`.
   - Solid arrows to the embedding box.

7. **Six context embeddings**
   - Label: `6 × 16-D embeddings`.
   - Output: `[B,6,16]`.
   - Split into two solid outputs.

8. **Token projection path**
   - Label: `Token projection: 16 → 32`.
   - Output: `six context tokens [B,6,32]`.
   - Solid arrows to cross-attention K/V.

9. **Global-context path**
   - Label: `Concatenate: 6 × 16 = 96-D` then `Global context projection: 96 → 32 → 32`.
   - Output: `[B,32]`.
   - Solid arrows to additive fusion and dynamic gate.

10. **Dynamic Multimodal Fusion container**
    - Title: `DYNAMIC MULTIMODAL FUSION`.
    - Three parallel operation boxes:
      - `Operation 1: Identity` — `[B,15,32] → [B,15,32]`.
      - `Operation 2: Additive fusion` — trajectory + broadcast global context `[B,32]` → `[B,15,32]`.
      - `Operation 3: Cross-attention fusion` — `Q: trajectory [B,15,32]; K,V: six context tokens [B,6,32]`; residual + norm + FFN → `[B,15,32]`.

11. **Dynamic gate**
    - Label: `Per-sample gate: pooled trajectory [B,32] + global context [B,32] → 64 → 32 → 3`.
    - Output: `one operation decision per sample`.
    - Dashed orange arrows to operation-selection merge.
    - Note: `Training: hard Gumbel-softmax; evaluation: argmax. All branches currently execute before selection.`

12. **Selected fused trajectory memory**
    - Label: `Fused trajectory memory [B,15,32]`.
    - Solid arrow to decoder memory input.

13. **Original flow inputs**
    - Boxes: `Ego flow → [B,64,32]`; `Pedestrian flow → [B,9,32]`.
    - Solid arrows to corresponding decoder-attention inputs.

14. **MTN Decoder ×3**
    - Label: `DecoderLayer ×3: src_attn + spd_attn + ped_attn + FFN`.
    - Input/output: fused memory and flows → `[B,45,32]`.

15. **Prediction head**
    - Label: `Linear(32,4) + final-observed-box residual`.
    - Output: `Future bounding-box prediction [B,45,4]`.
