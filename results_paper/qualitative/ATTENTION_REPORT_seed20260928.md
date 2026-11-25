# Decoder attention and gate-mask analysis (seed 20260928)

Analysis uses saved E1 predictions and attention arrays; no training or checkpoint selection was performed.

## Saved E1 gate availability

| Model | g=0 windows | g=1 windows |
|---|---:|---:|
| PACE-Net without SATIE | 217 | 0 |
| PACE-Net | 217 | 0 |

Neither learned SAGE checkpoint admitted context on any of the 217 test windows. Consequently, the saved E1 attention has 15 trajectory-memory keys, not 16 keys; there is no observed learned-gate `g=1` context-attention weight or naturally admitted window to mask. The context column in the selected-window CSV is explicitly marked as omitted, not misreported as a measured softmax value. The all-closed forward path bypasses token concatenation, so its effective context-attention mass is exactly zero by absence.

## Attention summaries

`ATTENTION_SELECTED_seed20260928.csv` reports each selected window's mean across 45 decoder queries for all 4 heads, all 3 decoder layers, and each available trajectory key. `ATTENTION_TEST_AGGREGATE_seed20260928.csv` reports the same per-head means averaged over all 217 test windows; the context-key row has zero observed admitted windows and no measured attention value.

## Explicit mixed-mask implementation check

A post-hoc forward-only check loaded the saved PACE-Net checkpoint and forced a two-row gate batch `[1, 0]`; no weights were updated. The decoder then had 16 keys. In every decoder layer and head, the g=0 row's context attention was exactly zero (`count_nonzero == 0`), while the forced-open row had positive context weight. These forced values verify masking semantics only; they are not learned-gate results and are excluded from the qualitative comparison.

The mask-probe details are in `GATE_MASK_PROBE_seed20260928.csv`. The scene-stratified masking table reports the actual saved-gate population and marks the g=1 masking effect not estimable because the count is zero.

## Scene-stratified context masking

| Scene field | Label code | Test windows | g=0 | g=1 | Natural g=1 masking effect |
|---|---:|---:|---:|---:|---|
| intersection_type | 0 | 217 | 217 | 0 | not estimable: no naturally admitted windows |
| signalized_type | 0 | 12 | 12 | 0 | not estimable: no naturally admitted windows |
| signalized_type | 1 | 14 | 14 | 0 | not estimable: no naturally admitted windows |
| signalized_type | 3 | 191 | 191 | 0 | not estimable: no naturally admitted windows |
| road_type | 0 | 217 | 217 | 0 | not estimable: no naturally admitted windows |
| location_type | 0 | 217 | 217 | 0 | not estimable: no naturally admitted windows |
| motion_direction | 0 | 217 | 217 | 0 | not estimable: no naturally admitted windows |
