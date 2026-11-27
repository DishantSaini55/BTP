# Differentiable SAGE V2: formulation and implementation

## Frozen components and scope

SAGE V2 is a separate experimental implementation in `MTN/sage_v2.py`. It leaves the historical SAGE gate, MASC masking, E1 models, and SATIE helper unchanged. The P2 models use the frozen GP6 split, training-only box/speed normalization, cached optical flow, Adam at `1e-3`, batch size 8, and normalized-coordinate MSE. The screening runner loads training and validation only.

## Admission prior

Let the encoder produce the trajectory memory

`H = [h_1, ..., h_15]`

and let the scene-context encoder produce context token `c`. The learned two-class gate emits logits `z`; the continuous admission probability is

`p = softmax(z)_context`.

The decoder memory is `[h_1, ..., h_15, c]` for every training sample. For each decoder source-attention layer and query, the unnormalized attention score for the context key is modified as

`s'_context = s_context + log(p + epsilon)`, with `epsilon = 1e-8`.

Trajectory-key logits are unchanged. Existing Boolean masks for padding or invalid sequence positions are still applied after the context prior. Attention is then normalized by the usual softmax:

`alpha_i = exp(s'_i) / sum_j exp(s'_j)`.

Thus, before the epsilon floor is material, the context's unnormalized mass is multiplied by `p`; the relative weights among trajectory tokens are not changed by admission. The prior is installed in every decoder source-attention layer that reads the augmented memory.

The context token is **not** also multiplied by `p`, and its Boolean mask remains open in the differentiable training path. This avoids applying admission twice. The original trajectory-memory mask remains intact.

## Training and inference behavior

- **Training:** continuous `p` and the differentiable log-prior are always used. No all-closed early return occurs.
- **Soft inference:** uses the learned continuous probability and the same prior.
- **Hard inference:** uses the gate argmax. Open samples use a context prior of exactly 1; rejected samples take the original motion-only `IndividualTF` path. Mixed batches are split so rejected predictions are bitwise identical to motion-only predictions.
- **Forced-open/forced-closed validation:** implemented as explicit diagnostic policies, never as training settings.
- **P2-A:** continues to use the historical SAGE/MASC implementation; no V2 behavior is retrofitted into it.

## Controlled configurations and measured parameters

All counts below are unique trainable parameter elements after SATIE sharing has been installed and were measured directly from the model objects:

| Configuration | Definition | Unique trainable parameters |
|---|---|---:|
| P2-A | Original SAGE + MASC + SATIE | 130,694 |
| P2-B | Independently initialized always-open context + MASC + SATIE; no gate module | 130,068 |
| P2-C | Differentiable SAGE V2; original gate inputs | 130,694 |
| P2-D | Differentiable SAGE V2; normalized final-observation bbox clipped to `[-5, 5]` for gate input | 130,694 |
| P2-E | Differentiable SAGE V2; context-only gate input | 130,630 |

P2-A/C/D share the same initialization digest for seed 20260928. P2-B's underlying motion model and context encoder are initialized with the same seed/order as the corresponding components in P2-C; P2-B intentionally has no gate parameters. The exact initialization digests and run configurations are recorded per run.

## Correctness evidence

Deterministic tests verify: finite attention near admission boundaries; normalized attention; no mass on unrelated masked positions; exact no-prior attention equivalence; nonzero low-probability gate/context gradients; mixed probabilities; P2-B/P2-C always-open equivalence; exact hard-closed motion fallback; SATIE identity sharing; and output shape `[batch, 45, 4]`.

The exact gradient measurements, including a saturated-gate zero-gradient case, are in `diagnostics/P2/sage_v2_gradient_diagnostics_v2.json` and summarized in `SAGE_V2_GRADIENT_REPORT.md`. These tests establish implementation pathways, not a prediction-quality improvement or the sole explanation for historical gate collapse.
