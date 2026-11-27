# SAGE gradient propagation audit

## Scope and reproducibility

This audit follows the SAGE prediction path in `MTN/context_TF.py`, the
attention mask in `MTN/transformer/functional.py`, the context-memory append in
`MTN/transformer/encoder_decoder.py`, and the E1 training loss in
`MTN/run_from_scratch_e1.py`. It makes no claim that gate collapse has one
single cause.

The deterministic mechanism probes are in
`MTN/sage_gradient_diagnostics.py`; measured results are preserved in
`results_paper/diagnostics/phase1/sage_gradient_measurements.json`. They use
seed 20260928, CPU PyTorch, one thread, a small deterministic three-sample
model (`d_model=8`, dropout 0), and actual prediction MSE gradients. To
reproduce:

```powershell
python -m MTN.sage_gradient_diagnostics --output results_paper\diagnostics\phase1\sage_gradient_measurements_repro.json
python -m pytest -q tests\test_sage_gradients.py
```

The gradients below are measured pathway diagnostics, not gradients recovered
from historical E1 checkpoints or evidence of an accuracy change.

## Result

| Forward decisions | Gate MLP gradient L2 | Gate params absent / zero / nonzero | Context encoder gradient L2 | Context params absent / zero / nonzero |
|---|---:|---:|---:|---:|
| All open `[1,1,1]` | 0.0094741145 | 0 / 0 / 4 | 0.8636352628 | 0 / 0 / 12 |
| All closed `[0,0,0]` | absent | 4 / 0 / 0 | absent | 12 / 0 / 0 |
| Mixed `[1,0,1]` (batch aggregate) | 0.0060631965 | 0 / 0 / 4 | 0.6736701989 | 0 / 0 / 12 |
| Soft admission `[0.468991,0.428161,0.457377]` | 0.0021512256 | 0 / 0 / 4 | 0.3531348171 | 0 / 0 / 12 |

For the mixed case, the loss from the closed middle sample alone produced **0
nonzero gradients**, **0 absent gradients**, and **16 present-but-exactly-zero
parameter gradients** across the 4 gate and 12 context-encoder parameters.
The full mixed-batch loss has nonzero gradients because its two open samples
contribute.

All predictions and present gradients were finite. The all-closed inference
path was bitwise equal to the base motion-only model. SATIE-sharing evidence
verified all encoder self-attention Q/K/V/O parameters share by object and
data pointer, with no duplicate named parameter objects; matching unshared
weights and shared execution produced bitwise-equal predictions (max absolute
difference 0).

## Why all-closed prediction loss disconnects SAGE

`ContextIndividualTF.forward` first computes a context token and gate, but
when the entire batch has `gate == 0`, `if not bool(gate.any())` returns
`self.base(...)` directly. The returned prediction has no computational
dependency on either `self.gate` or `self.context_encoder`. Backpropagation
therefore leaves their `.grad` fields as `None` (absent), not as allocated
zero tensors. The surrounding E1 trainer optimizes prediction MSE only and
does not add `context_cost_loss()`, so that prediction loss cannot revive the
gate or context encoder from an all-closed batch.

When a batch mixes open and closed decisions, it takes the context-aware path.
The open examples provide SAGE gradients; a closed example in that mixed batch
has a graph reaching the SAGE parameters, but its per-example gradients are
exactly zero. For a batch with even one open example, the closed-sample
context-token attention key is masked, so the closed sample itself still
does not update the gate or context encoder through prediction loss.

This is a verified possible self-reinforcing failure mode after an all-closed
batch. It does **not** establish that the mechanism alone caused any observed
gate behavior or that a different loss would improve results.

## Hard attention mask and numerical context scaling

`attention()` computes scores and applies
`scores.masked_fill_(mask == 0, value=-1e9)` before softmax. In the SAGE path,
the context token is appended to memory and its mask entry is the per-sample
gate. The comparison `mask == 0` is boolean and non-differentiable:

- The mask decision itself has no gradient path to gate logits.
- A masked context score receives no gradient through `masked_fill`; its
  post-softmax attention is effectively zero because the trajectory keys
  remain unmasked.
- The context token is also numerically scaled as `context_token * gate`.
  This multiplication is differentiable. For an open hard decision, its
  straight-through gate value preserves a surrogate gradient through the
  scaled token and back into the gate MLP and context encoder.
- For a closed decision, the key is masked, so the prediction does not depend
  on the scaled context token; the scaling route contributes zero gradient.
  The straight-through derivative cannot recover a path that the attention
  mask has removed.
- With soft admission, any strictly positive gate is considered open by
  `mask == 0`, while the token remains scaled by a continuous value. The
  prediction can then differentiate through the numerical scale. Measured
  soft-case gate and context gradient norms are nonzero.

## Test coverage

`tests/test_sage_gradients.py` covers all-open, all-closed, mixed, continuous
soft admission, gate/context gradient norms, finite outputs/gradients, exact
motion-only fallback, and SATIE parameter-sharing invariance. Tests explicitly
distinguish absent gradients (`grad is None`) from allocated zero gradients.

## Limitations

These are deterministic small-model mechanism tests. They do not estimate
gradient distributions across the E1 seeds, quantify collapse probability,
validate test-set performance, or establish that a soft gate or auxiliary
regularizer is preferable. Keep those as separate, pre-specified Phase 2
questions rather than inferring a remedy from this audit.
