# Existing manuscript claim audit

## Finding

`paper/` describes an earlier GP6 full-input PACE-Net proof-of-concept and
claims architecture-oriented improvements. Phase 6B is a later annotation-only
official-test study. The manuscript must not be silently updated or treated as
reporting Phase 6B.

| Current claim family | Audit | Required disposition |
|---|---|---|
| PACE-Net is the strongest predictor | Contradicted by Phase 6B: ridge MSE 2889.718 is lower than all neural means | Remove/reframe before submission |
| SATIE improves accuracy | Unsupported by official test; AO-3 is close to AO-2 and neither beats CV conclusively | Retain only as compactness result |
| SAGE/MASC/ROUTE provide adaptive-context gains | Earlier gate analyses found unreliable/no natural admission benefit | Remove causal/performance claim |
| Native scene labels solve unstructured context | AO protocol excluded scene labels because availability was not established | Remove for AO paper; label as earlier exploratory work |
| Literature-derived novelty of gates/routing/sharing | Overstated: conditional computation and transformer sharing are established | Reframe as implementation variants |

No `paper/` file was edited by Phase 6C.
