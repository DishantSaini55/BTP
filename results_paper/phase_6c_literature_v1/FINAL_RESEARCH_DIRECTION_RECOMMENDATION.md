# Recommended paper direction

## Recommendation: B, comparative empirical paper

Use **“Physics-Anchored Pedestrian Trajectory Prediction in Unstructured
Traffic: A Comparative Study of Classical and Transformer-Based Models.”**

It aligns with the official test: frozen ridge is best (2889.718 MSE; 27.707%
better than CV), AO-2/AO-3 have marginal mean improvements over CV with
seed-level intervals spanning zero, and AO-1 establishes the value of the CV
anchor relative to the constant-position residual formulation. This is a
valuable result when stated as protocol-specific evidence.

Do not choose A as the primary framing: it implies an architecture win that the
official test rejects. C is intellectually honest but less specific and risks
overgeneralizing from one benchmark. B can include the compact transformer and
negative gate findings as secondary analyses, while centring the classical
baseline lesson.

Submission needs: a manuscript rewrite outside this phase, explicit unmatched
published-baseline table, released evaluator/protocol, and careful limitation
language. It does not need retrospective neural model selection.
