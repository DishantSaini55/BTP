# Stimulus readiness and blinding specification

**Readiness: BLOCKED.** No official-test coordinate prediction arrays exist in the canonical saved bundle; only metrics and sample identifiers are available. Do not reconstruct or invent tracks from MSE. Do not use historical E1 overlays as AO stimuli.

## Required stimulus record
`stimulus_id, source_sample_id, video_id, track_id, stratum, candidate_A_model, candidate_B_model, candidate_C_model, candidate_order, random_seed, source_prediction_sha256, ground_truth_sha256, permission_status, anonymization_status, reviewer_status`

## Planned visual form
- Plot normalized image-plane coordinates only if dimensions and transform are documented; otherwise use absolute pixel axes with common bounds within each trial.
- Show observed trajectory in a neutral solid style and forecast candidates with accessible, distinct line/marker encodings.
- Hide model names, numerical errors, and future truth until after the choice is locked.
- Use deterministic randomization seeded and recorded; preserve a private answer key separate from the participant packet.

## Readiness gates
1. Existing saved AO prediction coordinates located and hash-verified; no inference rerun.
2. Balanced sample selection frozen before examining trajectory appearance or model ranking.
3. Dataset and image-use permissions documented; faces/plates handled under approved privacy protocol.
4. Independent visual QA verifies axes, IDs, line styles, answer-key separation, and no leaked metrics.
5. Ethics approval, consent, preregistration, and response storage plan complete.

No stimuli were rendered because the required prediction coordinates are absent. The response CSV is a blank schema only.
