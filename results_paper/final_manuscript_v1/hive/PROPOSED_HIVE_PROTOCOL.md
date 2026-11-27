# HIVE-style proposed evaluation protocol (design only)

## Status
No human participants have been recruited, no responses collected, and no trial stimuli are currently cleared or ready. This document adapts the distinction/agreement logic of HIVE to pedestrian forecasts; it is not a completed user study or validation result.

## Scope and questions
The proposed study asks whether a human can distinguish the more accurate forecast among blinded candidate trajectories, and whether a neutral supplementary temporal explanation changes confidence or accuracy. It does not test road safety, model causality, or deployment suitability.

## Experimental tasks
- **Distinction:** show one observed history and ground-truth future only after response (if the approved design permits scoring disclosure), with 2-3 forecast candidates under randomized neutral IDs. Ask which candidate is closest to the future. Candidate order and IDs are counterbalanced.
- **Agreement/confidence:** show one candidate in isolation and collect confidence that it is the best forecast, with randomized model identity hidden.
- Primary endpoint: distinction accuracy on predeclared eligible trials. Secondary: confidence, response time, and comprehension. Model identities decoded only after data validation.

## Design requirements before stimuli exist
1. Recover saved AO official-test coordinate predictions without running inference, if such coordinates are subsequently located in an already-authorized artifact. Current canonical v2 only stores errors and IDs, so this requirement is blocked.
2. Freeze a stratified selection protocol across videos and tracks, independent of model advantage; include ordinary, low-error, and high-error strata for descriptive balance, with fixed seed and all selections logged.
3. Create side-by-side trajectory diagrams with identical axes, time direction, line styles, accessibility-safe colors, and no model names, MSE values, future-ground-truth reveal, or answer cues before response.
4. Keep each track/video leakage group in one condition; counterbalance candidate order and random IDs.
5. Obtain dataset-owner permission for all video imagery, institutional ethics review, informed consent, privacy review, participant data retention/deletion plan, and preregistration before recruitment.

## Proposed protocol
- Within-subject, randomized and counterbalanced conditions: forecast-only and forecast-plus-explanation, only if an explanation condition is separately validated and visually neutral.
- Training: brief practice using separate non-study examples; exclude practice trials.
- Trials: target 24-36 balanced examples with optional rest breaks; this is a design placeholder to be finalized in a preregistration/power plan, not an executed sample-size decision.
- Recruitment, exclusion rules, sample size and statistical model require supervisor/ethics approval before any participants are approached.

## Analysis plan (future only)
Prespecify trial-level mixed-effects logistic regression for distinction accuracy with participant and stimulus random effects; compare conditions with confidence intervals. Report confidence calibration and response-time distributions as secondary outcomes. Do not treat trials as independent participants. Do not claim human interpretability from attention visualizations alone.

## Current blocker
Historical E1 overlays concern another model/input family, one test video, and are not suitable substitutes for final AO comparison stimuli. They are excluded. No official coordinate prediction file was found in the canonical v2 bundle. The existing error CSVs cannot draw a forecast path. Therefore the package contains protocol documents and empty templates only, not fabricated or participant-ready stimulus images.
