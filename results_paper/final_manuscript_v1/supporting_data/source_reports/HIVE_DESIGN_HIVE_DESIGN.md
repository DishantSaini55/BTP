# HIVE-style study design for trajectory forecasts

**Status: design only.** No participants were recruited, no survey was deployed, and no participant responses were generated, simulated, or analyzed. Do not launch without ethics review/approval and the data-owner permissions required for the video stimuli.

## Framework and adaptation

Read the [HIVE repository](https://github.com/brandonyzhao/HIVE) and Kim et al., [*HIVE: Evaluating the Human Interpretability of Visual Explanations*](https://arxiv.org/abs/2112.03184) (ECCV 2022). HIVE uses falsifiable behavioral tasks, notably **distinction** (select the correct prediction from several prediction/explanation options) and **agreement** (rate confidence in one prediction/explanation pair), with subjective understanding questions as secondary measures. The paper emphasizes neutral presentation, digestible explanations, and avoiding prior-knowledge shortcuts.

This design adapts those tasks to forecast selection and decoder temporal-attention displays. It evaluates whether the saved attention display helps a person choose among forecasts; it does **not** test whether attention is causal, faithful, or a semantic explanation.

## Evidence and stimulus boundary

- The stimulus bank is the 12 frozen seed-20260928 windows in [SELECTION.md](./SELECTION.md) and [OVERLAYS_seed20260928.csv](./OVERLAYS_seed20260928.csv): six RNG-selected random windows from six pedestrians, two largest improvements, two largest degradations, and two closed-gate cases. The last six are diagnostic cases, not representative sampling.
- The windows come from one held-out GP6 video. All 12 have the same recorded scene tuple, and the saved PACE-Net and PACE-Net-without-SATIE gates are `g=0`. There are **no natural `g=1` test examples**.
- Use the saved MTN, PACE-Net without SATIE, PACE-Net, and constant-velocity forecasts from the step-2 outputs. The primary matched-attention task uses the three learned models; retain constant velocity as an external benchmark in the saved analysis materials, not as a participant choice, because it has no learned attention map.
- Use the step-3 `ATTENTION_SELECTED_seed20260928.csv` values for the same selected IDs. In the participant UI, the simple attention strip may be the mean over the 3 decoder layers and 4 heads at each of the 15 trajectory-memory steps. The saved per-layer/per-head values remain available for an optional detail view.
- **Do not use the forced mixed-gate rows** in `GATE_MASK_PROBE_seed20260928.csv` as participant stimuli. They are implementation diagnostics, not predictions made by the learned gate. Never show a context-token weight for these windows: natural learned-gate outputs have no admitted context token.
- The existing overlay PDF contains ground truth, model names, MSE, and gate labels. It is **not** participant-ready. Derive a neutral version from the saved frames/predictions: show the last observed frame, the 15 observed boxes, and candidate future trajectories only. Hide future ground truth, per-window metrics, selection group, model name, and answer key until all responses are submitted. Randomize candidate letters, line colors, and display order by trial.

The six random windows are the only primary stimulus stratum. The two improvement, two degradation, and two closed-gate cases can be included as clearly pre-specified diagnostic trials, but must be analyzed separately. With only six primary windows from one video, this bank is suitable for a **feasibility pilot only**, not a confirmatory or cross-video claim. A confirmatory study needs a pre-frozen larger bank spanning independent pedestrians and multiple held-out videos.

## Study tasks

### A. Distinction task (primary behavioral task)

Randomly assign participants to **attention shown** or **attention hidden**; all other UI elements are identical. Each trial presents one scene and three unlabeled candidate forecasts from MTN, PACE-Net without SATIE, and PACE-Net. Ask:

> Which forecast do you think is closest to where the pedestrian actually travels next? Choose A, B, or C, or “cannot tell.”

In the attention-shown arm, attach the same-format temporal-attention strip to each learned-model candidate. Explain neutrally: “The strip summarizes how the model's decoder weighted the 15 observed trajectory steps; it is a model-internal attention weight, not proof of causal importance.” Show `context not admitted (g=0)` for both PACE-Net candidates in **both** arms, so gate status is not the randomized treatment.

The correct candidate set is computed after collection from the saved per-window raw-coordinate MSEs. Any candidate within a pre-registered `1e-6` of the minimum is considered tied and correct; “cannot tell” is incorrect for the primary score and reported separately. Chance is item-specific (`number_of_tied_winners / 3`). Do not reveal the ground truth or answer between trials.

### B. Agreement task (secondary; separate randomized block)

For one PACE-Net forecast at a time, randomly show or hide its temporal-attention strip. Balance each sample across shown/hidden assignments between participants; do not show the same sample twice to one participant. Ask:

1. “How confident are you that this forecast will be close to the pedestrian's actual path?” (1 = not at all confident, 5 = very confident.)
2. “Would you use this forecast as one input to a low-stakes planning decision?” (Yes / No / Unsure.)

This is a calibration/trust measure, not a correctness judgment. Compare confidence and acceptance with the continuous saved per-window forecast error after the study; report possible over-trust even for high-error forecasts.

### C. Explanation comprehension and self-report (secondary)

After a small number of trials, ask participants to identify which observed-time segment received the greatest average attention (early / middle / late / approximately even / unsure). Score against the saved attention strip, while making clear that this checks reading of the display, not causal understanding. Ask for self-rated understanding of the attention display (1–5) before the task, after the task, and after showing an end-of-study explanation of what attention does and does not establish. One optional free-text item may ask what information was useful or confusing.

## Participant-facing survey text

### Introduction and consent

> This study asks how people interpret pedestrian trajectory forecasts. You will see a road scene, a short observed pedestrian path, and one or more possible future paths. Some versions also show a summary of which observed time steps a model attended to. These weights describe the model's internal computation; they do not prove that the highlighted steps caused its prediction. The future path will not be shown until you finish the forecast questions. Participation is voluntary, and you may stop at any time.

Collect explicit informed consent before continuing. Record no names, contact details, exact location, or unnecessary demographic information. Optional machine-learning familiarity may be collected using broad categories only.

### Instructions

> The colored outline marks the observed pedestrian. The forecast lines are alternatives produced by different systems. The letters and colors do not indicate quality. Choose the path you expect to be closest to what happens next. Some cases may be difficult; “cannot tell” is available. You will not receive correctness feedback until all trials are complete.

### Per-trial questions

- Distinction: candidate A / candidate B / candidate C / cannot tell.
- Confidence in the choice: 1–5.
- Optional short reason: trajectory continuity / scene context / attention display / other / unsure.
- Agreement block: confidence 1–5 and use / do not use / unsure as above.
- Comprehension: early / middle / late / approximately even / unsure.

### Closing

Ask whether any interface element was unclear and the participant's overall self-rated understanding (1–5). Only after all choices are locked may the UI reveal the ground-truth path and explain that attention weights are not causal explanations. Do not give trial-by-trial feedback.

## Randomization, blinding, and safeguards

- Between-participant randomization to attention shown/hidden avoids showing the same forecast twice to one person. Counterbalance trial order and candidate letter/color assignments across participants.
- Do not display model names, test-set MSE, best/worst group, scene-label codes, selection rationale, or ground truth before responses are locked.
- Use the six random cases for the pre-specified primary endpoint; keep the six outcome-selected diagnostic cases separately labeled in the analysis data and report them as exploratory only.
- Anonymize/blur identifiable faces and license plates before creating participant materials; verify the dataset license and video-sharing permissions. Provide accessibility alternatives for color-only distinctions.
- This is not a safety evaluation or a deployment recommendation. Because the bank is from one video and the model's gate is closed on every selected test case, do not generalize to other traffic scenes or claim a human evaluation of context-token explanations.
- Obtain appropriate institutional ethics review, informed consent, a data-retention plan, and a pre-registration before recruitment. No HIVE AWS/AMT deployment or account operation is authorized by this design.

## Analysis-script specification (future responses only)

No response file exists and no analysis was run. A later analysis script should accept a UTF-8 response CSV with:

`participant_id,condition,trial_id,sample_id,selection_stratum,candidate_order,chosen_candidate,confidence,use_decision,comprehension_answer,response_time_ms`

It should join `sample_id` and the randomized candidate map to the frozen overlay metrics, validate IDs and answer ranges, and preserve the original response rows unchanged. Pseudocode:

```python
for response in responses:
    metrics = overlay_metrics[response.sample_id]
    candidates = ("MTN", "PACE-Net without SATIE", "PACE-Net")
    mse = {name: metrics[name + "_mse"] for name in candidates}
    minimum = min(mse.values())
    correct_set = {name for name, value in mse.items()
                   if value <= minimum + 1e-6}
    chosen_model = reverse_candidate_map(
        response.sample_id, response.candidate_order, response.chosen_candidate
    )
    correct = chosen_model in correct_set
    save_derived_row(response, mse=mse, correct_set=correct_set, correct=correct)

report participant-level distinction accuracy by condition, with item-level
accuracy, item-specific chance, confidence, opt-out, and comprehension shown
separately. Estimate the randomized attention-shown versus attention-hidden
contrast with a pre-registered participant/stimulus-aware model or two-way
cluster bootstrap (participant and pedestrian); show all individual stimulus
outcomes and uncertainty. Analyze improvement/degradation/closed-gate cases
only as exploratory strata. For agreement, plot confidence and acceptance
against continuous per-window error and report calibration/over-trust
descriptively. Do not treat repeated trials as independent participants,
do not tune thresholds after seeing responses, and state the one-video caveat
with every interval.
```

Do not run this analysis until real, consented responses exist. Never fabricate, simulate, or backfill participant records.

## References and saved inputs

- Kim et al., ECCV 2022: [arXiv:2112.03184](https://arxiv.org/abs/2112.03184).
- HIVE implementation: [brandonyzhao/HIVE](https://github.com/brandonyzhao/HIVE) (fork of Princeton Visual AI's HIVE).
- Frozen windows and overlays: [SELECTION.md](./SELECTION.md), [OVERLAYS_seed20260928.pdf](./OVERLAYS_seed20260928.pdf), [OVERLAYS_seed20260928.csv](./OVERLAYS_seed20260928.csv).
- Saved attention and natural-gate limitations: [ATTENTION_REPORT_seed20260928.md](./ATTENTION_REPORT_seed20260928.md), [ATTENTION_SELECTED_seed20260928.csv](./ATTENTION_SELECTED_seed20260928.csv), and [MASKING_BY_SCENE_seed20260928.csv](./MASKING_BY_SCENE_seed20260928.csv).

No participant materials were rendered or deployed, no participant responses were generated, and no files under `paper/` were changed.
