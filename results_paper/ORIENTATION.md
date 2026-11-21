# Orientation audit - PACE-Net

**Status: complete; read this audit before starting any training.**  This report records what is implemented, what the existing artifacts establish, and what remains missing for the prescribed experiment plan. It does not treat prior POC results as new results.

Audit date: 2026-11-21.  Repository commit observed: `2572a86567f9d7ef6b6d690ca83f06579288c0b9`.

## 1. Entry points and current reproducibility coverage

| Paper model | Construction / training path | Evaluation path | Seed support today |
|---|---|---|---|
| MTN (baseline) | `MTN/train_iddped.py:58-67` with no `--context`; old CPU POC `MTN/cpu_poc_experiment.py:37-59` with `context=False` | `MTN/test_iddped.py:52-128`; POC `cpu_poc_experiment.py:60-69` | Neither generic script accepts a seed CLI argument. The POC hard-codes `20260928` (`cpu_poc_experiment.py:19,23`). No matched 32/8/24, ten-seed baseline runner exists. |
| PACE-Net without SATIE (SAGE + MASC) | `MTN/cpu_targeted_weight_sharing.py:new("none")` at lines 25-35; unified runner model `B` at `MTN/final_four_way_runner.py:21-30` | Custom `evaluate` in `cpu_targeted_weight_sharing.py:52-58` | Unified runner uses only hard-coded seeds 20260928-30 (`final_four_way_runner.py:10`); targeted runner hard-codes one seed (`cpu_targeted_weight_sharing.py:15,17`). |
| PACE-Net (SAGE + MASC + SATIE) | `new("encoder_self_attn")` at `cpu_targeted_weight_sharing.py:25-35`; `make_c4_encoder_self_attention_shared` at `MTN/c4_encoder_shared.py:11-19`; unified runner model `C4` | Same custom evaluator | Same three-seed limitation. |
| PACE-Net with ROUTE | `MTN/dynamic_fusion_TF.py:56-72`; unified runner model `F` at `final_four_way_runner.py:12-24` | Same custom evaluator | Same three-seed limitation. |
| PACE-Net with ROUTE and SATIE | ROUTE construction then SATIE aliasing in `final_four_way_runner.py:12-19`; unified runner model `F_C4` | Same custom evaluator | Same three-seed limitation. |

The existing closest runner is `python MTN/final_four_way_runner.py`, but it is **not** an E1 entry point: it excludes MTN, has four models rather than five, hard-codes three seeds, writes outside the repository to `D:\IDD-PeD-Evaluation\outputs\final_validation`, lacks deterministic CUDA controls, and does not save predictions, gate decisions, or attention weights. It must not be used as-is for the requested ten-seed experiment.

The generic official-style scripts also are not sufficient for the fixed protocol: `train_iddped.py:105-112` shuffles data, uses AdamW plus schedulers, and defaults to 100 epochs; `test_iddped.py:122-129` reports only MSE and C-MSE then exits.

## 2. Metrics and comparison with official IDD-PeD evaluation

### Implemented definitions

The custom POC evaluator at `MTN/cpu_targeted_weight_sharing.py:52-58` denormalizes boxes and computes, per sample,

\[
\mathrm{MSE}=\operatorname{mean}_{t\in[45],c\in[4]}(Y_{t,c}-\hat Y_{t,c})^2,
\]

\[
\mathrm{C\text{-}MSE}=\operatorname{mean}_{t\in[45],c\in\{x,y\}}(\operatorname{centre}(Y)_{t,c}-\operatorname{centre}(\hat Y)_{t,c})^2,
\]

and

\[
\mathrm{CF\text{-}MSE}=\operatorname{mean}_{c\in\{x,y\}}(\operatorname{centre}(Y)_{45,c}-\operatorname{centre}(\hat Y)_{45,c})^2.
\]

Dataset-level values are the arithmetic mean of these per-sample values (`cpu_targeted_weight_sharing.py:58`), which is equivalent to an elementwise mean because all samples have 45 future steps. The generic evaluator uses the same denormalization and MSE/C-MSE arrays at `MTN/test_iddped.py:113-128`.

The manuscript equations in `paper/sections/experimental_results.tex:7-13` agree with these definitions.

### Official comparison

The official IDD-PeD MTN evaluator uses the same 15 observed / 45 future box protocol, denormalizes boxes, defines centres as `(x1+x2)/2,(y1+y2)/2`, then averages squared box errors and squared centre errors ([official `test_iddped.py`, lines 12-18 and 88-101](https://github.com/Ruthvik9/IDD-PeD/blob/main/MTN/test_iddped.py)). The official paper calls all three errors squared-pixel metrics and reports MSE, C-MSE and CF-MSE ([IDD-PeD paper, Evaluation Metrics](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/IDD-PeD.pdf)).

**Mismatch / limitation:** the official public `test_iddped.py` prints MSE and C-MSE only; it contains no CF-MSE calculation or print. CF-MSE is therefore a compatible explicit extension in the local custom evaluator, rather than a byte-for-byte output of that official script. No discrepancy was found in the MSE or C-MSE formula. The official paper does not publish a more detailed algebraic CF-MSE formula than its name.

## 3. What reaches SAGE

SAGE forms its logits with `torch.cat((context_token, observed_boxes[:, -1]), dim=-1)` (`MTN/context_TF.py:86-93`). It is a 36→16→2 MLP: 32-dimensional context plus four box coordinates, then `Linear(36,16)`, GELU, `Linear(16,2)` (`context_TF.py:74-81`). This is 626 parameters.

The box input is **normalised, not raw pixels**. `baselineUtils.get_tracks` normalises every box as `(box - mean) / std` when `normalize_bbox=True` (`MTN/baselineUtils.py:154-161`); the standard experiment options set it true (`MTN/cpu_targeted_weight_sharing.py:15`). The unnormalised coordinates are retained separately only for visualisation (`baselineUtils.py:234-240`).

For the exact current 32/8/24 cached split, measured from the cached arrays using the training mean and standard deviation, the final observed box passed to the gate has these ranges:

| Split | Samples | Overall minimum / maximum | Per-coordinate minimum | Per-coordinate maximum |
|---|---:|---:|---|---|
| train | 32 | −1.88885 / 2.30023 | [−1.88885, −1.21614, −1.86404, −1.71651] | [1.96841, 2.30023, 2.03894, 1.79893] |
| validation | 8 | −18.16263 / 5.35232 | [−2.84587, −18.16263, −2.60790, −2.73094] | [4.83545, −9.26669, 5.35232, 3.04305] |
| test | 24 | −50.55189 / 8.60318 | [−0.49768, −50.55189, −0.34594, −30.39507] | [8.01961, −41.41455, 8.60318, −21.06417] |

The very different vertical coordinate scales across these three local videos explain the large test z-scores; they are not raw-pixel inputs. Accordingly, E8's stated precondition (“if x_T enters raw”) is false for the current default model. This should be recorded and E8 should not be run unless the author explicitly broadens it to a robustness study.

MASC is implemented as a sixteenth source-memory item: context is appended to the 15-token encoder memory and the source mask is concatenated with `g` (`MTN/transformer/encoder_decoder.py:32-37`, `MTN/context_TF.py:107-114`). A rejected hard gate supplies a zero token and zero source-mask entry.

## 4. Exact local matched split

The matched 32/8/24 split is a local CPU/Farneback POC, not the official 70/30 IDD-PeD split. It is defined by `MTN/cpu_poc_prepare.py:146-170` and `MTN/prepare_c4_larger_test.py:15-35`:

- train: 32 deterministically stratified 60-frame trajectories from `gp_set_0006_vid_0002`;
- validation: 8 deterministically stratified trajectories from `gp_set_0006_vid_0001`, restricted to windows ending before frame 4000 because of a decode corruption;
- test: **all 24 eligible** 60-frame trajectories in source order from `gp_set_0006_vid_0003`.

Thus the test set has **one distinct video**, not 24 videos. The exact manifests are external source artifacts:

- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\train_manifest.json`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\validation_manifest.json`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_c4_larger\test_full_manifest.json`

Track identities (manifest order) are:

- train source indices: 81, 109, 95, 93, 90, 82, 111, 99, 83, 112, 100, 84, 113, 106, 85, 114, 86, 87, 88, 89, 91, 92, 94, 96, 97, 98, 101, 102, 103, 104, 105, 107; PIDs: gp_4223, gp_4288, gp_4260, gp_4253, gp_4241, gp_4228, gp_4291, gp_4269, gp_4229, gp_4292, gp_4270, gp_4231, gp_4294, gp_4282, gp_4232, gp_4296, gp_4235, gp_4236, gp_4239, gp_4240, gp_4248, gp_4249, gp_4257, gp_4261, gp_4264, gp_4266, gp_4271, gp_4274, gp_4275, gp_4276, gp_4280, gp_4284.
- validation source indices: 13, 0, 16, 17, 12, 11, 20, 27; PIDs: gp_1739, gp_1646, gp_1776, gp_1777, gp_1714, gp_1710, gp_1789, gp_1840.
- test source indices: 124–147 inclusive; PIDs: gp_6933, gp_6934, gp_6935, gp_6948, gp_6949, gp_6950, gp_6951, gp_6952, gp_6953, gp_6954, gp_6955, gp_6956, gp_6957, gp_6958, gp_6959, gp_6960, gp_6961, gp_6962, gp_6963, gp_6964, gp_6965, gp_6966, gp_6967, gp_6968.

The exact selection rule is stored in the saved C4 result configuration at `D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted\C4_larger_validation_results.json` and is “all 24 eligible video-3 trajectories in source order; no outcome-based selection.”

## 5. Initialisation from the official MTN checkpoint

The relevant checkpoint is `D:\IDD-PeD-Evaluation\checkpoints\mtn\mtn\epoch_latest.pth` (`MTN/cpu_targeted_weight_sharing.py:13-15`). For PACE-Net without SATIE and PACE-Net, `new()` creates a context model, strips `base.` from each model state key, and loads every same-name, same-shape official tensor (`cpu_targeted_weight_sharing.py:25-30`). The saved runs report 242 official tensors loaded. The 16 context-branch tensors—scene embeddings, projection and gate—have no official counterparts and retain the constructor’s random initialisation.

For SATIE, all independent layer weights are loaded first, then layer 1 (`layers[0]`) `self_attn` is aliased into layers 2 and 3 (`MTN/c4_encoder_shared.py:11-19`). This is consistent with the requested layer-1 initialisation; the no-longer-referenced later self-attention tensors are not optimised. The verifier confirms common module identity, parameter object identity, `data_ptr()`, and no duplicate named parameters (`c4_encoder_shared.py:22-50`). The instantiation audit reproduced all required counts: 133,700 / 139,142 / 130,694 / 154,007 / 145,559.

ROUTE uses the same `base.` key-matching process in `MTN/final_four_way_runner.py:12-19`; its fusion/context modules remain random. **MTN itself is not currently constructed by that runner.** Its earlier CPU POC does load the official checkpoint exactly with `IndividualTF.load_state_dict(ck)` (`MTN/cpu_poc_experiment.py:37-42`), but does not meet the matched one-epoch/ten-seed protocol. The E1 implementation must initialise MTN by the same official checkpoint, use the same shuffle/order policy as every other E1 model, and record the exact loaded-key list rather than only a count.

## 6. Earlier targeted-sharing study

The previously unverified Table IV values are now traceable, but they remain **not a replacement for E4**. The source is:

`D:\IDD-PeD-Evaluation\outputs\weight_sharing_targeted\targeted_weight_sharing_results.json`

with implementation in `MTN/cpu_targeted_weight_sharing.py:68-77` and checkpoints in the same directory (`B_no_sharing.pth`, `decoder_src_attn.pth`, `decoder_src_qkv.pth`, `decoder_src_kv.pth`, `encoder_self_attn.pth`). That JSON records one seed (20260928), 32/8/**10** trajectories, one epoch, cached Farneback flow, Adam 1e-3, batch 8 and four targeted variants. It gives exactly:

| Variant | Parameters | MSE | CF-MSE |
|---|---:|---:|---:|
| No sharing | 139,142 | 972.5131 | 2802.6661 |
| Decoder source attention Q,K,V,O | 130,694 | 1081.4412 | 3369.5764 |
| Decoder source attention Q,K,V | 132,806 | 1188.9212 | 3940.0977 |
| Decoder source attention K,V | 134,918 | 1152.4858 | 3807.0238 |
| Encoder self-attention Q,K,V,O | 130,694 | 819.9889 | 2441.1601 |

Thus the manuscript's 972.51 and 819.99 values were transcribed accurately, but they came from a one-seed 10-test-trajectory study and are not statistically adequate. The full encoder+decoder result comes from the separate 12-epoch `D:\IDD-PeD-Evaluation\outputs\weight_sharing_results.json`, with code in `MTN/cpu_weight_sharing_experiment.py`; it is likewise not E4.

## 7. Existing timing values and why they disagree

They are measurements of different checkpoints, test-set sizes, warm-up regimes, repetitions and—in one case—training budgets. They must not be pooled.

| Artifact | Models / data | Timing protocol | Why it differs |
|---|---|---|---|
| `outputs/weight_sharing_results.json`; `MTN/weight_sharing_report.md:76-85` | Old B vs fully shared C, 10 test trajectories, 12 epochs | 5 warm-up passes, 200 forwards; B 47.20 ms, C 50.13 ms | Full layer sharing and different training budget. |
| `outputs/weight_sharing_targeted/targeted_weight_sharing_results.json`; `cpu_targeted_weight_sharing.py:59-67` | One seed, B and targeted variants, 10 tests | 5 warm-up passes, 200 single-sample forwards | Targeted variants and a different retained checkpoint; B mean is 47.79 ms. |
| `outputs/weight_sharing_targeted/C4_larger_validation_results.json`; `cpu_c4_larger_validation.py:30-39` | B/C4, three seeds, 24 tests | 2 warm-up traversals, then 5 complete test traversals = 120 forwards per model/seed | B/C4 32/8/24 evaluation; resulting aggregate latency differs from the earlier 10-test study. |
| `outputs/final_validation/final_four_way_results.json`; `final_four_way_runner.py:43` | B/C4/F/F_C4, three seeds, 24 tests | same 2-warm-up/120-forward helper | Means are B 45.58, C4 47.38, F 50.68, F_C4 52.07 ms; no IQR and not the requested E7 benchmark. |
| `ARCHITECTURE_COMPARISON.md` | synthetic architecture comparison | different model sizes and synthetic inputs | It is not labelled-data benchmark timing and must not be cited as E7. |

The supplied paper correctly says its shared-CPU timing was too noisy to support a conclusion (`paper/sections/experimental_results.tex:61-62`). E7 must supersede these with the specified 50 warm-ups, at least 500 iterations, fixed thread count, median and IQR.

## 8. Saved predictions, gates, attention, and visual assets

- The current 24-test result JSONs retain per-sample error summaries and source metadata for the B/C4 and F/F_C4 runs, but **not predictions, targets, gate outputs, gate logits, or attention tensors**. `final_four_way_runner.py:43` receives `rows` but does not serialise them at all.
- `ContextIndividualTF` does expose the latest gate in memory as `last_context_gate` (`context_TF.py:83-96`), but it is overwritten each forward and is not persisted. It does not retain logits.
- `MultiHeadAttention` exposes only the most recent attention matrix as `self.attn` (`MTN/transformer/multihead_attention.py:19-36`). Each decoder layer has `src_attn` (`MTN/transformer/decoder_layer.py:14-35`), so E3/Q2 can be implemented by collecting copies after each batch, but no history is saved now.
- The old 10-test CPU POC stores predictions and targets inside `D:\IDD-PeD-Evaluation\outputs\results.json` (`MTN/cpu_poc_experiment.py:60-74`) and produced a PDF. This is not compatible with the 24-test E1 experiment and contains no attention weights or decisions.
- Raw cached selected tracks and flow are under `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\` and `...\cpu_farneback_c4_larger\`. The raw annotation/image paths are rooted at `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian`; manifests record the frame path for each selected sample. `cpu_poc_prepare.py:36-40` maps those paths to videos under `D:\IDD-PeD-Evaluation\videos\gopro\...`. XML annotations are parsed from `annotations\IDDPedestrian\annotations\gopro\<set>\<video>.xml` (`iddped_interface_traj.py:684-732`).

## 9. Baselines available now

Only MTN/PACE-Net code is present in this repository. The local tree does **not** contain PIEtraj/PIEPredict, BiTraP, or SGNet implementations, adapters, or matched-split result files.

The official IDD-PeD repository documents public run paths for PIEPredict, MTN, BiTraP and SGNet ([official README](https://github.com/Ruthvik9/IDD-PeD#trajectory-prediction)); the source paper reports the published IDD-PeD values and specifies best-of-20 for stochastic methods ([Table III and metrics](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/IDD-PeD.pdf)). None has been demonstrated to run on this local 32/8/24 cache, and their native data/flow contracts should be checked before a broader-comparison run. No B2 result may be claimed yet.

## 10. Existing three-seed four-model evidence

The four paper means are verified against saved machine-readable data, not newly recalculated:

- PACE-Net without SATIE: 1137.1083 MSE from `C4_larger_validation_results.json`.
- PACE-Net: 1080.1606 MSE from the same file.
- PACE-Net with ROUTE: 1154.7783 MSE from `fusion_validation_results.json`.
- PACE-Net with ROUTE and SATIE: 1126.6693 MSE from the same file.

These match `paper/sections/tab_unified.tex:5-8` after rounding. They are the three seeds 20260928, 20260929 and 20260930, and were run on the same 32/8/24 cache. MTN has no corresponding saved matched three-seed result. No claim about reproducing ten-seed means is possible yet.

## Mandatory implementation work before E1

1. Create a new repository-local runner that covers **all five** prescribed models, all ten seeds, deterministic Python/NumPy/Torch/CUDA settings, the fixed split and a common loading/init routine.
2. Make `results_paper/` the sole output location, refuse to overwrite, and emit the requested config, checkpoint, metrics, sample IDs/hash, predictions, gates/logits and attention arrays per run.
3. Add a baseline MTN path, preserve exact matching checkpoint loading across all five models, and smoke-test every model before any long run.
4. Add a unified evaluator that explicitly saves MSE, C-MSE and CF-MSE and validates g=0 against the motion-only path. Do not use validation/test data for model selection.
5. Do not begin training until the author confirms this audit has been read and resolves any change to the fixed matched protocol.
