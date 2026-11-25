# CRAFT-inspired trajectory concept analysis

## Decision and scope

Proceeding with a **bounded exploratory adaptation**, not claiming a faithful reproduction of CRAFT. The CRAFT paper and repository describe decomposing image-model activations into concepts, estimating concept importance with Sobol indices, and producing concept attribution maps for classifier decisions. PACE-Net is a trajectory regressor, not an image classifier: it has no class logit or image-region concept map to explain directly. The adaptation therefore factors non-negative trajectory-model activations and probes the final predicted center displacement by intervening on those factors. It does not implement CRAFT's recursive image-concept discovery or implicit-differentiation attribution maps.

Sources read:

- [CRAFT GitHub repository](https://github.com/deel-ai/craft) and its PyTorch API/README.
- [Fel et al., CRAFT, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fel_CRAFT_Concept_Recursive_Activation_FacTorization_for_Explainability_CVPR_2023_paper.html).
- [arXiv:2211.10154](https://arxiv.org/abs/2211.10154).

## Protocol

- Used the saved seed-20260928 PACE-Net checkpoint, the 212 training windows and 43 validation windows in the frozen split, with no weight updates. The split cache is one pickle containing all roles, so it was loaded as a whole; only train and validation datasets were iterated. **No test sample, prediction, metric, or label was accessed or used.**
- Audited the post-ReLU FFN activations in all three encoder box streams, encoder speed streams, and decoder layers. For concepts aligned with the 15 observed time steps, fit NMF on the final encoder box-stream FFN activations (`212 x 15 x 128`). Separately factorized the context projection's post-GELU hidden layer after clamping negative values (`212 x 32`).
- For each representation, fit NMF bases on training activations only. Chose rank from validation reconstruction error among ranks 2–16: the smallest rank within 1% of the best validation relative Frobenius error. Validation was used only for this unsupervised rank choice; it was not used to fit the bases.
- Generated top-activation training-window examples, per-window/per-step NMF scores, scene/gate associations, and functional probes. The fixed Sobol reference window is the lexicographically first training sample ID, not an outcome-selected case.

## Findings

### Activation signs and clamping

All audited encoder and decoder FFN activation tensors were finite and non-negative, as expected from the ReLU FFNs. Across those audited tensors, negative-entry fraction and negative absolute-mass fraction were both zero. The context post-GELU layer was not non-negative: 61.03% of train entries and 59.59% of validation entries were negative. Clamping removed 13.81% of train absolute activation mass and 13.58% of validation absolute activation mass.

### Factorization and rank

| Representation | Validation-selected rank | Validation relative Frobenius error |
|---|---:|---:|
| Final encoder box-stream FFN | 15 | 0.3397 |
| Context post-GELU, clamped | 14 | 0.1334 |

These are high ranks for a small data set and do not yield a compact decomposition. The encoder FFN's validation reconstruction error remains 0.3397 at rank 15; the context representation's is 0.1334 at rank 14. A candidate rank-5 context fit did not converge within 1,500 iterations; the selected rank-14 fit did converge. The initial rank-2–8 sweep and its results are preserved separately as preliminary; it saturated at the rank ceiling, so the search range was expanded before selecting the reported ranks.

### Examples, scenes, and gate

[CRAFT_NMF_EXAMPLES_seed20260928.png](./craft_seed20260928_rank2_16/CRAFT_NMF_EXAMPLES_seed20260928.png) shows one top-activation training frame/window per selected encoder concept, with observed trajectory, ground truth, and the saved PACE-Net prediction. These examples show different pedestrian paths and image locations in the same GP6 training video, but **do not reveal a clear, stable semantic concept vocabulary**. They are illustrative maxima, not evidence that a human observer would assign these factor labels consistently.

The saved checkpoint admitted context on 49/212 training windows and 17/43 validation windows. Scene/gate tables are descriptive only: for example, all 212 training windows have `road_type=main` and `motion_direction=OW`; 208/212 have `intersection_type=NI`, and all observed occlusion labels are zero. The sparse, imbalanced labels and overlapping stride-30 windows from 43 pedestrians do not support claims that a factor represents a scene category or causes gate admission.

### Functional probes

The fixed local reference window is `gp_set_0006/gp_set_0006_vid_0002/gp_4223/00362/00421`, with learned gate `g=0`. Its natural final-center displacement is `(-77.2581, 0.1397)` pixels. Replacing the final encoder FFN activation with its NMF reconstruction changes that to `(-77.1646, 0.1318)` pixels (absolute change `0.0935` px in x and `0.0079` px in y).

Per-concept/per-observed-step ablations on that one window changed final displacement by at most `0.0386` px in x and `0.0082` px in y. The per-step activation scores for all 15 observed time steps are saved separately. For the one-window Sobol probe, 1,024 scrambled Sobol samples were transformed through the empirical training-window concept-score marginals, with concept dimensions varied independently. The largest total-order indices were concept 4 for `dy` (0.4299) and `dx` (0.3738), and concept 15 for `dx` (0.3153). These values are **local to one gate-closed training window and this intervention distribution**; they are not global feature importance or confirmatory statistics. The small ablation effects and high-rank factorization limit their explanatory value.

## What worked and what did not

- **Worked:** activation capture was finite; ReLU FFN non-negativity was confirmed; the context clamping mass was measured; training-only NMF bases and validation-only rank selection were completed; top-activation windows were visualized; scene/gate associations, per-step NMF scores, a local per-step ablation, and a Sobol sensitivity probe were produced.
- **Did not work as a semantic explanation:** the selected decompositions are high-rank, the visual examples do not form clearly interpretable concepts, scene labels are highly imbalanced, and the functional importance probe is only for one gate-closed training window. The CRAFT repository recommends a substantially larger image set (about 500 images); this analysis has 212 overlapping train windows from 43 pedestrians and is not an image-classifier task.
- **Conclusion:** retain these as exploratory artifacts only. They do not establish that PACE-Net has human-interpretable concepts, that scene labels drive its predictions, or that its gate is causally explained by these factors.

## Artifacts

- Reproducible analyzer: [analyze_craft_trajectory.py](../../MTN/analyze_craft_trajectory.py).
- Final outputs: [rank/validation errors](./craft_seed20260928_rank2_16/CRAFT_RANK_SELECTION_seed20260928.csv), [activation checks](./craft_seed20260928_rank2_16/CRAFT_ACTIVATION_CHECKS_seed20260928.csv), [concept scores](./craft_seed20260928_rank2_16/CRAFT_CONCEPT_SCORES_seed20260928.csv), [15-step scores](./craft_seed20260928_rank2_16/CRAFT_TIME_ATTRIBUTION_seed20260928.csv), [scene/gate associations](./craft_seed20260928_rank2_16/CRAFT_SCENE_GATE_ASSOCIATION_seed20260928.csv), [Sobol results](./craft_seed20260928_rank2_16/CRAFT_SOBOL_seed20260928.csv), [time ablations](./craft_seed20260928_rank2_16/CRAFT_TIME_ABLATION_seed20260928.csv), [run metadata](./craft_seed20260928_rank2_16/CRAFT_RUN_METADATA_seed20260928.json), and [visual examples](./craft_seed20260928_rank2_16/CRAFT_NMF_EXAMPLES_seed20260928.png).
- Preliminary rank-2–8 outputs are preserved under [craft_seed20260928_rank2_8_preliminary/](./craft_seed20260928_rank2_8_preliminary/); they are not used for the reported conclusions.

No files under `paper/` were changed.
