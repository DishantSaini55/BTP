# CRAFT final experiment report — Stage 3

**Status:** bounded exploratory development analysis completed in 102.65 seconds (four CPU threads maximum). The frozen AO-3 seed-20260928 model and canonical Stage 2B rank-4 bank were used. No model weights were changed, no final-layer activation extraction was repeated, and official-test data was not accessed.

This is a trajectory-regression adaptation inspired by the supervisor-provided CRAFT paper, not a faithful reproduction of its image-crop pipeline, implicit differentiation, attribution maps, or human utility experiments.

## Recursive decomposition

See `RECURSIVE_CONCEPT_RESULTS.md` and `PARENT_TOP_TOKEN_MANIFEST.csv`. Each parent selected the top 10% of train token coefficients and used aligned preceding-layer ReLU FFN outputs. Child factors remained numeric; no labels were assigned from visualization.

## Motion association

Six observation-derived descriptors were tested against four temporal-mean parent coefficients. Benjamini-Hochberg correction covered all 24 tests; association flags required |Spearman rho|≥0.30 and q<0.05. See `MOTION_ASSOCIATIONS.csv`. The 988 windows include repeated/overlapping samples from 261 video-track groups; window-level p-values are descriptive and do not account for all within-track dependence.
Associations meeting the fixed threshold: C1 vs mean_velocity_px_per_observed_step (rho -0.519, q 9.52e-68, n=988); C1 vs mean_acceleration_px_per_step2 (rho -0.386, q 2.56e-35, n=988); C3 vs mean_velocity_px_per_observed_step (rho 0.304, q 1e-21, n=988)

## Sobol design and results

The 32-window GP7 sample comprises 16 windows from each validation video, selected by sorted video and sample IDs without consulting predictions, errors, or concept importance. Four independent Uniform(0,1) amplitude multipliers scale the frozen parent contribution fields. Jansen total-order indices were estimated at nested base budgets 32, 64, and 128 using one scrambled 8-D Sobol net split into A and B blocks. This is relative sensitivity within the reconstructed FFN branch under that intervention distribution; it is not natural joint concept variation or complete-model global importance.

The synthetic additive sensitivity check had maximum absolute index error 0.01376 (expected indices `[1,4,9,16]/30`). Sobol indices and output variances are in `SOBOL_IMPORTANCE.csv`; budget traces are in `SOBOL_CONVERGENCE.csv`.

| Scalar output | Units | Output variance (N=128) | Converged by fixed rule? |
|---|---|---:|---|
| final_center_x | pixels | 1.10832 | True |
| final_center_y | pixels | 0.0243637 | True |
| final_displacement_norm | pixels | 0.012173 | True |
| cv_correction_energy | pixels_squared | 390.089 | True |

## Absolute effects and fidelity

Original, identity, full reconstruction, zero activation, concept-only, and concept-deleted scalar outputs are in `ABSOLUTE_INTERVENTION_EFFECTS.csv`. Full-reconstruction deviations from original are recorded separately from concept interventions. Insertion/deletion curves compare Sobol order, reverse order, and five fixed random orders; see `FIDELITY_RESULTS.csv`. Normalized distances are undefined when the full-versus-original denominator is ≤1e-10. A fidelity-order screen passes only under the predeclared AUC margins against both controls; see `STAGE3_SUMMARY.json`.

| Output | Full-vs-original absolute scalar difference | Insertion screen | Deletion screen |
|---|---:|---|---|
| final_center_x | 0.484375 | True | True |
| final_center_y | 0.0880127 | True | True |
| final_displacement_norm | 0.37175 | True | False |
| cv_correction_energy | 0.949127 | True | True |

## Figures and limitations

`FIGURE_MANIFEST.csv` lists PNG, editable SVG, and PDF exports generated from completed numeric results. Parent profiles are activation coefficients, not input attribution maps. Trajectory overlays are development examples only. No human evaluation, implicit differentiation, semantic validation, or additional rank search was performed. Sobol indices depend on the declared independent uniform scaling of fixed concept profiles and only 32 GP7 windows; high relative indices with small absolute output variance/effects are not evidence of strong model-level importance. Recursive factors, motion correlations, and fidelity controls should be interpreted within these limits.

No follow-on phase was started.


## Detailed numerical results and audit notes

### Recursive factor fits

| Parent | Training tokens | Fit error, seed 20260928 | Fit error, seed 20260929 | Matched cosine | Iterations | Numerical support |
|---|---:|---:|---:|---:|---|---|
| C1 | 1,482 | 0.078481 | 0.078481 | 1.000000 | 200 / 200 | No: both runs reached the iteration cap with convergence warnings |
| C2 | 1,482 | 0.165391 | 0.165391 | 1.000000 | 146 / 146 | Yes |
| C3 | 1,482 | 0.361628 | 0.361628 | 1.000000 | 29 / 29 | Yes |
| C4 | 1,482 | 0.202348 | 0.202348 | 1.000000 | 92 / 92 | Yes |

The recursive error and stability criteria support numeric child banks for C2–C4 only. C1 remains unsupported because convergence warnings fail the predeclared gate, despite low reconstruction error and seed agreement.

### Sobol indices at N=128

| Output | Units | Variance | C1 | C2 | C3 | C4 | N=64→128 max index change |
|---|---|---:|---:|---:|---:|---:|---:|
| Final predicted center x | px² | 1.108320 | 0.000024 | 0.991480 | 0.000000 | 0.009114 | 0.000917 |
| Final predicted center y | px² | 0.024364 | 0.080864 | 0.900494 | 0.000000 | 0.021793 | 0.000893 |
| Final displacement norm | px² | 0.012173 | 0.004057 | 0.206009 | 0.000000 | 0.792439 | 0.001675 |
| CV-correction energy | (px²)² | 390.088740 | 0.004491 | 0.981160 | 0.000000 | 0.018125 | 0.000592 |

All four satisfy the predeclared numerical convergence rule. C2 dominates sensitivity of the center coordinates and CV-correction energy under the fixed independent-amplitude intervention; C4 dominates displacement norm. These are branch-relative sensitivities. C3 has zero estimated sensitivity for these outputs under this design. The scalar-output variances and full/zero model controls provide the relevant scale context.

### Absolute control effects on the 32-window GP7 sample

| Output | Full reconstruction vs original | Zero activation vs original |
|---|---:|---:|
| Final center x | 0.484375 px | 2.813110 px |
| Final center y | 0.088013 px | 0.661865 px |
| Final displacement norm | 0.371750 px | 0.224533 px |
| CV-correction energy | 0.949127 px² | 64.771088 px² |

Identity-hook maximum output error was exactly zero. Thus the final-layer branch's absolute effect is small for these center/displacement means but larger for the CV-correction energy; the prior Stage 2B output-norm result and these scalar summaries are different summaries on different subsets/functionalities and should not be conflated.

### Fidelity AUC summaries

AUC is trapezoidal area under normalized distance from the full-reconstruction scalar output over intervention fraction 0–1. AUCs may be large because the normalization denominator is the small full-reconstruction-to-original scalar gap. Random controls are mean ± sample SD over seeds 3101–3105.

| Output | Mode | Sobol order AUC | Reverse order AUC | Random mean ± SD | Fixed screen |
|---|---|---:|---:|---:|---|
| Center x | Insertion | 1.0584 | 6.0848 | 3.4365 ± 1.8473 | Pass |
| Center x | Deletion | 6.0848 | 1.0584 | 3.7778 ± 1.9538 | Pass |
| Center y | Insertion | 1.8779 | 6.4761 | 4.0341 ± 1.3536 | Pass |
| Center y | Deletion | 6.4761 | 1.8779 | 4.2916 ± 1.3311 | Pass |
| Displacement norm | Insertion | 0.1624 | 0.5212 | 0.2988 ± 0.1644 | Pass |
| Displacement norm | Deletion | 0.5212 | 0.1624 | 0.4225 ± 0.1101 | Fail |
| CV-correction energy | Insertion | 11.2259 | 55.8651 | 32.2753 ± 15.9137 | Pass |
| CV-correction energy | Deletion | 55.8651 | 11.2259 | 35.6362 ± 16.7833 | Pass |

The numerical AUC screen passes 7 of 8 output/mode combinations. This is not a general fidelity validation: the normalized denominator can be much smaller than single-concept effects (for example center-y full-vs-original is 0.0880 px while zero activation differs by 0.6619 px), and the displacement-norm deletion comparison fails. `FIDELITY_AUC_SUMMARY.csv` contains the AUC values and random-control summaries; `FIDELITY_RESULTS.csv` contains per-step curves. The latter's original mixed-row writer did not preserve AUC columns, so the separate summary was computed from its saved per-step numeric curves without rerunning inference.

### Motion associations meeting the fixed threshold

| Numeric concept | Descriptor | Spearman rho | BH q | n windows |
|---|---|---:|---:|---:|
| C1 | Mean velocity | -0.519 | 9.52×10⁻⁶⁸ | 988 |
| C1 | Mean acceleration | -0.386 | 2.56×10⁻³⁵ | 988 |
| C3 | Mean velocity | 0.304 | 1.00×10⁻²¹ | 988 |

These correlations do not justify naming the concepts “slow”, “acceleration”, or any other semantic category; windows overlap and p-values are not clustered by track.

### Tests and execution record

The final focused Stage 3 suite passed **4/4** (Sobol synthetic estimator, coefficient-basis reconstruction, BH monotonicity, deterministic top-decile selection). The initial 4-D independent A/B synthetic design failed before AO-3 evaluation; its log, status, and test output are retained as `INITIAL_PREMODEL_GATE_*`. It was replaced before model evaluation by the documented 8-D Sobol net split into A/B blocks; the corrected synthetic maximum index error was 0.013756. Two execution attempts that stopped at hash-format and NumPy/PyTorch scalar-helper issues are preserved under `ATTEMPT_HASHFIX_RECURSION_*`; no test-driven tuning or official-test results were involved. The completed run took 102.65 seconds (the final process log measures approximately 105 seconds including final writes).
