# CRAFT implementation gap audit

| Original CRAFT component | Existing saved analysis | Gap / status |
|---|---|---|
| Activation set and intermediate extraction | Hooks captured old PACE-Net ReLU FFNs on 212 train/43 validation windows | Not AO-2/AO-3; no model-defined crop/subsequence set |
| Non-negative activations | FFN captures are finite/non-negative; context post-GELU was clamped, losing 13.81% train absolute mass | ReLU FFN is acceptable; clamped signed context is not a faithful CRAFT representation |
| NMF / CAVs / coefficients | scikit-learn NMF on final encoder box FFN; ranks 2--16, selected 15 | Coefficients exist, but rank is high and no NNLS for independently held-out explanations is documented |
| Recursive earlier-layer decomposition | None | Missing |
| Total Sobol | 1,024 samples on one training reference; independent empirical coefficient marginals; final dx/dy only | Local exploratory probe; no multi-output contract, no dependence treatment, no global aggregation |
| Insertion/deletion fidelity | Reconstruction shift and single-step ablations only | Missing ranked concept insertion/deletion curves and uncertainty |
| Implicit-differentiation attribution | None; sklearn NMF lacks required KKT/dual solver state | Missing |
| Human-centred utility | HIVE-style design only; no participants | Missing and must remain so without approval |

The saved artefacts are useful feasibility evidence, not a faithful CRAFT
reproduction. No latent factor may be called velocity, acceleration, turning or
stationarity: the current outputs contain no preregistered quantitative
association/stability test sufficient for those labels.
