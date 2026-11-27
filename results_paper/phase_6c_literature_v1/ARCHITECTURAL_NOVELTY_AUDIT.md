# Architectural novelty audit

| Proposed item | Closest established work | Evidence here | Defensible wording | Decision |
|---|---|---|---|---|
| CV-anchored residual learning | kinematic/CV baselines and residual regression | AO-2/3 beat AO-1; ridge wins official test | A tested formulation for this annotation-only task, not a novel principle | Retain empirically |
| SATIE attention sharing | Universal Transformer/ALBERT parameter sharing | 11% fewer trainable parameters; no reliable accuracy advantage over AO-2 | Compact sharing variant | Retain compactness only |
| SAGE admission gate | conditional computation/MoE/gating | learned admission unreliable; no confirmed gain | Exploratory gate implementation | Remove as contribution |
| MASC context memory | memory-token/context-attention augmentation | no reliable benefit and no compatible AO context inputs | Architectural exploratory mechanism | Remove as contribution |
| ROUTE operator routing | dynamic fusion/MoE | no supported official-test advantage | Exploratory routing mechanism | Remove as contribution |
| Annotation-only forecasting | trajectory-only baselines are established | rigorous held-out annotation-only protocol and classical comparison | Evaluation contribution, not a new modality | Retain |
| Classical vs neural residual comparison | standard baseline practice | ridge +27.707% vs CV; neural-CV t intervals include zero | Main empirical contribution | Retain prominently |

Methodological novelty is not established for any acronym solely by naming or
combination. The strongest supported contribution is transparent negative/positive
comparative evidence under a fixed protocol.
