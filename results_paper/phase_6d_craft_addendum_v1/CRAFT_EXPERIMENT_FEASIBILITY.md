# CRAFT experiment feasibility and approval gate

## Saved-artifact decision

Saved artefacts are insufficient for a meaningful AO-2/AO-3 CRAFT experiment:
they are generated from old PACE-Net, omit AO activations, recursion, fidelity
curves and implicit differentiation, and only probe one reference window. New
development-only forwards are necessary. No such forward pass has been run.

## Required work and dependencies

| Work | New computation | Dependencies | CPU estimate |
|---|---|---|---:|
| Capture AO-2/AO-3 encoder activations | 2×(988+432) forward windows, batch 8 | PyTorch, existing checkpoints/caches | 5–15 min |
| NMF + stability | ranks 2–16; bootstrap refits | numpy, scikit-learn | 10–30 min/model |
| Sobol + fidelity | 4 functionals, 2,048 base QMC, selected parents | scipy, numpy, PyTorch hooks | 1–3 h/model |
| Temporal implicit attribution | implementation/verification | Jax/Jaxopt or validated implicit NNLS | development work; not bounded safely |

Estimates are CPU-only, four threads, and exclude engineering/debug time. Store
only development artifacts; no raw video is required. Scikit-learn is already
used. A faithful implicit-differentiation component would add JAX/Jaxopt (or a
tested equivalent); do not install it or modify the environment without a
separate approval.

## Stop condition

Approval is required before any activation extraction or forward pass. Approve
only the bounded AO-development plan above, or direct a smaller audit-only
alternative. The experiment must not touch official-test records, retrain a
model, name latent factors from visual intuition, or claim CRAFT replication.
