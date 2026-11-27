# Bounded CRAFT trajectory adaptation plan — not executed

## Scope

Use frozen AO-2 and AO-3, seed 20260928 checkpoints, and only the frozen AO
development train/validation caches (988/432 windows; no official-test access).
AO-2 and AO-3 take 15 observed `xyxy` boxes; their post-ReLU encoder FFNs are
the primary valid activation site. AO-3's shared attention is an architectural
difference, not an activation site to pool indiscriminately.

## Predeclared pipeline

1. Forward each development window once per model in `eval()`/`no_grad()` with
   hooks at each encoder FFN ReLU output; save arrays, IDs, checkpoint hash and
   split digest in a new immutable output directory. Do not inspect targets for
   selection. Audit signs before factorization.
2. Treat each `(window, observed step)` as a temporal analogue of an auxiliary
   example. Fit NMF only on training activations at a chosen final encoder layer;
   select a low rank by train reconstruction plus a *predeclared* validation
   reconstruction/stability rule, then refit only on train. Do not clamp signed
   layers; exclude them or use a separately named signed-factorization study.
3. Compute held-out NNLS coefficients with fixed `W`. Quantify reconstruction
   fidelity in activation space and output fidelity: absolute errors for the
   four predeclared scalar functionals after replacement by `UWᵀ`.
4. Recursively refine at most two globally important parent concepts: select
   training temporal examples above the parent coefficient's 90th percentile,
   factorize the immediately earlier encoder ReLU layer, and report membership
   overlap/stability across bootstrap resamples. Call children numeric factors
   unless a preregistered quantitative motion association supports a label.
5. Estimate total Sobol indices with coefficient masks and Jansen/Sobol QMC for
   each scalar functional. Use 2,048 base samples, fixed seed, and report all
   four outputs; no aggregate winner. The intervention must preserve the chosen
   temporal profile rather than sample independent coefficients per time step.
6. Fidelity: rank concepts using training-derived Sobol importance, then on
   validation windows plot insertion/deletion curves by restoring/removing
   reconstructed concept contributions. Report area, bootstrap CIs clustered by
   pedestrian, and a random-rank control. This is trajectory-output fidelity,
   not the ImageNet logit experiment in CRAFT.
7. Do **not** promise pixel maps. A temporal attribution extension requires a
   differentiable NNLS/NMF layer with KKT/dual variables (e.g., Jaxopt ADMM or
   an independently validated PyTorch implicit solver). If implemented, map
   `∂u_j/∂box[t,c]` to a 15×4 temporal-coordinate heatmap; explicitly distinguish
   it from CRAFT's spatial image map.

This is a trajectory-specific adaptation, not a faithful reproduction of CRAFT.
