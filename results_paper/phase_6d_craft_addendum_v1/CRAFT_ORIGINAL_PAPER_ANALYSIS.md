# CRAFT original-paper analysis

**Controlling source:** supervisor-provided `C:\Users\PC\Downloads\CRAFT.pdf`,
Fel et al., *CRAFT: Concept Recursive Activation FacTorization for
Explainability*, CVPR 2023. This analysis was made from that PDF, including
Sections 3.1--3.4, Section 4, Figure 4 and supplementary Sections C--F.

## Method as specified by the paper

CRAFT splits a model as `f=h∘g`, selects a model-defined set of image examples,
forms an auxiliary set of crop-and-resize inputs, and obtains non-negative
intermediate activations `A=g(X)`. It factorizes `A≈UWᵀ` with NMF: columns of
`W` are Concept Activation Vectors (the concept bank); rows of `U` are the
per-example concept coefficients. For a new input, coefficients are obtained
by NNLS with the fixed bank.

Figure 4 adds three ingredients. First, recursive refinement: choose examples
with a high coefficient (the paper uses a 90th-percentile criterion) and
factorize an earlier layer to produce sub-concepts. Second, perturb concept
coefficients with masks, reconstruct `Ũ=(U⊙M)Wᵀ`, propagate through `h`, and
estimate total Sobol indices, including interactions. Third, concept
attribution maps: differentiating NNLS/NMF through the KKT system via implicit
differentiation enables `∂U/∂X`, then standard forward/backward attribution to
image pixels. The paper explicitly says common scikit-learn coordinate-descent
NMF does not expose the dual variables needed for this step; its implementation
uses an ADMM/Jaxopt solver.

Section 4 evaluates human utility, recursivity, Sobol importance and local
maps. Supplementary fidelity evaluates concept insertion/deletion curves over
100,000 ImageNet images and five seeds, ranking concepts by Sobol versus TCAV.
Those experiments are not interchangeable with reconstruction error.

## Regression translation boundary

AO produces a 45×4 box sequence, not a class logit. A valid adaptation must
predeclare scalar functionals of a trajectory, rather than silently treating a
vector output as a class score. Candidate functionals are: final centre-x,
final centre-y, final centre displacement norm relative to the final observed
centre, and correction-energy `mean_{t,c}(ŷ−CV)^2`. Report them separately;
never average them post hoc to manufacture an importance ranking. Temporal
attribution can identify observed-box time steps or coordinates, but it is not
a spatial image attribution map and must not be presented as one.
