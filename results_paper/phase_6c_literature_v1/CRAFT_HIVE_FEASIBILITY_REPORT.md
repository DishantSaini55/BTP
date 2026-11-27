# CRAFT/HIVE feasibility report

CRAFT (Fel et al., CVPR 2023) factorizes non-negative intermediate activations,
recursively refines concepts, and estimates importance with Sobol indices. The
existing CRAFT-inspired work captured trajectory-transformer activations, used
NMF and local Sobol probes, but it is not a faithful end-to-end CRAFT study:
the source method is image/classification-oriented, the context representation
needed clamping, ranks were high, labels were imbalanced, and only 212
overlapping training windows/43 pedestrians were available. Its own report
correctly concludes no stable semantic concept vocabulary.

HIVE (Kim et al., ECCV 2022) is a human interpretability evaluation framework,
not an automated score. A valid HIVE-style deployment requires a participant
protocol, ethics/data permission review, consent, anonymised stimuli,
pre-registration and real responses. None exists. The saved predictions,
overlays and attention arrays can create stimuli without rerunning inference,
but no human-study result may be claimed.
