# Legacy warm-start / adaptation artifacts

These files are preserved only for provenance. Every listed experiment initialised matching tensors from the official MTN checkpoint. The official training split includes all GP6 videos, including the local GP6 video-3 evaluation material. They are therefore **contaminated/adaptation artifacts**, not leakage-free held-out results.

They must not enter the from-scratch E1/B1 summary tables, figures, hypothesis tests, bootstrap intervals, manuscript results, or model selection. The original files remain under `D:\IDD-PeD-Evaluation\outputs\`; the copies here are non-executable result/configuration records.

Archived records:

- `C4_larger_validation_results.json` and its manifest/summary: three-seed B/C4 GP6 warm-start study.
- `fusion_validation_results.json`: three-seed ROUTE / ROUTE+SATIE GP6 warm-start study.
- `final_four_way_results.json`: consolidated warm-start four-model output.
- `targeted_weight_sharing_results.json`: one-seed, ten-test-window targeted-sharing POC; superseded by E4 and not evidence.
