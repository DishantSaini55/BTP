# Provenance and integrity addendum

**Package:** `results_paper/final_manuscript_v1/`  
**Prepared:** 2026-11-27 (Asia/Calcutta)

The manuscript's official-test values are derived from the corrected Phase 6B `OFFICIAL_TEST_RESULTS_v2.json` and existing v3 model, paired-comparison and track-bootstrap summaries. The corrected JSON SHA-256 is `6a96d14b857452a454ceeab76954bdb5f9dfc15d11de18c4ff6500e7d8558b03`, identical to the Phase 6B integrity addendum. All 15 neural checkpoints and frozen ridge coefficient hashes are documented in the original `results_paper/FINAL_CHECKPOINT_HASHES.csv`; that historical source was not modified.

All canonical v2 per-window/per-video/per-track/seed CSVs, v3 summaries and relevant Phase 6A/6B/6C/6D reports are copied or referenced under `supporting_data/`. The package preserves the v1 ridge result as superseded history and records the corrected v2 unit-consistency path. Manuscript images are based on those saved metrics or copied from the verified Stage 3 CRAFT figure set. The five diagrams are source-backed illustrations of the existing implementation.

`PROVENANCE_AND_INTEGRITY_MANIFEST.csv` lists SHA-256 and byte size for canonical source inputs and files in the delivery directory. The manifest intentionally does not hash itself. The Overleaf ZIP was checked to contain the final `main.tex`, bibliography and required figure/diagram assets. `QA_VALIDATION.md` records data-row reconciliation and compilation checks.

No prediction or checkpoint was regenerated. No model inference, training, selection, tuning or raw-video access occurred for this package. The official-test bundle has no coordinate-level predictions; the absent trajectory traces were not fabricated. Human-study materials are protocol drafts only.
