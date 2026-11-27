# Final package QA validation

- Canonical corrected Phase 6B v2 JSON SHA-256 matches the existing integrity addendum: `6a96d14b857452a454ceeab76954bdb5f9dfc15d11de18c4ff6500e7d8558b03`.
- Reconciled canonical data rows: 18 seed/model records; 234 per-video rows (18 runs x 13 videos); 15,498 per-track rows (18 runs x 861 clusters); 47,070 per-window rows (18 runs x 2,615 windows); 18 macro rows.
- Confirmed canonical summary MSE for all six model identities against values rendered in the paper.
- Confirmed required source and figure files are present in `OVERLEAF_SOURCE.zip`.
- Verified final PDF exists (15 pages) and final release log has no unresolved citations/references, fatal errors or overfull boxes.
- Final artifact QA status: PASS.

This QA validates document/data packaging and transcription. It does not rerun or independently recompute official-test predictions or any model inference.
