# Final delivery report

Date: 2026-11-27 (Asia/Calcutta)

## Delivered

- Rewritten 15-page LaTeX manuscript: `main.tex`, with verified bibliography in `references.bib`.
- Compiled paper: `FINAL_RESEARCH_PAPER.pdf` (pdfLaTeX, 15 pages; BibTeX references resolved).
- Overleaf source archive: `OVERLEAF_SOURCE.zip` with all source figure dependencies.
- Five architecture diagram sets: editable draw.io XML and SVG plus PNG; combined multi-page diagram PDF and TeX source.
- Five publication-style metrics figures based on the canonical saved Phase 6B error CSVs, plus three copied CRAFT Stage 3 figures.
- Full canonical official-test JSON/CSV evidence and source reports in `supporting_data/`.
- Prospective HIVE study materials under `hive/`, with explicit ethics, permission and data limitations.
- Changelog, supervisor requirement status, scientific integrity checklist and artifact provenance manifest.

## Evidence conclusions presented

AO-RIDGE is the strongest observed predictor (MSE 2,889.718; 27.707% below CV). AO-2 and AO-3 are 0.555% and 0.880% below CV, respectively, and their seed-paired intervals versus CV include zero. AO-3 ties encoder attention across three layers and reduces parameters by 11.0% versus AO-2; a predictive gain over AO-2 is not established by seed-level intervals. All values are reported against the frozen 2,615-window, 13-video, 861-track annotation-only protocol.

The ridge v1 defect and v2 correction are described as a coordinate-consistency bug fix: v1 mixed normalized residuals with raw-pixel CV forecasts, while v2 adds the residual in normalized coordinates and applies the same frozen inverse normalization. No test-driven tuning or inference rerun was performed during manuscript preparation.

The CRAFT-inspired section reports the exploratory temporal factorization, recursive convergence warning, numeric-only associations, branch-relative Sobol outcomes, limited branch influence and failed displacement deletion screen. It does not claim validated semantic concepts, a faithful CRAFT reproduction, implicit attribution maps or human evaluation.

## Unresolved blocker

The corrected official-test bundle contains only per-window errors and identifiers, not observed/ground-truth/prediction coordinates. Therefore official-test trajectory success/failure overlays and participant-ready blinded AO stimuli could not be created without repeating inference, which was prohibited. The HIVE package is a protocol and template only. Historical E1 materials were excluded as model/protocol mismatches.

The author and institutional metadata are placeholders because those details were not supplied. Any human study requires ethics approval, dataset-owner permission, approved consent, privacy handling and preregistration before recruitment.

## Quality assurance

Direct pdfLaTeX and BibTeX compilation completed successfully. Final release passes contain no unresolved citations, references, fatal errors or overfull-box warnings. MiKTeX emitted its informational reminder that updates were not checked. `latexmk` itself could not run because its Perl script engine is absent; this did not prevent direct compilation. The output PDFs are present and the Overleaf ZIP includes its required source and figures.

No existing `paper/` files were edited. No training, inference, CRAFT experiment, raw-video download, hyperparameter selection, commit, push or subsequent research phase was started.
