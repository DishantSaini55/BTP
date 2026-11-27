# PACE-Net Phase 2 research report

## Status and scope

| Work item | Status | Evidence |
|---|---|---|
| Corrected E3 observed-input diagnostic | Complete | `E3_CORRECTED_REPORT.md`; JSON/CSV under `diagnostics/` |
| Differentiable SAGE V2 and controlled P2-A–P2-E | Complete | `MTN/sage_v2.py`; formulation below |
| Deterministic math/gradient tests | Complete | 24 repository tests passed |
| Resumable, validation-only screening runner | Complete | `MTN/run_p2_screen.py`; budget-stop/resume smoke passed |
| Seed 20260928 full-split screening | Complete | Five configurations, 10 epochs each, 212 train / 43 validation windows |
| Test-set evaluation | Not performed | Runner did not load the test split |
| Additional seeds / Phase 3 | Not started | This report stops at the requested initial screen |

The five screening runs used the frozen split and cached optical flow, training-only normalization, Adam `1e-3`, batch size 8, four CPU threads, normalized MSE, and a maximum of 10 epochs. The validation checkpoint was selected without test data. These are exploratory one-seed results, not statistically conclusive findings.

## Corrected E3 diagnostic

The historical `MTN/run_e3.py:x_t_ranges` used `sample["bbox"][-1]`, which selects the last frame of the full 60-frame sample. The gate's actual input is the final observed bbox, `sample["bbox"][observe_length - 1]` (index 14 for 15 observed frames). The correction is a diagnostic frame-selection fix; it does not by itself establish training leakage.

Train-only normalization produced these aggregate normalized observed-input ranges:

| Split | Normalized min | Normalized max | Mean | Population SD | Outside `[-5, 5]` |
|---|---:|---:|---:|---:|---:|
| Train (212 windows) | -2.4068 | 3.3708 | 0.0633 | 0.7386 | 0.0000 |
| Validation (43 windows) | -8.3610 | 3.1456 | -1.8946 | 2.6329 | 0.1860 |
| Test (217 windows) | -30.8120 | 4.9415 | -7.2188 | 9.4670 | 0.4977 |

The test range is descriptive only and was not used to select clipping thresholds, gate inputs, or hyperparameters. Per-coordinate min/max, mean, SD, quantiles, and outside fractions are in `diagnostics/E3_v2_observed_input_ranges.json` and `diagnostics/E3_v2_observed_input_ranges.csv`. The historical report/output is preserved.

## SAGE V2 implementation and verification

V2 leaves the historical SAGE/MASC implementations unchanged. It retains the context token in decoder memory during differentiable training and adds `log(p + 1e-8)` to that token's pre-softmax decoder source-attention score in every decoder layer. It does not scale the token a second time. Ordinary masks remain in force for unrelated invalid/padded keys. Soft inference uses continuous admission; hard inference uses argmax, with exact motion-only fallback for rejected samples.

All five models had SATIE Q/K/V/O identity sharing verified. Actual unique trainable parameter counts:

| Configuration | Parameters |
|---|---:|
| P2-A: original SAGE + MASC + SATIE | 130,694 |
| P2-B: independently trained always-open context + MASC + SATIE | 130,068 |
| P2-C: differentiable gate, original inputs | 130,694 |
| P2-D: differentiable gate, normalized `x_T` clipped to `[-5, 5]` | 130,694 |
| P2-E: differentiable gate, context-only input | 130,630 |

Deterministic tests verify normalized attention, zero attention on unrelated masked positions, finite values near probability boundaries, always-open equivalence, exact hard-closed fallback, mixed hard inference, SATIE sharing, and output shape `[batch, 45, 4]`. The full suite passed: **24 tests**. A separate budget-limited P2-C run was marked `incomplete` after its configured budget and then resumed from its saved model, optimizer, random states, epoch and batch position; it completed the capped diagnostic run.

The actual learned gate MLP at admission probability `0.00245011` had measured gradient L2 `7.55293e-6`, with all 4 gate parameter tensors receiving nonzero gradients; the context encoder gradient L2 was `0.00107058`, with all 12 tensors receiving nonzero gradients. In a separate extreme saturation probe (`p = 3.78e-44`), the gate gradient was present but exactly zero. Therefore V2 resolves the demonstrated low-but-nonzero admission gradient path, but numerical saturation can still produce zero gate gradients. See `SAGE_V2_GRADIENT_REPORT.md` and `diagnostics/P2/sage_v2_gradient_diagnostics_v2.json`.

## Full-split validation results

The table gives metrics under each configuration's checkpoint-selection inference policy: hard for P2-A, always-open for P2-B, and soft for P2-C/D/E. Errors are validation-only; MSE values are raw pixel-coordinate MSE, C-MSE is center-coordinate MSE, and CF-MSE is final-step center MSE. `Train MSE` is normalized-coordinate MSE on the full training split using the selected checkpoint and policy.

| Configuration | Selected epoch | Train normalized MSE | Validation normalized MSE | Raw pixel MSE | C-MSE | CF-MSE | Gate grad L2 mean | Context grad L2 mean | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P2-A | 10 | 0.112770 | 0.999297 | 8,010.10 | 7,344.73 | 17,571.08 | 0.001730 | 0.025688 | 130,694 |
| P2-B | 6 | 0.150390 | 1.056828 | 7,808.88 | 7,047.01 | 26,897.85 | — (no gate) | 0.037485 | 130,068 |
| P2-C | 5 | 0.160862 | 1.066509 | 7,835.28 | 7,101.23 | 25,788.04 | 0.003609 | 0.041446 | 130,694 |
| P2-D | 5 | 0.160862 | 1.066693 | 7,832.09 | 7,097.79 | 25,778.42 | 0.003609 | 0.041446 | 130,694 |
| P2-E | 9 | 0.112845 | 1.021382 | 8,015.25 | 7,393.63 | 19,123.48 | 0.002154 | 0.034194 | 130,630 |

Full per-policy metrics (including normalized validation MSE, raw MSE/C-MSE/CF-MSE, probability SD and quantiles, hard admission rate, entropy, logits, attention weight, gradient summaries, runtime, and memory) are in `P2_VALIDATION_SUMMARY.csv`. The selected model's per-window predictions, gate logits/decisions, and decoder context-attention weights are saved under each run directory.

### Soft versus hard inference

| Configuration | Soft validation normalized MSE | Hard validation normalized MSE | Hard admission rate |
|---|---:|---:|---:|
| P2-A | 0.995217 | 0.999297 | 1.0000 |
| P2-C | 1.066509 | 1.050885 | 0.9070 |
| P2-D | 1.066693 | 1.059914 | 0.9535 |
| P2-E | 1.021382 | 0.987263 | 1.0000 |

P2-A's historical soft-gate behavior is retained as implemented in the original model and is not the V2 attention-prior formulation. P2-E's hard predictions are exactly equal to forced-open predictions because all 43 validation windows are admitted by the hard decision. P2-A likewise has hard predictions exactly equal to forced-open predictions. P2-C/D reject only a small fraction of validation windows.

### Forced-open versus forced-closed validation diagnostic

The per-window context-benefit diagnostic is `forced-closed raw MSE - forced-open raw MSE`; positive means forced-open context reduced error. This is exploratory validation-only analysis, not a gate-selection objective.

| Configuration | Mean context benefit (raw MSE) | Pearson r: soft probability vs. benefit | Mean benefit, admitted | Mean benefit, rejected |
|---|---:|---:|---:|---:|
| P2-A | 1,319.55 | 0.257 | 1,319.55 | unavailable (no rejected windows) |
| P2-C | 333.15 | -0.010 | 303.44 | 622.83 (4 windows) |
| P2-D | 333.15 | 0.001 | 345.41 | 81.86 (2 windows) |
| P2-E | 2,575.75 | -0.148 | 2,575.75 | unavailable (no rejected windows) |

These are descriptive statistics over 43 validation windows from the frozen validation video. The very small rejected subsets for P2-C/D and the absence of rejected windows for P2-A/E preclude strong conclusions. No confidence interval or significance claim is made.

## Scientific interpretation

1. **Gradient pathway:** The differentiable log-prior yields nonzero measured gradients through the actual gate MLP and context encoder at low but nonzero admission. Full-screen training also recorded nonzero gate/context gradient norms for P2-C/D/E. Extreme softmax saturation remains a zero-gradient edge case.
2. **Sample-dependent admission:** Probabilities varied, but hard decisions admitted 90.7% (C), 95.3% (D), and 100% (E) of validation windows. P2-A also admitted all validation windows. The screen does not show useful hard sample-dependent rejection.
3. **Clipping:** Train gate-input values were wholly inside `[-5, 5]`, so clipping made no change to training inputs. P2-C and P2-D selected model states were bitwise identical; their forced-open and forced-closed validation predictions were also bitwise identical. The validation gate probabilities differed slightly, but the soft validation MSE difference was negligible (1.066509 vs. 1.066693). This predefined clipping candidate did not demonstrate improved stability or accuracy.
4. **Always-open comparison:** P2-B achieved validation normalized MSE 1.056828. The differentiable models' selected soft results were 1.066509 (C), 1.066693 (D), and 1.021382 (E); E's hard result was 0.987263, but it was identical to forced-open because its hard gate admitted every validation window. P2-A's hard result was 0.999297 and also admitted every window. These one-seed results do not demonstrate that learned sample-dependent admission improves over always-open context.
5. **Hard inference:** Relative to soft inference, hard inference was better for C/D/E by 0.015624/0.006779/0.034118 normalized MSE and slightly worse for A by 0.004080. These differences are validation-only and one-seed.
6. **Recommendation:** The gradient-path repair is technically verified. Accuracy and admission behavior do not yet justify claiming an advantage or statistical significance. Do not start multi-seed training or Phase 3 without explicit approval; if authorized later, first pre-register a confirmatory comparison against P2-B and inspect admission behavior without tuning on test data.

## Limitations and integrity notes

- One seed and a 43-window validation split are insufficient for statistical conclusions.
- No test predictions, test metrics, or test-informed decisions were made.
- Capped smoke losses are infrastructure checks only; they are not full-split accuracy.
- The corrected E3 test range is descriptive and was not used to choose the `[-5, 5]` candidate.
- The earlier smoke directories with failed report serialization or policy dispatch are kept separately versioned and are not screening evidence; the final capped smoke is `P2_smoke_v3`.
- Historical E1/E2/E3/E7/E8 results and original model implementations remain unchanged. No files under `paper/` were edited.

## Reproduction

```powershell
python MTN\run_e3_v2_diagnostics.py
python -m unittest discover -s tests -v
python MTN\sage_v2_gradient_diagnostics.py
python MTN\run_p2_screen.py --configurations p2_a p2_b p2_c p2_d p2_e --seeds 20260928 --max-epochs 10 --patience 10 --batch-size 8 --cpu-threads 4 --wall-clock-budget-seconds 3600 --checkpoint-every-batches 5
```

The screening output path is `results_paper/P2_initial_screen_v1/`; the machine-readable policy summary is `results_paper/P2_VALIDATION_SUMMARY.csv`.
