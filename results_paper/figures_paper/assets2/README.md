# Second real-data architecture asset pack

## Frozen window selection

Before viewing any candidate image, selected from the frozen GP6 test split using metadata only:

1. Keep test windows whose mean raw OBD speed over the first 15 observed frames is nonzero and whose maximum observed occlusion over those frames is at least 1.
2. Sort the qualifying sample IDs lexicographically and select the first.
3. Only if no window satisfies both conditions, relax to nonzero mean observed OBD speed alone and select the lexicographically first match.

The strict rule found 5 candidates, so no relaxation was needed. The selected ID is:

`gp_set_0006/gp_set_0006_vid_0003/gp_6933/01205/01264`

Its metadata-only selection values are mean observed OBD speed `1.5726103782653809` and maximum observed occlusion `2`. The ID and rule are frozen; do not replace this window based on its appearance.

## Sources and construction

Assets use real source-video frames, cached arrays from `results_paper/splits/gp6_from_scratch_v1/flow_splits.pkl`, and saved E1 model outputs/checkpoints. No image-generation model or synthetic scene pixels are used. Photo assets contain only source pixels plus colored box outlines; analytical assets are direct color mappings or plots of saved values. All PNGs are 300 DPI; text labels remain editable and are documented in `scene_labels.json`.

The source video is `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0003.MP4`. The saved seed-20260928 E1 PACE-Net and ROUTE outputs/checkpoints provide predictions, gate values, attention, and operator activations. Saved PACE-Net gate is `g=0`; ROUTE selects additive `[0,1,0]`.

| File | Source and construction |
|---|---|
| `generate_asset_pack.py` | Small wrapper around the first pack's shared renderer, with this README's frozen sample ID and this output directory. Inference only; refuses to overwrite output files. |
| `observed_frames.png` | Five real observed frames at window-relative indices 0, 3, 7, 11, and 14; pedestrian box red, 30% per-side margin, fixed source-frame aspect ratio. |
| `observed_strip.png` | Same real-frame treatment for all 15 observed frames in temporal order. |
| `predicted_frames.png` | Five real future frames at forecast indices 0, 11, 22, 33, and 44; ground truth yellow and saved PACE-Net purple. |
| `predicted_strip.png` | All 45 real future frames with the same ground-truth and PACE-Net outlines. |
| `ego_flow.png` | Cached Farneback ego-flow for transitions 1216→1217, 1217→1218, and 1218→1219. The cached 8×8 mean-flow grids are nearest-neighbor enlarged to 240×240; these are coarse features, not dense flow. |
| `ped_flow.png` | Cached Farneback pedestrian-flow for the same transitions. The 3×3 mean-flow grids are nearest-neighbor enlarged to 240×240. |
| `flow_strip.png` | Ego-flow maps above pedestrian-flow maps, with the three transitions in temporal order. |
| `speed_bars.png` | All 15 raw observed OBD speeds, without axes or text. Values are recorded exactly in `scene_labels.json`; the mean is nonzero. |
| `frame_with_labels.png` | Real last-observed frame 1219 with its observed pedestrian box in red. Scene/occlusion chips are represented as editable JSON values, not rasterized text. |
| `scene_labels.json` | Exact sample ID, scene codes/names, 60-frame occlusion series, 15 observed speeds, saved gates, flow/token/attention ranges, and PACE-Net inverted-corner forecast-step indices. |
| `memory_tokens_15.png` | Norm-colored strip of the 15 actual decoder memory tokens from saved PACE-Net. |
| `memory_tokens_16.png` | The 15 decoder tokens plus the pre-gate context candidate; because `g=0`, this is an architecture candidate, not the memory used in the saved forward pass. |
| `decoder_attention_strip.png` | Saved decoder memory attention, three layers stacked with four-head mean and 15 memory keys. |
| `route_operators.png` | Identity, additive, and cross-attention operator outputs from saved ROUTE checkpoint, as token-norm strips. |
| `params_bar.png` | Independent versus SATIE-shared encoder self-attention parameter counts (12,672 vs 4,224), rendered without text. |

Observed boxes are red `#E63946`, ground truth yellow `#FFD400`, and PACE-Net purple `#7B2CBF`. All analytical PNGs omit text so labels can be added in figure files. Flow uses HSV direction/magnitude coloring and is independently scaled per flow type; maxima are in `scene_labels.json`.

**Prediction-coordinate note:** the saved PACE-Net forecast has inverted horizontal `xyxy` endpoints at steps listed in `scene_labels.json` (steps are zero-based). In visual assets only, rectangle outlines use the min/max envelope of the saved endpoints so they can be drawn; the saved predictions are not modified. The original exporter previously failed on reversed endpoints; its shared box drawing/crop helper now handles them explicitly.

The exporter requires the local source video, cached split, and saved E1 checkpoints. Run from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe results_paper\figures_paper\assets2\generate_asset_pack.py
```

It performs inference only and refuses to overwrite an existing output.
