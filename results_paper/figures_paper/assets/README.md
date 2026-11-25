# Architecture-figure asset pack

This pack uses one fixed real test window throughout:

`gp_set_0006/gp_set_0006_vid_0003/gp_6933/01415/01474`

Video frames are decoded from `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0003.MP4`. Split inputs and optical-flow features come from `results_paper/splits/gp6_from_scratch_v1/flow_splits.pkl`. No image-generation model or synthetic scene pixels were used. Photo assets are crops/resizes of source-video frames with only the specified colored box outlines; analytical assets are direct color mappings or plots of saved arrays/model activations.

All PNGs are 300 DPI. Photo-based assets are opaque because their pixels are source frames. Data plots use transparent backgrounds where practical. The PNGs contain no added words, titles, axes, or legends; use this README and `scene_labels.json` for editable labels in figure files. `frame_with_labels.png` is the actual last-observed frame with the pedestrian box; its five scene-label chips and occlusion chip are data in the JSON, intentionally not rasterized into the PNG.

## Files

| File | Source and construction |
|---|---|
| `generate_asset_pack.py` | Reproducible exporter. Reads the fixed flow split, source video, seed-20260928 checkpoints, and saved predictions/attention; refuses to overwrite an existing generated asset. |
| `observed_frames.png` | Five real observed frames at window-relative indices 0, 3, 7, 11, and 14. Each tile is cropped around that frame's annotated pedestrian box with a 30% per-side margin, fixed to the source-frame aspect ratio, then outlined red. |
| `observed_strip.png` | The same treatment for all 15 observed frames, in temporal order, as a thin filmstrip. |
| `predicted_frames.png` | Five real future frames at forecast indices 0, 11, 22, 33, and 44. Each crop contains the ground-truth box (yellow) and saved PACE-Net box (purple), with a 30% per-side margin and source-frame aspect ratio. |
| `predicted_strip.png` | All 45 real future frames in temporal order, with ground truth yellow and PACE-Net purple. |
| `ego_flow.png` | Farneback ego-flow maps for cached transition indices 11–13: frames 1426→1427, 1427→1428, and 1428→1429. The cached 8×8 mean-flow grids are nearest-neighbor upsampled to 240×240 for display; these are coarse cached features, not dense flow. |
| `ped_flow.png` | Farneback pedestrian-flow maps for the same three transitions. Each cached 3×3 mean-flow grid (computed inside the annotated pedestrian crop) is nearest-neighbor upsampled to 240×240 for display; the enlargement adds no flow detail. |
| `flow_strip.png` | Combined flow plate: ego maps in the top row and pedestrian maps in the bottom row; columns are the three consecutive transitions in time. |
| `speed_bars.png` | The 15 raw cached observed ego-speed values, with axes and text removed. All 15 values are exactly zero in this window; zero-height bars are indicated by markers at zero. |
| `frame_with_labels.png` | Real frame 1429, the last observed frame, with its annotated pedestrian box in red. Scene/occlusion values for editable text chips are in `scene_labels.json`; no text is baked into this image. |
| `scene_labels.json` | Exact sample ID, raw scene codes and names, all 60 occlusion labels, observed-15 and whole-window occlusion summaries, raw observed speed values, and saved SAGE/ROUTE gate decisions. Scene tuple: `intersection_type=NI`, `signalized_type=CS`, `road_type=main`, `location_type=urban`, `motion_direction=OW`. Whole-window maximum occlusion is 2; the first 15 observed frames have maximum 0. |
| `memory_tokens_15.png` | Norm-colored strip of the 15 actual decoder memory tokens (each 32-D), captured from the saved seed-20260928 PACE-Net checkpoint. Color is each token's L2 norm, using the shared token color range `[0, 7.25115442276001]`. |
| `memory_tokens_16.png` | The same 15 decoder tokens plus the checkpoint's pre-gate context-token candidate combined with the speed-memory vector, as it would enter position 16 if admitted. The saved gate is `g=0`, so this 16-token strip is a candidate/architecture view, **not** the memory used by the actual E1 forward pass; that pass had 15 keys. |
| `decoder_attention_strip.png` | Saved PACE-Net decoder attention for this sample: three consecutive 45-query blocks (layers 1–3, top to bottom), each averaged over its four heads, with the 15 memory keys as columns. The color range is `[0, 0.06708124279975891]`; the range is also recorded in `scene_labels.json`. |
| `route_operators.png` | Three 15-token ROUTE operator outputs from the saved `pace_net_with_route` seed-20260928 checkpoint, shown as token-norm strips in top-to-bottom order: identity, additive, cross-attention. The saved route decision is `[0,1,0]` (additive). Shared norm color range: `[0, 5.736114025115967]`. |
| `params_bar.png` | Two unlabeled bars, left to right: independent encoder self-attention parameters (12,672) and SATIE-shared encoder self-attention parameters (4,224). For `d_model=32`, one Q/K/V/O module has `4 × (32×32 + 32) = 4,224` parameters; three independent layers have 12,672. |

## Color and scale conventions

- Observed boxes are red `#E63946`, ground truth is yellow `#FFD400`, and PACE-Net is purple `#7B2CBF`.
- Flow uses the standard HSV directional encoding: hue is vector direction and value/brightness is magnitude. Ego and pedestrian maps are each scaled by their own maximum over the three displayed transitions, respectively `11.003376960754395` and `0.14580456912517548`; these values are in `scene_labels.json`. The flow grids are the cached Farneback means on the 320×180 processing image (8×8 full-frame grid and 3×3 pedestrian-crop grid).
- Token and ROUTE strips encode per-token L2 norm with the `viridis` palette; decoder attention uses `magma`. Colorbar numbers are omitted from PNGs so figure files can add editable labels; numeric ranges are listed above and in `scene_labels.json`.
- When a saved predicted `xyxy` box has reversed endpoints, crops and outlines use the min/max endpoint envelope so it can be drawn; the underlying saved prediction is unchanged.
- No text labels are rasterized into the PNGs. Row ordering, selected times, model names, gate values, source paths, and exact numeric data are documented here or in `scene_labels.json`.

## Regeneration

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe results_paper\figures_paper\assets\generate_asset_pack.py
```

The script requires the local source video, frozen cached flow split, and saved E1 checkpoints listed above. It performs inference only for activation/operator capture; it does not train, update, or select a checkpoint. Remove or rename specific existing generated files before an intentional re-export; the script otherwise refuses to overwrite them.
