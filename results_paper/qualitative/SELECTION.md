# Qualitative window selection

This selection protocol and the selected IDs were frozen before opening or viewing any image. All choices use only the saved seed-20260928 E1 test outputs, per-sample metrics, split metadata, and gate arrays. No model is retrained and no selection decision changes a checkpoint or reported aggregate.

## Models and selection rules

- Compare the saved `mtn_baseline` checkpoint with the saved `pace_net_sage_plus_masc_plus_satie` checkpoint (the primary PACE-Net). Also include saved `pace_net_without_satie_sage_plus_masc` outputs and the deterministic constant-velocity baseline in the figures.
- Random subset RNG: Python `random.Random(20261125)`. Sort distinct test pedestrian IDs, sample six IDs without replacement, then choose one eligible window uniformly from each ID's sample-ID-sorted windows using the same RNG. This gives six random windows from six distinct pedestrians.
- From windows not already selected, choose the two greatest improvements and the two greatest degradations by `delta = PACE-Net-with-SATIE per-window MSE - MTN per-window MSE`. Improvement uses the two smallest deltas; degradation uses the two largest. Ties are broken by ascending sample ID. The two groups cannot overlap the random group or one another.
- From remaining windows, choose up to two examples with `g=0` and up to two with `g=1`, preferring the PACE-Net-with-SATIE gate and falling back to PACE-Net-without-SATIE only if a gate state is unavailable in the primary model. Within a gate state, choose ascending sample ID and do not duplicate previously selected windows.
- Gate convention: `g=1` means the context token is admitted; `g=0` means closed. MSE is in the pixel-coordinate scale used by the E1 per-sample result files. Figures show the 15 observed boxes, the 45-step ground truth, MTN, PACE-Net without SATIE, PACE-Net with SATIE, and constant velocity. Each figure annotates both PACE-Net gates (or N/A), scene labels, and per-window MSE for all five plotted series.

## Frozen selections

All samples are from test video `gp_set_0006_vid_0003`; all have scene codes `[intersection_type=NI, signalized_type=CS, road_type=main, location_type=urban, motion_direction=OW]`. Occlusion below is the maximum over the 60-frame sample window. The full model's saved gate was `g=0` for all 217 test windows; the without-SATIE saved gate was also `g=0` for all 217. Thus no `g=1` sample is available in seed 20260928's learned-gate outputs. Do not substitute another seed or tune the gate to obtain an admitted example.

| Group | Sample ID | PID | Frames | PACE-Net − MTN MSE | MTN MSE | PACE-Net MSE | Without-SATIE MSE | Full PACE gate | Max occlusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6933/01415/01474` | gp_6933 | 1415–1474 | 68.686 | 1655.796 | 1724.482 | 2562.021 | 0 | 2 |
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6955/02200/02259` | gp_6955 | 2200–2259 | 664.104 | 509.028 | 1173.132 | 3014.718 | 0 | 0 |
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6959/02510/02569` | gp_6959 | 2510–2569 | -543.813 | 4301.429 | 3757.616 | 4493.737 | 0 | 0 |
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6962/02790/02849` | gp_6962 | 2790–2849 | 194.366 | 793.496 | 987.862 | 1544.253 | 0 | 2 |
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6952/02185/02244` | gp_6952 | 2185–2244 | 763.729 | 929.047 | 1692.776 | 3527.350 | 0 | 1 |
| Random | `gp_set_0006/gp_set_0006_vid_0003/gp_6934/01265/01324` | gp_6934 | 1265–1324 | 495.205 | 540.495 | 1035.699 | 1858.502 | 0 | 2 |
| Largest improvements | `gp_set_0006/gp_set_0006_vid_0003/gp_6966/02675/02734` | gp_6966 | 2675–2734 | -15851.208 | 23707.025 | 7855.817 | 11818.865 | 0 | 2 |
| Largest improvements | `gp_set_0006/gp_set_0006_vid_0003/gp_6957/02705/02764` | gp_6957 | 2705–2764 | -12461.131 | 21952.559 | 9491.428 | 13378.203 | 0 | 0 |
| Largest degradations | `gp_set_0006/gp_set_0006_vid_0003/gp_6959/02450/02509` | gp_6959 | 2450–2509 | 3494.287 | 14722.082 | 18216.369 | 24800.449 | 0 | 2 |
| Largest degradations | `gp_set_0006/gp_set_0006_vid_0003/gp_6959/02720/02779` | gp_6959 | 2720–2779 | 2872.352 | 6782.172 | 9654.524 | 10238.688 | 0 | 1 |
| Gate closed | `gp_set_0006/gp_set_0006_vid_0003/gp_6933/01175/01234` | gp_6933 | 1175–1234 | 603.286 | 1172.373 | 1775.659 | 2931.520 | 0 | 2 |
| Gate closed | `gp_set_0006/gp_set_0006_vid_0003/gp_6933/01205/01264` | gp_6933 | 1205–1264 | -470.271 | 26382.738 | 25912.467 | 29467.656 | 0 | 2 |
| Gate open | None available | — | — | — | — | — | — | — | — |

Frame extraction target is the last observed frame, `frame_start + 14`, not the sample's final frame. The raw split cache points to extracted PNG paths, which are absent locally; the corresponding GP6 video file is the available source for frame extraction. If a selected frame cannot be decoded, report the missing figure rather than changing the selection.
