# Phase 3 validation plot selection

## Selection rule

Selection (frozen before plotting): use only the fixed validation split and seed 20260928. Select six windows by a fixed RNG seed (20260928), choosing at most one window per pedestrian; then select the two largest P3-E improvements and two largest degradations versus P3-A constant velocity by per-window raw-pixel MSE. Ties are resolved by lexicographic sample ID. No image or plot is viewed before this rule and exact IDs are written. P3-E uses hard inference. This is descriptive selection, not model selection.

## Selection (frozen before plotting)

```json
{
  "random_distinct_pedestrians": [
    "gp_set_0006/gp_set_0006_vid_0001/gp_1769/01596/01655",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1694/00571/00630",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1710/00686/00745",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1764/01553/01612",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1783/02193/02252",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1840/03341/03400"
  ],
  "largest_validation_improvements_over_cv": [
    "gp_set_0006/gp_set_0006_vid_0001/gp_1667/00203/00262",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1646/00066/00125"
  ],
  "largest_validation_degradations_vs_cv": [
    "gp_set_0006/gp_set_0006_vid_0001/gp_1792/02614/02673",
    "gp_set_0006/gp_set_0006_vid_0001/gp_1710/00746/00805"
  ]
}
```
