# MTN scene-context ablation

Two compatible models are available.

| Variant | Inputs | Fusion |
| --- | --- | --- |
| `IndividualTF` | observed boxes, OBD speed, ego optical flow, pedestrian optical flow | Original MTN transformer cross-attention |
| `ContextIndividualTF` | Baseline inputs plus scene context and observation-period occlusion | Scene/occlusion encoder emits one decoder-memory token; a per-sample gate admits or skips it |

The context vector has this fixed order:

`intersection_type, signalized_type, road_type, location_type, motion_direction, time_of_day`

Occlusion is a separate sequence over only the 15 observed frames. Valid levels are `0` none, `1` partial, `2` full; `3` means unavailable. Every scene field likewise reserves a final unknown category. IDD-PeD uses day/night as the illumination label; it does not expose a separate illumination category.

Pass time-of-day labels as `--time_of_day_map labels.json`, where the JSON maps each video id (for example, `gp_set_0003_vid_0005`) to `"day"`/`"night"` or `0`/`1`. This avoids guessing illumination from future frames. Videos absent from the map use the explicit unknown value.

The gate is inspired by DynMM's modality-level decision: it can skip the extra context token for samples where trajectory/flow evidence suffices. It is trained with a straight-through Gumbel gate (`gate_mode="hard"`) or can be run as a soft ablation (`gate_mode="soft"`). `--context_cost_weight` adds DynMM's resource-aware penalty for using the extra branch; set it to `0` for a pure accuracy ablation.

For a fair experiment, train each model with identical splits, seed, optimisation schedule, and observed inputs. Report MSE at 1.5 s plus slices for night/day, occluded/unoccluded, and signalized/unsignalized scenes. Do not feed future occlusion labels or labels derived from the future trajectory.
