# Phase 5B annotation availability

## Result

The local nested `reference_IDD-PeD` repository contains individual pedestrian and vehicle/OBD XML annotations for the selected development videos. No acquisition was needed. The development annotation source repository was recorded at revision `62466ce24cc30b665825864f197df39782fb087c`.

The machine-readable [ANNOTATION_INVENTORY.csv](./ANNOTATION_INVENTORY.csv) contains 20 candidate-video rows: 19 development annotation files were parsed, and the historical GP6 video-0003 test-role row contains filesystem metadata only. Its XML contents, frame records, or caches were not opened. Official test-set records from GP3, GP5, GP8, and GP9 were not queried.

The selected GP6 video-0002 pedestrian and OBD XMLs in `D:\IDD-PeD-Evaluation` were previously verified byte-identical to the nested reference-repository copies used here.

## Schema and quality checks

- Pedestrian boxes use CVAT `xtl, ytl, xbr, ybr` pixel coordinates and zero-based frame indices.
- CVAT track IDs identify annotation tracks. Manually entered pedestrian IDs are not used as track keys because some are missing or duplicated.
- A valid window contains 60 consecutive in-frame, finite, positive-area boxes; it is split into 15 observed and 45 future frames. Invalid/out-of-frame boxes break continuity. Window stride is 30 frames.
- Selected video annotations include full-frame OBD-speed records with zero-based frame coverage. Speed units are not documented consistently enough to use as an input; ego speed is excluded.
- Pedestrian boxes carry a per-box `occluded` flag, and nested attributes include a categorical `occlusion` label. The two forms disagree in some records and are excluded from the primary model inputs.
- GP2 video 0001 contains 6 out-of-bounds boxes and GP4 video 0001 contains 17. Those records are excluded and break their tracks. No invalid-coordinate records were found for the selected videos.
- GP7 video 0005 has no POI tracks or matched static scene labels. The GP7 validation split therefore cannot support a complete, matched scene-context evaluation.
- Annotation fields describing behavior, occlusion, or future trajectory state are not used as inputs.

## Frozen candidate membership and counts

The membership was selected by annotation availability and video-level separation, before model comparison:

| Role | Video | Eligible tracks | Valid windows |
|---|---|---:|---:|
| Train | GP1 video 0001 | 25 | 68 |
| Train | GP2 video 0001 | 157 | 555 |
| Train | GP4 video 0001 | 34 | 149 |
| Train | GP6 video 0002 | 45 | 216 |
| **Train total** | **4 videos** | **261** | **988** |
| Validation | GP7 video 0004 | 31 | 228 |
| Validation | GP7 video 0005 | 89 | 204 |
| **Validation total** | **2 videos** | **120** | **432** |

These are overlapping windows, not independent samples. Distinct video and track counts are retained with all evaluation tables. GP6 video 0003 is excluded because it is the historical held-out video in the existing GP6 protocol.

## Provenance and boundaries

Train and validation caches are separate NPZ files under `splits/ao_annotation_only_v1/`. Their manifest records source XML SHA-256 values, per-role counts and sample-ID digests, exact video membership, and `official_test_sets_read: []`. The runner reads only these two caches and rejects a manifest that does not certify the annotation-only schema and historical GP6-test exclusion.

No raw videos were downloaded, no optical flow was generated, no monolithic cache was opened, and no training/test-role cache or official-test annotation record was read for this inventory.
