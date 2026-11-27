# Annotation-only Phase 5B protocol v1

## Frozen split

The split is frozen in `splits/ao_annotation_only_v1/split_manifest.json`. It is independent of model predictions and contains four train videos and two validation videos:

- Train: GP1 video 0001, GP2 video 0001, GP4 video 0001, GP6 video 0002.
- Validation: GP7 videos 0004 and 0005.
- Official test partition: untouched; manifest `official_test_sets_read` is empty.
- Historical GP6 video 0003: excluded; only its file-presence and byte-size metadata were recorded in the inventory.

The split has 988 training windows from 261 tracks and 432 validation windows from 120 tracks. Overlapping windows are not treated as independent tracks or videos.
The exact sample IDs are stored in the separate role NPZ caches; the ordered train and validation sample-ID SHA-256 digests are `9b3c69920c67f33836fd4811da045b8905fbcdb793950ffb69aea1a877fc82cf` and `eb78190af81de2588ae222bd81472bcf36fd25979f19dae0eb105e56d4d783ee`, respectively.

## Window and coordinate protocol

- GoPro camera: 30 FPS, as identified for GoPro Hero 8 in the IDD-PeD dataset description.
- Input: 15 observed frames of absolute pixel-space `xyxy` bounding boxes.
- Target: the immediately following 45 frames from the same track.
- Window length: 60 consecutive valid frames; stride: 30 within each contiguous valid segment.
- Outside, malformed, non-finite, non-positive-area, and out-of-frame boxes are ineligible and split a segment. No interpolation is performed.
- Track identity is the CVAT annotation track ID, not the manual pedestrian ID field.
- The legacy `crossing_motive`-presence filter is not used. It conditions inclusion on an event-related annotation and would remove GP7 video 0005's windows, collapsing validation to one video. The 60-frame/stride-30 rule is retained, with valid-box continuity made explicit.

## Normalization and comparison

Coordinate-wise mean and standard deviation are fitted only to unique box frames in the training role (observed and training targets). The same train-only transform is used for model inputs, targets, and validation; reported pixel metrics inverse-transform predictions and targets. Model initialization uses seed `20260928`; Adam learning rate is `1e-3`, batch size 8, four CPU threads by default, and at most ten epochs. Validation normalized MSE selects the checkpoint; no test evaluation is performed.

Raw-pixel MSE averages squared error across the 45 future frames and four `xyxy` coordinates. C-MSE averages squared center-coordinate error across all 45 steps; CF-MSE is the final-step center-coordinate error. Horizon MSE is also reported at steps 1, 15, 30, and 45.

## Cache and provenance

`train.npz` and `validation.npz` are independent files. The split manifest includes source annotation hashes and sample-ID digests. The runner verifies role membership, shapes, disjoint source videos, and manifest isolation before loading data. The runner never constructs a path to the historical test cache.

## Scope

This is a new annotation-only protocol, not a reproduction of historical full-input MTN or PACE-Net. It has no optical-flow input and no verified ego-speed input. Predictions and conclusions are limited to the selected videos and one exploratory seed.
