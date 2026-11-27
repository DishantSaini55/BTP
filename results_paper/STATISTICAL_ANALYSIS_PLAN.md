# Phase 6B Statistical Analysis Plan

## Predeclared comparisons

Report absolute and relative differences for:

1. AO-2 versus AO-1.
2. AO-2 versus AO-0.
3. AO-3 versus AO-0.
4. AO-3 versus AO-2.
5. AO-RIDGE versus AO-0.
6. AO-RIDGE versus AO-2.
7. AO-RIDGE versus AO-3.

Negative error differences favor the first-listed model.

## Primary inferential unit

Each neural seed is a repeated initialization/training realization on the
same test sample membership. Seed-level paired differences are summarized
with sample SD and t intervals when five seeds are complete.

Five seeds are not five independent test datasets and are not multiplied
by overlapping window counts.

## Clustering

The bootstrap resampling unit is the two-dimensional video-and-track
cluster `(video_id, track_id)`, with all windows from that track sampled
together. Overlapping windows within the cluster are not independently
resampled.

Bootstrap estimates are conditional on the observed official test videos
and tracks. They do not establish population-level video generalization.

## Fresh-seed sensitivity

The full five-seed table and a sensitivity table excluding resumed
seed 20260928 are required.

## Multiple comparisons

The seven comparisons are prespecified. Exploratory secondary metrics and
horizon/per-video/per-track analyses are descriptive unless a corrected
analysis is separately justified. No equivalence claim is permitted.
Claiming equivalence would require a prespecified practical margin and an
appropriate two-sided equivalence procedure.

## Limitations

Seed-level t intervals assume paired approximately symmetric differences.
Track bootstrap intervals may be unstable if cluster counts are small.
Video-level generalization cannot be established with a small fixed video
set. Ridge alpha was validation-selected and must remain descriptively
marked in official-test interpretation.
