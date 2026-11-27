# Minimum multi-video development pilot (not frozen)

## Scope and guardrails

This is a planning estimate only. No split is frozen, and no model accuracy was used to select videos. Keep official test sets GP3, GP5, GP8, and GP9 untouched. The estimates use the official-train annotation files for GP1/2/4/7 at Hugging Face revision `9be5757b550bc2e60581c1af2b6ce4b3a78bfd93`; only train-set annotation XMLs were streamed. The local GP6 estimate comes from the existing local annotations and may not match the Hugging Face annotation revision.

The existing sample-building rule uses 60-frame tracks, stride 30, and complete OBD speed coverage. A historical `crossing_motive`-presence filter materially changes counts and may condition inclusion on an outcome-related attribute. Both filtered and unfiltered counts are reported; eligibility must be resolved before freezing a split.

## Annotation-derived estimates

| Set | Matched annotated videos | Pedestrian tracks | Tracks with >=1 usable window | Usable windows, no `crossing_motive` filter | Usable windows, legacy filter |
|---|---:|---:|---:|---:|---:|
| GP1 | 7 | 1,416 | 566 | 1,701 | 1,687 |
| GP2 | 4 | 463 | 287 | 1,136 | 1,135 |
| GP4 | 4 | 961 | 739 | 3,311 | 3,203 |
| GP7 | 2 | 174 | 120 | 432 | 228 |

Window counts are overlapping windows, not independent observations. One GP2 OBD annotation file has no matching pedestrian annotation file and is excluded from matched-video counts.

## Small candidate meeting the initial volume target

**Candidate training videos**

- Local GP6 video 0002: 43 tracks with usable windows, 212 windows.
- GP1 video 0001: 68 windows without the legacy filter; 62 with it.
- GP1 video 0003: 240 windows without the legacy filter; 232 with it.
- Candidate total: 520 unfiltered or 506 with the legacy filter, across three training videos.

This reaches the approximate 500-window initial target with only two additional GP1 videos, but GP6 uses a local annotation version while GP1 counts use the pinned Hugging Face revision. Reconcile annotation versions and validate video/annotation correspondence before counting it as one coherent dataset.

**Candidate validation videos**

- GP7 videos 0004 and 0005: 432 windows without the legacy filter across two videos (228 and 204 respectively).
- The legacy filter retains 228 windows, all from video 0004; video 0005 contributes 204 windows unfiltered and zero under that filter.

The two-video validation option is preferable for video diversity, but only if the eligibility rule is set without reference to model results. The legacy filter would collapse validation to one video.

## Acquisition and storage

The minimum candidate requires `gp_set_0001.tar` and `gp_set_0007.tar`; only GP1 videos 0001/0003 and GP7 videos 0004/0005 need extraction if the tar member layout supports selection. Exact archive and MP4 sizes are unknown because the video host failed DNS resolution. Selected GP1/GP7 annotation XML metadata is 47,379,091 bytes; actual copies are not currently local. The earlier GP7 video-0001 reference was incorrect; the available validation annotations are for videos 0004 and 0005.

Do not download or freeze the split until:

1. Fresh HEAD and one-byte Range probes establish current accessibility, content length, redirects, and validator behavior.
2. The archive sizes and D: peak-storage requirement are calculated.
3. Selected archive members are listed and verified after full archive-readability inspection.
4. The team resolves whether the `crossing_motive` presence rule is retained, replaced by an outcome-independent eligibility criterion, or reported in parallel.
5. GP6 and Hugging Face annotation versions are reconciled.

The proposal intentionally does not include GP2/GP4 in the minimum acquisition. If GP1 fails, a new estimate can be made from GP2 or GP4 annotations, but their larger candidate sets do not make video acquisition feasible until their archive sizes and availability are known.

## Status

**Unfrozen; acquisition not approved or started.** No accuracy-based selection, split generation, flow-cache creation, or training was performed.
