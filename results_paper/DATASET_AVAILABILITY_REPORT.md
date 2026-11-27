# Phase 5 dataset availability report

**Status: blocked at feasibility.** The current local copy does not support a clean multi-video pilot. No new split, cache, flow extraction, training, or download was started.

## Inventory scope and safeguards

The inventory is a filesystem metadata check of the local dataset root `D:\IDD-PeD-Evaluation`. It records video/archive paths, byte sizes, annotation filenames, and whether corresponding raw videos exist. Video headers cited below came from the earlier container-metadata check; no full-video decode was performed for this report.

No experiment test cache was opened or deserialized. The local annotation database pickle was listed by path and size only. No test-derived values, sample windows, statistics, or flow features were generated.

The complete scoped file inventory is [DATASET_INVENTORY.csv](./DATASET_INVENTORY.csv).

## What is available locally

- The dataset root contains 125 files totalling 8,495,455,289 bytes at inventory time. The D: volume had 75,205,439,488 free bytes at that time; free space is dynamic.
- Three extracted GP6 MP4s are present and total 4,136,590,373 bytes. Previously observed container headers reported:

| Video | Bytes | Container metadata |
|---|---:|---|
| `gp_set_0006_vid_0001.MP4` | 1,945,038,178 | 1920x1440, 30 fps, 21,210 frames |
| `gp_set_0006_vid_0002.MP4` | 2,013,735,682 | 1920x1440, 30 fps, 21,210 frames |
| `gp_set_0006_vid_0003.MP4` | 177,816,513 | 1920x1080, 30 fps, 5,312 frames |

  Opening the containers and reading their headers does not establish that every frame decodes successfully.
- A `gp_set_0006.tar` archive is present (4,136,593,920 bytes). Prior member-name/size inspection found the three GP6 video names and sizes, but byte-for-byte archive-to-extracted-file identity has not been verified.
- Local pedestrian XMLs cover GP6 videos 0001/0002/0003 and GP8 videos 0001/0004/0005. Local OBD XMLs cover those same six video IDs. Two annotated-frame index CSVs are present for GP6 and GP8.
- GP8 has per-video pedestrian and OBD XMLs but none of its three locally annotated raw videos. No raw videos or per-video pedestrian/OBD XMLs for GP1, GP2, GP3, GP4, GP5, GP7, or GP9 were found in the filesystem inventory. The annotation database pickle was not deserialized, so this statement does not characterize its contents.

## Why the existing GP6 material is not a clean new pilot

The frozen historical split assigns GP6 video 0002 to train, video 0001 to validation, and video 0003 to test ([SPLIT.md](./config/SPLIT.md)). That experiment is a one-video-per-role GP6 study; its existing held-out video is not an untouched new multi-video test set. Quarantine GP6 video 0003 from any claim of a newly acquired or untouched final test.

The official IDD-PeD source documentation assigns GP1/2/4/6/7 to its training pool and GP3/5/8/9 to its test pool ([Hugging Face dataset README](https://huggingface.co/datasets/Ruthvik9/IDD-PeD), [source repository](https://github.com/Ruthvik9/IDD-PeD)). GP8 annotations alone cannot substitute for its absent videos. Keep every official test set untouched for the prospective development-only pilot.

The current P4 cache is a derivative of the historical GP6 split. It does not provide additional videos and cannot be treated as expanded or independent evidence. Existing P4 dataset-isolation constraints remain documented in [DATASET_ISOLATION_REPORT.md](./DATASET_ISOLATION_REPORT.md).

## Download feasibility

The official video archive naming pattern is `gp_set_XXXX.tar`; the source links are listed in [DATASET_EXPANSION_PLAN.md](./DATASET_EXPANSION_PLAN.md). The IIIT host failed DNS resolution during the earlier feasibility investigation. No archive lengths could therefore be confirmed from that host, and no download has been attempted in this phase.

The older local download script contains a hard-coded GP8 total-size value of 2,382,336,512 bytes. It is not independently verified and must not be used as an authoritative length. The previous GP8 attempt stopped after ranges through byte 201,326,591; neither the partial chunk nor a completed GP8 archive is present.

## Selection and preprocessing cautions

The existing GP6 sample materializer filters tracks on the presence of the track-level `crossing_motive` attribute ([prepare_from_scratch_gp6.py](../MTN/prepare_from_scratch_gp6.py)). That is a potential outcome-conditioned selection rule and should not be carried into an expanded pilot without an explicit justification and a frozen eligibility definition.

Scene fields are taken from track-level POI annotations; observation-time availability of `motion_direction` and `location_type` has not been established. Do not rely on them in a primary context comparison until their annotation timing is audited. The historical flow pipeline stores 60 time positions although the model consumes only the 14 flow pairs from the 15 observed frames; any future cache design should be observed-only and separately versioned.

## Decision

**Do not launch a multi-video pilot from the present local copy.** It lacks the additional raw video assets required for the proposed train/validation roles, and the available GP6 test video is historical rather than a new untouched holdout. The next step is a user-approved download and verification of the proposed official-train archives. The split and protocol in the expansion plan are drafts only.
