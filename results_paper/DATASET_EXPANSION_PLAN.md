# Dataset expansion plan (draft; not frozen)

## Gate

Do not download, extract, build a split/cache, compute flow, or train until the user explicitly approves dataset acquisition and the draft development protocol. The current local-data feasibility result is **no-go** for a multi-video pilot.

## Exact source archives

The official source documents link video archives at:

`https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_XXXX.tar`

They document pedestrian and vehicle/OBD annotations at:

- `https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations.tar`
- `https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations_vehicle.tar`

Prospective archive filenames are:

| Purpose | Exact archive filename | Official pool | Present locally |
|---|---|---|---|
| Development training | `gp_set_0001.tar` | train | no |
| Development training | `gp_set_0002.tar` | train | no |
| Development training | `gp_set_0004.tar` | train | no |
| Development validation | `gp_set_0007.tar` | train | no |
| Annotations | `annotations.tar` | annotations | no separately identified archive |
| OBD annotations | `annotations_vehicle.tar` | annotations | no separately identified archive |

The source README lists GP1/2/4/6/7 as official training sets and GP3/5/8/9 as official test sets. The draft above omits GP6 because the local historical split already used its three videos and the aim is an additional multi-video development pilot. It deliberately omits every official test-set video.

The exact video-archive sizes are **unverified**. Earlier DNS resolution for the IIIT host failed. Do not rely on the hard-coded GP8 size in `MTN/download_set0008_chunks.ps1`; it has not been independently checked. No download is authorized or in progress.

## Proposed development-only split

This is a proposal, not a frozen split:

- Train on all eligible, verified videos from GP1, GP2, and GP4.
- Validate on all eligible, verified videos from GP7.
- Exclude GP6 historical runs/caches from this pilot.
- Keep GP3, GP5, GP8, and GP9 completely untouched. Do not compute test statistics, select samples, extract test flow, or evaluate on them.
- Do not describe validation as a final test or claim generalisation to official test sets.

Before any split is frozen, inspect the downloaded archive manifests and annotation coverage. If a candidate set lacks matching pedestrian and OBD annotation coverage, stop and revise the proposal with approval; do not silently substitute videos.

## Required pre-cache verification after approval

1. Resolve and verify source byte lengths/checksums when available; retain the exact retrieval URL, timestamp, archive size, and SHA-256. Do not accept a partial response as a completed archive.
2. List archive members without extraction, check for path traversal and unexpected contents, then extract only approved official-train sets into a new versioned directory.
3. Match each MP4 to pedestrian and OBD annotation records; record video dimensions, frame rate, frame count, decode errors, annotation coverage, and checksums. Container metadata alone is not full decode validation.
4. Freeze an eligibility rule independent of future outcomes. Review the current `crossing_motive`-presence filter before reuse; verify which scene annotations are genuinely available at observation time.
5. Create a versioned split manifest with whole-video or whole-set roles, ordered sample IDs, source checksums, and disjointness assertions. Have the user review it before flow computation.
6. Generate only the 14 observed-frame flow pairs required by the 15-frame input, into a new cache namespace. Do not modify or reuse the historical GP6 cache as if it represented this expanded dataset.
7. Record compute/storage requirements from a bounded, approved pilot on one training video before scaling to the remaining approved videos. Do not infer full-dataset completion time from the earlier single-window probe.

## Stop conditions

Stop before training if source sizes/checksums cannot be established, archives are incomplete, annotations do not match the videos, decode integrity fails, the split cannot keep videos disjoint, or outcome-conditioned eligibility cannot be resolved. Report the exact blocker and preserve all existing historical results.

## Current status

No archive has been downloaded in this phase. No new split, cache, manifest, flow, or training output exists. This draft must be reviewed and approved before execution.
