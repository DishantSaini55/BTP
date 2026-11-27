# Phase 5A acquisition feasibility report

**Status: stopped at the pre-download approval gate.** No video archive was downloaded or extracted.

## Finding

The proposed smallest multi-video pilot is plausible from annotation counts but is not yet a safe acquisition: the official video host failed DNS resolution during all four metadata probes. Archive sizes, redirects, range support, resumability, checksums, and extracted-video sizes remain unknown. D: has 70.041 GiB free, but the peak storage needed cannot be calculated without the remote sizes. Do not start a large download on the evidence currently available.

## Candidate acquisition plan

| Order | Archive | Intended use | Remote size | Action |
|---:|---|---|---|---|
| 1 | `gp_set_0001.tar` | Extract GP1 videos 0001 and 0003 for training | Unknown | Do not download before a fresh successful probe and storage approval |
| 2 | `gp_set_0007.tar` | Extract GP7 videos 0004 and 0005 for validation | Unknown | Do not download before a fresh successful probe and storage approval |

The candidate provides 506 windows under the historical `crossing_motive` rule or 520 without it for training when combined with local GP6 video 0002. GP7 videos 0004 and 0005 provide 228 and 204 unfiltered windows respectively (432 total); the legacy filter retains 228 from video 0004 and none from video 0005. These are annotation-derived estimates, not verified usable-video counts. The earlier video-0001 reference was incorrect; the available GP7 annotations are for videos 0004 and 0005. See [MINIMUM_DATASET_EXPANSION_PLAN.md](./MINIMUM_DATASET_EXPANSION_PLAN.md).

**Storage estimate:** unknown. The two full archive sizes and the four selected MP4 sizes are unknown. Selected annotation XMLs total 47,379,091 bytes by remote metadata. Preserve archives for provenance unless an explicit later policy says otherwise; also reserve space for extraction and a safety margin. Current D: free space is 75,205,443,584 bytes (70.041 GiB), which is not sufficient evidence to approve an unknown total.

**Extraction:** after successful transfer, run the utility's `inspect` command first. It reads the tar and records a local SHA-256; it does not extract. Only after inspection should selected MP4 member paths, taken exactly from the inventory, be passed to `extract`. Do not trust an assumed internal path or extract the whole archive by default.

## Remote and corruption evidence

HEAD and one-byte Range probes for GP1, GP2, GP4, and GP7 returned no HTTP response because `cvit.iiit.ac.in` could not be resolved (curl exit 6, status `000`). No server-side size, redirect, validator, range, or checksum information was observed. Full rows are in [REMOTE_ARCHIVE_INVENTORY.csv](./REMOTE_ARCHIVE_INVENTORY.csv).

GitHub issue [Ruthvik9/IDD-PeD#4](https://github.com/Ruthvik9/IDD-PeD/issues/4) reports extraction failures for GP1–GP7 after retries/resume. This complaint has not been independently reproduced against current archives and is not proof of current archive corruption. The official repository's download script extracts and removes downloaded archives without recording integrity metadata. The acquisition utility instead preserves the archive, verifies tar readability, refuses overwrites, supports validator-aware resume, and records a local SHA-256; no trusted checksum is known.

## Alternatives if the official server remains unavailable

- Retry metadata-only probes later from a network that can resolve the official host; do not infer availability from the annotation host.
- Ask the dataset maintainers for an official mirror, current archive sizes, and trusted checksums.
- Ask the authors for access to selected official-train videos or a documented transfer method.
- Re-scope the pilot to locally available GP6 data only and label it single-video; this does not meet the multi-video diversity objective.

No verified alternate raw-video mirror was found. Hugging Face annotation availability is not an alternative video source. Do not use unofficial mirrors without provenance and integrity evidence.

## Utility and commands

Added [download_iddped_archives.py](../MTN/download_iddped_archives.py), with focused tests in [test_download_iddped_archives.py](../tests/test_download_iddped_archives.py). It only permits official development sets GP1/2/4/7 and D: destinations. Downloading requires both an explicit byte cap and `--execute`; partial transfers are retained for safe resume. The utility reports elapsed time and average speed, checks expected sizes when known, inspects tar readability before completion, and refuses to overwrite existing complete files.

Metadata-only probe (safe to repeat):

```powershell
python MTN\download_iddped_archives.py probe --sets 0001 0007
```

Only after fresh probes establish exact sizes, storage is approved, and an explicit per-archive cap is chosen:

```powershell
python MTN\download_iddped_archives.py download --sets 0001 --destination D:\IDD-PeD-Evaluation\archives\phase5a --max-bytes <approved-GP1-byte-cap> --execute
python MTN\download_iddped_archives.py download --sets 0007 --destination D:\IDD-PeD-Evaluation\archives\phase5a --max-bytes <approved-GP7-byte-cap> --execute
python MTN\download_iddped_archives.py inspect --archive D:\IDD-PeD-Evaluation\archives\phase5a\gp_set_0001.tar
python MTN\download_iddped_archives.py inspect --archive D:\IDD-PeD-Evaluation\archives\phase5a\gp_set_0007.tar
```

These are templates, not an authorization to run. The two caps and selected extraction members cannot safely be filled in until remote sizes and tar member paths are verified. No `download`, `inspect`, or `extract` command was run in Phase 5A.

## Approval gate and remaining decisions

Before any archive download, obtain fresh remote metadata, calculate peak storage for both full archives plus selected extraction and annotations, and get explicit user approval for the archive names, byte caps, and destination. Separately decide the outcome-related `crossing_motive` eligibility issue and reconcile GP6 annotation provenance with the pinned Hugging Face revision. The train/validation split remains unfrozen.

No training, optical-flow generation, test-cache access, `paper/` edit, commit, or push occurred.
