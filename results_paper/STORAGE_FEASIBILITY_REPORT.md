# Phase 5A storage feasibility

## Snapshot

Metadata-only snapshot of `D:\IDD-PeD-Evaluation` and D:; no files were deleted.

| Item | Observed |
|---|---:|
| D: free space | 75,205,443,584 bytes (70.041 GiB) |
| Existing dataset directory | 8,495,455,289 bytes across 125 files (7.912 GiB) |
| Extracted GP6 videos | 4,136,590,373 bytes (3.852 GiB) |
| Local GP6 tar | 4,136,593,920 bytes (3.853 GiB) |
| Flow directory files | 3,742,182 bytes (3.57 MiB) |

The GP6 archive lists the same three member names and sizes as the extracted GP6 MP4s. The size difference is 3,547 bytes, consistent with tar overhead, but no byte-for-byte comparison or content hashes were performed. Treat archive/video duplication as a strong candidate, not a verified duplicate.

The prior GP8 partial tar and chunk files are absent. The retained transfer log records byte ranges only through 201,326,591; the log is evidence of a prior partial transfer, not proof of a complete or resumable local archive.

## Local annotations and flow

The local tree contains GP6 and GP8 pedestrian/OBD annotation material. GP8 annotations do not have matching raw videos in the inventoried local video directory. GP1/2/4/7 annotation XMLs were streamed from Hugging Face for counts during this feasibility work; they were not saved locally. Metadata for the matching GP1/2/4/7 annotation files totals 334,895,281 bytes (about 319.38 MiB); this is not part of the current local dataset size.

Flow cache files were inventoried by names and filesystem sizes only. The files `flow\cpu_farneback_poc\train.pkl`, `validation.pkl`, and `test.pkl`, and `flow\cpu_farneback_c4_larger\test_full.pkl` were not opened or deserialized. The test-named cache files remain untouched.

## Remote-size and integrity feasibility

The metadata-only HEAD and one-byte Range probes for GP1, GP2, GP4, and GP7 all failed DNS resolution at `cvit.iiit.ac.in` (curl exit 6; reported HTTP status `000`). No redirect or HTTP response was received. Therefore each archive's `Content-Length`, range support, ETag/Last-Modified validator, resumability, and remote checksum are **unknown**, not negative or verified.

The four records are in [REMOTE_ARCHIVE_INVENTORY.csv](./REMOTE_ARCHIVE_INVENTORY.csv). The public report [Ruthvik9/IDD-PeD#4](https://github.com/Ruthvik9/IDD-PeD/issues/4) describes GP1–GP7 downloads that fail extraction after retries/resume. It is an unresolved user report, not independent evidence that the present remote objects are corrupt. The official repository download script uses `wget`, extracts archives, and deletes them without recording an expected size or checksum; do not use it as an integrity-verification procedure.

No trusted archive checksums or reliable alternate video mirror were identified. The Hugging Face source provides annotations; its working annotation range response does not establish video-host availability or integrity.

## Capacity conclusion

D: currently has about 70 GiB free, but sufficiency for GP1 and GP7 cannot be determined until archive and selected-MP4 sizes are obtained. The minimum candidate needs the complete GP1 and GP7 tar archives for acquisition, even if only selected MP4 members are extracted. If archives are retained for provenance, peak additional storage is approximately:

`size(gp_set_0001.tar) + size(gp_set_0007.tar) + size(selected extracted MP4s) + size(selected annotation XMLs) + safety margin`

The selected GP1/GP7 annotation XMLs are estimated from remote metadata at 47,379,091 bytes (about 45.2 MiB). All video terms and a safe margin remain unknown. If the official archives must be fully extracted, replace the selected-MP4 term with the full extracted-set sizes; those are also unknown. No claim that the available space is sufficient is justified yet.

## Safety status

- No archive download, extraction, optical-flow generation, training, or test-cache deserialization occurred.
- The new downloader refuses to overwrite a complete archive, retains partial transfers, enforces an explicit byte cap, and calculates local SHA-256 after a complete readable tar is obtained. A local hash is provenance only; it proves no match to an unavailable trusted digest.
- Extraction is a separate explicit operation, validates tar readability first, and is limited to caller-selected MP4 members.
- A new transfer should not be authorized until fresh probes return usable metadata, an explicit size cap/space plan is agreed, and archive readability is checked before extraction.
