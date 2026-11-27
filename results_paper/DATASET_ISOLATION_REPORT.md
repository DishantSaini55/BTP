# Phase 4 Dataset Isolation Report

## Cache migration record

The Phase 4 workflow uses the role-separated cache under
`splits/gp6_from_scratch_v1/roles_v1/`. It was created once by
`MTN/migrate_flow_cache_v1.py` after explicit acknowledgement that the trusted
source pickle would be deserialized. The source was
`splits/gp6_from_scratch_v1/flow_splits.pkl` (21,682,476 bytes; SHA-256
`3c888c7c089de88bffd44349c8336b4ca9b76177402d141e990ea79a1c306001`). The
migration verified role counts, IDs, video membership, per-sample content
hashes, and copied source metadata; it recomputed no optical flow.

The migrated roles contain 212 train samples from
`gp_set_0006_vid_0002`, 43 validation samples from
`gp_set_0006_vid_0001`, and 217 test samples from
`gp_set_0006_vid_0003`. Train and validation cache hashes are:

| Role | Pickle SHA-256 | Sample-ID SHA-256 |
|---|---|---|
| Train | `48755c1c62d6d15d47cbeee4bb0b1eb93b03ea9de987d0d97b7d7816e6cdc5fb` | `08123044b774f971e94dc557cbe2114e163f6aed2678e4decd7a3763cd83005c` |
| Validation | `1add14f3a4b8bb0826a70cf45460f9b9f805216f8ba1b3da829a4484d011b0bd` | `2b71f2f246fc2f7376c0d487208c843fabea3bb7a17266d1b3e6180625a2d81c` |

The independent `test.pkl` remains present as a historical frozen role. It
must not be opened by Phase 4 screening or result finalization. Its existence
does not imply it was loaded by the Phase 4 code.

## Phase 4 access boundary

`MTN/dataset_roles_v1.py` accepts only `"train"` and `"validation"` as loader
roles, constructs paths only for those two role names, verifies their own
manifests, and fits box and speed normalization statistics on train samples
only. `MTN/run_p4_screen.py` uses only this loader and asserts that the
returned dataset keys are exactly `{"train", "validation"}`. It evaluates
training loss and validation loss only; it has no test-prediction or test
metric path. The result finalizer reads only P4 run folders, their
validation-prediction outputs, and the deterministic validation CV baseline.

The legacy monolithic loader remains unchanged. In particular,
`run_from_scratch_e1.load_data`, `run_e8_v2.load_split`, `run_e3.py`, and
`run_e3_v2_diagnostics.py` still deserialize the monolithic flow cache; they
are not called by the Phase 4 screening runner. The migration utility is the
only Phase 4 setup step that reads the monolithic source, and its one-time
full-role deserialization is explicitly recorded in
`roles_v1/migration_manifest.json`.

## Testless-cache execution check

Infrastructure validation copied only the train and validation pickles and
their manifests to a temporary cache directory. That directory contained
exactly these four files and no test role. All six capped smoke configurations
completed, followed by full-split one-epoch benchmark runs. The temporary
directory was removed after the successful checks. The independent test cache
was not opened for either screening validation.

The capped smoke run used only two train and two validation batches per model;
its losses are infrastructure checks and are not full-split accuracy results.
The benchmark used all 212 training and 43 validation windows for one epoch
per model. Neither run accessed test predictions, test metrics, or test
normalization statistics.

## Limitations

- The original trusted migration necessarily deserialized all three roles
  once. Role separation prevents subsequent Phase 4 screening code from
  reading the test artifact; it does not retroactively change that migration
  event.
- Historical E1/E3/E8/P2/P3 loaders are not made test-cache-safe by this
  migration. Their existing behavior and outputs are preserved.
- Validation comprises one video and 43 overlapping windows. Any
  track-cluster uncertainty interval must carry the one-video caveat.
