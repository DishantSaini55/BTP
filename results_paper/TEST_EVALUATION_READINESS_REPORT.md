# Phase 6B Test-Evaluation Readiness Report

## Readiness state

`NOT_AUTHORIZED`: official-test preparation is complete, but held-out test
records remain untouched.

## Verified prerequisites

- Phase 6A neural matrix completed 15/15 runs.
- AO-CP/AO-RIDGE baseline records are present.
- 20 Phase 6A completed experiment records reconcile to 15 neural runs,
  5 constant-velocity records and one constant-position/ridge baseline
  record each.
- Selected neural checkpoints exist for AO-1, AO-2 and AO-3 across five
  seeds.
- Frozen checkpoint hashes are listed in `FINAL_CHECKPOINT_HASHES.csv`.
- The ridge coefficients are fixed at alpha 100.
- Final report, runtime, latency and metadata artifacts exist.
- 21 AO regression tests passed.
- Python compilation passed for the Phase 6A and Phase 6B preparation code.

## Evaluation implementation

`MTN/official_test_evaluator_v1.py`:

- Loads only frozen best-model or coefficient artifacts.
- Uses `torch.no_grad()` and `model.eval()`.
- Rejects model/seed/parameter/SATIE mismatches.
- Produces identical sample order within a cache.
- Has no optimizer or gradient-update path.
- Refuses `--allow-official-test` until separately authorized.

## Test annotation requirements

Before approval, the test source must provide:

- Bounding-box XML or equivalent per-pedestrian tracks.
- Frame indices and continuous track membership.
- Video IDs for GP3, GP5, GP8 and GP9.
- Coordinates in absolute pixel `xyxy` form.
- 60-frame eligible contiguous segments with 30-frame stride.

Official test archives or annotation completeness remain unverified locally.
Raw image/video assets are not required for annotation-only prediction.

## Expected resources

No test size is claimed. The numerical cache size can only be measured after
the test annotations are authorized and a fixed subset/member manifest is
materialized. CPU sequential evaluation is expected to require checkpoint
loading, 432-plus synthetic/development validation tests and per-sample
aggregation; official-test storage is not expected to require optical flow
or extracted PNG frames.

## Unresolved limitations

- Official test annotations or caches are unavailable or not authorized in
  the current workspace.
- GP8/GP9 and GP3/GP5 membership counts cannot be reported yet.
- Test window eligibility cannot be verified before access.
- Seed 20260928 remains a predeclared resumed checkpoint case.
- Ridge alpha was selected on GP7 validation and is not test-selected.
- The frozen matrix covers only two GP7 validation videos.
