# Official-Test Membership Plan

## Authoritative membership

The authoritative dataset sources assign GP1, GP2, GP4, GP6 and GP7 to
training/development material and GP3, GP5, GP8 and GP9 to the official
test partition:

- Hugging Face dataset README:
  https://huggingface.co/datasets/Ruthvik9/IDD-PeD
- Official repository:
  https://github.com/Ruthvik9/IDD-PeD

The repository states a roughly 70/30 train/test split and names GP3,
GP5, GP8 and GP9 as test sets.

## Approved test scope

After explicit authorization, use all eligible annotation-only windows from
GP3, GP5, GP8 and GP9 where complete 60-frame tracks and required bounding
boxes are available.

Use only the frozen 15/45 observation/prediction protocol and stride 30.
Do not add, remove or replace windows after reviewing model performance.

## Complete-data fallback

If complete annotations for all official test sets are unavailable, the
fallback subset must be frozen before test access:

1. Prefer complete official test videos from at least two official test sets.
2. Require at least 15 complete observed frames and 45 following frames for
   every window.
3. Freeze the exact video IDs and window count solely from availability,
   independent of model predictions.
4. Report all unavailable sets and videos.
5. Do not use a subset chosen by predictive accuracy.

No subset is approved yet.

## Historical GP6 exclusion

The local GP6 split is a historical development protocol: GP6-0002 was
training, GP6-0001 validation and GP6-0003 the local test role. That GP6
test video is not the official held-out IDD-PeD test partition and must not
be described as official-test evidence.

## Current authorization state

Official test membership is identified from authoritative documentation.
No official test annotation or cache has been opened. No test subset is
accessed or frozen from values. Explicit authorization is required before
opening held-out records.
