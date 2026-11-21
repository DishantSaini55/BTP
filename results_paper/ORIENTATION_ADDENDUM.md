# Orientation addendum - checkpoint leakage finding

**Status: BLOCKING. Do not start E1, B1, runner implementation, smoke tests, or any other experiment until the author decides how to handle this leakage.**

## Finding

The official MTN checkpoint at
`D:\IDD-PeD-Evaluation\checkpoints\mtn\mtn\epoch_latest.pth` is an official IDD-PeD training-checkpoint artifact. The official default trajectory split places the entire `gp_set_0006` set in **train**:

```python
image_set_nums = {
    'train': ['gp_set_0001', 'gp_set_0002', 'gp_set_0004', 'gp_set_0006', 'gp_set_0007'],
    'test':  ['gp_set_0003', 'gp_set_0005', 'gp_set_0008', 'gp_set_0009'],
}
```

Source: `MTN/iddped_interface_traj.py:76-91`; the official MTN data factory invokes `generate_data_trajectory_sequence('train', ...)` with `data_split_type='default'` at `MTN/baselineUtils.py:8-38` and `MTN/train_iddped.py:57-60`. The corresponding public source is the [official IDD-PeD MTN split implementation](https://github.com/Ruthvik9/IDD-PeD/blob/main/MTN/iddped_interface_traj.py).

The official annotation cache independently confirms that `gp_set_0006` comprises exactly:

- `gp_set_0006_vid_0001`
- `gp_set_0006_vid_0002`
- `gp_set_0006_vid_0003`

Source: `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\data_cache\iddp_database.pkl` (read-only inspection on 2026-11-21).

The local matched protocol uses precisely those three videos:

| Local role | Video | Relationship to official checkpoint training split |
|---|---|---|
| validation | `gp_set_0006_vid_0001` | overlap |
| local train | `gp_set_0006_vid_0002` | overlap |
| local test | `gp_set_0006_vid_0003` | **overlap / leakage** |

## Consequence

The official MTN checkpoint cannot be used to initialise any model evaluated on this local GP6 “test” video while preserving a held-out-test claim: that video belongs to the official checkpoint's training set. Fine-tuning for one epoch does not remove this leakage.

The checkpoint itself is a tensor state dictionary and contains no per-video training manifest, so it cannot prove which individual GP6 samples were seen. However, the official training code trains on the entire default `gp_set_0006` set, and the official checkpoint is supplied for that protocol. This is sufficient to treat all three GP6 videos as contaminated for the proposed evaluation.

## Decision required

Choose one of the following before any experiment proceeds:

1. **Leakage-free benchmark:** choose local train/validation/test videos that are all outside the official checkpoint training sets, or train every model from scratch with a split that is held out at video level.
2. **Warm-start study only:** retain the official checkpoint but relabel the exercise as an in-training-set adaptation analysis; do not claim held-out evaluation, benchmark comparison, or test-set generalisation.
3. **Different verified checkpoint:** use a checkpoint with a documented training manifest that excludes all selected evaluation videos.

No implementation or experiment work was performed after detecting this overlap.

## Follow-up: leakage-free official-test asset inventory (2026-11-21)

**Status: BLOCKING. The required minimum of three locally usable official-test videos is not met.**

The official default test sets are `gp_set_0003`, `gp_set_0005`, `gp_set_0008`, and `gp_set_0009` (`MTN/iddped_interface_traj.py:87-90`). Availability was checked read-only under `D:\IDD-PeD-Evaluation` before writing a split configuration or computing any flow.

| Official test set | Local pedestrian annotation XML | Local extracted images | Local video file | Flow-usable videos |
|---|---|---|---|---:|
| `gp_set_0003` | none | none | none | 0 |
| `gp_set_0005` | none | none | none | 0 |
| `gp_set_0008` | `gp_set_0008_vid_0001.xml`, `_0004.xml`, `_0005.xml` | none | none | 0 |
| `gp_set_0009` | none | none | none | 0 |

The only locally available official-test annotations are therefore the three GP8 XMLs at:

`D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0008\`

They cannot be turned into trajectory windows with Farneback flow because neither the corresponding video files under `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0008\` nor extracted image directories under `...\annotations\IDDPedestrian\images\gopro\gp_set_0008\` exist. The local annotation cache does not substitute for frame pixels.

`gp_set_0006_vid_0003.MP4` is present, but it is in the official checkpoint training split and is not an eligible leakage-free test asset.

Per the pre-registered decision rule, `results_paper/config/SPLIT.md` has **not** been written: no valid split can be frozen before the three-video availability condition is satisfied. No flow, model, baseline, runner, smoke test, or result migration has been performed after this check.

### Required author decision

Provide or authorise acquisition of at least three official-test videos with matching annotations (across at least two sets), or revise the benchmark-design requirement. The local GP8 annotations alone are insufficient.
