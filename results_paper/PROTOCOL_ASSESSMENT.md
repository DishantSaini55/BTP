# Protocol assessment

## Why learned models underperform constant velocity

The fixed GP6 protocol contains 212 training windows, 43 validation windows,
and 217 test windows. This is a small training set for transformer and
recurrent trajectory models, especially when the split is video-level rather
than pedestrian-window-level.

The three GP6 videos are used as separate train, validation, and test roles:

- train: `gp_set_0006_vid_0002`
- validation: `gp_set_0006_vid_0001`
- test: `gp_set_0006_vid_0003`

This creates a cross-video distribution shift. The validation and test
normalized `x_T` ranges differ substantially:

- train: `[-8.497092, 8.950606]`
- validation: `[-23.226851, 12.111341]`
- test: `[-38.396156, 4.739763]`

The constant-velocity baseline measures 1496.018040 test MSE, 655.570860
C-MSE, and 1948.649039 CF-MSE. It is strong because GP6 pedestrian motion is
well approximated by the last three observed boxes. The learned models must
estimate their own displacement against a residual head that starts from the
constant-position prediction. The small MTN and PACE-Net families also learn
45-step autoregressive/transformer trajectories from only 212 windows.

The E1 seed-level results and epoch diagnostics are consistent with this:
many validation/test-optimal epochs differ substantially, and seed 20260932
has a validation-selected epoch 34 versus test-optimal epoch 22 in the
diagnostic rerun. This is evidence of video shift and weak training signal,
not permission to tune against test curves.

The B1 fixed-LSTM reruns remain far worse than constant velocity. Their
diagnostic mean test MSE is 446028.863458 without scene labels and
430463.202011 with scene labels. This does not make the 371k original result
correct: normalization and denormalization were consistent, while the original
runner had a 45-step rollout/teacher-forcing and gradient-clipping instability.

## Local IDD-PeD video inventory

The following source videos are present locally:

- `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0001.MP4`
- `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0002.MP4`
- `D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0003.MP4`

No GP3, GP5, or GP9 source video was found in the local roots checked.

## Local pedestrian and vehicle annotation inventory

Pedestrian XML files:

- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0006\gp_set_0006_vid_0001.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0006\gp_set_0006_vid_0002.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0006\gp_set_0006_vid_0003.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0008\gp_set_0008_vid_0001.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0008\gp_set_0008_vid_0004.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations\gopro\gp_set_0008\gp_set_0008_vid_0005.xml`

Vehicle/ego OBD XML files:

- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0006\gp_set_0006_vid_0001_obd.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0006\gp_set_0006_vid_0002_obd.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0006\gp_set_0006_vid_0003_obd.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0008\gp_set_0008_vid_0001_obd.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0008\gp_set_0008_vid_0004_obd.xml`
- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\annotations_vehicle\gopro\gp_set_0008\gp_set_0008_vid_0005_obd.xml`

Additional local data-cache files:

- `D:\IDD-PeD-Evaluation\annotations\IDDPedestrian\data_cache\iddp_database.pkl`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\validation.pkl`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\train.pkl`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_poc\test.pkl`
- `D:\IDD-PeD-Evaluation\flow\cpu_farneback_c4_larger\test_full.pkl`
- `D:\DOWNLOADS\BTP\results_paper\splits\gp6_from_scratch_v1\raw_splits.pkl`
- `D:\DOWNLOADS\BTP\results_paper\splits\gp6_from_scratch_v1\flow_splits.pkl`

GP8 XML files and OBD files are present, but GP8 source videos were not found
locally. No locally usable GP8 flow cache is present in the frozen GP6 protocol.

## Protocol guard

No new training protocol change is recommended or applied here. The frozen
GP6 split, 50-epoch maximum, validation-only patience-10 early stopping, Adam
1e-3, batch 8, and MSE objective remain unchanged. This assessment is diagnostic
and requires approval before any protocol change.
