# Observation-time feature safety audit

## Primary model inputs

The only model input is the first 15 observed bounding boxes for each sample (`[15,4]`, absolute pixel `xyxy`). The next 45 boxes are supervision only. Input and target normalization is estimated from unique training-role frames only.

## Excluded fields

- **Ego speed:** selected OBD XMLs have complete frame-ID coverage, but the unit is not verified. It is not input to any primary model.
- **Optical flow:** raw videos are unavailable and flow is not computed or synthesized.
- **Occlusion, crossing behavior, and future state:** excluded; annotations may refer to later frames or events and are not needed for the matched motion-only comparison.
- **Occlusion-label consistency:** the Boolean `occluded` field and categorical `occlusion` field disagree in some annotations. Neither is used as input or as a substituted label.
- **Static scene labels:** present on POI tracks for some selected videos, but availability at the exact prediction time is not independently established. GP7 video 0005 has no POI tracks or scene labels. AO-4 is therefore skipped rather than evaluating context on a selected subset or imputing missing labels.
- **Scene-ID linkage:** the selected GP2 video has one duplicated manual pedestrian ID; the selected GP6 video has one missing and one duplicated manual ID. Scene-to-pedestrian linkage is therefore not assumed to be complete or unambiguous.
- **Official-test records and historical GP6 test data:** not used. GP6 video 0003 appears only as a metadata-only exclusion row in the annotation inventory; the official test sets were not read.

No missing values are fabricated or filled with zeros. The primary models receive identical motion information.
