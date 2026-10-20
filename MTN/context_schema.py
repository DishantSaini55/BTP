"""Stable categorical schema for the IDD-PeD context ablation."""

SCENE_FIELDS = (
    "intersection_type", "signalized_type", "road_type", "location_type",
    "motion_direction",
)
# Valid native IDD-PeD labels plus one UNKNOWN value.  Time of day and
# illumination are not native IDD-PeD annotation fields, so they deliberately
# are not part of this experiment's context contract.
SCENE_VOCAB_SIZES = (7, 5, 5, 5, 3)
SCENE_UNKNOWN_VALUES = (6, 4, 4, 4, 2)
OCCLUSION_VOCAB_SIZE = 4
OCCLUSION_UNKNOWN_VALUE = 3
