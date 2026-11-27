import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from MTN.ao_dataset_v1 import (
    _safe_video_path,
    parse_video,
    prepare_split,
)


def _write_video(root, set_id, video_id, frames=121, invalid_frame=None):
    directory = root / "annotations" / "gopro" / set_id
    directory.mkdir(parents=True, exist_ok=True)
    boxes = []
    for frame in range(frames):
        if frame == invalid_frame:
            boxes.append(
                f'<box frame="{frame}" outside="0" xtl="-1" ytl="2" '
                'xbr="10" ybr="20"><attribute name="id">p1</attribute></box>'
            )
        else:
            boxes.append(
                f'<box frame="{frame}" outside="0" xtl="{frame}" ytl="2" '
                f'xbr="{frame + 10}" ybr="20">'
                '<attribute name="id">p1</attribute></box>'
            )
    xml = (
        f'<annotations><meta><task><name>{video_id}</name><size>{frames}</size>'
        f'<original_size><width>200</width><height>100</height></original_size>'
        '</task></meta><track id="1" label="pedestrian">'
        + "".join(boxes)
        + '</track><track id="2" label="POI"><box frame="0" outside="0" '
        'xtl="0" ytl="0" xbr="1" ybr="1">'
        '<attribute name="id">p1</attribute>'
        '<attribute name="intersection_type">crossroad</attribute>'
        '</box></track></annotations>'
    )
    (directory / f"{video_id}.xml").write_text(xml, encoding="utf-8")

    obd_directory = root / "annotations_vehicle" / "gopro" / set_id
    obd_directory.mkdir(parents=True, exist_ok=True)
    obd = "<annotations>" + "".join(
        f'<frame id="{frame}" OBD_speed="0.5"/>' for frame in range(frames)
    ) + "</annotations>"
    (obd_directory / f"{video_id}_obd.xml").write_text(obd, encoding="utf-8")


class AnnotationOnlyDatasetTests(unittest.TestCase):
    def test_windows_are_contiguous_and_use_observe_15_predict_45(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_video(root, "gp_set_0001", "gp_set_0001_vid_0001")
            report, samples = parse_video(
                root, "gp_set_0001", "gp_set_0001_vid_0001"
            )
        self.assertEqual(report["eligible_windows_stride30"], 3)
        self.assertEqual([sample["frame_start"] for sample in samples], [0, 30, 60])
        first = samples[0]["boxes"]
        self.assertEqual(first.shape, (60, 4))
        np.testing.assert_array_equal(first[:15, 0], np.arange(15))
        np.testing.assert_array_equal(first[15:, 0], np.arange(15, 60))

    def test_cvat_task_name_with_mp4_suffix_matches_video_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_id = "gp_set_0001_vid_0001"
            _write_video(root, "gp_set_0001", video_id)
            path = (
                root / "annotations" / "gopro" / "gp_set_0001"
                / f"{video_id}.xml"
            )
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    f"<name>{video_id}</name>",
                    f"<name>{video_id}.MP4</name>",
                ),
                encoding="utf-8",
            )
            report, _ = parse_video(root, "gp_set_0001", video_id)
        self.assertEqual(report["cvat_task_name"], f"{video_id}.MP4")

    def test_invalid_box_breaks_continuity_and_is_not_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_video(
                root, "gp_set_0001", "gp_set_0001_vid_0001",
                invalid_frame=40,
            )
            report, samples = parse_video(
                root, "gp_set_0001", "gp_set_0001_vid_0001"
            )
        self.assertEqual(report["out_of_bounds_bbox_records"], 1)
        self.assertEqual(report["eligible_windows_stride30"], 1)
        self.assertEqual(samples[0]["frame_start"], 41)
        self.assertEqual(samples[0]["frame_end"], 100)

    def test_train_and_validation_caches_are_video_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = ("gp_set_0001", "gp_set_0001_vid_0001")
            validation = ("gp_set_0007", "gp_set_0007_vid_0004")
            _write_video(root, *train)
            _write_video(root, *validation)
            output = root / "split"
            with patch("MTN.ao_dataset_v1.TRAIN_VIDEOS", (train,)), patch(
                "MTN.ao_dataset_v1.VALIDATION_VIDEOS", (validation,)
            ):
                manifest = prepare_split(root, output)

            train_cache = np.load(output / "train.npz", allow_pickle=False)
            validation_cache = np.load(
                output / "validation.npz", allow_pickle=False
            )
            saved_manifest = json.loads(
                (output / "split_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(train_cache["observed_boxes"].shape[1:], (15, 4))
            self.assertEqual(train_cache["future_boxes"].shape[1:], (45, 4))
            self.assertFalse(
                set(train_cache["video_id"]).intersection(
                    set(validation_cache["video_id"])
                )
            )
            self.assertEqual(
                manifest["normalization_fit_role"], "train only, unique training observations/targets"
            )
            self.assertEqual(saved_manifest["official_test_sets_read"], [])
            train_cache.close()
            validation_cache.close()

    def test_non_development_set_is_rejected_without_path_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                _safe_video_path(
                    Path(temporary), "gp_set_0003",
                    "gp_set_0003_vid_0001", "annotations",
                )

    def test_historical_gp6_test_annotation_content_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "test-role"):
                _safe_video_path(
                    Path(temporary), "gp_set_0006",
                    "gp_set_0006_vid_0003", "annotations",
                )


if __name__ == "__main__":
    unittest.main()
