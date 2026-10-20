"""Read-only readiness probe for official IDD-PeD MTN test evaluation.

No videos, flows, checkpoints, or model parameters are created or downloaded.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference_IDD-PeD"
DATA = REFERENCE / "data" / "IDDPedestrian"
TEST_SETS = ("gp_set_0003", "gp_set_0005", "gp_set_0008", "gp_set_0009")


def main():
    if not DATA.is_dir():
        raise SystemExit(f"Missing local annotations: {DATA}")
    sys.path.insert(0, str(REFERENCE / "MTN"))
    from iddped_interface_traj import IDDPedestrian

    options = {
        "fstride": 1, "sample_type": "all", "height_rng": [0, float("inf")],
        "squarify_ratio": 0, "data_split_type": "default", "seq_type": "trajectory",
        "interaction": "y", "min_track_size": 60,
        "random_params": {"ratios": None, "val_data": True, "regen_data": True},
        "kfold_params": {"num_folds": 5, "fold": 1},
    }
    dataset = IDDPedestrian(data_path=str(DATA))
    trajectories = dataset.generate_data_trajectory_sequence("test", **options)
    count = len(trajectories["bbox"])
    frames = sum(len(track) for track in trajectories["bbox"])
    expected_video_dirs = [DATA / "videos" / "gopro" / name for name in TEST_SETS]
    missing_video_dirs = [str(path) for path in expected_video_dirs if not path.is_dir()]
    print(f"official_test_sets={','.join(TEST_SETS)}")
    print(f"default_mtn_interaction_test_tracks={count}")
    print(f"default_mtn_interaction_test_frames={frames}")
    print(f"annotations_ready=True")
    print(f"videos_ready={not missing_video_dirs}")
    print(f"flows_ready=False")
    print("next_blocker=" + ("test videos and generated flow arrays" if missing_video_dirs else "generated flow arrays"))
    if count == 0:
        raise SystemExit("The official test trajectory loader produced zero samples.")


if __name__ == "__main__":
    main()
