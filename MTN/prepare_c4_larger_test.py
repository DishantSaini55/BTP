"""Prepare the complete locally available GP6 video-3 held-out test set once.

The existing 32/8 train/validation POC files are intentionally untouched.
All 24 eligible trajectories from the separate video-3 holdout are retained;
there is no outcome-based selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import cpu_poc_prepare as poc

OUTPUT = Path(r"D:\IDD-PeD-Evaluation\flow\cpu_farneback_c4_larger")

def main():
    raw = poc._load_set("gp_set_0006", "train")
    candidates = [i for i, images in enumerate(raw["image"])
                  if "gp_set_0006_vid_0003" in str(images[0])]
    # This is the full available video-level holdout, retained in source order.
    data = poc._subset(raw, candidates)
    poc._add_flows(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import pickle
    with (OUTPUT / "test_full.pkl").open("wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
    manifest=[]
    for local_index, source_index in enumerate(candidates):
        manifest.append({"local_index":local_index,"source_index":source_index,
                         "pid":str(data["pid"][local_index][0]),
                         "first_frame":str(data["image"][local_index][0]),
                         "scene_context":[int(v) for v in data["scene_context"][local_index][0]],
                         "max_occlusion":int(max(data["occlusion"][local_index])),
                         "track_frames":len(data["image"][local_index])})
    (OUTPUT / "test_full_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    (OUTPUT / "farneback_config.json").write_text(json.dumps(poc.CONFIG.as_dict(),indent=2),encoding="utf8")
    print(json.dumps({"selection_rule":"all eligible gp_set_0006_vid_0003 trajectories in source order","trajectories":len(candidates),"videos":1,"output":str(OUTPUT)},indent=2))
if __name__ == "__main__": main()
