"""Export the frozen second-window assets using the shared asset renderer."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent
SAMPLE_ID = "gp_set_0006/gp_set_0006_vid_0003/gp_6933/01205/01264"
SOURCE = ASSET_DIR.parent / "assets" / "generate_asset_pack.py"

spec = importlib.util.spec_from_file_location("first_window_asset_exporter", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load shared asset exporter: {SOURCE}")

exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
exporter.ASSET_DIR = ASSET_DIR
exporter.SAMPLE_ID = SAMPLE_ID
exporter.main()
