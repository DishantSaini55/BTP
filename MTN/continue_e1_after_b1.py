"""Fail-closed B1-to-E1 hand-off for the frozen from-scratch protocol."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"
STATUS = RESULTS / "orchestration_from_scratch_v2.json"
EXPECTED_B1_METRICS = 22  # 2 deterministic + 2 LSTMs x 10 seeds


def write_status(**fields) -> None:
    if STATUS.exists():
        raise FileExistsError(f"Refusing to overwrite {STATUS}")
    STATUS.write_text(json.dumps(fields, indent=2), encoding="utf-8")


def process_running(pid: int) -> bool:
    if sys.platform == "win32":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-pid", required=True, type=int)
    args = parser.parse_args()
    while process_running(args.b1_pid):
        time.sleep(20)
    b1_metrics = list((RESULTS / "runs" / "B1").glob("**/metrics.json"))
    stderr = (RESULTS / "b1_stderr.log").read_text(encoding="utf-8", errors="replace")
    if len(b1_metrics) != EXPECTED_B1_METRICS or "Traceback" in stderr:
        write_status(status="blocked", reason="B1 did not complete cleanly", b1_metric_files=len(b1_metrics))
        raise SystemExit(1)
    e1_root = RESULTS / "runs" / "E1"
    if e1_root.exists():
        write_status(status="blocked", reason="E1 output directory already exists")
        raise SystemExit(1)
    write_status(status="starting_e1", b1_metric_files=len(b1_metrics), initialization="from scratch only")
    command = [sys.executable, "MTN/run_from_scratch_e1.py", "run"]
    result = subprocess.run(command, cwd=ROOT)
    # Preserve the initial immutable status record. E1's own outputs carry its completion state.
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
