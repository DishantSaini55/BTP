"""Wait for all immutable E1 runs, then create the one-time E1 summary."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results_paper" / "runs" / "E1"
SUMMARY = ROOT / "results_paper" / "summary"


def main() -> None:
    while True:
        if RUNS.exists() and len(list(RUNS.glob("**/metrics.json"))) == 50:
            if SUMMARY.exists():
                raise FileExistsError(f"Refusing to overwrite {SUMMARY}")
            completed = subprocess.run([sys.executable, "MTN/summarize_e1.py"], cwd=ROOT)
            raise SystemExit(completed.returncode)
        time.sleep(30)


if __name__ == "__main__":
    main()
