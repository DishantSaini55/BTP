"""Resume only E1 runs absent after the audited attention-serialization failure."""
from __future__ import annotations

import subprocess
import sys

try:
    from .run_from_scratch_e1 import NAMES, RESULTS, SEEDS, slug
except ImportError:
    from run_from_scratch_e1 import NAMES, RESULTS, SEEDS, slug


def main() -> None:
    # Seed 20260936 had no metrics/predictions, and only the remaining seeds
    # and models are eligible. Existing completed runs are never revisited.
    jobs = [
        (NAMES[1], [20260936, 20260937]),
        (NAMES[2], SEEDS),
        (NAMES[3], SEEDS),
        (NAMES[4], SEEDS),
    ]
    for model, seeds in jobs:
        for seed in seeds:
            run = RESULTS / "runs" / "E1" / slug(model) / str(seed)
            config = RESULTS / "config" / f"E1_{slug(model)}_{seed}.json"
            if run.exists() or config.exists():
                raise FileExistsError(f"Refusing to overwrite or rerun {model}, seed {seed}")
        command = [sys.executable, "MTN/run_from_scratch_e1.py", "run", "--models", model,
                   "--seeds", *map(str, seeds)]
        result = subprocess.run(command)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
