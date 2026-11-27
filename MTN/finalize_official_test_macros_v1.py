"""Build macro-video, macro-track and runtime reports for Phase 6B."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_macro_rows(
    video_rows: list[dict[str, str]],
    track_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Return per-model/seed macro-video and macro-track MSE rows."""
    macro_rows: list[dict[str, object]] = []
    for model_id in sorted(set(row["model_id"] for row in video_rows)):
        seeds = sorted(set(
            int(row["seed"]) for row in video_rows if row["model_id"] == model_id
        ))
        for seed in seeds:
            video_values = np.asarray([
                float(row["mse"]) for row in video_rows
                if row["model_id"] == model_id and int(row["seed"]) == seed
            ])
            track_values = np.asarray([
                float(row["mse"]) for row in track_rows
                if row["model_id"] == model_id and int(row["seed"]) == seed
            ])
            if not len(video_values) or not len(track_values):
                raise ValueError(f"Missing video or track rows for {model_id} seed {seed}.")
            macro_rows.append({
                "model_id": model_id,
                "seed": seed,
                "video_count": len(video_values),
                "macro_video_mse": float(video_values.mean()),
                "track_cluster_count": len(track_values),
                "macro_track_mse": float(track_values.mean()),
            })
    return macro_rows


def main() -> int:
    root = Path("results_paper/AO_confirmation_v1")
    result_dir = root / "OFFICIAL_TEST_RESULTS_v2"
    video_rows = read_csv(result_dir / "OFFICIAL_TEST_PER_VIDEO.csv")
    track_rows = read_csv(result_dir / "OFFICIAL_TEST_PER_TRACK.csv")

    macro_rows = build_macro_rows(video_rows, track_rows)
    write_csv(root / "OFFICIAL_TEST_MACRO_RESULTS.csv", macro_rows)

    report = [
        "# Phase 6B Official-Test Runtime Report",
        "",
        "## Official-test inference",
        "",
        "- Eligible windows: 2,615",
        "- Eligible videos: 13",
        "- Track clusters: 861",
        "- Models/records: 15 neural seed evaluations plus AO-0, AO-CP and AO-RIDGE.",
        "- Official-test wall-clock execution time was not instrumented in this "
        "evaluation invocation and is therefore not reported.",
        "",
        "## Phase 6A matched CPU latency",
        "",
        "| Model | Batch | Median ms | IQR ms | Peak sampled process bytes |",
        "|---|---:|---:|---:|---:|",
        "| AO-2 | 1 | 99.049 | 181.088 | 240713728 |",
        "| AO-2 | 8 | 114.574 | 186.512 | 264704000 |",
        "| AO-3 | 1 | 49.395 | 39.370 | 272629760 |",
        "| AO-3 | 8 | 102.235 | 117.540 | 297631744 |",
        "",
        "Measured on Intel64 Family 6 Model 142 Stepping 12, four threads, "
        "50 warm-ups and 500 timed iterations.",
        "",
        "## Checkpoint sizes",
        "",
        "- AO-2/AO-3 seed-20260928 best checkpoints: 1,624,085 bytes each.",
        "- Neural latest checkpoints remain approximately 3.89–3.97 MB.",
        "- Ridge coefficient archive: "
        "`classical/AO-RIDGE/seed_20260928/ridge_coefficients.npz`.",
    ]
    (root / "OFFICIAL_TEST_RUNTIME_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
