"""Post-hoc E3 gate and route behaviour analysis for saved E1 outputs."""
from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"
SPLIT = RESULTS / "splits" / "gp6_from_scratch_v1"
SEEDS = list(range(20260928, 20260938))
SCENE_FIELDS = ("intersection_type", "signalized_type", "road_type", "location_type", "motion_direction")


def load_samples() -> dict:
    with (SPLIT / "flow_splits.pkl").open("rb") as handle:
        return pickle.load(handle)


def train_x_t_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.concatenate([sample["bbox"] for sample in samples])
    mean = boxes.mean(0)
    std = np.maximum(boxes.std(0), 1e-6)
    return mean, std


def x_t_ranges(split: dict) -> dict:
    mean, std = train_x_t_stats(split["train"])
    result = {}
    for role in ("train", "validation", "test"):
        values = np.array([((sample["bbox"][-1] - mean) / std) for sample in split[role]])
        result[role] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "abs_max": float(np.abs(values).max()),
            "exceeds_plus_minus_5": bool(np.abs(values).max() > 5),
        }
    return result


def main() -> None:
    split = load_samples()
    x_ranges = x_t_ranges(split)
    rows = []
    scene_counts = defaultdict(int)

    for role in ("test",):
        for sample in split[role]:
            scene_counts[tuple(int(value) for value in sample["scene_context"][0])] += 1

    models = (
        ("PACE-Net without SATIE (SAGE + MASC)", "pace_net_without_satie_sage_plus_masc", "sage"),
        ("PACE-Net (SAGE + MASC + SATIE)", "pace_net_sage_plus_masc_plus_satie", "sage"),
        ("PACE-Net with ROUTE", "pace_net_with_route", "route"),
        ("PACE-Net with ROUTE and SATIE", "pace_net_with_route_and_satie", "route"),
    )

    for model, slug, family in models:
        for seed in SEEDS:
            run = RESULTS / "runs" / "E1" / slug / str(seed)
            metrics = json.loads((run / "metrics.json").read_text())
            predictions = np.load(run / "predictions.npz")
            with (run / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
                per_sample = list(csv.DictReader(handle))
            if family == "sage":
                gate = predictions["gate"]
                gate_logits = predictions["gate_logits"]
                admitted = gate.astype(bool)
                admitted_mse = float(np.mean([float(row["mse"]) for row, value in zip(per_sample, admitted) if value]))
                rejected_mse = float(np.mean([float(row["mse"]) for row, value in zip(per_sample, admitted) if not value]))
            else:
                gate = predictions["gate"]
                chosen = np.argmax(gate, axis=1)
                admitted = np.ones(len(per_sample), dtype=bool)
                admitted_mse = float(np.mean([float(row["mse"]) for row in per_sample]))
                rejected_mse = None
            admission_by_scene = {}
            for scene_index, scene_name in enumerate(SCENE_FIELDS):
                values = defaultdict(list)
                for sample_index, sample in enumerate(split["test"]):
                    label = int(sample["scene_context"][0][scene_index])
                    if family == "sage":
                        values[label].append(bool(admitted[sample_index]))
                    else:
                        values[label].append(int(chosen[sample_index]))
                if family == "sage":
                    admission_by_scene[scene_name] = {
                        str(label): float(np.mean(items)) for label, items in sorted(values.items())
                    }
                else:
                    admission_by_scene[scene_name] = {
                        str(label): {
                            "identity": float(np.mean(np.array(items) == 0)),
                            "additive": float(np.mean(np.array(items) == 1)),
                            "cross_attention": float(np.mean(np.array(items) == 2)),
                        }
                        for label, items in sorted(values.items())
                    }
            occlusion_values = defaultdict(list)
            for sample_index, sample in enumerate(split["test"]):
                occlusion = int(sample["occlusion"][:15][-1])
                occlusion_values[occlusion].append(bool(admitted[sample_index]) if family == "sage" else int(chosen[sample_index]))
            if family == "sage":
                admission_by_occlusion = {
                    str(label): float(np.mean(items)) for label, items in sorted(occlusion_values.items())
                }
                gate_mean = float(np.mean(gate))
                gate_sd = float(np.std(gate, ddof=1))
                gate_logit_mean = float(np.mean(gate_logits))
                gate_logit_sd = float(np.std(gate_logits, ddof=1))
            else:
                admission_by_occlusion = {
                    str(label): {
                        "identity": float(np.mean(np.array(items) == 0)),
                        "additive": float(np.mean(np.array(items) == 1)),
                        "cross_attention": float(np.mean(np.array(items) == 2)),
                    }
                    for label, items in sorted(occlusion_values.items())
                }
                gate_mean = gate_sd = gate_logit_mean = gate_logit_sd = None
            rows.append({
                "model": model,
                "seed": seed,
                "family": family,
                "admission_rate": float(np.mean(admitted)) if family == "sage" else None,
                "admitted_mse": admitted_mse,
                "rejected_mse": rejected_mse,
                "gate_mean": gate_mean,
                "gate_sd": gate_sd,
                "gate_logit_mean": gate_logit_mean,
                "gate_logit_sd": gate_logit_sd,
                "admission_by_occlusion": json.dumps(admission_by_occlusion),
                "admission_by_scene": json.dumps(admission_by_scene),
            })

    summary = RESULTS / "summary"
    summary.mkdir(exist_ok=True)
    (summary / "E3_gate_behaviour.csv").write_text("")
    with (summary / "E3_gate_behaviour.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (summary / "E3_gate_behaviour.json").write_text(json.dumps({
        "rows": rows,
        "x_t_normalized_ranges": x_ranges,
        "e8_trigger": any(value["exceeds_plus_minus_5"] for value in x_ranges.values()),
        "occlusion_states": "native 0=fully visible, 1=partly, 2=fully/strongly occluded, 3=unknown",
        "scene_fields": SCENE_FIELDS,
    }, indent=2), encoding="utf-8")

    lines = [
        "# E3 gate and route behaviour",
        "",
        "Post-hoc analysis of immutable E1 saved outputs; no training.",
        "",
        "| Model | Seed | Admission rate | Admitted MSE | Rejected MSE | Gate mean ± SD | Gate logit mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["family"] == "sage":
            gate = "n/a" if row["gate_mean"] is None else f"{row['gate_mean']:.6f} ± {row['gate_sd']:.6f}"
            logit = "n/a" if row["gate_logit_mean"] is None else f"{row['gate_logit_mean']:.6f} ± {row['gate_logit_sd']:.6f}"
            rejected = "n/a" if row["rejected_mse"] is None else f"{row['rejected_mse']:.6f}"
            lines.append(
                f"| {row['model']} | {row['seed']} | {row['admission_rate']:.6f} | "
                f"{row['admitted_mse']:.6f} | {rejected} | {gate} | {logit} |"
            )
    lines += [
        "",
        "## Normalized x_T range",
        "",
        f"- Train: {x_ranges['train']}",
        f"- Validation: {x_ranges['validation']}",
        f"- Test: {x_ranges['test']}",
        f"- E8 trigger: `{any(value['exceeds_plus_minus_5'] for value in x_ranges.values())}`",
    ]
    (summary / "E3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (summary / "E3_summary.tex").open("x", encoding="utf-8") as handle:
        handle.write(r"\begin{table}[t]" + "\n")
        handle.write(r"\centering" + "\n")
        handle.write(r"\begin{tabular}{llrrr}" + "\n")
        handle.write(r"\hline" + "\n")
        handle.write(r"Model & Seed & Admission rate & Admitted MSE & Rejected MSE \\" + "\n")
        handle.write(r"\hline" + "\n")
        for row in rows:
            if row["family"] == "sage":
                handle.write(
                    f"{row['model']} & {row['seed']} & {row['admission_rate']:.6f} & "
                    f"{row['admitted_mse']:.6f} & "
                    f"{'n/a' if row['rejected_mse'] is None else f'{row['rejected_mse']:.6f}'} \\\\\n"
                )
        handle.write(r"\hline" + "\n")
        handle.write(r"\end{tabular}" + "\n")
        handle.write(r"\caption{E3 SAGE gate behaviour.}" + "\n")
        handle.write(r"\end{table}" + "\n")


if __name__ == "__main__":
    main()
