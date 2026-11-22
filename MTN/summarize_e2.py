"""Summarize the E2 gate-ablation inference variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_paper"


def paired(left: np.ndarray, right: np.ndarray) -> dict:
    difference = left - right
    n = len(difference)
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    se = sd / np.sqrt(n)
    critical = float(stats.t.ppf(.975, n - 1))
    test = stats.ttest_rel(left, right)
    return {
        "n_seeds": n,
        "mean_difference_left_minus_right": mean,
        "ci95": [mean - critical * se, mean + critical * se],
        "paired_t_p": float(test.pvalue),
        "cohen_dz": mean / sd if sd else None,
        "left_lower_error_seeds": int((difference < 0).sum()),
    }


def main() -> None:
    root = RESULTS / "runs" / "E2"
    records = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        records.append(metrics)
    if len(records) != 80:
        raise ValueError(f"Expected 80 E2 records, found {len(records)}")

    summary = RESULTS / "summary"
    summary.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(field for record in records for field in record))
    with (summary / "E2_all_runs.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    group = []
    for model in sorted({record["model"] for record in records}):
        for ablation in ("always_open", "always_closed", "soft_sigmoid", "context_only"):
            subset = [record for record in records if record["model"] == model and record["ablation"] == ablation]
            if not subset:
                continue
            group.append({
                "model": model,
                "ablation": ablation,
                "runs": len(subset),
                "mse_mean": float(np.mean([record["mse"] for record in subset])),
                "mse_sd": float(np.std([record["mse"] for record in subset], ddof=1)),
                "cmse_mean": float(np.mean([record["cmse"] for record in subset])),
                "cfmse_mean": float(np.mean([record["cfmse"] for record in subset])),
                "epochs": "reused E1 checkpoint",
            })
    with (summary / "E2_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group[0]))
        writer.writeheader()
        writer.writerows(group)

    h3 = {}
    for model in sorted({record["model"] for record in records}):
        learned = []
        for seed in range(20260928, 20260938):
            e1_slug = ("pace_net_without_satie_sage_plus_masc" if "without SATIE" in model
                       else "pace_net_sage_plus_masc_plus_satie")
            metrics = json.loads((RESULTS / "runs" / "E1" / e1_slug / str(seed) / "metrics.json").read_text())
            learned.append(metrics["mse"])
        opened = [
            record["mse"] for record in records
            if record["model"] == model and record["ablation"] == "always_open"
        ]
        h3[model] = paired(np.array(learned), np.array(opened))

    (summary / "E2_statistics.json").write_text(json.dumps({
        "H3_learned_gate_vs_always_open": h3,
        "note": "E2 variants are post-hoc checkpoint evaluations; H3 is confirmatory but is assessed in the combined H1/H2/H3 Holm family only.",
    }, indent=2), encoding="utf-8")

    lines = [
        "# E2 SAGE gate ablation report",
        "",
        "Post-hoc evaluations of the matching E1 checkpoints; no E2 run retrained any model.",
        "",
        "| Model | Ablation | Runs | MSE mean ± SD | C-MSE mean | CF-MSE mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in group:
        lines.append(
            f"| {item['model']} | {item['ablation']} | {item['runs']} | "
            f"{item['mse_mean']:.6f} ± {item['mse_sd']:.6f} | "
            f"{item['cmse_mean']:.6f} | {item['cfmse_mean']:.6f} |"
        )
    lines += ["", "## H3 learned gate vs always-open", ""]
    for model, item in h3.items():
        ci = item["ci95"]
        lines.append(
            f"- **{model}**: mean difference {item['mean_difference_left_minus_right']:.6f}; "
            f"95% t CI [{ci[0]:.6f}, {ci[1]:.6f}]; d_z="
            f"{item['cohen_dz']:.6f}; paired t p={item['paired_t_p']:.6g}; "
            f"learned gate lower-error seeds={item['left_lower_error_seeds']}/{item['n_seeds']}."
        )
    (summary / "E2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Model & Ablation & Runs & MSE mean $\pm$ SD & C-MSE mean & CF-MSE mean \\",
        r"\hline",
    ]
    for item in group:
        tex.append(
            f"{item['model']} & {item['ablation']} & {item['runs']} & "
            f"${item['mse_mean']:.6f} \\pm {item['mse_sd']:.6f}$ & "
            f"${item['cmse_mean']:.6f}$ & ${item['cfmse_mean']:.6f}$ \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", r"\caption{E2 SAGE gate-ablation seed summary.}", r"\end{table}"]
    (summary / "E2_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
