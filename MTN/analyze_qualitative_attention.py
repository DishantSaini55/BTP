"""Summarize saved E1 attention and verify the closed-context mask."""
from __future__ import annotations

import csv
import json
import pickle
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .render_qualitative import MODEL_SPECS, QUALITATIVE, SEED, load_selection, read_rows
    from .run_from_scratch_e1 import NAMES, RESULTS, load_data, model_for
except ImportError:
    from render_qualitative import MODEL_SPECS, QUALITATIVE, SEED, load_selection, read_rows
    from run_from_scratch_e1 import NAMES, RESULTS, load_data, model_for

SELECTED_CSV = QUALITATIVE / f"ATTENTION_SELECTED_seed{SEED}.csv"
AGGREGATE_CSV = QUALITATIVE / f"ATTENTION_TEST_AGGREGATE_seed{SEED}.csv"
MASKING_CSV = QUALITATIVE / f"MASKING_BY_SCENE_seed{SEED}.csv"
PROBE_CSV = QUALITATIVE / f"GATE_MASK_PROBE_seed{SEED}.csv"
REPORT = QUALITATIVE / f"ATTENTION_REPORT_seed{SEED}.md"
GATED_MODELS = {
    "PACE-Net without SATIE": "pace_net_without_satie_sage_plus_masc",
    "PACE-Net": "pace_net_sage_plus_masc_plus_satie",
}
SCENE_FIELD_NAMES = (
    "intersection_type",
    "signalized_type",
    "road_type",
    "location_type",
    "motion_direction",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_saved_attention() -> tuple[list[dict], list[dict], dict, dict, dict]:
    selected_ids = [item["sample_id"] for item in load_selection()]
    samples_path = RESULTS / "splits" / "gp6_from_scratch_v1" / "raw_splits.pkl"
    with samples_path.open("rb") as handle:
        samples = pickle.load(handle)["test"]
    sample_index = {sample["id"]: index for index, sample in enumerate(samples)}
    if any(sample_id not in sample_index for sample_id in selected_ids):
        raise ValueError("Selection contains an ID absent from the frozen test split.")

    selected_rows = []
    aggregate_rows = []
    gate_counts = {}
    gates_by_model = {}
    for label, (_slug, relative) in MODEL_SPECS.items():
        if label == "Constant velocity":
            continue
        directory = Path(relative)
        archive = np.load(directory / "predictions.npz", allow_pickle=False)
        rows = read_rows(directory / "per_sample.csv")
        row_ids = [row["sample_id"] for row in rows]
        if row_ids != [sample["id"] for sample in samples]:
            raise ValueError(f"Saved prediction order differs from frozen test split: {label}")

        gates = archive["gate"] if "gate" in archive.files else None
        if label in GATED_MODELS:
            if gates is None or gates.shape != (len(samples),):
                raise ValueError(f"Missing per-sample gate decisions for {label}")
            gate_counts[label] = {
                "g0": int(np.count_nonzero(gates == 0)),
                "g1": int(np.count_nonzero(gates == 1)),
            }
            gates_by_model[label] = gates.copy()
        attention_keys = archive["attention_memory_tokens"] if "attention_memory_tokens" in archive.files else None

        for layer in range(3):
            key = f"attention_layer_{layer}"
            attention = archive[key]
            if attention.ndim != 4 or attention.shape[:3] != (len(samples), 4, 45):
                raise ValueError(f"Unexpected attention dimensions for {label}, {key}: {attention.shape}")
            if not np.isfinite(attention).all():
                raise ValueError(f"Non-finite saved attention in {label}, {key}")
            if np.max(np.abs(attention.sum(axis=-1) - 1.0)) > 1e-5:
                raise ValueError(f"Attention does not sum to one in {label}, {key}")

            for sample_id in selected_ids:
                index = sample_index[sample_id]
                memory_keys = int(attention_keys[index]) if attention_keys is not None else attention.shape[-1]
                for token_index in range(16):
                    if token_index < memory_keys:
                        values = attention[index, :, :, token_index].mean(axis=1)
                        status = "observed"
                    elif token_index == 15 and label in GATED_MODELS and gates[index] == 0:
                        values = np.full(4, np.nan)
                        status = "context key omitted in all-closed E1 forward"
                    else:
                        continue
                    selected_rows.append({
                        "sample_id": sample_id,
                        "model": label,
                        "layer": layer + 1,
                        "token_index": token_index,
                        "token_type": "context" if token_index == 15 else "trajectory",
                        "gate": int(gates[index]) if gates is not None else "N/A",
                        "mean_attention_head_0": float(values[0]) if np.isfinite(values[0]) else "",
                        "mean_attention_head_1": float(values[1]) if np.isfinite(values[1]) else "",
                        "mean_attention_head_2": float(values[2]) if np.isfinite(values[2]) else "",
                        "mean_attention_head_3": float(values[3]) if np.isfinite(values[3]) else "",
                        "mean_attention_heads": float(np.nanmean(values)) if np.isfinite(values).any() else "",
                        "status": status,
                    })

            for token_index in range(attention.shape[-1]):
                values = attention[:, :, :, token_index].mean(axis=(0, 2))
                aggregate_rows.append({
                    "model": label,
                    "layer": layer + 1,
                    "token_index": token_index,
                    "token_type": "trajectory",
                    "n_windows": len(samples),
                    "n_query_steps_per_window": 45,
                    "mean_attention_head_0": float(values[0]),
                    "mean_attention_head_1": float(values[1]),
                    "mean_attention_head_2": float(values[2]),
                    "mean_attention_head_3": float(values[3]),
                    "mean_attention_heads": float(values.mean()),
                })
            if label in GATED_MODELS:
                aggregate_rows.append({
                    "model": label,
                    "layer": layer + 1,
                    "token_index": 15,
                    "token_type": "context",
                    "n_windows": int(np.count_nonzero(gates == 1)),
                    "n_query_steps_per_window": 45,
                    "mean_attention_head_0": "",
                    "mean_attention_head_1": "",
                    "mean_attention_head_2": "",
                    "mean_attention_head_3": "",
                    "mean_attention_heads": "",
                })
    return selected_rows, aggregate_rows, gate_counts, gates_by_model, sample_index


def verify_mixed_gate_mask(sample_indices: list[int], samples: list[dict]) -> list[dict]:
    if len(sample_indices) != 2 or sample_indices[0] == sample_indices[1]:
        raise ValueError("Mask probe requires two distinct test windows.")
    datasets, _ = load_data()
    test = datasets["test"]
    batch = next(iter(DataLoader(Subset(test, sample_indices), batch_size=2, shuffle=False)))
    checkpoint_path = (
        RESULTS / "runs" / "E1" / GATED_MODELS["PACE-Net"] / SEED / "checkpoint.pth"
    )
    model = model_for(NAMES[2]).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    requested_gate = torch.tensor([1.0, 0.0], dtype=torch.float32)

    def forced_gate(self, context_token, observed_boxes):
        gate = requested_gate.to(context_token.device)
        self.last_context_gate = gate.detach()
        self.last_context_gate_logits = torch.stack((1 - gate, gate), dim=-1)
        self._context_gate_for_loss = gate
        return gate

    model._context_gate = types.MethodType(forced_gate, model)
    with torch.no_grad():
        model.forward_batch(batch)

    probe_rows = []
    for layer_index, layer in enumerate(model.base.model.decoder.layers):
        attention = layer.src_attn.attn
        if attention is None or attention.shape != (2, 4, 45, 16):
            raise AssertionError(f"Expected a mixed-gate 16-key attention tensor, got {None if attention is None else attention.shape}")
        context_weights = attention[:, :, :, 15]
        if torch.count_nonzero(context_weights[1]).item() != 0:
            raise AssertionError("Closed gate had nonzero context-token attention.")
        if not bool(torch.any(context_weights[0] > 0)):
            raise AssertionError("Forced-open diagnostic did not attend to the context key.")
        for row_index, forced_value in enumerate((1, 0)):
            for head in range(4):
                probe_rows.append({
                    "sample_id": samples[sample_indices[row_index]]["id"],
                    "forced_gate_for_mask_validation_only": forced_value,
                    "saved_learned_gate": 0,
                    "layer": layer_index + 1,
                    "head": head,
                    "context_attention_mean_over_45_queries": float(
                        context_weights[row_index, head].mean()
                    ),
                    "context_attention_max_over_45_queries": float(
                        context_weights[row_index, head].max()
                    ),
                    "closed_context_weight_exactly_zero": bool(
                        torch.count_nonzero(context_weights[row_index]).item() == 0
                    ),
                })
    return probe_rows


def write_report(gate_counts: dict, masking_rows: list[dict]) -> None:
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite {REPORT}")
    lines = [
        f"# Decoder attention and gate-mask analysis (seed {SEED})",
        "",
        "Analysis uses saved E1 predictions and attention arrays; no training or checkpoint selection was performed.",
        "",
        "## Saved E1 gate availability",
        "",
        "| Model | g=0 windows | g=1 windows |",
        "|---|---:|---:|",
    ]
    for label, counts in gate_counts.items():
        lines.append(f"| {label} | {counts['g0']} | {counts['g1']} |")
    lines.extend([
        "",
        "Neither learned SAGE checkpoint admitted context on any of the 217 test windows. Consequently, the saved E1 attention has 15 trajectory-memory keys, not 16 keys; there is no observed learned-gate `g=1` context-attention weight or naturally admitted window to mask. The context column in the selected-window CSV is explicitly marked as omitted, not misreported as a measured softmax value. The all-closed forward path bypasses token concatenation, so its effective context-attention mass is exactly zero by absence.",
        "",
        "## Attention summaries",
        "",
        "`ATTENTION_SELECTED_seed20260928.csv` reports each selected window's mean across 45 decoder queries for all 4 heads, all 3 decoder layers, and each available trajectory key. `ATTENTION_TEST_AGGREGATE_seed20260928.csv` reports the same per-head means averaged over all 217 test windows; the context-key row has zero observed admitted windows and no measured attention value.",
        "",
        "## Explicit mixed-mask implementation check",
        "",
        "A post-hoc forward-only check loaded the saved PACE-Net checkpoint and forced a two-row gate batch `[1, 0]`; no weights were updated. The decoder then had 16 keys. In every decoder layer and head, the g=0 row's context attention was exactly zero (`count_nonzero == 0`), while the forced-open row had positive context weight. These forced values verify masking semantics only; they are not learned-gate results and are excluded from the qualitative comparison.",
        "",
        "The mask-probe details are in `GATE_MASK_PROBE_seed20260928.csv`. The scene-stratified masking table reports the actual saved-gate population and marks the g=1 masking effect not estimable because the count is zero.",
        "",
        "## Scene-stratified context masking",
        "",
        "| Scene field | Label code | Test windows | g=0 | g=1 | Natural g=1 masking effect |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in masking_rows:
        lines.append(
            f"| {row['scene_field']} | {row['label_code']} | {row['n_windows']} | {row['n_gate_0']} | "
            f"{row['n_gate_1']} | {row['masked_prediction_effect']} |"
        )
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = (SELECTED_CSV, AGGREGATE_CSV, MASKING_CSV, PROBE_CSV, REPORT)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing attention outputs: {existing}")
    selected_rows, aggregate_rows, gate_counts, gates_by_model, sample_index = summarize_saved_attention()
    samples_path = RESULTS / "splits" / "gp6_from_scratch_v1" / "raw_splits.pkl"
    with samples_path.open("rb") as handle:
        samples = pickle.load(handle)["test"]
    selected_ids = [item["sample_id"] for item in load_selection()]
    probe_rows = verify_mixed_gate_mask(
        [sample_index[selected_ids[0]], sample_index[selected_ids[1]]], samples
    )

    masking_rows = []
    full_gates = gates_by_model["PACE-Net"]
    for field_index, field_name in enumerate(SCENE_FIELD_NAMES):
        labels = sorted({int(sample["scene_context"][0][field_index]) for sample in samples})
        for label_code in labels:
            indices = [
                index for index, sample in enumerate(samples)
                if int(sample["scene_context"][0][field_index]) == label_code
            ]
            n_gate_1 = int(np.count_nonzero(full_gates[indices] == 1))
            n_gate_0 = int(np.count_nonzero(full_gates[indices] == 0))
            masking_rows.append({
                "scene_field": field_name,
                "label_code": label_code,
                "n_windows": len(indices),
                "n_gate_0": n_gate_0,
                "n_gate_1": n_gate_1,
                "mean_mse_g1_observed": "",
                "masked_prediction_effect": "not estimable: no naturally admitted windows" if n_gate_1 == 0 else "requires per-window counterfactual",
            })

    write_csv(SELECTED_CSV, selected_rows)
    write_csv(AGGREGATE_CSV, aggregate_rows)
    write_csv(MASKING_CSV, masking_rows)
    write_csv(PROBE_CSV, probe_rows)
    write_report(gate_counts, masking_rows)
    print(json.dumps({
        "selected_attention_rows": len(selected_rows),
        "aggregate_attention_rows": len(aggregate_rows),
        "mask_probe_rows": len(probe_rows),
        "gate_counts": gate_counts,
        "mask_probe_context_weight_for_closed_rows": "exactly zero",
        "natural_g1_effect": "not estimable",
    }, indent=2))


if __name__ == "__main__":
    main()
