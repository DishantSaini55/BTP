"""Exploratory CRAFT-inspired concepts for the saved seed-20260928 PACE-Net."""
from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import qmc
from torch.utils.data import DataLoader

try:
    from .context_schema import SCENE_FIELDS
    from .run_from_scratch_e1 import (
        NAMES, RESULTS, SEEDS, SPLIT, load_data, model_for, set_seed, slug,
    )
except ImportError:
    from context_schema import SCENE_FIELDS
    from run_from_scratch_e1 import (
        NAMES, RESULTS, SEEDS, SPLIT, load_data, model_for, set_seed, slug,
    )


SEED = SEEDS[0]
OUT = RESULTS / "qualitative" / "craft_seed20260928_rank2_16"
VIDEO = Path(r"D:\IDD-PeD-Evaluation\videos\gopro\gp_set_0006\gp_set_0006_vid_0002.MP4")
RANKS = tuple(range(2, 17))
RANK_TOLERANCE = 0.01
SOBOL_BASE_SAMPLES = 1024
SOBOL_BATCH_SIZE = 32
SCENE_LABELS = {
    "intersection_type": ("NI", "U-turn", "T-right", "T-left", "four-way", "Y-intersection", "UNKNOWN"),
    "signalized_type": ("N/A", "C", "S", "CS", "UNKNOWN"),
    "road_type": ("main", "secondary", "street", "lane", "UNKNOWN"),
    "location_type": ("urban", "rural", "commercial", "residential", "UNKNOWN"),
    "motion_direction": ("OW", "TW", "UNKNOWN"),
}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_nmf(train: np.ndarray, validation: np.ndarray):
    candidates = []
    for rank in RANKS:
        model = NMF(
            n_components=rank, init="nndsvda", solver="cd", max_iter=1500,
            tol=1e-4, random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train)
        val_scores = model.transform(validation)
        error = float(
            np.linalg.norm(validation - val_scores @ model.components_)
            / max(np.linalg.norm(validation), 1e-12)
        )
        candidates.append({
            "rank": rank, "validation_relative_frobenius_error": error,
            "iterations": int(model.n_iter_),
            "convergence_warning": any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
        })
    best_error = min(item["validation_relative_frobenius_error"] for item in candidates)
    chosen = min(
        item["rank"] for item in candidates
        if item["validation_relative_frobenius_error"] <= best_error * (1 + RANK_TOLERANCE)
    )
    final = NMF(
        n_components=chosen, init="nndsvda", solver="cd", max_iter=1500,
        tol=1e-4, random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final.fit(train)
    return final, candidates, chosen, [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]


def activation_stats(values: np.ndarray, name: str, split: str) -> dict:
    finite = values[np.isfinite(values)]
    negative = finite[finite < 0]
    return {
        "split": split, "activation": name, "shape": "x".join(map(str, values.shape)),
        "finite_values": int(finite.size), "nonfinite_values": int(values.size - finite.size),
        "minimum": float(finite.min()), "maximum": float(finite.max()),
        "negative_fraction": float(np.mean(finite < 0)),
        "zero_fraction": float(np.mean(finite == 0)),
        "negative_absolute_mass_fraction": float(
            np.abs(negative).sum() / max(np.abs(finite).sum(), 1e-12)
        ),
    }


def capture_split(model, dataset):
    encoder_layers = model.base.model.encoder.layers
    decoder_layers = model.base.model.decoder.layers
    current: dict[str, torch.Tensor] = {}
    call_counts = {"encoder_" + str(i): 0 for i in range(len(encoder_layers))}
    override = {"value": None}
    activations = defaultdict(list)

    def make_encoder_hook(index):
        key = "encoder_" + str(index)

        def hook(_module, inputs):
            call = call_counts[key]
            call_counts[key] += 1
            value = inputs[0]
            if call == 0:
                replacement = override["value"]
                if replacement is not None and index == len(encoder_layers) - 1:
                    value = replacement.to(device=value.device, dtype=value.dtype)
                    return (value,)
                current[key + "_boxes"] = value.detach().cpu()
            elif call == 1:
                current[key + "_speed"] = value.detach().cpu()
            else:
                raise RuntimeError(f"Unexpected encoder FFN call count for {key}: {call}")
            return None

        return hook

    def make_decoder_hook(index):
        def hook(_module, inputs):
            current["decoder_" + str(index)] = inputs[0].detach().cpu()
            return None

        return hook

    handles = []
    for index, layer in enumerate(encoder_layers):
        handles.append(layer.feed_forward.w_2.register_forward_pre_hook(make_encoder_hook(index)))
    for index, layer in enumerate(decoder_layers):
        handles.append(layer.feed_forward.w_2.register_forward_pre_hook(make_decoder_hook(index)))
    handles.append(
        model.context_encoder.projection[2].register_forward_hook(
            lambda _module, _inputs, output: current.__setitem__(
                "context_post_gelu", output.detach().cpu()
            )
        )
    )

    out = {
        "ids": [], "predictions": [], "gates": [], "gate_logits": [],
        "encoder_boxes": defaultdict(list), "encoder_speed": defaultdict(list),
        "decoder": defaultdict(list), "context": [],
    }
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            current.clear()
            for key in call_counts:
                call_counts[key] = 0
            prediction = model.forward_batch(batch)
            out["ids"].extend(batch["sample_id"])
            out["predictions"].append(prediction.cpu().numpy())
            out["gates"].append(model.last_context_gate.cpu().numpy())
            out["gate_logits"].append(model.last_context_gate_logits.cpu().numpy())
            out["context"].append(current["context_post_gelu"].numpy())
            for index in range(len(encoder_layers)):
                out["encoder_boxes"][index].append(current[f"encoder_{index}_boxes"].numpy())
                out["encoder_speed"][index].append(current[f"encoder_{index}_speed"].numpy())
            for index in range(len(decoder_layers)):
                out["decoder"][index].append(current[f"decoder_{index}"].numpy())
    for handle in handles:
        handle.remove()
    for key in ("predictions", "gates", "gate_logits", "context"):
        out[key] = np.concatenate(out[key], axis=0)
    for key in ("encoder_boxes", "encoder_speed", "decoder"):
        out[key] = {index: np.concatenate(chunks, axis=0) for index, chunks in out[key].items()}
    return out


def save_rank_results(path: Path, records: list[dict]) -> None:
    write_csv(
        path,
        ["representation", "candidate_rank", "validation_relative_frobenius_error",
         "fit_iterations", "convergence_warning", "selected_rank",
         "selection_rule", "final_fit_convergence_warnings"],
        records,
    )


def scene_label(field: str, value: int) -> str:
    labels = SCENE_LABELS[field]
    return labels[value] if 0 <= value < len(labels) else f"invalid:{value}"


def run_sobol(model, dataset, stats, reference_index, reference_acts,
              components, train_window_scores, component_count):
    reference = dataset[reference_index]
    inputs = {
        key: value.unsqueeze(0)
        for key, value in reference.items() if isinstance(value, torch.Tensor)
    }
    train_means = train_window_scores.mean(axis=1)
    samples = SOBOL_BASE_SAMPLES
    uniforms = qmc.Sobol(
        d=2 * component_count, scramble=True, seed=SEED + 99
    ).random_base2(m=int(np.log2(samples)))
    A = np.column_stack([
        np.quantile(train_means[:, j], uniforms[:, j])
        for j in range(component_count)
    ])
    B = np.column_stack([
        np.quantile(train_means[:, j], uniforms[:, component_count + j])
        for j in range(component_count)
    ])
    profiles = reference_acts / np.maximum(reference_acts.mean(axis=0, keepdims=True), 1e-12)
    override = {"value": None}

    def repeat_inputs(count: int):
        return {
            key: value.repeat((count,) + (1,) * (value.ndim - 1))
            for key, value in inputs.items()
        }

    def evaluate(coefficients: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, len(coefficients), SOBOL_BATCH_SIZE):
            block = coefficients[start:start + SOBOL_BATCH_SIZE]
            forced = np.einsum("bk,tk,kd->btd", block, profiles, components)
            override["value"] = torch.from_numpy(forced.astype(np.float32))
            batch = repeat_inputs(len(block))
            for key in call_counts:
                call_counts[key] = 0
            with torch.no_grad():
                prediction = model.forward_batch(batch).cpu().numpy()
            raw = prediction * stats[1] + stats[0]
            last_box = reference["boxes"][-1].numpy() * stats[1] + stats[0]
            last_center = np.array([
                (last_box[0] + last_box[2]) / 2,
                (last_box[1] + last_box[3]) / 2,
            ])
            final_center = np.column_stack((
                (raw[:, -1, 0] + raw[:, -1, 2]) / 2,
                (raw[:, -1, 1] + raw[:, -1, 3]) / 2,
            ))
            outputs.append(final_center - last_center)
        override["value"] = None
        return np.concatenate(outputs, axis=0)

    encoder_layers = model.base.model.encoder.layers
    call_counts = {"encoder_" + str(i): 0 for i in range(len(encoder_layers))}
    handles = []

    def make_hook(index):
        key = "encoder_" + str(index)

        def hook(_module, inputs):
            call = call_counts[key]
            call_counts[key] += 1
            if call == 0 and index == len(encoder_layers) - 1:
                return (override["value"].to(device=inputs[0].device, dtype=inputs[0].dtype),)
            return None

        return hook

    for index, layer in enumerate(encoder_layers):
        handles.append(layer.feed_forward.w_2.register_forward_pre_hook(make_hook(index)))

    y_a = evaluate(A)
    y_b = evaluate(B)
    y_ab = []
    for concept in range(component_count):
        mixed = A.copy()
        mixed[:, concept] = B[:, concept]
        y_ab.append(evaluate(mixed))
    for handle in handles:
        handle.remove()

    rows = []
    for axis, axis_name in enumerate(("dx_pixels", "dy_pixels")):
        variance = max(float(np.var(np.concatenate((y_a[:, axis], y_b[:, axis])))), 1e-12)
        for concept in range(component_count):
            rows.append({
                "reference_sample_id": reference["sample_id"],
                "output": axis_name, "concept": f"encoder_ffn_{concept + 1}",
                "first_order_sobol": float(
                    1 - np.mean((y_b[:, axis] - y_ab[concept][:, axis]) ** 2)
                    / (2 * variance)
                ),
                "total_order_sobol": float(
                    np.mean((y_a[:, axis] - y_ab[concept][:, axis]) ** 2) / (2 * variance)
                ),
                "base_samples": samples,
                "interpretation": "single deterministic training reference window; empirical independent concept coefficients",
            })
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    set_seed(SEED)
    torch.set_num_threads(4)
    datasets, stats = load_data()
    model = model_for(NAMES[2])
    checkpoint = RESULTS / "runs" / "E1" / slug(NAMES[2]) / str(SEED) / "checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    captured = {
        split: capture_split(model, datasets[split])
        for split in ("train", "validation")
    }
    if len(captured["train"]["ids"]) != 212 or len(captured["validation"]["ids"]) != 43:
        raise AssertionError("Unexpected frozen split cardinality")

    activation_check_rows = []
    for split, result in captured.items():
        for layer, values in result["encoder_boxes"].items():
            activation_check_rows.append(activation_stats(values, f"encoder_layer_{layer + 1}_box_ffn_relu", split))
        for layer, values in result["encoder_speed"].items():
            activation_check_rows.append(activation_stats(values, f"encoder_layer_{layer + 1}_speed_ffn_relu", split))
        for layer, values in result["decoder"].items():
            activation_check_rows.append(activation_stats(values, f"decoder_layer_{layer + 1}_ffn_relu", split))
        activation_check_rows.append(activation_stats(result["context"], "context_projection_post_gelu", split))
    write_csv(
        OUT / "CRAFT_ACTIVATION_CHECKS_seed20260928.csv",
        list(activation_check_rows[0]), activation_check_rows,
    )

    train_encoder = captured["train"]["encoder_boxes"][2]
    val_encoder = captured["validation"]["encoder_boxes"][2]
    train_encoder_matrix = train_encoder.reshape(-1, train_encoder.shape[-1])
    val_encoder_matrix = val_encoder.reshape(-1, val_encoder.shape[-1])
    train_context_raw = captured["train"]["context"].reshape(len(datasets["train"]), -1)
    val_context_raw = captured["validation"]["context"].reshape(len(datasets["validation"]), -1)
    train_context = np.maximum(train_context_raw, 0)
    val_context = np.maximum(val_context_raw, 0)

    encoder_nmf, encoder_errors, encoder_rank, encoder_warnings = fit_nmf(
        train_encoder_matrix, val_encoder_matrix
    )
    context_nmf, context_errors, context_rank, context_warnings = fit_nmf(
        train_context, val_context
    )
    rank_rows = []
    for representation, candidates, chosen, fit_warnings in (
        ("final_encoder_box_ffn", encoder_errors, encoder_rank, encoder_warnings),
        ("context_post_gelu_clamped", context_errors, context_rank, context_warnings),
    ):
        for candidate in candidates:
            rank_rows.append({
                "representation": representation, "candidate_rank": candidate["rank"],
                "validation_relative_frobenius_error": candidate["validation_relative_frobenius_error"],
                "fit_iterations": candidate["iterations"],
                "convergence_warning": candidate["convergence_warning"],
                "selected_rank": chosen,
                "selection_rule": f"smallest rank within {RANK_TOLERANCE:.0%} of best validation reconstruction error",
                "final_fit_convergence_warnings": " | ".join(fit_warnings),
            })
    save_rank_results(OUT / "CRAFT_RANK_SELECTION_seed20260928.csv", rank_rows)

    train_encoder_h = encoder_nmf.transform(train_encoder_matrix).reshape(
        len(datasets["train"]), train_encoder.shape[1], encoder_rank
    )
    val_encoder_h = encoder_nmf.transform(val_encoder_matrix).reshape(
        len(datasets["validation"]), val_encoder.shape[1], encoder_rank
    )
    train_context_h = context_nmf.transform(train_context)
    val_context_h = context_nmf.transform(val_context)
    all_scores = []
    time_rows = []
    for split, result, encoder_h, context_h in (
        ("train", captured["train"], train_encoder_h, train_context_h),
        ("validation", captured["validation"], val_encoder_h, val_context_h),
    ):
        source_samples = datasets[split].samples
        for index, sample in enumerate(source_samples):
            labels = sample["scene_context"][0].astype(int).tolist()
            common = {
                "split": split, "sample_id": sample["id"], "pid": sample["pid"],
                "gate": int(result["gates"][index]),
                **{field: scene_label(field, value) for field, value in zip(SCENE_FIELDS, labels)},
            }
            for concept in range(encoder_rank):
                time_scores = encoder_h[index, :, concept]
                all_scores.append({
                    **common, "representation": "final_encoder_box_ffn",
                    "concept": f"encoder_ffn_{concept + 1}",
                    "mean_activation": float(time_scores.mean()),
                    "max_activation": float(time_scores.max()),
                })
                for step, value in enumerate(time_scores, start=1):
                    time_rows.append({
                        "split": split, "sample_id": sample["id"], "pid": sample["pid"],
                        "observed_step": step, "concept": f"encoder_ffn_{concept + 1}",
                        "nmf_activation": float(value), "gate": int(result["gates"][index]),
                    })
            for concept in range(context_rank):
                all_scores.append({
                    **common, "representation": "context_post_gelu_clamped",
                    "concept": f"context_{concept + 1}",
                    "mean_activation": float(context_h[index, concept]),
                    "max_activation": float(context_h[index, concept]),
                })
    write_csv(
        OUT / "CRAFT_CONCEPT_SCORES_seed20260928.csv",
        ["split", "sample_id", "pid", *SCENE_FIELDS, "gate", "representation",
         "concept", "mean_activation", "max_activation"], all_scores,
    )
    write_csv(
        OUT / "CRAFT_TIME_ATTRIBUTION_seed20260928.csv",
        ["split", "sample_id", "pid", "observed_step", "concept", "nmf_activation", "gate"],
        time_rows,
    )

    association = defaultdict(list)
    for row in all_scores:
        if row["split"] != "train":
            continue
        for field in (*SCENE_FIELDS, "gate"):
            value = row[field]
            association[(row["representation"], row["concept"], field, value)].append(row)
    association_rows = []
    for (representation, concept, field, value), group in sorted(association.items()):
        pids = {row["pid"] for row in group}
        association_rows.append({
            "representation": representation, "concept": concept, "field": field,
            "value": value,
            "windows": len(group), "unique_pedestrians": len(pids),
            "mean_activation": float(np.mean([row["mean_activation"] for row in group])),
            "admission_rate": float(np.mean([int(row["gate"]) for row in group])),
        })
    write_csv(
        OUT / "CRAFT_SCENE_GATE_ASSOCIATION_seed20260928.csv",
        ["representation", "concept", "field", "value", "windows", "unique_pedestrians",
         "mean_activation", "admission_rate"], association_rows,
    )

    reference_index = min(
        range(len(datasets["train"])),
        key=lambda index: datasets["train"].samples[index]["id"],
    )
    reference_sample = datasets["train"].samples[reference_index]
    reference_h = train_encoder_h[reference_index]
    reference = datasets["train"][reference_index]
    reference_batch = {
        key: value.unsqueeze(0)
        for key, value in reference.items() if isinstance(value, torch.Tensor)
    }
    time_contribution_rows = []
    encoder_layers = model.base.model.encoder.layers
    original_reference_prediction = captured["train"]["predictions"][reference_index]
    original_raw = original_reference_prediction * stats[1] + stats[0]
    last_box = reference_sample["bbox"][14]
    last_center = np.array([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2])
    original_center = np.array([
        (original_raw[-1, 0] + original_raw[-1, 2]) / 2,
        (original_raw[-1, 1] + original_raw[-1, 3]) / 2,
    ])
    original_displacement = original_center - last_center
    override = {"value": None}
    calls = {"count": 0}
    hooks = []

    def intervene(_module, inputs):
        call = calls["count"]
        calls["count"] += 1
        if call == 0 and override["value"] is not None:
            return (override["value"].to(inputs[0]),)
        return None

    hooks.append(encoder_layers[-1].feed_forward.w_2.register_forward_pre_hook(intervene))
    components = encoder_nmf.components_

    def prediction_with_activation(activation):
        override["value"] = torch.from_numpy(activation.astype(np.float32))
        calls["count"] = 0
        with torch.no_grad():
            value = model.forward_batch(reference_batch).cpu().numpy()[0]
        raw = value * stats[1] + stats[0]
        center = np.array([(raw[-1, 0] + raw[-1, 2]) / 2, (raw[-1, 1] + raw[-1, 3]) / 2])
        return center - last_center

    reconstructed = np.einsum("tk,kd->td", reference_h, components)
    reconstructed_displacement = prediction_with_activation(reconstructed[None])
    for concept in range(encoder_rank):
        for step in range(reference_h.shape[0]):
            ablated = reference_h.copy()
            ablated[step, concept] = 0
            ablated_activation = np.einsum("tk,kd->td", ablated, components)
            displacement = prediction_with_activation(ablated_activation[None])
            time_contribution_rows.append({
                "reference_sample_id": reference_sample["id"],
                "concept": f"encoder_ffn_{concept + 1}", "observed_step": step + 1,
                "delta_dx_pixels_vs_reconstructed": float(reconstructed_displacement[0] - displacement[0]),
                "delta_dy_pixels_vs_reconstructed": float(reconstructed_displacement[1] - displacement[1]),
                "reference_gate": int(captured["train"]["gates"][reference_index]),
            })
    for hook in hooks:
        hook.remove()
    override["value"] = None
    sobol_rows = run_sobol(
        model, datasets["train"], stats, reference_index, reference_h, components,
        train_encoder_h, encoder_rank,
    )
    write_csv(
        OUT / "CRAFT_SOBOL_seed20260928.csv",
        ["reference_sample_id", "output", "concept", "first_order_sobol",
         "total_order_sobol", "base_samples", "interpretation"], sobol_rows,
    )
    write_csv(
        OUT / "CRAFT_TIME_ABLATION_seed20260928.csv",
        ["reference_sample_id", "concept", "observed_step",
         "delta_dx_pixels_vs_reconstructed", "delta_dy_pixels_vs_reconstructed",
         "reference_gate"], time_contribution_rows,
    )

    ranking = train_encoder_h.mean(axis=1)
    used = set()
    example_indices = []
    for concept in range(encoder_rank):
        for candidate in np.argsort(-ranking[:, concept], kind="stable"):
            if int(candidate) not in used:
                used.add(int(candidate))
                example_indices.append((concept, int(candidate)))
                break
    metadata_examples = []
    video = None
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)
    video = cv2.VideoCapture(str(VIDEO))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open local training video: {VIDEO}")
    cell_width, image_height, header_height = 720, 405, 86
    columns = 2
    rows = (len(example_indices) + columns - 1) // columns
    canvas = np.full(
        (rows * (image_height + header_height), columns * cell_width, 3),
        245, dtype=np.uint8,
    )
    colors = ((251, 166, 0), (10, 214, 255), (110, 0, 255))

    def draw_path(image, points, color, frame_shape):
        scale_x = cell_width / frame_shape[1]
        scale_y = image_height / frame_shape[0]
        scaled = np.round(points * np.array([scale_x, scale_y])).astype(int)
        for first, second in zip(scaled, scaled[1:]):
            cv2.line(image, tuple(first), tuple(second), color, 2, cv2.LINE_AA)

    for concept, index in example_indices:
        sample = datasets["train"].samples[index]
        frame_index = int(sample["frame_start"]) + 14
        frame_path = Path(sample["image"][14])
        frame = cv2.imread(str(frame_path)) if frame_path.is_file() else None
        if frame is None:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = video.read()
            if not ok:
                raise RuntimeError(f"Unable to decode selected training frame {frame_index}")
        frame_shape = frame.shape
        frame = cv2.resize(frame, (cell_width, image_height), interpolation=cv2.INTER_AREA)
        raw_pred = captured["train"]["predictions"][index] * stats[1] + stats[0]
        boxes = sample["bbox"]
        observed, future = boxes[:15], boxes[15:]
        observed_centers = np.column_stack(((observed[:, 0] + observed[:, 2]) / 2,
                                            (observed[:, 1] + observed[:, 3]) / 2))
        future_centers = np.column_stack(((future[:, 0] + future[:, 2]) / 2,
                                          (future[:, 1] + future[:, 3]) / 2))
        predicted_centers = np.column_stack(((raw_pred[:, 0] + raw_pred[:, 2]) / 2,
                                             (raw_pred[:, 1] + raw_pred[:, 3]) / 2))
        draw_path(frame, observed_centers, colors[0], frame_shape)
        draw_path(frame, future_centers, colors[1], frame_shape)
        draw_path(frame, predicted_centers, colors[2], frame_shape)
        scale_x = cell_width / frame_shape[1]
        scale_y = image_height / frame_shape[0]
        x1, y1, x2, y2 = observed[-1]
        cv2.rectangle(
            frame, (int(x1 * scale_x), int(y1 * scale_y)),
            (int(x2 * scale_x), int(y2 * scale_y)), (255, 255, 255), 2,
        )
        row, column = divmod(concept, columns)
        x_offset = column * cell_width
        y_offset = row * (image_height + header_height)
        mse = float(np.mean((raw_pred - future) ** 2))
        title = (
            f"Encoder FFN concept {concept + 1} | score={ranking[index, concept]:.4g} "
            f"| gate={int(captured['train']['gates'][index])} | MSE={mse:.1f}"
        )
        cv2.putText(canvas, title, (x_offset + 8, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "/".join(sample["id"].split("/")[-3:]),
                    (x_offset + 8, y_offset + 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Observed", (x_offset + 8, y_offset + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[0], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Ground truth", (x_offset + 110, y_offset + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[1], 1, cv2.LINE_AA)
        cv2.putText(canvas, "PACE-Net", (x_offset + 245, y_offset + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[2], 1, cv2.LINE_AA)
        canvas[y_offset + header_height:y_offset + header_height + image_height,
               x_offset:x_offset + cell_width] = frame
        metadata_examples.append({
            "concept": concept + 1, "sample_id": sample["id"],
            "pid": sample["pid"], "frame": frame_index,
            "mean_nmf_activation": float(ranking[index, concept]),
            "gate": int(captured["train"]["gates"][index]),
        })
    if not cv2.imwrite(str(OUT / "CRAFT_NMF_EXAMPLES_seed20260928.png"), canvas):
        raise RuntimeError("Unable to write the NMF concept example image")
    video.release()

    clamp = {
        split: activation_stats(result["context"], "context_post_gelu", split)
        for split, result in captured.items()
    }
    summary = {
        "analysis": "exploratory CRAFT-inspired NMF and functional concept interventions; not the original image-classifier CRAFT pipeline",
        "checkpoint": str(checkpoint),
        "split": str(SPLIT),
        "training_windows": len(datasets["train"]),
        "training_pedestrians": len({sample["pid"] for sample in datasets["train"].samples}),
        "validation_windows": len(datasets["validation"]),
        "shared_split_pickle_loaded": True,
        "test_dataset_iterated": False,
        "test_predictions_or_labels_used": False,
        "model_weights_updated": False,
        "rank_tolerance": RANK_TOLERANCE,
        "encoder_rank": encoder_rank,
        "context_rank": context_rank,
        "encoder_rank_errors": encoder_errors,
        "context_rank_errors": context_errors,
        "encoder_final_fit_convergence_warnings": encoder_warnings,
        "context_final_fit_convergence_warnings": context_warnings,
        "context_post_gelu_clamping": {
            split: {
                "negative_entry_fraction": values["negative_fraction"],
                "negative_absolute_mass_fraction": values["negative_absolute_mass_fraction"],
            }
            for split, values in clamp.items()
        },
        "train_admission_count": int(captured["train"]["gates"].sum()),
        "train_admission_windows": len(captured["train"]["gates"]),
        "validation_admission_count": int(captured["validation"]["gates"].sum()),
        "validation_admission_windows": len(captured["validation"]["gates"]),
        "time_ablation_reference": {
            "sample_id": reference_sample["id"],
            "natural_prediction_displacement_xy_pixels": original_displacement.tolist(),
            "nmf_reconstruction_prediction_displacement_xy_pixels": reconstructed_displacement.tolist(),
            "absolute_reconstruction_shift_xy_pixels": np.abs(
                original_displacement - reconstructed_displacement
            ).tolist(),
            "gate": int(captured["train"]["gates"][reference_index]),
        },
        "sobol_base_samples": SOBOL_BASE_SAMPLES,
        "visual_examples": metadata_examples,
    }
    (OUT / "CRAFT_RUN_METADATA_seed20260928.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUT), "encoder_rank": encoder_rank, "context_rank": context_rank,
        "train_gate_admissions": summary["train_admission_count"],
        "validation_gate_admissions": summary["validation_admission_count"],
        "clamp": summary["context_post_gelu_clamping"],
        "examples": metadata_examples,
    }, indent=2))


if __name__ == "__main__":
    main()
