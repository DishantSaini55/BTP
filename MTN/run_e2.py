"""Post-hoc E2 gate ablations for the learned PACE-Net E1 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
    from .context_TF import ContextIndividualTF
    from .run_from_scratch_e1 import ROOT, RESULTS, SPLIT, SEEDS, NAMES, WindowDataset, load_data
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared
    from context_TF import ContextIndividualTF
    from run_from_scratch_e1 import ROOT, RESULTS, SPLIT, SEEDS, NAMES, WindowDataset, load_data

BASE_VARIANTS = (
    ("PACE-Net without SATIE (SAGE + MASC)", "pace_net_without_satie_sage_plus_masc", False),
    ("PACE-Net (SAGE + MASC + SATIE)", "pace_net_sage_plus_masc_plus_satie", True),
)
ABLATIONS = ("learned", "always_open", "always_closed", "soft_sigmoid", "context_only")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def build_model(satie: bool) -> torch.nn.Module:
    model = ContextIndividualTF(15, 4, 4, 4, N=3, d_model=32, d_ff=128, h=4, dropout=.1)
    if satie:
        model = make_c4_encoder_self_attention_shared(model)
    return model


def apply_ablation(model: ContextIndividualTF, ablation: str) -> None:
    base = model if hasattr(model, "context_encoder") else model.base
    if ablation == "always_open":
        base._context_gate = lambda context_token, observed_boxes: torch.ones(
            context_token.shape[0], dtype=context_token.dtype, device=context_token.device
        )
    elif ablation == "always_closed":
        base._context_gate = lambda context_token, observed_boxes: torch.zeros(
            context_token.shape[0], dtype=context_token.dtype, device=context_token.device
        )
    elif ablation == "soft_sigmoid":
        def soft_gate(context_token, observed_boxes):
            logits = base.gate(torch.cat((context_token, observed_boxes[:, -1]), dim=-1))
            gate = torch.sigmoid(logits[:, 1])
            base.last_context_gate = gate.detach()
            base.last_context_gate_logits = logits.detach()
            base._context_gate_for_loss = gate
            return gate
        base._context_gate = soft_gate
        base.gate_mode = "soft"
    elif ablation == "context_only":
        gate_weight = base.gate[-1].weight
        with torch.no_grad():
            gate_weight[:, -4:] = 0
    elif ablation != "learned":
        raise ValueError(ablation)


def evaluate(model: torch.nn.Module, loader: DataLoader, stats_config: tuple) -> tuple[dict, np.ndarray, np.ndarray]:
    mean, std, _, _ = stats_config
    predictions, targets, gates, logits = [], [], [], []
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model.forward_batch(batch).cpu()
            target = batch["pred_target"].cpu()
            raw_prediction = prediction.numpy() * std + mean
            raw_target = target.numpy() * std + mean
            predictions.append(raw_prediction)
            targets.append(raw_target)
            if hasattr(model, "last_context_gate") and model.last_context_gate is not None:
                gates.append(model.last_context_gate.cpu().numpy())
                logits.append(model.last_context_gate_logits.cpu().numpy())
            for index in range(prediction.shape[0]):
                centre_prediction = (raw_prediction[index, :, :2] + raw_prediction[index, :, 2:4]) / 2
                centre_target = (raw_target[index, :, :2] + raw_target[index, :, 2:4]) / 2
                rows.append({
                    "sample_id": batch["sample_id"][index],
                    "video": batch["video"][index],
                    "pid": batch["pid"][index],
                    "frame_start": int(batch["frame_start"][index]),
                    "frame_end": int(batch["frame_end"][index]),
                    "mse": float(np.mean((raw_prediction[index] - raw_target[index]) ** 2)),
                    "cmse": float(np.mean((centre_prediction - centre_target) ** 2)),
                    "cfmse": float(np.mean((centre_prediction[-1] - centre_target[-1]) ** 2)),
                })
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    metrics = {
        "mse": float(np.mean([row["mse"] for row in rows])),
        "cmse": float(np.mean([row["cmse"] for row in rows])),
        "cfmse": float(np.mean([row["cfmse"] for row in rows])),
    }
    if gates:
        gate_array = np.concatenate(gates)
        logits_array = np.concatenate(logits)
        metrics["admission_rate"] = float(np.mean(gate_array))
        metrics["gate_mean"] = float(np.mean(gate_array))
        metrics["gate_std"] = float(np.std(gate_array, ddof=1))
        metrics["gate_logit_mean"] = float(np.mean(logits_array))
        metrics["gate_logit_std"] = float(np.std(logits_array, ddof=1))
    return metrics, rows, prediction_array


def run_variant(model_name: str, slug: str, satie: bool, ablation: str, seed: int, datasets: dict, stats_config: tuple) -> None:
    out = RESULTS / "runs" / "E2" / f"{slug}_{ablation}" / str(seed)
    if out.exists():
        print(f"RESUME: reuse existing {out}")
        return
    out.mkdir(parents=True)
    set_seed(seed)
    model = build_model(satie)
    checkpoint = RESULTS / "runs" / "E1" / slug / str(seed) / "checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    apply_ablation(model if hasattr(model, "context_encoder") else model.base, ablation)
    metrics, rows, predictions = evaluate(model, DataLoader(datasets["test"], batch_size=8, shuffle=False), stats_config)
    metrics.update({
        "model": model_name,
        "ablation": ablation,
        "seed": seed,
        "reuse_e1_checkpoint": str(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", predictions=predictions)
    with (out / "per_sample.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    datasets, stats_config = load_data()
    for model_name, slug, satie in BASE_VARIANTS:
        for seed in args.seeds:
            for ablation in ABLATIONS:
                if ablation == "learned":
                    continue
                run_variant(model_name, slug, satie, ablation, seed, datasets, stats_config)


if __name__ == "__main__":
    main()
