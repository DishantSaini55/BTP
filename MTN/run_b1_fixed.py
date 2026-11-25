"""B1 diagnostic rerun with autoregressive training and gradient clipping.

Creates an independent run directory; never overwrites existing B1 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .run_b1_from_scratch import VELOCITY_FRAMES
    from .run_from_scratch_e1 import MAX_EPOCHS, PATIENCE, RESULTS, SEEDS, SPLIT, load_data, source_provenance
    from .context_schema import SCENE_VOCAB_SIZES
except ImportError:
    from run_b1_from_scratch import VELOCITY_FRAMES
    from run_from_scratch_e1 import MAX_EPOCHS, PATIENCE, RESULTS, SEEDS, SPLIT, load_data, source_provenance
    from context_schema import SCENE_VOCAB_SIZES

TRAINED = ("LSTM (boxes + ego speed)", "LSTM (boxes + ego speed + scene labels)")


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_").lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


class FixedLSTMBaseline(nn.Module):
    def __init__(self, use_scene: bool) -> None:
        super().__init__()
        self.use_scene = use_scene
        self.encoder = nn.LSTM(5, 32, batch_first=True)
        self.decoder = nn.LSTM(4, 32, batch_first=True)
        self.output = nn.Linear(32, 4)
        if use_scene:
            scene_sizes = SCENE_VOCAB_SIZES
            self.scene_embeddings = nn.ModuleList([nn.Embedding(size, 4) for size in scene_sizes])
            self.scene_projection = nn.Linear(20, 32)

    def forward(self, boxes: torch.Tensor, speed: torch.Tensor, scene_context: torch.Tensor,
                targets: torch.Tensor | None = None) -> torch.Tensor:
        encoder_input = torch.cat((boxes, speed[:, :boxes.shape[1]]), dim=-1)
        encoded, (hidden, cell) = self.encoder(encoder_input)
        if encoded.shape[1] == 0:
            raise ValueError("Empty observed sequence")
        if self.use_scene:
            pieces = [
                embedding(scene_context[:, index].long().clamp(0, embedding.num_embeddings - 1))
                for index, embedding in enumerate(self.scene_embeddings)
            ]
            context = self.scene_projection(torch.cat(pieces, dim=-1)).unsqueeze(0)
            hidden = hidden + context
        previous = boxes[:, -1]
        outputs = []
        for step in range(45):
            if targets is not None:
                previous = boxes[:, -1] if step == 0 else targets[:, step - 1]
            decoded, (hidden, cell) = self.decoder(previous.unsqueeze(1), (hidden, cell))
            displacement = self.output(decoded[:, 0])
            prediction = previous + displacement
            outputs.append(prediction)
            if targets is None:
                previous = prediction
        return torch.stack(outputs, dim=1)


def mean_loss(model: nn.Module, loader: DataLoader, loss: nn.Module) -> tuple[float, float]:
    total, count = 0.0, 0
    train_loss = 0.0
    for batch in loader:
        prediction = model(
            batch["boxes"].float(), batch["speed"].float(),
            batch["scene_context"],
        )
        value = loss(prediction, batch["pred_target"].float())
        batch_count = batch["boxes"].shape[0]
        total += value.item() * batch_count
        count += batch_count
    return total / count, train_loss


def train_loss(model: nn.Module, loader: DataLoader, loss: nn.Module) -> float:
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["boxes"].float(), batch["speed"].float(),
                batch["scene_context"],
            )
            value = loss(prediction, batch["pred_target"].float())
            batch_count = batch["boxes"].shape[0]
            total += value.item() * batch_count
            count += batch_count
    return total / count


def evaluate(model: nn.Module, loader: DataLoader, stats: tuple) -> tuple[dict, list[dict], dict]:
    mean, std, _, _ = stats
    rows, predictions, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["boxes"].float(), batch["speed"].float(), batch["scene_context"]
            ).cpu()
            target = batch["pred_target"].cpu()
            raw_prediction = prediction.numpy() * std + mean
            raw_target = target.numpy() * std + mean
            predictions.append(raw_prediction)
            targets.append(raw_target)
            for index in range(prediction.shape[0]):
                cp = (raw_prediction[index, :, :2] + raw_prediction[index, :, 2:]) / 2
                ct = (raw_target[index, :, :2] + raw_target[index, :, 2:]) / 2
                rows.append({
                    "sample_id": batch["sample_id"][index],
                    "video": batch["video"][index],
                    "pid": batch["pid"][index],
                    "frame_start": int(batch["frame_start"][index]),
                    "frame_end": int(batch["frame_end"][index]),
                    "mse": float(np.mean((raw_prediction[index] - raw_target[index]) ** 2)),
                    "cmse": float(np.mean((cp - ct) ** 2)),
                    "cfmse": float(np.mean((cp[-1] - ct[-1]) ** 2)),
                })
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in ("mse", "cmse", "cfmse")}
    return metrics, rows, {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets)}


def run_one(name: str, seed: int, datasets: dict, stats: tuple) -> None:
    run = RESULTS / "runs" / "B1_AUTOREGRESSIVE" / slug(name) / str(seed)
    config = RESULTS / "config" / f"B1_AUTOREGRESSIVE_{slug(name)}_{seed}.json"
    if run.exists() or config.exists():
        print(f"RESUME: reuse existing {run}")
        return
    run.mkdir(parents=True)
    set_seed(seed)
    model = FixedLSTMBaseline(use_scene=name == TRAINED[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    train = DataLoader(datasets["train"], batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    valid = DataLoader(datasets["validation"], batch_size=8, shuffle=False)
    test = DataLoader(datasets["test"], batch_size=8, shuffle=False)
    history = []
    best = float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch in train:
            optimizer.zero_grad()
            prediction = model(
                batch["boxes"].float(), batch["speed"].float(),
                batch["scene_context"],
            )
            value = loss(prediction, batch["pred_target"].float())
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_count = batch["boxes"].shape[0]
            epoch_loss += value.item() * batch_count
            epoch_count += batch_count
        model.eval()
        validation = mean_loss(model, valid, loss)[0]
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_count,
            "validation_mse": validation,
        })
        if validation < best:
            best = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    checkpoint = run / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    test_metrics, rows, arrays = evaluate(model, test, stats)
    metadata = json.loads((SPLIT / "split_metadata.json").read_text())
    test_metrics.update({
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "best_validation_mse": best,
        "epochs_completed": len(history),
        "stopping_reason": "early stopping" if len(history) < MAX_EPOCHS else "maximum epochs",
        "training_seconds": time.perf_counter() - started,
        "test_sample_id_sha256": metadata["test_sample_id_sha256"],
    })
    (run / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(run / "predictions.npz", **arrays)
    with (run / "per_sample.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run / "train_log.txt").write_text(
        "\n".join(
            f"{item['epoch']},{item['train_loss']:.10f},{item['validation_mse']:.10f}"
            for item in history
        ) + "\n",
        encoding="utf-8",
    )
    initial = [
        {"name": key, "shape": list(value.shape),
         "sha256": hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()}
        for key, value in model.state_dict().items()
    ]
    config.write_text(json.dumps({
        "model": name,
        "seed": seed,
        "initialization": "from scratch; no official checkpoint loaded",
        "loaded_keys": [],
        "initial_state": initial,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": {
            "encoder": "LSTM(5,32)",
            "decoder": "autoregressive LSTM(4,32)",
            "scene_variant": bool(model.use_scene),
            "training": "autoregressive normalized-box rollout",
            "gradient_clip_norm": 1.0,
        },
        "protocol": {
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": 8,
            "loss": "MSE",
            "validation_rollout": "autoregressive; matches test inference",
        },
        "provenance": source_provenance(),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"model": name, "seed": seed, "metrics": test_metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    datasets, stats = load_data()
    for name in TRAINED:
        for seed in args.seeds:
            run_one(name, seed, datasets, stats)


if __name__ == "__main__":
    main()
