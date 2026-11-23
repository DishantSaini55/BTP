"""B1 baselines on the frozen GP6 from-scratch split.

The fixed B1 contract is deliberately small: deterministic last-box and
three-frame constant-velocity predictors, plus two 32-hidden-unit LSTM
encoder/decoders (box+speed and box+speed+five static scene labels).  The
trained baselines use exactly the split, normalization, Adam schedule, and
validation-only early stopping rule defined in results_paper/config/SPLIT.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .run_from_scratch_e1 import (
        MAX_EPOCHS, NAMES, PATIENCE, RESULTS, SEEDS, SPLIT, WindowDataset,
        load_data, source_provenance,
    )
    from .context_schema import SCENE_VOCAB_SIZES
except ImportError:
    from run_from_scratch_e1 import (
        MAX_EPOCHS, NAMES, PATIENCE, RESULTS, SEEDS, SPLIT, WindowDataset,
        load_data, source_provenance,
    )
    from context_schema import SCENE_VOCAB_SIZES

VELOCITY_FRAMES = 3
TRAINED = ("LSTM (boxes + ego speed)", "LSTM (boxes + ego speed + scene labels)")
DETERMINISTIC = ("Constant position", "Constant velocity (last 3 frames)")


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_").lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


class LSTMBaseline(nn.Module):
    """A plain autoregressive LSTM decoder; optional static label embeddings."""
    def __init__(self, use_scene: bool) -> None:
        super().__init__()
        self.use_scene = use_scene
        self.encoder = nn.LSTM(5, 32, batch_first=True)
        self.decoder = nn.LSTM(4, 32, batch_first=True)
        self.output = nn.Linear(32, 4)
        if use_scene:
            # The known-cardinality scene codes match context_schema.py.
            self.scene_embeddings = nn.ModuleList([nn.Embedding(size, 4) for size in SCENE_VOCAB_SIZES])
            self.scene_projection = nn.Linear(20, 32)

    def forward(self, boxes: torch.Tensor, speed: torch.Tensor, scene_context: torch.Tensor,
                targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded, (hidden, cell) = self.encoder(torch.cat((boxes, speed[:, :boxes.shape[1]]), dim=-1))
        if self.use_scene:
            pieces = [embedding(scene_context[:, index].long().clamp(0, embedding.num_embeddings - 1))
                      for index, embedding in enumerate(self.scene_embeddings)]
            context = self.scene_projection(torch.cat(pieces, dim=-1)).unsqueeze(0)
            hidden = hidden + context
        previous = boxes[:, -1]
        outputs = []
        for _ in range(45):
            decoded, (hidden, cell) = self.decoder(previous.unsqueeze(1), (hidden, cell))
            # Predict an offset from the previous predicted normalised box.
            previous = previous + self.output(decoded[:, 0])
            outputs.append(previous)
            if targets is not None:
                previous = targets[:, len(outputs) - 1]
        return torch.stack(outputs, dim=1)


def metric_rows(prediction: np.ndarray, target: np.ndarray, batch: dict) -> list[dict]:
    rows = []
    for index in range(prediction.shape[0]):
        centre_prediction = (prediction[index, :, :2] + prediction[index, :, 2:]) / 2
        centre_target = (target[index, :, :2] + target[index, :, 2:]) / 2
        rows.append({
            "sample_id": batch["sample_id"][index], "video": batch["video"][index],
            "pid": batch["pid"][index], "frame_start": int(batch["frame_start"][index]),
            "frame_end": int(batch["frame_end"][index]),
            "mse": float(np.mean((prediction[index] - target[index]) ** 2)),
            "cmse": float(np.mean((centre_prediction - centre_target) ** 2)),
            "cfmse": float(np.mean((centre_prediction[-1] - centre_target[-1]) ** 2)),
        })
    return rows


def evaluate(model: nn.Module | None, loader: DataLoader, stats, kind: str):
    mean, std, _, _ = stats
    rows, predictions, targets = [], [], []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for batch in loader:
            if kind == DETERMINISTIC[0]:
                prediction = batch["boxes"][:, -1:, :].repeat(1, 45, 1)
            elif kind == DETERMINISTIC[1]:
                velocity = (batch["boxes"][:, -1] - batch["boxes"][:, -VELOCITY_FRAMES]).div(VELOCITY_FRAMES - 1)
                steps = torch.arange(1, 46, dtype=velocity.dtype).view(1, 45, 1)
                prediction = batch["boxes"][:, -1:, :] + steps * velocity.unsqueeze(1)
            else:
                prediction = model(batch["boxes"].float(), batch["speed"].float(), batch["scene_context"])
            target = batch["pred_target"].float()
            raw_prediction = prediction.numpy() * std + mean
            raw_target = target.numpy() * std + mean
            predictions.append(raw_prediction)
            targets.append(raw_target)
            rows.extend(metric_rows(raw_prediction, raw_target, batch))
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in ("mse", "cmse", "cfmse")}
    return metrics, rows, {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets)}


def mean_loss(model: nn.Module, loader: DataLoader, loss: nn.Module) -> float:
    total, count = 0.0, 0
    for batch in loader:
        value = loss(
            model(batch["boxes"].float(), batch["speed"].float(),
                  batch["scene_context"], batch["pred_target"].float()),
            batch["pred_target"].float(),
        )
        total += value.item() * batch["boxes"].shape[0]
        count += batch["boxes"].shape[0]
    return total / count


def write_outputs(run: Path, metrics: dict, rows: list[dict], arrays: dict) -> None:
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(run / "predictions.npz", **arrays)
    with (run / "per_sample.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic(name: str, datasets, stats) -> None:
    run = RESULTS / "runs" / "B1" / slug(name) / "deterministic"
    config = RESULTS / "config" / f"B1_{slug(name)}_deterministic.json"
    if run.exists() or config.exists():
        raise FileExistsError(f"Refusing to overwrite {run}")
    metrics, rows, arrays = evaluate(None, DataLoader(datasets["test"], batch_size=8, shuffle=False), stats, name)
    metadata = json.loads((SPLIT / "split_metadata.json").read_text())
    metrics.update({"parameters": 0, "checkpoint_bytes": 0, "test_sample_id_sha256": metadata["test_sample_id_sha256"]})
    config.write_text(json.dumps({
        "model": name, "initialization": "deterministic non-trained baseline", "loaded_keys": [],
        "velocity_frames": VELOCITY_FRAMES if name == DETERMINISTIC[1] else None,
        "provenance": source_provenance(),
    }, indent=2), encoding="utf-8")
    write_outputs(run, metrics, rows, arrays)
    print(json.dumps({"model": name, "metrics": metrics}, indent=2))


def trained(name: str, seed: int, datasets, stats) -> None:
    run = RESULTS / "runs" / "B1" / slug(name) / str(seed)
    config = RESULTS / "config" / f"B1_{slug(name)}_{seed}.json"
    if run.exists() or config.exists():
        raise FileExistsError(f"Refusing to overwrite {run}")
    run.mkdir(parents=True)
    set_seed(seed)
    model = LSTMBaseline(use_scene=name == TRAINED[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    initial = [{"name": key, "shape": list(value.shape), "sha256": hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()}
               for key, value in model.state_dict().items()]
    config.write_text(json.dumps({
        "model": name, "seed": seed, "initialization": "from scratch; no official checkpoint loaded",
        "loaded_keys": [], "initial_state": initial, "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": {"encoder": "LSTM(5,32)", "decoder": "autoregressive LSTM(4,32)",
                         "scene_variant": bool(model.use_scene), "scene_embedding_dim": 4 if model.use_scene else None},
        "protocol": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": 0, "optimizer": "Adam", "lr": 1e-3, "batch_size": 8, "loss": "MSE"},
        "provenance": source_provenance(),
    }, indent=2), encoding="utf-8")
    train = DataLoader(datasets["train"], batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    valid = DataLoader(datasets["validation"], batch_size=8, shuffle=False)
    best, best_state, stale, history = float("inf"), None, 0, []
    started = time.perf_counter()
    for _epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train:
            optimizer.zero_grad()
            value = loss(
                model(batch["boxes"].float(), batch["speed"].float(),
                      batch["scene_context"], batch["pred_target"].float()),
                batch["pred_target"].float(),
            )
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = mean_loss(model, valid, loss)
        history.append(validation)
        if validation < best:
            best, best_state, stale = validation, {key: value.detach().clone() for key, value in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    checkpoint = run / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    metrics, rows, arrays = evaluate(model, DataLoader(datasets["test"], batch_size=8, shuffle=False), stats, name)
    metadata = json.loads((SPLIT / "split_metadata.json").read_text())
    metrics.update({"parameters": sum(p.numel() for p in model.parameters()), "checkpoint_bytes": checkpoint.stat().st_size,
                    "best_validation_mse": best, "epochs_completed": len(history),
                    "stopping_reason": "early stopping" if len(history) < MAX_EPOCHS else "maximum epochs",
                    "training_seconds": time.perf_counter() - started, "test_sample_id_sha256": metadata["test_sample_id_sha256"]})
    # run already exists, so write remaining artifacts without a second mkdir.
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(run / "predictions.npz", **arrays)
    with (run / "per_sample.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (run / "train_log.txt").write_text("\n".join(map(str, history)) + "\n", encoding="utf-8")
    print(json.dumps({"model": name, "seed": seed, "metrics": metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "run"))
    parser.add_argument("--models", nargs="*", default=DETERMINISTIC + TRAINED)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    datasets, stats = load_data()
    if args.action == "smoke":
        output = RESULTS / "smoke" / "b1_from_scratch_gp6_smoke.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        batch = next(iter(DataLoader(datasets["train"], batch_size=2, shuffle=False)))
        checks = {}
        for name in TRAINED:
            set_seed(SEEDS[0])
            model = LSTMBaseline(name == TRAINED[1]).eval()
            with torch.no_grad():
                prediction = model(batch["boxes"].float(), batch["speed"].float(), batch["scene_context"])
            checks[name] = {"parameters": sum(p.numel() for p in model.parameters()), "shape": list(prediction.shape), "finite": bool(torch.isfinite(prediction).all())}
            if not checks[name]["finite"]:
                raise AssertionError(name)
        for name in DETERMINISTIC:
            _, rows, _ = evaluate(None, DataLoader(datasets["validation"], batch_size=8, shuffle=False), stats, name)
            checks[name] = {"parameters": 0, "validation_windows": len(rows), "finite": bool(np.isfinite([row["mse"] for row in rows]).all())}
            if not checks[name]["finite"]:
                raise AssertionError(name)
        payload = {"checks": checks, "test_sample_id_sha256": json.loads((SPLIT / "split_metadata.json").read_text())["test_sample_id_sha256"]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    for name in args.models:
        if name in DETERMINISTIC:
            deterministic(name, datasets, stats)
        elif name in TRAINED:
            for seed in args.seeds:
                trained(name, seed, datasets, stats)
        else:
            raise ValueError(name)


if __name__ == "__main__":
    main()
