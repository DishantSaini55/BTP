"""Post-protocol E1 epoch diagnostics: train, validation, and test curves.

Test curves are recorded only as diagnostics. Checkpoint selection remains
validation-only and never uses test MSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .run_from_scratch_e1 import (
        MAX_EPOCHS, PATIENCE, RESULTS, SEEDS, load_data, model_for,
        set_seed, forward, mean_loss, NAMES, slug,
    )
except ImportError:
    from run_from_scratch_e1 import (
        MAX_EPOCHS, PATIENCE, RESULTS, SEEDS, load_data, model_for,
        set_seed, forward, mean_loss, NAMES, slug,
    )

DIAGNOSTIC_NAMES = (NAMES[0], NAMES[1], NAMES[2])


def raw_test_mse(model: torch.nn.Module, loader: DataLoader, stats: tuple) -> float:
    mean, std, _, _ = stats
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            normalized_prediction = forward(model, batch).cpu()
            normalized_target = batch["pred_target"].cpu()
            prediction = normalized_prediction.numpy() * std + mean
            target = normalized_target.numpy() * std + mean
            batch_count = prediction.shape[0]
            total += float(np.mean((prediction - target) ** 2, axis=(1, 2)).sum())
            count += batch_count
    return total / count


def run_one(name: str, seed: int, datasets: dict, stats: tuple) -> None:
    run = RESULTS / "runs" / "E1_DIAGNOSTICS" / slug(name) / str(seed)
    if run.exists():
        print(f"RESUME: reuse existing {run}")
        return
    run.mkdir(parents=True)
    set_seed(seed)
    model = model_for(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()
    train = DataLoader(datasets["train"], batch_size=8, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(datasets["validation"], batch_size=8, shuffle=False)
    test = DataLoader(datasets["test"], batch_size=8, shuffle=False)

    history = []
    best_validation = float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train:
            optimizer.zero_grad()
            value = loss(forward(model, batch), batch["pred_target"])
            value.backward()
            optimizer.step()
            batch_count = batch["pred_target"].shape[0]
            train_total += value.item() * batch_count
            train_count += batch_count
        model.eval()
        with torch.no_grad():
            validation_mse = mean_loss(model, validation, loss)
            test_normalized_mse = mean_loss(model, test, loss)
            test_raw_mse = raw_test_mse(model, test, stats)
        train_mse = train_total / train_count
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "test_normalized_mse": test_normalized_mse,
            "test_raw_mse": test_raw_mse,
        })
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    checkpoint = run / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    selected_epoch = int(np.argmin([row["validation_mse"] for row in history]) + 1)
    test_optimal_epoch = int(np.argmin([row["test_normalized_mse"] for row in history]) + 1)
    (run / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (run / "epoch_history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model": name,
        "seed": seed,
        "epochs_completed": len(history),
        "selected_epoch_validation_only": selected_epoch,
        "test_optimal_epoch_not_used_for_selection": test_optimal_epoch,
        "selected_validation_mse": history[selected_epoch - 1]["validation_mse"],
        "selected_test_normalized_mse": history[selected_epoch - 1]["test_normalized_mse"],
        "selected_test_raw_mse": history[selected_epoch - 1]["test_raw_mse"],
        "test_optimal_normalized_mse": history[test_optimal_epoch - 1]["test_normalized_mse"],
        "test_optimal_raw_mse": history[test_optimal_epoch - 1]["test_raw_mse"],
        "epoch_gap": test_optimal_epoch - selected_epoch,
        "stopping_reason": "early stopping" if len(history) < MAX_EPOCHS else "maximum epochs",
        "training_seconds": time.perf_counter() - started,
    }
    (run / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    datasets, stats = load_data()
    for name in DIAGNOSTIC_NAMES:
        for seed in args.seeds:
            run_one(name, seed, datasets, stats)


if __name__ == "__main__":
    main()
