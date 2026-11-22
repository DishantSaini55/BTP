"""E8 gate robustness experiments triggered by the E3 test-range audit."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .c4_encoder_shared import make_c4_encoder_self_attention_shared
    from .context_TF import ContextIndividualTF
    from .run_from_scratch_e1 import load_data, ROOT, RESULTS, SEEDS, mean_loss, forward
except ImportError:
    from c4_encoder_shared import make_c4_encoder_self_attention_shared
    from context_TF import ContextIndividualTF
    from run_from_scratch_e1 import load_data, ROOT, RESULTS, SEEDS, mean_loss, forward

MAX_EPOCHS, PATIENCE = 50, 10
BASES = (
    ("PACE-Net without SATIE (SAGE + MASC)", "pace_net_without_satie_sage_plus_masc", False),
    ("PACE-Net (SAGE + MASC + SATIE)", "pace_net_sage_plus_masc_plus_satie", True),
)
VARIANTS = ("clipped_normalized_x_t", "context_only")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def build_model(satie: bool) -> ContextIndividualTF:
    model = ContextIndividualTF(15, 4, 4, 4, N=3, d_model=32, d_ff=128, h=4, dropout=.1)
    if satie:
        model = make_c4_encoder_self_attention_shared(model)
    return model


def configure_variant(model: ContextIndividualTF, variant: str) -> None:
    base = model if hasattr(model, "context_encoder") else model.base
    if variant == "context_only":
        base.gate = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 2),
        )
        base._context_gate = lambda context_token, observed_boxes: _context_only_gate(base, context_token)
    elif variant == "clipped_normalized_x_t":
        original = base._context_gate
        def clipped_gate(context_token, observed_boxes):
            logits = base.gate(torch.cat((context_token, observed_boxes[:, -1].clamp(-5, 5)), dim=-1))
            if base.training:
                use_context = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
            else:
                use_context = logits.argmax(dim=-1).eq(1).to(context_token.dtype)
            base.last_context_gate = use_context.detach()
            base.last_context_gate_logits = logits.detach()
            base._context_gate_for_loss = use_context
            return use_context
        base._context_gate = clipped_gate


def _context_only_gate(base: ContextIndividualTF, context_token: torch.Tensor) -> torch.Tensor:
    logits = base.gate(context_token)
    if base.training:
        use_context = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
    else:
        use_context = logits.argmax(dim=-1).eq(1).to(context_token.dtype)
    base.last_context_gate = use_context.detach()
    base.last_context_gate_logits = logits.detach()
    base._context_gate_for_loss = use_context
    return use_context


def evaluate(model: torch.nn.Module, loader: DataLoader, stats_config: tuple) -> tuple[dict, list[dict]]:
    mean, std, _, _ = stats_config
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = forward(model, batch).cpu()
            target = batch["pred_target"].cpu()
            raw_prediction = prediction.numpy() * std + mean
            raw_target = target.numpy() * std + mean
            for index in range(prediction.shape[0]):
                cp = (raw_prediction[index, :, :2] + raw_prediction[index, :, 2:4]) / 2
                ct = (raw_target[index, :, :2] + raw_target[index, :, 2:4]) / 2
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
    return {
        "mse": float(np.mean([row["mse"] for row in rows])),
        "cmse": float(np.mean([row["cmse"] for row in rows])),
        "cfmse": float(np.mean([row["cfmse"] for row in rows])),
    }, rows


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_").lower()


def run_one(model_name: str, base_slug: str, satie: bool, variant: str, seed: int, datasets: dict, stats_config: tuple) -> None:
    run = RESULTS / "runs" / "E8" / f"{base_slug}_{variant}" / str(seed)
    if run.exists():
        print(f"RESUME: reuse existing {run}")
        return
    run.mkdir(parents=True)
    set_seed(seed)
    model = build_model(satie)
    configure_variant(model, variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()
    train = DataLoader(datasets["train"], batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(datasets["validation"], batch_size=8, shuffle=False)
    history = []
    best = float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train:
            optimizer.zero_grad()
            value = loss(forward(model, batch), batch["pred_target"])
            value.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_mse = mean_loss(model, validation, loss)
        history.append(validation_mse)
        if validation_mse < best:
            best = validation_mse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), run / "checkpoint.pth")
    metrics, rows = evaluate(model, DataLoader(datasets["test"], batch_size=8, shuffle=False), stats_config)
    metrics.update({
        "model": model_name,
        "variant": variant,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "epochs_completed": len(history),
        "stopping_reason": "early stopping" if len(history) < MAX_EPOCHS else "maximum epochs",
        "training_seconds": time.perf_counter() - started,
        "best_validation_mse": best,
    })
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run / "train_log.txt").write_text("\n".join(map(str, history)) + "\n", encoding="utf-8")
    with (run / "per_sample.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    datasets, stats_config = load_data()
    for model_name, base_slug, satie in BASES:
        for variant in VARIANTS:
            for seed in args.seeds:
                run_one(model_name, base_slug, satie, variant, seed, datasets, stats_config)


if __name__ == "__main__":
    main()
