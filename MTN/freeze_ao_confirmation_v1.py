"""Freeze a testless manifest for the Phase 6A AO confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from . import run_ao_v1
    from .ao_models_v1 import (
        MODEL_CONFIG,
        build_ao_model,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )
except ImportError:
    import run_ao_v1
    from ao_models_v1 import (
        MODEL_CONFIG,
        build_ao_model,
        unique_trainable_parameter_count,
        verify_satie_identity,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results_paper" / "splits" / "ao_annotation_only_v1"
DEFAULT_OUTPUT = ROOT / "results_paper" / "AO_CONFIRMATION_MANIFEST.json"
SEEDS = (20260928, 20260929, 20260930, 20260931, 20260932)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_membership(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(data["sample_id"][index]),
            "video_id": str(data["video_id"][index]),
            "track_id": str(data["track_id"][index]),
            "frame_start": int(data["frame_start"][index]),
        }
        for index in range(len(data["sample_id"]))
    ]


def build_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    train, validation, split = run_ao_v1._read_split(cache_dir)
    cache_hashes = {
        name: _file_sha256(cache_dir / name)
        for name in ("train.npz", "validation.npz", "split_manifest.json")
    }
    if split.get("official_test_sets_read") != []:
        raise ValueError("Frozen split manifest does not certify test isolation.")
    if set(train["sample_id"].tolist()).intersection(
        set(validation["sample_id"].tolist())
    ):
        raise ValueError("Train and validation sample IDs overlap.")
    expected_train_videos = {
        "gp_set_0001_vid_0001",
        "gp_set_0002_vid_0001",
        "gp_set_0004_vid_0001",
        "gp_set_0006_vid_0002",
    }
    expected_validation_videos = {
        "gp_set_0007_vid_0004",
        "gp_set_0007_vid_0005",
    }
    if set(train["video_id"].tolist()) != expected_train_videos:
        raise ValueError("Training video membership differs from Phase 5B.")
    if set(validation["video_id"].tolist()) != expected_validation_videos:
        raise ValueError("Validation video membership differs from Phase 5B.")
    if len(train["sample_id"]) != 988 or len(validation["sample_id"]) != 432:
        raise ValueError("Frozen split sample counts differ from Phase 5B.")

    model_specs = {
        "AO-1": {
            "anchor": "constant_position",
            "share_encoder_attention": False,
        },
        "AO-2": {
            "anchor": "constant_velocity",
            "share_encoder_attention": False,
        },
        "AO-3": {
            "anchor": "constant_velocity",
            "share_encoder_attention": True,
        },
    }
    source_hash = _file_sha256(ROOT / "MTN" / "ao_models_v1.py")
    model_configs = {}
    for model_id, spec in model_specs.items():
        model = build_ao_model(model_id)
        if model_id == "AO-3" and not verify_satie_identity(model):
            raise AssertionError("AO-3 attention module identity is not shared.")
        config = {
            "model_id": model_id,
            "model_source_sha256": source_hash,
            "model_dimensions": MODEL_CONFIG,
            "input": "15 observed absolute pixel xyxy boxes",
            "output": "45 future absolute pixel xyxy boxes",
            **spec,
            "unique_trainable_parameters": unique_trainable_parameter_count(model),
        }
        model_configs[model_id] = {
            **config,
            "configuration_sha256": _canonical_sha256(config),
        }
    expected_parameters = {"AO-1": 76804, "AO-2": 76804, "AO-3": 68356}
    if {
        model_id: value["unique_trainable_parameters"]
        for model_id, value in model_configs.items()
    } != expected_parameters:
        raise AssertionError("Model unique parameter counts changed.")
    ao2_common = dict(model_configs["AO-2"])
    ao3_common = dict(model_configs["AO-3"])
    for key in (
        "model_id", "share_encoder_attention",
        "unique_trainable_parameters", "configuration_sha256",
    ):
        ao2_common.pop(key)
        ao3_common.pop(key)
    if ao2_common != ao3_common:
        raise AssertionError("AO-2 and AO-3 differ beyond encoder attention sharing.")

    training_config = {
        "seeds": list(SEEDS),
        "maximum_epochs": 30,
        "early_stopping_patience": 10,
        "batch_size": 8,
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "learning_rate": 0.001,
        "loss": "mean squared error",
        "cpu_threads": 4,
        "device": "cpu",
        "data_order": "permutation seeded by seed + epoch; matched across model IDs",
        "checkpoint_selection": "minimum validation normalized MSE",
        "primary_model_metric": "pooled raw-pixel validation MSE",
        "secondary_metric_aggregation": (
            "macro-video mean reported separately; track-level summaries "
            "cluster windows within video and track"
        ),
    }
    training_config_hash = _canonical_sha256(training_config)
    script_files = (
        "MTN/ao_dataset_v1.py",
        "MTN/ao_models_v1.py",
        "MTN/run_ao_v1.py",
        "MTN/ao_confirmation_v1.py",
        "MTN/freeze_ao_confirmation_v1.py",
        "MTN/run_ao_confirmation_v1.py",
        "MTN/benchmark_ao_latency_v1.py",
        "results_paper/summary/E7_summary.csv",
        "results_paper/summary/E7_REPORT.md",
    )
    source_hashes = {
        name: _file_sha256(ROOT / Path(name))
        for name in script_files
        if (ROOT / Path(name)).is_file()
    }

    resume_sources = {}
    screen_root = ROOT / "results_paper" / "AO_initial_screen_v1"
    for model_id in model_specs:
        run = screen_root / model_id / "seed_20260928"
        config_path = run / "run_config.json"
        checkpoint_path = run / "latest_checkpoint.pt"
        if not (config_path.is_file() and checkpoint_path.is_file()):
            raise FileNotFoundError(
                f"Seed-20260928 continuation source is incomplete: {run}"
            )
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "seed": 20260928,
            "batch_size": 8,
            "learning_rate": 0.001,
            "cpu_threads": 4,
            "patience": 10,
        }
        if any(old_config.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Seed-20260928 pilot config is not resumable under the frozen protocol: {run}"
            )
        if old_config.get("cache_sha256") != cache_hashes:
            raise ValueError(f"Pilot checkpoint uses different data: {run}")
        resume_sources[model_id] = {
            "source_run": str(run.relative_to(ROOT)),
            "status": json.loads(
                (run / "status.json").read_text(encoding="utf-8")
            ).get("state"),
            "epochs_completed": 10,
            "optimizer_and_rng_state_sha256": _file_sha256(checkpoint_path),
            "checkpoint_config_sha256": _file_sha256(config_path),
            "continuation_rule": (
                "fork into a new output directory; resume saved optimizer/RNG "
                "state through the 30-epoch maximum; never modify the pilot run"
            ),
        }

    return {
        "schema": "ao_confirmation_protocol_v1",
        "frozen_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cache_root": str(cache_dir),
        "cache_sha256": cache_hashes,
        "split_manifest": split,
        "train_samples": _sample_membership(train),
        "validation_samples": _sample_membership(validation),
        "annotation_provenance": split["sources"],
        "official_test_sets_read": [],
        "historical_gp6_test_video_used": False,
        "normalization_fit_role": "train only; unique train-role frames",
        "model_input_fields": ["observed_boxes[0:15]"],
        "future_fields_are_supervision_only": ["future_boxes[15:60]"],
        "model_source_sha256": source_hash,
        "model_configs": model_configs,
        "AO2_AO3_only_architecture_difference": {
            "field": "share_encoder_attention",
            "AO-2": False,
            "AO-3": True,
            "verified_shared_attention_identity": verify_satie_identity(
                build_ao_model("AO-3")
            ),
        },
        "training_config": training_config,
        "training_config_sha256": training_config_hash,
        "source_sha256": source_hashes,
        "seed_20260928_resumable_checkpoint_sources": resume_sources,
        "predeclared_model_selection": {
            "best_predictive_accuracy": (
                "lowest mean pooled raw-pixel validation MSE across matched seeds"
            ),
            "compact_neural": (
                "AO-3 is a compact candidate only if it remains on the "
                "accuracy/parameter Pareto frontier; report its measured "
                "accuracy trade-off without claiming equivalence"
            ),
            "all_metrics_are_validation_only": True,
            "five_seeds_are_not_five_independent_datasets": True,
            "video_generalization_limit": "two fixed validation videos",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {args.output}")
    manifest = build_manifest(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen confirmation manifest: {args.output}")
    print(f"Cache hashes: {manifest['cache_sha256']}")
    print(f"Model configuration hashes: " + json.dumps({
        key: value["configuration_sha256"]
        for key, value in manifest["model_configs"].items()
    }, sort_keys=True))
    print(f"Training configuration SHA-256: {manifest['training_config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
