import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from MTN.run_e8_v2 import (
    _checkpoint_payload,
    _prepare_run,
    _restore_checkpoint,
    _status,
    atomic_torch_save,
    atomic_write_json,
    parse_args,
)


class E8RunnerInfrastructureTest(unittest.TestCase):
    def test_existing_directory_is_not_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "incomplete"
            run.mkdir()
            (run / "partial.txt").write_text("unfinished", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no complete status/config"):
                _prepare_run(run, {"runner_version": "E8_v2"}, resume=False)

    def test_atomic_writers_leave_complete_files_without_temp_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            json_path = directory / "state.json"
            checkpoint_path = directory / "latest.pt"
            atomic_write_json(json_path, {"state": "running", "epoch": 1})
            atomic_torch_save(checkpoint_path, {"epoch": 1, "tensor": torch.ones(2)})
            self.assertEqual(json_path.read_text(encoding="utf-8"), '{\n  "state": "running",\n  "epoch": 1\n}\n')
            self.assertEqual(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["epoch"], 1)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["latest.pt", "state.json"])

    def test_explicit_run_states_are_serialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for state in ("running", "completed", "interrupted", "failed"):
                _status(run, state, epoch=1)
                self.assertEqual(
                    json.loads((run / "status.json").read_text(encoding="utf-8"))["state"],
                    state,
                )
            with self.assertRaises(ValueError):
                _status(run, "unknown")

    def test_checkpoint_roundtrip_restores_optimizer_rng_and_stopping_state(self):
        torch.manual_seed(11)
        random.seed(12)
        np.random.seed(13)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        loader_generator = torch.Generator().manual_seed(14)
        saved_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
        payload = _checkpoint_payload(
            model,
            optimizer,
            loader_generator,
            epoch=4,
            history=[{"epoch": 4, "validation_loss": 0.25}],
            best_validation_loss=0.25,
            best_epoch=3,
            stale_epochs=1,
            best_model_state=saved_model,
        )
        expected_rng = (
            random.random(),
            float(np.random.random()),
            torch.rand(1),
            torch.rand(1, generator=loader_generator),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            atomic_torch_save(checkpoint, payload)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=False)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100.0)
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        loader_generator.manual_seed(99)
        start, history, best_loss, best_epoch, stale, best_state = _restore_checkpoint(
            restored, model, optimizer, loader_generator
        )
        actual_rng = (
            random.random(),
            float(np.random.random()),
            torch.rand(1),
            torch.rand(1, generator=loader_generator),
        )
        self.assertEqual(start, 5)
        self.assertEqual(history, [{"epoch": 4, "validation_loss": 0.25}])
        self.assertEqual(best_loss, 0.25)
        self.assertEqual(best_epoch, 3)
        self.assertEqual(stale, 1)
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in saved_model.items()))
        self.assertEqual(expected_rng[0], actual_rng[0])
        self.assertEqual(expected_rng[1], actual_rng[1])
        self.assertTrue(torch.equal(expected_rng[2], actual_rng[2]))
        self.assertTrue(torch.equal(expected_rng[3], actual_rng[3]))
        self.assertIsNotNone(best_state)
        self.assertTrue(optimizer.state)

    def test_cli_defaults_are_bounded_and_cpu_only(self):
        args = parse_args([])
        self.assertEqual(args.max_epochs, 2)
        self.assertEqual(args.cpu_threads, 4)
        self.assertEqual(args.seeds, [20260928])
        self.assertEqual(args.models, ["without_satie"])

    def test_validation_only_and_smoke_caps_are_configurable(self):
        args = parse_args([
            "--validation-only",
            "--max-epochs", "2",
            "--max-train-batches", "2",
            "--max-validation-batches", "2",
            "--variants", "context_only",
        ])
        self.assertTrue(args.validation_only)
        self.assertEqual(args.max_train_batches, 2)
        self.assertEqual(args.max_validation_batches, 2)
        self.assertEqual(args.variants, ["context_only"])


if __name__ == "__main__":
    unittest.main()
