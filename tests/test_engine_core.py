from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_trainer.dataset_manager import DatasetManager
from local_trainer.engine import (
    LlamaFactoryTrainingEngine,
    build_engine,
    parse_trainer_log,
)
from local_trainer.experiment_service import ExperimentService
from local_trainer.hardware import select_precision
from local_trainer.persistence import Database


class TrainerLogParsingTests(unittest.TestCase):
    def test_parses_progress_loss_and_eta(self) -> None:
        text = "\n".join(
            [
                '{"current_steps": 1, "loss": 2.5, "epoch": 0.4, "percentage": 33.3, "remaining_time": "0:02:00"}',
                'not json, should be ignored',
                '{"current_steps": 2, "loss": 1.1, "epoch": 1.2, "percentage": 66.6, "remaining_time": "0:01:00"}',
            ]
        )

        parsed = parse_trainer_log(text)

        self.assertEqual(parsed["progress"], 66)
        self.assertEqual(parsed["loss"], [2.5, 1.1])
        self.assertEqual(parsed["eta"], "0:01:00")
        self.assertEqual(parsed["epoch"], 2)

    def test_progress_is_capped_at_99(self) -> None:
        parsed = parse_trainer_log('{"percentage": 100.0}')
        self.assertEqual(parsed["progress"], 99)

    def test_empty_log_is_safe(self) -> None:
        parsed = parse_trainer_log("")
        self.assertEqual(parsed["progress"], 0)
        self.assertEqual(parsed["loss"], [])
        self.assertIsNone(parsed["eta"])

    def test_collects_eval_loss(self) -> None:
        text = "\n".join(
            [
                '{"current_steps": 5, "loss": 1.2, "epoch": 1.0, "percentage": 10.0}',
                '{"current_steps": 5, "eval_loss": 1.5, "epoch": 1.0, "percentage": 10.0}',
                '{"current_steps": 10, "eval_loss": 1.1, "epoch": 2.0, "percentage": 20.0}',
            ]
        )
        parsed = parse_trainer_log(text)
        self.assertEqual(parsed["eval_loss"], [1.5, 1.1])


class CheckpointCleanupTests(unittest.TestCase):
    def _make_checkpoint(self, output_dir: Path, step: int) -> None:
        ckpt = output_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    def test_discard_removes_intermediate_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            for step in (5, 10, 15):
                self._make_checkpoint(out, step)
            LlamaFactoryTrainingEngine._discard_checkpoints(out)
            remaining = sorted(p.name for p in out.glob("checkpoint-*"))
            self.assertEqual(remaining, [])


class PrecisionPolicyTests(unittest.TestCase):
    def test_mps_and_cpu_use_fp32(self) -> None:
        for device in ("mps", "cpu"):
            precision = select_precision(device)
            self.assertFalse(precision["bf16"])
            self.assertFalse(precision["fp16"])

    def test_cuda_uses_bf16(self) -> None:
        precision = select_precision("cuda")
        self.assertTrue(precision["bf16"])
        self.assertFalse(precision["fp16"])


class EngineFactoryTests(unittest.TestCase):
    def _build_services(self, temp_dir: str) -> tuple[ExperimentService, DatasetManager]:
        db = Database(db_path=Path(temp_dir) / "workbench.db")
        datasets = DatasetManager(db, root=Path(temp_dir) / "datasets")
        experiments = ExperimentService(db, datasets)
        return experiments, datasets

    def test_build_engine_returns_real_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments, datasets = self._build_services(temp_dir)
            engine = build_engine(experiments, datasets)
        self.assertIsInstance(engine, LlamaFactoryTrainingEngine)
        self.assertEqual(engine.name, "llamafactory")


if __name__ == "__main__":
    unittest.main()
