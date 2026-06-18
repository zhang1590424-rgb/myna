from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_trainer.dataset_manager import DatasetManager
from local_trainer.engine import (
    LlamaFactoryTrainingEngine,
    MockTrainingEngine,
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

    def test_returns_real_engine_when_ready(self) -> None:
        import local_trainer.engine as engine_module

        original = engine_module.real_engine_ready
        engine_module.real_engine_ready = lambda: True
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                experiments, datasets = self._build_services(temp_dir)
                engine = build_engine(experiments, datasets)
        finally:
            engine_module.real_engine_ready = original

        self.assertIsInstance(engine, LlamaFactoryTrainingEngine)
        self.assertEqual(engine.name, "llamafactory")

    def test_falls_back_to_mock_when_not_ready(self) -> None:
        import local_trainer.engine as engine_module

        original = engine_module.real_engine_ready
        engine_module.real_engine_ready = lambda: False
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                experiments, datasets = self._build_services(temp_dir)
                engine = build_engine(experiments, datasets)
        finally:
            engine_module.real_engine_ready = original

        self.assertIsInstance(engine, MockTrainingEngine)
        self.assertEqual(engine.name, "mock")


if __name__ == "__main__":
    unittest.main()
