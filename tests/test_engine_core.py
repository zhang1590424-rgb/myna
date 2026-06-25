from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_trainer.dataset_manager import DatasetManager
from local_trainer.diagnostics import compute_diagnostics, compute_live_diagnostics
from local_trainer.domain import Experiment, ExperimentParams
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


def _diagnostic_experiment(
    *,
    loss: list[float],
    eval_loss: list[float] | None = None,
    method: str = "sft",
    epochs: int = 5,
    dataset_count: int = 80,
) -> Experiment:
    return Experiment(
        id="diag-test",
        name="诊断测试",
        method=method,  # type: ignore[arg-type]
        model_id="qwen3.5-0.8b",
        dataset_id="dataset-test",
        dataset_count=dataset_count,
        params=ExperimentParams(epochs=epochs),
        loss=loss,
        eval_loss=eval_loss or [],
        created_at="2026-01-01T00:00:00Z",
    )


class DiagnosticsTests(unittest.TestCase):
    def test_plateau_curve_gets_plain_language_ok_card(self) -> None:
        exp = _diagnostic_experiment(
            loss=[4.0, 3.0, 2.6, 2.3, 2.20, 2.19, 2.18]
        )

        cards = compute_diagnostics(exp)

        self.assertEqual(cards[0].level, "ok")
        self.assertIn("趋稳", cards[0].title)
        self.assertIsNotNone(cards[0].observation)
        self.assertIsNotNone(cards[0].interpretation)
        self.assertIsNotNone(cards[0].next_step)

    def test_rising_loss_is_primary_error(self) -> None:
        exp = _diagnostic_experiment(loss=[2.0, 2.2, 2.5, 2.8])

        cards = compute_diagnostics(exp)

        self.assertEqual(cards[0].level, "error")
        self.assertIn("往上走", cards[0].title)
        self.assertEqual(cards[0].action.action, "retrain")

    def test_overfit_is_primary_when_validation_loss_rises(self) -> None:
        exp = _diagnostic_experiment(
            loss=[4.0, 3.0, 2.5, 2.0],
            eval_loss=[2.0, 1.5, 1.7],
        )

        cards = compute_diagnostics(exp)

        self.assertEqual(cards[0].level, "error")
        self.assertIn("过拟合", cards[0].title)
        self.assertIn("验证 loss", cards[0].observation)

    def test_underfit_explains_model_may_not_have_learned_much(self) -> None:
        exp = _diagnostic_experiment(loss=[4.0, 3.95, 3.9, 3.85], epochs=5)

        cards = compute_diagnostics(exp)

        self.assertEqual(cards[0].level, "warn")
        self.assertIn("没学到多少", cards[0].title)
        self.assertIn("检查数据质量", cards[0].suggestion)

    def test_live_diagnostics_uses_lightweight_process_copy(self) -> None:
        exp = _diagnostic_experiment(loss=[2.0, 2.2, 2.5, 2.8])

        cards = compute_live_diagnostics(exp)

        self.assertEqual(cards[0].level, "warn")
        self.assertIn("训练可能有点不稳", cards[0].title)
        self.assertIsNone(cards[0].action)


if __name__ == "__main__":
    unittest.main()
