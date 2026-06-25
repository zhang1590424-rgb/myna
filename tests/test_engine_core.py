from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from local_trainer.dataset_manager import DatasetManager
from local_trainer.diagnostics import compute_diagnostics, compute_live_diagnostics
from local_trainer.domain import Experiment, ExperimentParams, ExperimentStatus
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


class OrphanReconcileTests(unittest.TestCase):
    """孤儿训练自检：跨服务重启或子进程被 kill 后能把状态扳成 failed。"""

    def _build_engine(self, temp_dir: str) -> tuple[ExperimentService, LlamaFactoryTrainingEngine]:
        db = Database(db_path=Path(temp_dir) / "workbench.db")
        datasets = DatasetManager(db, root=Path(temp_dir) / "datasets")
        experiments = ExperimentService(db, datasets)
        engine = LlamaFactoryTrainingEngine(experiments, datasets, runs_dir=Path(temp_dir) / "runs")
        return experiments, engine

    def _make_running_experiment(
        self,
        experiments: ExperimentService,
        *,
        pid: int | None,
        output_dir: Path | None,
        started_at: str = "2026-01-01T00:00:00+00:00",
    ) -> Experiment:
        exp = Experiment(
            id=os.urandom(4).hex(),
            name="orphan-case",
            method="sft",  # type: ignore[arg-type]
            model_id="qwen3.5-0.8b",
            dataset_id="dataset-test",
            dataset_count=10,
            params=ExperimentParams(epochs=1),
            status=ExperimentStatus.running,
            pid=pid,
            output_dir=str(output_dir) if output_dir else None,
            started_at=started_at,
            created_at="2026-01-01T00:00:00+00:00",
        )
        experiments.db.upsert_experiment(exp)
        return exp

    def test_dead_pid_with_no_log_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments, engine = self._build_engine(temp_dir)
            exp = self._make_running_experiment(experiments, pid=999999, output_dir=None)

            recovered = engine.reconcile_orphans()

            self.assertEqual(recovered, [exp.id])
            self.assertEqual(experiments.get(exp.id).status, ExperimentStatus.failed.value)

    def test_stale_trainer_log_with_dead_pid_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments, engine = self._build_engine(temp_dir)
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            log = output_dir / "trainer_log.jsonl"
            log.write_text("{}", encoding="utf-8")
            old = time.time() - 3600
            os.utime(log, (old, old))
            exp = self._make_running_experiment(experiments, pid=999999, output_dir=output_dir)

            recovered = engine.reconcile_orphans()

            self.assertEqual(recovered, [exp.id])
            self.assertEqual(experiments.get(exp.id).status, ExperimentStatus.failed.value)

    def test_fresh_trainer_log_with_live_pid_is_not_touched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments, engine = self._build_engine(temp_dir)
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            log = output_dir / "trainer_log.jsonl"
            log.write_text("{}", encoding="utf-8")
            exp = self._make_running_experiment(
                experiments, pid=os.getpid(), output_dir=output_dir
            )

            recovered = engine.reconcile_orphans()

            self.assertEqual(recovered, [])
            self.assertEqual(experiments.get(exp.id).status, ExperimentStatus.running.value)

    def test_tracked_experiments_are_skipped(self) -> None:
        """本进程还在看管的实验不应被错误回收。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments, engine = self._build_engine(temp_dir)
            exp = self._make_running_experiment(experiments, pid=999999, output_dir=None)
            engine._procs[exp.id] = object()  # type: ignore[assignment]

            recovered = engine.reconcile_orphans()

            self.assertEqual(recovered, [])
            self.assertEqual(experiments.get(exp.id).status, ExperimentStatus.running.value)


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

    def test_no_validation_split_explained_for_small_dataset(self) -> None:
        exp = _diagnostic_experiment(
            loss=[3.0, 2.5, 2.2, 2.0, 1.9, 1.85, 1.84],
            dataset_count=20,
        )

        cards = compute_diagnostics(exp)

        no_eval = [c for c in cards if c.topic == "eval_loss" and c.level == "ok"]
        self.assertTrue(no_eval, "数据 < 30 时应给出未切验证集的解释卡")
        self.assertIn("没有验证 loss", no_eval[0].title)

    def test_validation_stalled_signals_to_stop(self) -> None:
        exp = _diagnostic_experiment(
            loss=[3.0, 2.4, 1.9, 1.6, 1.4, 1.25, 1.15, 1.05, 0.96, 0.88],
            eval_loss=[1.5, 1.42, 1.40, 1.39, 1.395, 1.392, 1.391, 1.390],
        )

        cards = compute_diagnostics(exp)

        stalled = [c for c in cards if "训练还在降，验证已经不动" in c.title]
        self.assertTrue(stalled, "训练降但验证停滞时应给出 stalled 提示")

    def test_diagnostics_carry_topic_for_frontend_filtering(self) -> None:
        exp = _diagnostic_experiment(loss=[4.0, 3.0, 2.6, 2.3, 2.20, 2.19, 2.18])

        cards = compute_diagnostics(exp)

        self.assertTrue(all(c.topic for c in cards), "所有诊断卡都应带 topic 用于前端按主题过滤")


if __name__ == "__main__":
    unittest.main()
