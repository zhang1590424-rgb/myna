from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_trainer.dataset_manager import DatasetManager
from local_trainer.domain import Experiment
from local_trainer.experiment_service import ExperimentService
from local_trainer.persistence import Database


def _experiment(exp_id: str, status: str = "completed") -> Experiment:
    return Experiment(
        id=exp_id,
        name="实验",
        model_id="qwen3-0.6b",
        dataset_id="dataset-1",
        status=status,
        created_at="2026-01-01T10:00:00",
    )


class ExperimentDeleteTests(unittest.TestCase):
    def test_delete_removes_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = Database(db_path=root / "workbench.db")
            datasets = DatasetManager(db, root=root / "datasets")
            service = ExperimentService(db, datasets)
            run_root = root / "runs"
            artifact = run_root / "exp-1" / "llamafactory" / "output" / "adapter_model.safetensors"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("adapter", encoding="utf-8")
            try:
                db.upsert_experiment(_experiment("exp-1"))
                with patch("local_trainer.experiment_service.RUNS_DIR", run_root):
                    deleted = service.delete("exp-1")
            finally:
                db.close()

        self.assertTrue(deleted)
        self.assertFalse((run_root / "exp-1").exists())

    def test_delete_rejects_live_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = Database(db_path=root / "workbench.db")
            datasets = DatasetManager(db, root=root / "datasets")
            service = ExperimentService(db, datasets)
            try:
                db.upsert_experiment(_experiment("exp-1", status="running"))

                with self.assertRaises(RuntimeError):
                    service.delete("exp-1")
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
