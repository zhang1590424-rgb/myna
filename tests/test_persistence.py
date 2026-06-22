from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_trainer.domain import Experiment, LabResult
from local_trainer.persistence import Database


def _lab_result(result_id: str, created_at: str) -> LabResult:
    return LabResult(
        id=result_id,
        experiment_id="exp-1",
        experiment_name="实验 1",
        kind="compare",
        prompt=f"问题 {result_id}",
        created_at=created_at,
    )


def _experiment(exp_id: str = "exp-1") -> Experiment:
    return Experiment(
        id=exp_id,
        name="实验 1",
        model_id="qwen3-0.6b",
        dataset_id="dataset-1",
        created_at="2026-01-01T10:00:00",
    )


class LabResultPersistenceTests(unittest.TestCase):
    def test_lists_recent_lab_results_across_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(db_path=Path(temp_dir) / "workbench.db")
            try:
                db.upsert_lab_result(_lab_result("old", "2026-01-01T10:00:00"))
                db.upsert_lab_result(_lab_result("new", "2026-01-03T10:00:00"))
                db.upsert_lab_result(_lab_result("mid", "2026-01-02T10:00:00"))

                results = db.list_recent_lab_results(limit=2)
            finally:
                db.close()

        self.assertEqual([item.id for item in results], ["new", "mid"])

    def test_deleting_experiment_removes_lab_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(db_path=Path(temp_dir) / "workbench.db")
            try:
                db.upsert_experiment(_experiment())
                db.upsert_lab_result(_lab_result("result-1", "2026-01-01T10:00:00"))

                deleted = db.delete_experiment("exp-1")
                results = db.list_lab_results("exp-1")
            finally:
                db.close()

        self.assertTrue(deleted)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
