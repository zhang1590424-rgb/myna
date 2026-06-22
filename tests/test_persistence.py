from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_trainer.domain import LabResult
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


if __name__ == "__main__":
    unittest.main()
