"""SQLite persistence for experiments and datasets.

Single-file DB at runtime/workbench.db. WAL mode so the polling frontend and a
background training task can read/write concurrently without locking. Each row
stores indexed columns for filtering plus a full JSON blob for the model.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .domain import DatasetInfo, Experiment, LabResult
from .paths import WORKBENCH_DB, ensure_runtime_dirs


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    method       TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    dataset_id   TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);

CREATE TABLE IF NOT EXISTS datasets (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    format          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    data            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lab_results (
    id                TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    kind              TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    data              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lab_results_exp_created
    ON lab_results(experiment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_lab_results_created
    ON lab_results(created_at);
"""


class Database:
    """Thread-safe SQLite access. One connection guarded by a lock.

    The training engine runs in the asyncio loop and the queue worker in the
    same loop, so a single connection with a lock is enough and avoids
    cross-thread connection issues.
    """

    def __init__(self, db_path: Path = WORKBENCH_DB) -> None:
        ensure_runtime_dirs()
        self._path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- experiments ---- #
    def upsert_experiment(self, exp: Experiment) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO experiments (id, name, method, model_id, dataset_id, status, created_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, method=excluded.method, model_id=excluded.model_id,
                    dataset_id=excluded.dataset_id, status=excluded.status, data=excluded.data
                """,
                (
                    exp.id,
                    exp.name,
                    exp.method,
                    exp.model_id,
                    exp.dataset_id,
                    exp.status,
                    exp.created_at,
                    exp.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get_experiment(self, exp_id: str) -> Experiment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
        if row is None:
            return None
        return Experiment.model_validate_json(row["data"])

    def list_experiments(self) -> list[Experiment]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [Experiment.model_validate_json(row["data"]) for row in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            self._conn.execute("DELETE FROM lab_results WHERE experiment_id = ?", (exp_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def count_method_prefix(self, method: str, model_id: str) -> int:
        """How many experiments share this method+model, for auto-naming (#N)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM experiments WHERE method = ? AND model_id = ?",
                (method, model_id),
            ).fetchone()
        return int(row["n"])

    # ---- datasets ---- #
    def upsert_dataset(self, info: DatasetInfo) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO datasets (id, name, format, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, format=excluded.format, data=excluded.data
                """,
                (info.id, info.name, info.format, info.created_at, info.model_dump_json()),
            )
            self._conn.commit()

    def get_dataset(self, dataset_id: str) -> DatasetInfo | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            return None
        return DatasetInfo.model_validate_json(row["data"])

    def list_datasets(self) -> list[DatasetInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM datasets ORDER BY created_at DESC"
            ).fetchall()
        return [DatasetInfo.model_validate_json(row["data"]) for row in rows]

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---- lab results ---- #
    def upsert_lab_result(self, result: LabResult) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO lab_results (id, experiment_id, kind, prompt, created_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    experiment_id=excluded.experiment_id, kind=excluded.kind,
                    prompt=excluded.prompt, created_at=excluded.created_at, data=excluded.data
                """,
                (
                    result.id,
                    result.experiment_id,
                    result.kind,
                    result.prompt,
                    result.created_at,
                    result.model_dump_json(),
                ),
            )
            self._conn.commit()

    def list_lab_results(self, experiment_id: str, limit: int = 50) -> list[LabResult]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT data FROM lab_results
                WHERE experiment_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (experiment_id, limit),
            ).fetchall()
        return [LabResult.model_validate_json(row["data"]) for row in rows]

    def list_lab_results_batch(self, experiment_ids: list[str], limit: int = 100) -> list[LabResult]:
        """一次查询多个实验的测评历史。"""
        if not experiment_ids:
            return []
        placeholders = ",".join("?" for _ in experiment_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT data FROM lab_results
                WHERE experiment_id IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*experiment_ids, limit),
            ).fetchall()
        return [LabResult.model_validate_json(row["data"]) for row in rows]

    def list_recent_lab_results(self, limit: int = 6) -> list[LabResult]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT data FROM lab_results
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [LabResult.model_validate_json(row["data"]) for row in rows]

    def get_lab_result(self, result_id: str) -> LabResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM lab_results WHERE id = ?", (result_id,)
            ).fetchone()
        if row is None:
            return None
        return LabResult.model_validate_json(row["data"])

    def delete_lab_result(self, result_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM lab_results WHERE id = ?", (result_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
