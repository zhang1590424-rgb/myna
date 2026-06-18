"""Dataset management: multiple datasets, both SFT (alpaca) and DPO (preference).

Records are stored as JSON files under DATASET_DIR (one file per dataset).
The DatasetInfo metadata lives in SQLite via persistence.Database so the rest
of the workbench can list/look-up datasets without scanning the filesystem.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from .data_validation import (
    DatasetValidationError,
    parse_dataset_bytes,
    parse_preference_bytes,
)
from .domain import (
    DatasetFormat,
    DatasetInfo,
    DatasetRecord,
    DatasetUploadResult,
    PreferenceRecord,
)
from .paths import DATASET_DIR, ensure_runtime_dirs
from .persistence import Database


class DatasetManager:
    def __init__(self, db: Database, root: Path = DATASET_DIR) -> None:
        self.db = db
        self.root = root
        ensure_runtime_dirs()

    # ---- upload ---- #
    def save_upload(
        self,
        filename: str,
        content: bytes,
        fmt: DatasetFormat,
        name: str | None = None,
    ) -> DatasetUploadResult:
        if fmt == "dpo_pairs":
            return self._save_preference(filename, content, name)
        return self._save_alpaca(filename, content, name)

    def _save_alpaca(self, filename: str, content: bytes, name: str | None) -> DatasetUploadResult:
        parsed = parse_dataset_bytes(filename, content)
        dataset_id = uuid.uuid4().hex
        self._write_records(dataset_id, [record.model_dump() for record in parsed.records])
        info = self._persist_info(
            dataset_id=dataset_id,
            name=name or self._default_name(filename),
            filename=filename,
            fmt="alpaca",
            row_count=parsed.valid_count,
        )
        return DatasetUploadResult(
            dataset_id=info.id,
            name=info.name,
            filename=filename,
            source_format=parsed.source_format,  # type: ignore[arg-type]
            format="alpaca",
            valid_count=parsed.valid_count,
            skipped_count=parsed.skipped_count,
            warnings=parsed.warnings,
            preview=[record.model_dump() for record in parsed.preview],
            human_summary=parsed.human_summary,
        )

    def _save_preference(self, filename: str, content: bytes, name: str | None) -> DatasetUploadResult:
        parsed = parse_preference_bytes(filename, content)
        dataset_id = uuid.uuid4().hex
        self._write_records(dataset_id, [record.model_dump() for record in parsed.records])
        info = self._persist_info(
            dataset_id=dataset_id,
            name=name or self._default_name(filename),
            filename=filename,
            fmt="dpo_pairs",
            row_count=parsed.valid_count,
        )
        return DatasetUploadResult(
            dataset_id=info.id,
            name=info.name,
            filename=filename,
            source_format=parsed.source_format,  # type: ignore[arg-type]
            format="dpo_pairs",
            valid_count=parsed.valid_count,
            skipped_count=parsed.skipped_count,
            warnings=parsed.warnings,
            preview=[record.model_dump() for record in parsed.preview],
            human_summary=parsed.human_summary,
        )

    # ---- read ---- #
    def list_datasets(self) -> list[DatasetInfo]:
        return self.db.list_datasets()

    def get_info(self, dataset_id: str) -> DatasetInfo:
        info = self.db.get_dataset(dataset_id)
        if info is None:
            raise KeyError(dataset_id)
        return info

    def read_records(self, dataset_id: str) -> list[DatasetRecord]:
        info = self.get_info(dataset_id)
        if info.format != "alpaca":
            raise DatasetValidationError("这个数据集是偏好数据，不能当作问答数据使用。")
        return [DatasetRecord.model_validate(row) for row in self._read_rows(dataset_id)]

    def read_preferences(self, dataset_id: str) -> list[PreferenceRecord]:
        info = self.get_info(dataset_id)
        if info.format != "dpo_pairs":
            raise DatasetValidationError("这个数据集是问答数据，不能当作偏好数据使用。")
        return [PreferenceRecord.model_validate(row) for row in self._read_rows(dataset_id)]

    def delete(self, dataset_id: str) -> bool:
        removed = self.db.delete_dataset(dataset_id)
        path = self._records_path(dataset_id)
        if path.exists():
            path.unlink()
        return removed

    # ---- helpers ---- #
    def _persist_info(
        self,
        dataset_id: str,
        name: str,
        filename: str,
        fmt: DatasetFormat,
        row_count: int,
    ) -> DatasetInfo:
        info = DatasetInfo(
            id=dataset_id,
            name=name,
            source_filename=filename,
            format=fmt,
            row_count=row_count,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.db.upsert_dataset(info)
        return info

    def _write_records(self, dataset_id: str, rows: list[dict]) -> None:
        self._records_path(dataset_id).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _read_rows(self, dataset_id: str) -> list[dict]:
        path = self._records_path(dataset_id)
        if not path.exists():
            raise KeyError(dataset_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _records_path(self, dataset_id: str) -> Path:
        return self.root / f"{dataset_id}.json"

    @staticmethod
    def _default_name(filename: str) -> str:
        stem = Path(filename).stem.strip()
        return stem or "未命名数据集"
