from __future__ import annotations

import json
import uuid
from pathlib import Path

from .data_validation import ParsedDataset, parse_dataset_bytes
from .domain import DatasetRecord, DatasetUploadResult
from .paths import DATASET_DIR, ensure_runtime_dirs


class DatasetStore:
    def __init__(self, root: Path = DATASET_DIR) -> None:
        self.root = root
        ensure_runtime_dirs()

    def save_upload(self, filename: str, content: bytes, template_id: str) -> DatasetUploadResult:
        parsed = parse_dataset_bytes(filename, content)
        dataset_id = uuid.uuid4().hex
        dataset_path = self.root / f"{dataset_id}.json"
        metadata_path = self.root / f"{dataset_id}.meta.json"

        dataset_path.write_text(
            json.dumps([record.model_dump() for record in parsed.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "dataset_id": dataset_id,
            "filename": filename,
            "template_id": template_id,
            "source_format": parsed.source_format,
            "valid_count": parsed.valid_count,
            "skipped_count": parsed.skipped_count,
            "warnings": parsed.warnings,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._result_from_parsed(dataset_id, filename, parsed)

    def read_records(self, dataset_id: str) -> list[DatasetRecord]:
        dataset_path = self.root / f"{dataset_id}.json"
        if not dataset_path.exists():
            raise KeyError(dataset_id)
        rows = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [DatasetRecord.model_validate(row) for row in rows]

    def read_metadata(self, dataset_id: str) -> dict[str, object]:
        metadata_path = self.root / f"{dataset_id}.meta.json"
        if not metadata_path.exists():
            raise KeyError(dataset_id)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _result_from_parsed(dataset_id: str, filename: str, parsed: ParsedDataset) -> DatasetUploadResult:
        return DatasetUploadResult(
            dataset_id=dataset_id,
            filename=filename,
            source_format=parsed.source_format,  # type: ignore[arg-type]
            valid_count=parsed.valid_count,
            skipped_count=parsed.skipped_count,
            warnings=parsed.warnings,
            preview=parsed.preview,
            human_summary=parsed.human_summary,
        )
