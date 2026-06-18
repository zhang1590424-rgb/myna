from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from .domain import DatasetRecord, PreferenceRecord


class DatasetValidationError(ValueError):
    def __init__(self, message: str, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.warnings = warnings or []


@dataclass(frozen=True)
class ParsedDataset:
    records: list[DatasetRecord]
    skipped_count: int
    warnings: list[str]
    source_format: str

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def preview(self) -> list[DatasetRecord]:
        return self.records[:3]

    @property
    def human_summary(self) -> str:
        if self.skipped_count:
            return f"已识别 {self.valid_count} 条有效数据，跳过 {self.skipped_count} 行不完整内容。"
        return f"已识别 {self.valid_count} 条有效数据，格式没问题。"


@dataclass(frozen=True)
class ParsedPreferenceDataset:
    records: list[PreferenceRecord]
    skipped_count: int
    warnings: list[str]
    source_format: str

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def preview(self) -> list[PreferenceRecord]:
        return self.records[:3]

    @property
    def human_summary(self) -> str:
        if self.skipped_count:
            return f"已识别 {self.valid_count} 组偏好对，跳过 {self.skipped_count} 行不完整内容。"
        return f"已识别 {self.valid_count} 组偏好对，格式没问题。"


HEADER_ALIASES = {
    "instruction": {
        "instruction",
        "prompt",
        "question",
        "query",
        "问题",
        "提问",
        "用户问题",
        "用户指令",
        "输入",
    },
    "input": {"input", "context", "上下文", "补充信息", "背景"},
    "output": {
        "output",
        "answer",
        "response",
        "completion",
        "回答",
        "回复",
        "答案",
        "期望回答",
        "客服回复",
        "输出",
    },
    "system": {"system", "系统", "系统提示词", "角色设定"},
}


PREFERENCE_ALIASES = {
    "instruction": {
        "instruction",
        "prompt",
        "question",
        "query",
        "问题",
        "提问",
        "用户问题",
        "用户指令",
        "输入",
    },
    "chosen": {"chosen", "good", "preferred", "better", "偏好回答", "更好回答", "采用", "好的回答", "正例"},
    "rejected": {"rejected", "bad", "worse", "不想要", "差的回答", "拒绝", "负例", "更差回答"},
}


def _read_rows(filename: str, content: bytes) -> tuple[list[dict[str, Any]], str]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix == "csv":
        return _read_csv(content), "csv"
    if suffix == "json":
        return _read_json(content), "json"
    if suffix == "jsonl":
        return _read_jsonl(content), "jsonl"
    if suffix == "xlsx":
        return _read_xlsx(content), "xlsx"
    raise DatasetValidationError("暂时只支持 CSV、JSON、JSONL 和 XLSX。请换一种格式再上传。")


def parse_dataset_bytes(filename: str, content: bytes) -> ParsedDataset:
    rows, source_format = _read_rows(filename, content)
    return _normalize_rows(rows, source_format=source_format)


def parse_preference_bytes(filename: str, content: bytes) -> ParsedPreferenceDataset:
    rows, source_format = _read_rows(filename, content)
    return _normalize_preference_rows(rows, source_format=source_format)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DatasetValidationError("文件编码无法识别。请导出为 UTF-8 CSV 后再上传。")


def _read_csv(content: bytes) -> list[dict[str, Any]]:
    text = _decode_text(content)
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise DatasetValidationError("CSV 第一行需要是表头，例如 question,answer。")
    return [dict(row) for row in reader]


def _read_json(content: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_decode_text(content))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError("JSON 格式无法解析。请检查逗号、引号和括号是否完整。") from exc

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise DatasetValidationError("JSON 需要是数组，或包含 data 数组。")
    return _ensure_dict_rows(payload)


def _read_jsonl(content: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(_decode_text(content).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"JSONL 第 {index} 行无法解析。") from exc
        if not isinstance(value, dict):
            raise DatasetValidationError(f"JSONL 第 {index} 行不是对象。")
        rows.append(value)
    return rows


def _ensure_dict_rows(payload: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"第 {index} 条数据不是对象。")
        rows.append(item)
    return rows


def _read_xlsx(content: bytes) -> list[dict[str, Any]]:
    try:
        with ZipFile(BytesIO(content)) as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_name = _first_sheet_name(archive)
            sheet_xml = archive.read(sheet_name)
    except KeyError as exc:
        raise DatasetValidationError("XLSX 结构不完整。请重新导出后再上传。") from exc
    except Exception as exc:
        raise DatasetValidationError("XLSX 文件无法读取。请确认它不是加密文件。") from exc

    rows = _sheet_xml_to_rows(sheet_xml, shared_strings)
    if not rows:
        raise DatasetValidationError("XLSX 里没有可读取的数据。")
    headers = [str(value).strip() for value in rows[0]]
    if not any(headers):
        raise DatasetValidationError("XLSX 第一行需要是表头，例如 question 和 answer。")

    dict_rows: list[dict[str, Any]] = []
    for row in rows[1:]:
        dict_rows.append({header: row[index] if index < len(row) else "" for index, header in enumerate(headers)})
    return dict_rows


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for item in root.findall("x:si", namespace):
        values.append("".join(text.text or "" for text in item.findall(".//x:t", namespace)))
    return values


def _first_sheet_name(archive: ZipFile) -> str:
    if "xl/worksheets/sheet1.xml" in archive.namelist():
        return "xl/worksheets/sheet1.xml"
    for name in archive.namelist():
        if name.startswith("xl/worksheets/") and name.endswith(".xml"):
            return name
    raise KeyError("worksheet")


def _sheet_xml_to_rows(sheet_xml: bytes, shared_strings: list[str]) -> list[list[str]]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(sheet_xml)
    rows: list[list[str]] = []
    for row in root.findall(".//x:sheetData/x:row", namespace):
        cells: dict[int, str] = {}
        for cell in row.findall("x:c", namespace):
            ref = cell.attrib.get("r", "")
            column_index = _column_index(ref)
            cells[column_index] = _cell_value(cell, namespace, shared_strings)
        if cells:
            max_index = max(cells)
            rows.append([cells.get(index, "") for index in range(max_index + 1)])
    return rows


def _cell_value(cell: ET.Element, namespace: dict[str, str], shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//x:t", namespace)).strip()

    value_node = cell.find("x:v", namespace)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except (IndexError, ValueError):
            return ""
    return raw.strip()


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    if not letters:
        return 0
    index = 0
    for char in letters.upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _normalize_rows(rows: list[dict[str, Any]], source_format: str) -> ParsedDataset:
    if not rows:
        raise DatasetValidationError("文件里没有数据。请至少保留一条问题和回答。")

    warnings: list[str] = []
    records: list[DatasetRecord] = []
    skipped_count = 0
    field_map = _field_map(rows[0])

    if "instruction" not in field_map or "output" not in field_map:
        raise DatasetValidationError("没有找到“问题”和“回答”两列。请使用 question,answer 或 instruction,output。")

    for index, row in enumerate(rows, start=2):
        instruction = _clean_cell(row.get(field_map["instruction"], ""))
        output = _clean_cell(row.get(field_map["output"], ""))
        if not instruction or not output:
            skipped_count += 1
            if len(warnings) < 5:
                warnings.append(f"第 {index} 行缺少问题或回答，已跳过。")
            continue

        records.append(
            DatasetRecord(
                instruction=instruction,
                input=_clean_cell(row.get(field_map.get("input", ""), "")),
                output=output,
                system=_optional_clean_cell(row.get(field_map.get("system", ""), "")),
            )
        )

    if not records:
        raise DatasetValidationError("没有找到可训练数据。请保留完整的问题和回答。", warnings)
    return ParsedDataset(records=records, skipped_count=skipped_count, warnings=warnings, source_format=source_format)


def _field_map(first_row: dict[str, Any]) -> dict[str, str]:
    return _build_field_map(first_row, HEADER_ALIASES)


def _build_field_map(first_row: dict[str, Any], aliases_table: dict[str, set[str]]) -> dict[str, str]:
    normalized_headers = {_normalize_header(header): header for header in first_row.keys()}
    result: dict[str, str] = {}
    for canonical, aliases in aliases_table.items():
        for alias in aliases:
            header = normalized_headers.get(_normalize_header(alias))
            if header is not None:
                result[canonical] = header
                break
    return result


def _normalize_preference_rows(rows: list[dict[str, Any]], source_format: str) -> ParsedPreferenceDataset:
    if not rows:
        raise DatasetValidationError("文件里没有数据。请至少保留一组偏好对。")

    warnings: list[str] = []
    records: list[PreferenceRecord] = []
    skipped_count = 0
    field_map = _build_field_map(rows[0], PREFERENCE_ALIASES)

    if "instruction" not in field_map or "chosen" not in field_map or "rejected" not in field_map:
        raise DatasetValidationError(
            "偏好数据需要三列：问题、偏好回答、不想要的回答。请使用 instruction,chosen,rejected。"
        )

    for index, row in enumerate(rows, start=2):
        instruction = _clean_cell(row.get(field_map["instruction"], ""))
        chosen = _clean_cell(row.get(field_map["chosen"], ""))
        rejected = _clean_cell(row.get(field_map["rejected"], ""))
        if not instruction or not chosen or not rejected:
            skipped_count += 1
            if len(warnings) < 5:
                warnings.append(f"第 {index} 行缺少问题、偏好或拒绝回答，已跳过。")
            continue

        records.append(PreferenceRecord(instruction=instruction, chosen=chosen, rejected=rejected))

    if not records:
        raise DatasetValidationError("没有找到可训练的偏好对。请保留完整的三列内容。", warnings)
    return ParsedPreferenceDataset(
        records=records, skipped_count=skipped_count, warnings=warnings, source_format=source_format
    )


def _normalize_header(value: Any) -> str:
    return re.sub(r"[\s_\-—:：]+", "", str(value).strip().lower())


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_clean_cell(value: Any) -> str | None:
    cleaned = _clean_cell(value)
    return cleaned or None
