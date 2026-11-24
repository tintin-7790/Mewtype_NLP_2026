import json
from dataclasses import dataclass
from pathlib import Path

from .errors import CaseError, TaskError


@dataclass
class Source:
    source_id: str
    content: str
    field_path: str = ""
    source_id_origin: str = "provided"


@dataclass
class Case:
    case_id: str
    record_cutoff: str
    sources: list[Source]


def load_standard_cases(path: Path) -> list[Case | CaseError]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskError("INPUT_NOT_FOUND", f"输入文件不存在：{path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskError("INPUT_INVALID_JSON", f"输入不是有效 UTF-8 JSON：{exc}") from exc
    return parse_standard_cases(raw)


def parse_standard_cases(raw) -> list[Case | CaseError]:
    """Parse an already-loaded competition payload without changing protocol semantics."""
    if not isinstance(raw, list):
        raise TaskError("INPUT_PROTOCOL_ERROR", "正式输入顶层必须是 JSON 数组。")
    ids = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str) or not item["case_id"].strip():
            raise TaskError("INPUT_PROTOCOL_ERROR", "每例必须包含非空字符串 case_id。")
        ids.append(item["case_id"])
    if len(ids) != len(set(ids)):
        raise TaskError("DUPLICATE_CASE_ID", "case_id 不得重复。")
    cases: list[Case | CaseError] = []
    for item in raw:
        try:
            cases.append(_parse_case(item))
        except CaseError as exc:
            exc.case_id = item["case_id"]
            cases.append(exc)
    return cases


def _parse_case(item: dict) -> Case:
    cutoff = item.get("record_cutoff")
    if not isinstance(cutoff, str) or not cutoff.strip():
        raise CaseError("INVALID_RECORD_CUTOFF", "record_cutoff 必须为非空字符串。")
    raw_sources = item.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CaseError("INVALID_SOURCES", "sources 必须为非空数组。")
    sources = []
    seen = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise CaseError("INVALID_SOURCE", "source 必须为对象。")
        source_id, content = raw.get("source_id"), raw.get("content")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in seen:
            raise CaseError("INVALID_SOURCE_ID", "source_id 必须在病例内唯一且非空。")
        if not isinstance(content, str) or not content.strip():
            raise CaseError("INVALID_SOURCE_CONTENT", f"{source_id} 的 content 必须为非空字符串。")
        seen.add(source_id)
        sources.append(Source(source_id, content, str(raw.get("field_path", "")), str(raw.get("source_id_origin", "provided"))))
    return Case(item["case_id"], cutoff, sources)


def failed_result(case_id: str, code: str, message: str) -> dict:
    return {"case_id": case_id, "status": "failed", "medical_record": None,
            "error": {"code": code, "message": message}}


def success_result(case_id: str, record: dict) -> dict:
    return {"case_id": case_id, "status": "success", "medical_record": record, "error": None}
