"""Strict input detection and in-memory practice-record adaptation."""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .errors import CaseError, TaskError
from .practice_adapter import adapt_practice_data
from .protocol import Case, parse_standard_cases


SUPPORTED_INPUT_FORMATS = ("auto", "competition", "practice")
_FORMAL_KEYS = frozenset({"case_id", "record_cutoff", "sources"})
_PRACTICE_SECTIONS = frozenset({
    "主诉", "现病史", "既往史", "个人史", "婚育史", "月经史生育史", "家族史",
    "体格检查", "辅助检查", "初步诊断", "记录医师",
})
_DASHES = str.maketrans({char: "-" for char in "‐‑‒–—―−﹘﹣－"})
_RECORD_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(:\d{2})?$")
_PRACTICE_SEQUENCE = re.compile(r"^(\d{1,4})_")


@dataclass(frozen=True)
class InputAdapterResult:
    cases: list[Case | CaseError]
    detected_format: str
    audit: dict


def load_input_cases(path: Path, input_format: str = "auto", *,
                     practice_case_id: str | None = None,
                     practice_record_cutoff: str | None = None) -> InputAdapterResult:
    if input_format not in SUPPORTED_INPUT_FORMATS:
        raise TaskError("INPUT_FORMAT_INVALID", f"不支持的输入格式选项：{input_format}")
    raw = _read_json(path)
    detected = _detect_format(raw, input_format)
    if detected == "competition":
        if practice_case_id is not None or practice_record_cutoff is not None:
            raise TaskError("PRACTICE_OVERRIDE_SCOPE_INVALID",
                            "--practice-case-id/--practice-record-cutoff 只允许练习输入。")
        cases = parse_standard_cases(raw)
        return InputAdapterResult(cases, detected, {
            "requested_format": input_format,
            "detected_format": detected,
            "adapted": False,
            "case_count": len(cases),
        })

    records = raw if isinstance(raw, list) else [raw]
    if len(records) > 1 and (practice_case_id is not None or practice_record_cutoff is not None):
        raise TaskError(
            "PRACTICE_OVERRIDE_SCOPE_INVALID",
            "--practice-case-id/--practice-record-cutoff 只允许单份练习输入。",
        )
    adapted_records = []
    mappings = []
    for index, record in enumerate(records, start=1):
        _validate_practice_shape(record)
        case_id, case_id_origin = _practice_case_id(
            path, practice_case_id, record,
            array_index=index if isinstance(raw, list) else None,
        )
        original_cutoff, cutoff_origin = _practice_cutoff(record, practice_record_cutoff)
        normalized_cutoff = normalize_record_cutoff(original_cutoff)
        adapted = adapt_practice_data(record, case_id, normalized_cutoff)
        adapted_records.append(adapted)
        mappings.append({
            "case_id": case_id,
            "case_id_origin": case_id_origin,
            "record_cutoff_original": original_cutoff,
            "record_cutoff_normalized": normalized_cutoff,
            "record_cutoff_origin": cutoff_origin,
            "source_count": len(adapted["sources"]),
        })
    cases = parse_standard_cases(adapted_records)
    return InputAdapterResult(cases, detected, {
        "requested_format": input_format,
        "detected_format": detected,
        "adapted": True,
        "case_count": len(cases),
        "case_mappings": mappings,
    })


def normalize_record_cutoff(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskError("PRACTICE_RECORD_CUTOFF_MISSING", "练习病例缺少可靠的记录时间。")
    normalized = " ".join(unicodedata.normalize("NFKC", value).translate(_DASHES).split())
    match = _RECORD_TIME.fullmatch(normalized)
    if not match:
        raise TaskError("PRACTICE_RECORD_CUTOFF_INVALID",
                        "记录时间必须为 YYYY-MM-DD HH:MM 或 YYYY-MM-DD HH:MM:SS。")
    pattern = "%Y-%m-%d %H:%M:%S" if match.group(3) else "%Y-%m-%d %H:%M"
    try:
        parsed = datetime.strptime(normalized.replace("T", " "), pattern)
    except ValueError as exc:
        raise TaskError("PRACTICE_RECORD_CUTOFF_INVALID", "记录时间不是有效日历时间。") from exc
    return parsed.strftime(pattern)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskError("INPUT_NOT_FOUND", f"输入文件不存在：{path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskError("INPUT_INVALID_JSON", f"输入不是有效 UTF-8 JSON：{exc}") from exc


def _detect_format(raw, requested: str) -> str:
    if requested == "competition":
        if not isinstance(raw, list):
            raise TaskError("INPUT_PROTOCOL_ERROR", "正式输入顶层必须是 JSON 数组。")
        if raw and all(_is_practice_shape(item) for item in raw):
            raise TaskError("INPUT_PROTOCOL_ERROR", "显式 competition 模式不接受练习病历数组。")
        return "competition"
    if requested == "practice":
        _validate_practice_payload(raw)
        return "practice"
    if isinstance(raw, list):
        if not raw:
            return "competition"
        practice_flags = [_is_practice_shape(item) for item in raw]
        formal_flags = [_is_formal_like(item) for item in raw]
        if all(practice_flags):
            return "practice"
        if all(formal_flags):
            return "competition"
        if any(practice_flags) and any(formal_flags):
            raise TaskError("INPUT_FORMAT_MIXED", "同一输入数组不得混合正式 Sources 病例与练习嵌套病历。")
        if any(practice_flags):
            raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习病例数组包含不符合练习协议的元素。")
        if any(formal_flags):
            raise TaskError("INPUT_PROTOCOL_ERROR", "正式病例数组包含无法识别的元素。")
        raise TaskError("INPUT_FORMAT_UNKNOWN", "无法识别输入数组格式。")
    if isinstance(raw, dict):
        if _FORMAL_KEYS & raw.keys():
            raise TaskError("INPUT_PROTOCOL_ERROR",
                            "检测到正式协议字段，但正式输入必须是 JSON 数组且字段完整；不会转为练习格式。")
        if "入院记录" in raw:
            _validate_practice_shape(raw)
            return "practice"
    raise TaskError("INPUT_FORMAT_UNKNOWN", "无法识别输入格式；仅支持正式病例数组或严格的练习嵌套病历。")


def _validate_practice_shape(raw) -> None:
    if not isinstance(raw, dict) or "入院记录" not in raw or not set(raw) <= {"入院记录", "case_id"}:
        raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习病例仅允许顶层字段“入院记录”和可选的 case_id。")
    if "case_id" in raw and (not isinstance(raw["case_id"], str) or not raw["case_id"].strip()):
        raise TaskError("PRACTICE_CASE_ID_INVALID", "练习病例显式 case_id 必须为非空字符串。")
    admission = raw.get("入院记录")
    if not isinstance(admission, dict):
        raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习输入的“入院记录”必须为对象。")
    if not isinstance(admission.get("一般项目"), dict):
        raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习输入必须包含对象型“入院记录.一般项目”。")
    if not (_PRACTICE_SECTIONS & admission.keys()):
        raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习输入缺少可识别的临床记录章节。")


def _validate_practice_payload(raw) -> None:
    if isinstance(raw, list):
        if not raw:
            raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习病例数组不得为空。")
        for item in raw:
            _validate_practice_shape(item)
        return
    _validate_practice_shape(raw)


def _is_practice_shape(raw) -> bool:
    try:
        _validate_practice_shape(raw)
        return True
    except TaskError:
        return False


def _is_formal_like(raw) -> bool:
    return isinstance(raw, dict) and bool(_FORMAL_KEYS & raw.keys()) and "入院记录" not in raw


def _practice_case_id(path: Path, explicit: str | None, raw: dict,
                      array_index: int | None = None) -> tuple[str, str]:
    if explicit is not None:
        if not explicit.strip():
            raise TaskError("PRACTICE_CASE_ID_INVALID", "显式练习病例 case_id 不得为空。")
        return explicit.strip(), "explicit_cli"
    raw_case_id = raw.get("case_id")
    if isinstance(raw_case_id, str) and raw_case_id.strip():
        return raw_case_id.strip(), "provided"
    if array_index is not None:
        return f"PRACTICE{array_index:03d}", "array_sequence"
    match = _PRACTICE_SEQUENCE.match(path.name)
    if match:
        return f"PRACTICE{match.group(1).zfill(2)}", "filename_sequence"
    fingerprint = hashlib.sha256(
        json.dumps(_redact_identity_fields(raw), ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12].upper()
    return f"PRACTICE_AUTO_{fingerprint}", "redacted_content_fingerprint"


_IDENTITY_FIELDS = frozenset({
    "姓名", "身份证", "身份证号", "证件号码", "住院号", "病案号", "联系电话",
    "手机号", "电话", "现住址", "家庭住址", "工作单位", "联系人",
})


def _redact_identity_fields(value):
    """Build a stable fingerprint payload without direct personal identifiers."""
    if isinstance(value, dict):
        return {key: ("<REDACTED>" if key in _IDENTITY_FIELDS else _redact_identity_fields(child))
                for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_identity_fields(child) for child in value]
    return value


def _practice_cutoff(raw: dict, explicit: str | None) -> tuple[str, str]:
    if explicit is not None:
        return explicit, "explicit_cli"
    general = raw["入院记录"]["一般项目"]
    value = general.get("记录时间")
    if not isinstance(value, str) or not value.strip():
        raise TaskError("PRACTICE_RECORD_CUTOFF_MISSING",
                        "练习病例缺少“入院记录.一般项目.记录时间”；请使用 --practice-record-cutoff 显式指定。")
    return value, "入院记录.一般项目.记录时间"
