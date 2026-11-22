import json
from pathlib import Path

from .errors import TaskError
from .io_utils import atomic_write_json


def adapt_file(input_path: Path, output_path: Path, case_id: str, record_cutoff: str) -> None:
    try:
        root = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskError("PRACTICE_INPUT_INVALID", f"练习 JSON 无法解析：{exc}") from exc
    if not isinstance(root, dict) or not isinstance(root.get("入院记录"), dict):
        raise TaskError("PRACTICE_PROTOCOL_ERROR", "练习输入必须包含对象型顶层字段“入院记录”。")
    if not case_id.strip() or not record_cutoff.strip():
        raise TaskError("PRACTICE_CONFIG_MISSING", "case_id 和 record_cutoff 必须显式提供。")
    sources = []
    for path, value in _leaves(root["入院记录"], "入院记录"):
        if value is None or value == "" or value == []:
            continue
        content = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        sources.append({"source_id": f"P{len(sources)+1:04d}", "content": content,
                        "field_path": path, "source_id_origin": "program_derived"})
    atomic_write_json(output_path, [{"case_id": case_id, "record_cutoff": record_cutoff, "sources": sources}])


def _leaves(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    else:
        yield path, value
