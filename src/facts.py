import re
from dataclasses import asdict, dataclass

from .protocol import Case, Source


@dataclass
class Fact:
    raw_text: str
    source_id: str
    field_path: str
    fact_text: str
    time: str | None
    value: str | None
    unit: str | None
    assertion: str
    status: str
    correction: dict | None


TIME_RE = re.compile(r"(?:\d{4}[-‑年]\d{1,2}(?:[-‑月]\d{1,2}日?)?|\d+(?:天|日|周|月|年|小时|分钟)前|入院时|目前|今日)")
VALUE_RE = re.compile(r"(?P<value>[<>≤≥]?\d+(?:\.\d+)?)\s*(?P<unit>℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%)")


def organize_facts(case: Case) -> list[dict]:
    facts = []
    for source in case.sources:
        clauses = [part.strip("，,。 ") for part in re.split(r"[；;]|，但", source.content) if part.strip("，,。 ")]
        for clause in clauses or [source.content]:
            facts.append(asdict(_source_fact(source, clause)))
    return facts


def _source_fact(source: Source, fact_text: str) -> Fact:
    text = fact_text
    if any(word in text for word in ("未提供", "不详", "未知", "未查")):
        assertion = "not_provided"
    elif any(word in text for word in ("待排", "考虑", "可能", "？", "?", "尚不能")):
        assertion = "uncertain"
    elif any(word in text for word in ("否认", "未见", "无明显", "未闻及", "阴性")):
        assertion = "negative"
    else:
        assertion = "positive"
    if any(word in text for word in ("待完善", "拟行", "计划", "建议", "待复查")):
        status = "planned"
    elif any(word in text for word in ("正在", "目前使用", "长期服用", "继续")):
        status = "ongoing"
    elif any(word in text for word in ("已行", "曾行", "检查提示", "予以", "既往予", "曾予", "服用", "完成")):
        status = "completed"
    else:
        status = "unknown"
    time = TIME_RE.search(text)
    value = VALUE_RE.search(text)
    correction = None
    if "原记录" in text and "复核" in text:
        correction = {"relation": "supersedes", "note": "原记录经复核更正；当前事实以复核后表述为准。"}
    return Fact(source.content, source.source_id, source.field_path, fact_text, time.group(0) if time else None,
                value.group("value") if value else None, value.group("unit") if value else None,
                assertion, status, correction)
