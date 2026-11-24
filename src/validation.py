import hashlib
import re
import unicodedata
from copy import deepcopy

from .errors import CaseError
from .protocol import Case


REQUIRED_TEXT = ("case_features", "diagnostic_discussion", "plan")
NUMBER_WITH_CONTEXT = re.compile(r"(?P<prefix>[\u4e00-\u9fffA-Za-z]{0,10})(?P<num>[<>≤≥]?\d+(?:\.\d+)?)(?P<suffix>\s*(?:℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%|岁|天|日|周|月|年)?)")


def validate_record(record, case: Case) -> list[dict]:
    issues: list[dict] = []
    if not isinstance(record, dict):
        raise CaseError("INVALID_MODEL_OUTPUT", "模型结果必须是 JSON 对象。")
    expected = {"case_features", "diagnostic_discussion", "plan", "evidence", "review_items"}
    if set(record) != expected:
        issues.append(_issue("error", "FORMAT", "病历顶层字段不完整或含未知字段。"))
    for field in REQUIRED_TEXT:
        if not isinstance(record.get(field), str) or not record[field].strip():
            issues.append(_issue("error", "FORMAT", f"{field} 必须是非空字符串。"))
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        issues.append(_issue("error", "FORMAT", "evidence 必须是数组。"))
        evidence = []
    if not isinstance(record.get("review_items"), list) or any(not isinstance(x, str) for x in record.get("review_items", [])):
        issues.append(_issue("error", "FORMAT", "review_items 必须是字符串数组。"))
    known = {s.source_id: s.content for s in case.sources}
    for index, item in enumerate(evidence):
        if (not isinstance(item, dict) or set(item) != {"statement", "source_ids", "kind"}
                or not isinstance(item.get("statement"), str) or not item["statement"].strip()):
            issues.append(_issue("error", "EVIDENCE_FORMAT", f"evidence[{index}] 结构无效。")); continue
        ids = item.get("source_ids")
        if not isinstance(ids, list) or not ids or any(x not in known for x in ids):
            issues.append(_issue("error", "SOURCE_ID", f"evidence[{index}] 引用了不存在或空的 source_ids。")); continue
        if item.get("kind") not in ("fact", "inference"):
            issues.append(_issue("error", "EVIDENCE_KIND", f"evidence[{index}].kind 非法。"))
        cited = "\n".join(known[x] for x in ids)
        issues.extend(_support_checks(item["statement"], cited, index, item.get("kind")))
    return issues


def _support_checks(statement: str, cited: str, index: int, kind: str | None) -> list[dict]:
    issues = []
    normalized_statement = _normalize_measurements(normalize_equivalent_dates(statement))
    normalized_cited = _normalize_measurements(normalize_equivalent_dates(cited))
    for match in NUMBER_WITH_CONTEXT.finditer(normalized_statement):
        token = match.group("num") + match.group("suffix").replace(" ", "")
        compact = normalized_cited.replace(" ", "")
        # A value with a unit must match that exact value/unit pair.  Falling
        # back to the bare number can confuse thresholds, dates and other tests.
        supported = token in compact if match.group("suffix").strip() else match.group("num") in compact
        if not supported:
            issues.append(_issue("error", "NUMERIC_SUPPORT", f"evidence[{index}] 的数值/单位“{token}”未在引用来源中找到。"))
    if any(x in statement for x in ("否认", "未见", "无明显", "阴性")) and not any(x in cited for x in ("否认", "未见", "无明显", "阴性", "未闻及")):
        issues.append(_issue("warning", "NEGATION_SUPPORT", f"evidence[{index}] 的否定表述需人工复核。"))
    if any(x in statement for x in ("已完成", "已予", "已行")) and any(x in cited for x in ("待完善", "拟行", "计划", "建议")):
        issues.append(_issue("error", "PLAN_AS_DONE", f"evidence[{index}] 可能把计划写成已完成。"))
    if kind == "fact" and not _has_lexical_support(statement, cited):
        issues.append(_issue("warning", "FACT_SUPPORT_LIMITED",
                             f"evidence[{index}] 与引用来源的字面重合较低，程序无法确认语义支持，需人工或模型核验。"))
    return issues


def repair_evidence_numeric_support(record: dict, case: Case,
                                    max_source_chars: int = 320) -> tuple[dict, dict]:
    """Safely replace invalid fact evidence with its short, exact cited text.

    No numeric value is guessed or mapped to a nearby value. Inference evidence
    and long/multi-purpose sources are left unresolved so final validation can
    fail explicitly instead of manufacturing provenance.
    """
    result = deepcopy(record)
    known = {source.source_id: source.content for source in case.sources}
    repaired = []
    unresolved = []
    for index, item in enumerate(result.get("evidence", []) if isinstance(result, dict) else []):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        ids = item.get("source_ids")
        if not isinstance(ids, list) or not ids or any(source_id not in known for source_id in ids):
            continue
        cited = "\n".join(known[source_id] for source_id in ids)
        numeric = [issue for issue in _support_checks(item["statement"], cited, index, item.get("kind"))
                   if issue["code"] == "NUMERIC_SUPPORT"]
        if not numeric:
            continue
        exact = "；".join(known[source_id].strip() for source_id in ids)
        can_repair = item.get("kind") == "fact" and 0 < len(exact) <= max_source_chars
        if can_repair:
            remaining = [issue for issue in _support_checks(exact, cited, index, "fact")
                         if issue["code"] == "NUMERIC_SUPPORT"]
            if not remaining:
                before = item["statement"]
                item["statement"] = exact
                repaired.append({"evidence_index": index, "strategy": "exact_cited_source",
                                 "source_ids": list(ids),
                                 "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                                 "after_sha256": hashlib.sha256(exact.encode()).hexdigest()})
                continue
        unresolved.append({"evidence_index": index, "code": "NUMERIC_SUPPORT",
                           "kind": item.get("kind"), "source_ids": list(ids)})
    return result, {"repaired_count": len(repaired), "unresolved_count": len(unresolved),
                    "repairs": repaired, "unresolved": unresolved}


def normalize_equivalent_dates(value: str) -> str:
    """Normalize common equivalent Chinese numeric dates without changing chronology."""
    def dotted(match):
        year, month, day = match.group(1), int(match.group(2)), match.group(3)
        return f"{year}年{month}月" + (f"{int(day)}日" if day else "")

    normalized = re.sub(r"(?<!\d)(\d{4})[./-](\d{1,2})(?:[./-](\d{1,2}))?(?!\d)", dotted, value)
    normalized = re.sub(r"(?<!\d)(\d{4})年0?(\d{1,2})月(?:0?(\d{1,2})日)?",
                        lambda match: f"{match.group(1)}年{int(match.group(2))}月" +
                        (f"{int(match.group(3))}日" if match.group(3) else ""), normalized)
    return normalized


def _normalize_measurements(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return (normalized.replace("µ", "μ").replace("\u00a0", " ")
            .replace("−", "-").replace("‑", "-").replace("–", "-").replace("—", "-"))


def _has_lexical_support(statement: str, cited: str) -> bool:
    clean = lambda value: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ]+", "", value)
    left, right = clean(statement), clean(cited)
    if left and left in right:
        return True
    grams = {left[i:i+2] for i in range(max(0, len(left)-1)) if not left[i:i+2].isdigit()}
    if not grams:
        return True
    return sum(gram in right for gram in grams) / len(grams) >= 0.25


def _issue(level: str, code: str, message: str) -> dict:
    return {"level": level, "code": code, "message": message}
