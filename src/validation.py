import re

from .errors import CaseError
from .protocol import Case


REQUIRED_TEXT = ("case_features", "diagnostic_discussion", "plan")
NUMBER_WITH_CONTEXT = re.compile(r"(?P<prefix>[\u4e00-\u9fffA-Za-z]{0,10})(?P<num>[<>≤≥]?\d+(?:\.\d+)?)(?P<suffix>\s*(?:℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%|岁|天|日|周|月|年)?)")


def validate_record(record, case: Case) -> list[dict]:
    issues: list[dict] = []
    if not isinstance(record, dict):
        raise CaseError("INVALID_MODEL_OUTPUT", "模型结果必须是 JSON 对象。")
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
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
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
    for match in NUMBER_WITH_CONTEXT.finditer(statement):
        token = match.group("num") + match.group("suffix").replace(" ", "")
        compact = cited.replace(" ", "")
        if token not in compact and match.group("num") not in compact:
            issues.append(_issue("error", "NUMERIC_SUPPORT", f"evidence[{index}] 的数值/单位“{token}”未在引用来源中找到。"))
    if any(x in statement for x in ("否认", "未见", "无明显", "阴性")) and not any(x in cited for x in ("否认", "未见", "无明显", "阴性", "未闻及")):
        issues.append(_issue("warning", "NEGATION_SUPPORT", f"evidence[{index}] 的否定表述需人工复核。"))
    if any(x in statement for x in ("已完成", "已予", "已行")) and any(x in cited for x in ("待完善", "拟行", "计划", "建议")):
        issues.append(_issue("error", "PLAN_AS_DONE", f"evidence[{index}] 可能把计划写成已完成。"))
    if kind == "fact" and not _has_lexical_support(statement, cited):
        issues.append(_issue("warning", "FACT_SUPPORT_LIMITED",
                             f"evidence[{index}] 与引用来源的字面重合较低，程序无法确认语义支持，需人工或模型核验。"))
    return issues


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
