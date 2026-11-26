"""Deterministic atomic-claim decomposition for all output sections."""

from __future__ import annotations

import re


TEXT_SECTIONS = ("case_features", "diagnostic_discussion", "plan")


def extract_claims(record: dict) -> list[dict]:
    claims = []
    for section in TEXT_SECTIONS:
        value = record.get(section)
        if isinstance(value, str):
            for statement, start, end in _split_with_spans(value):
                claims.append(_claim(len(claims) + 1, section, statement,
                                     _claim_type(section, statement), [], start, end, None))
    for item_index, item in enumerate(
            record.get("evidence", []) if isinstance(record.get("evidence"), list) else []):
        if isinstance(item, dict) and isinstance(item.get("statement"), str):
            kind = "inference" if item.get("kind") == "inference" else "patient_fact"
            claims.append(_claim(len(claims) + 1, "evidence", item["statement"], kind,
                                 item.get("source_ids") if isinstance(item.get("source_ids"), list) else [],
                                 None, None, item_index))
    for item_index, statement in enumerate(
            record.get("review_items", []) if isinstance(record.get("review_items"), list) else []):
        if isinstance(statement, str) and statement.strip():
            claims.append(_claim(len(claims) + 1, "review_items", statement.strip(),
                                 "missing_information", [], None, None, item_index))
    return claims


def _split_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split deterministic claims while retaining exact replacement spans.

    Only sentence/semicolon boundaries are editable Claim boundaries.  Earlier
    versions split long sentences on commas, which allowed a fallback patch to
    replace half of a parenthesised clinical analysis and corrupt adjacent text.
    Long sentences remain one Claim so a patch is either structurally complete
    or rejected; evidence details are still available in the original sources.
    """
    result = []
    for match in re.finditer(r"[^。！？!?；;\r\n]+(?:[。！？!?；;]+|$)", text):
        value = match.group(0)
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        if right <= left:
            continue
        start = match.start() + left
        end = match.start() + right
        result.append((text[start:end], start, end))
    return result


def _claim_type(section: str, statement: str) -> str:
    # Missing/unknown status remains a missing-information claim even when the
    # sentence ends with “需确认”; that phrase is not itself a clinical plan.
    if any(word in statement for word in ("未提供", "不详", "未知", "需确认", "待确认")):
        return "missing_information"
    if section == "plan" and any(word in statement for word in ("建议", "拟", "计划", "需", "可考虑")):
        return "clinical_plan"
    if section == "diagnostic_discussion" or any(word in statement for word in ("考虑", "可能", "提示", "待排")):
        return "inference"
    return "patient_fact"


def _claim(index: int, section: str, statement: str, claim_type: str, source_ids: list[str],
           span_start: int | None, span_end: int | None, item_index: int | None) -> dict:
    return {"claim_id": f"C{index:03d}", "section": section, "statement": statement,
            "claim_type": claim_type, "source_ids": list(source_ids),
            "span_start": span_start, "span_end": span_end, "item_index": item_index}
