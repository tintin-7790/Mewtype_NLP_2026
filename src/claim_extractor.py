"""Deterministic atomic-claim decomposition for all output sections."""

from __future__ import annotations

import re


TEXT_SECTIONS = ("case_features", "diagnostic_discussion", "plan")


def extract_claims(record: dict) -> list[dict]:
    claims = []
    for section in TEXT_SECTIONS:
        value = record.get(section)
        if isinstance(value, str):
            for statement in _split(value):
                claims.append(_claim(len(claims) + 1, section, statement, _claim_type(section, statement), []))
    for item in record.get("evidence", []) if isinstance(record.get("evidence"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("statement"), str):
            kind = "inference" if item.get("kind") == "inference" else "patient_fact"
            claims.append(_claim(len(claims) + 1, "evidence", item["statement"], kind,
                                 item.get("source_ids") if isinstance(item.get("source_ids"), list) else []))
    for statement in record.get("review_items", []) if isinstance(record.get("review_items"), list) else []:
        if isinstance(statement, str) and statement.strip():
            claims.append(_claim(len(claims) + 1, "review_items", statement.strip(), "missing_information", []))
    return claims


def _split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])|[\r\n]+", text) if part.strip()]


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


def _claim(index: int, section: str, statement: str, claim_type: str, source_ids: list[str]) -> dict:
    return {"claim_id": f"C{index:03d}", "section": section, "statement": statement,
            "claim_type": claim_type, "source_ids": list(source_ids)}
