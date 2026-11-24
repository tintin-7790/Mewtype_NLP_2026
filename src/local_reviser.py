"""One-pass localized revision that preserves unaffected text sections exactly."""

from __future__ import annotations

import json
import hashlib
from difflib import SequenceMatcher
from copy import deepcopy
from pathlib import Path

from .errors import CaseError
from .model_client import Budget, REVISER_SCHEMA, parse_model_json
from .protocol import Case


BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_SECTIONS = {"case_features", "diagnostic_discussion", "plan"}


def run_local_revision(case: Case, record: dict, clinical_plan: dict, issues: list[dict],
                       retriever, retrieval_top_k: int, client, response_format: str,
                       budget: Budget) -> tuple[dict, dict]:
    target_sections = {issue["section"] for issue in issues}
    evidence = []
    if retriever is not None:
        for issue in issues:
            evidence.append({
                "claim_id": issue["claim_id"],
                "candidates": [candidate.to_dict() for candidate in
                               retriever.retrieve(issue["statement"], top_k=retrieval_top_k)],
            })
    editable_text = {section: record[section] for section in TEXT_SECTIONS & target_sections}
    readonly_text = {section: record[section] for section in TEXT_SECTIONS - target_sections}
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "clinical_plan": clinical_plan,
        "issues": issues,
        "editable_sections": editable_text,
        "readonly_sections": readonly_text,
        "current_evidence": record.get("evidence", []),
        "current_review_items": record.get("review_items", []),
        "retrieval_candidates": evidence,
    }
    raw = client.call(_messages("revise_system.txt", payload), response_format, budget,
                      stage="reviser", schema=REVISER_SCHEMA)
    revision = parse_model_json(raw, stage="reviser", client=client)
    revised = apply_revision(record, revision, target_sections)
    meta = {"target_sections": sorted(target_sections),
            "preserved_sections": sorted(section for section in TEXT_SECTIONS if revised[section] == record[section]),
            "amplitude": revision_amplitude(record, revised, target_sections)}
    return revised, meta


def apply_conservative_fallback(record: dict, issues: list[dict]) -> tuple[dict, dict]:
    """Remove a located definite error when no safe model revision is available."""
    result = deepcopy(record)
    resolved = 0
    unresolved = []
    for issue in issues:
        section = issue.get("section")
        statement = issue.get("statement")
        replacement = _fallback_text(issue.get("error_type"))
        changed = False
        if section in TEXT_SECTIONS and isinstance(result.get(section), str) and isinstance(statement, str):
            if statement in result[section]:
                result[section] = result[section].replace(statement, replacement, 1)
                changed = True
        elif section == "evidence" and isinstance(result.get("evidence"), list):
            before = len(result["evidence"])
            result["evidence"] = [item for item in result["evidence"]
                                  if not (isinstance(item, dict) and item.get("statement") == statement)]
            changed = len(result["evidence"]) < before
        elif section == "review_items" and isinstance(result.get("review_items"), list):
            for index, item in enumerate(result["review_items"]):
                if item == statement:
                    result["review_items"][index] = replacement
                    changed = True
                    break
        if changed:
            resolved += 1
        else:
            unresolved.append(issue.get("claim_id"))
        hint = issue.get("revision_hint") or "相关事实需结合原始资料确认。"
        note = f"相关事实已作保守处理，仍需医生确认：{hint}"
        result.setdefault("review_items", [])
        if note not in result["review_items"]:
            result["review_items"].append(note)
    return result, {"strategy": "deterministic_conservative_fallback", "resolved_count": resolved,
                    "unresolved_claim_ids": unresolved,
                    "amplitude": revision_amplitude(record, result, set(issue.get("section") for issue in issues))}


def apply_revision(record: dict, revision, target_sections: set[str]) -> dict:
    if not isinstance(revision, dict) or not isinstance(revision.get("section_updates"), dict):
        raise CaseError("INVALID_LOCAL_REVISION", "局部修订必须包含 section_updates 对象。")
    updates = revision["section_updates"]
    if any(section not in TEXT_SECTIONS or section not in target_sections for section in updates):
        raise CaseError("INVALID_LOCAL_REVISION", "局部修订试图修改未定位的正文段落。")
    if any(not isinstance(value, str) or not value.strip() for value in updates.values()):
        raise CaseError("INVALID_LOCAL_REVISION", "局部修订后的正文必须是非空字符串。")
    required_updates = TEXT_SECTIONS & target_sections
    if not required_updates.issubset(updates):
        raise CaseError("INVALID_LOCAL_REVISION", "局部修订未返回所有被定位的正文段落。")
    if any(updates[section] == record[section] for section in required_updates):
        raise CaseError("INVALID_LOCAL_REVISION", "被定位的正文段落未发生实际修订。")
    if set(revision) != {"section_updates", "evidence", "review_items"}:
        raise CaseError("INVALID_LOCAL_REVISION", "局部修订字段不完整或含未知字段。")
    if not isinstance(revision["evidence"], list):
        raise CaseError("INVALID_LOCAL_REVISION", "修订后的 evidence 必须是完整数组。")
    if not isinstance(revision["review_items"], list) or any(not isinstance(item, str) for item in revision["review_items"]):
        raise CaseError("INVALID_LOCAL_REVISION", "修订后的 review_items 必须是完整字符串数组。")
    result = deepcopy(record)
    for section, value in updates.items():
        result[section] = value
    result["evidence"] = revision["evidence"]
    result["review_items"] = revision["review_items"]
    if "evidence" not in target_sections and result["evidence"] != record.get("evidence", []):
        raise CaseError("LOCAL_REVISION_SCOPE_ERROR", "未定位的 evidence 被意外修改。")
    if "review_items" not in target_sections and result["review_items"] != record.get("review_items", []):
        raise CaseError("LOCAL_REVISION_SCOPE_ERROR", "未定位的 review_items 被意外修改。")
    for section in TEXT_SECTIONS - target_sections:
        if result[section] != record[section]:
            raise CaseError("LOCAL_REVISION_SCOPE_ERROR", f"未定位段落 {section} 被意外修改。")
    return result


def revision_amplitude(before: dict, after: dict, target_sections: set[str]) -> dict:
    metrics = {}
    for section in sorted(TEXT_SECTIONS & target_sections):
        left, right = before.get(section, ""), after.get(section, "")
        metrics[section] = {
            "before_length": len(left), "after_length": len(right),
            "changed_char_estimate": _changed_chars(left, right),
            "sequence_similarity": round(SequenceMatcher(None, left, right).ratio(), 4),
            "before_sha256": hashlib.sha256(left.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(right.encode("utf-8")).hexdigest(),
        }
    for section in ("evidence", "review_items"):
        if section in target_sections:
            metrics[section] = {"before_count": len(before.get(section, [])),
                                "after_count": len(after.get(section, []))}
    return metrics


def _changed_chars(left: str, right: str) -> int:
    matches = sum(block.size for block in SequenceMatcher(None, left, right).get_matching_blocks())
    return len(left) + len(right) - 2 * matches


def _fallback_text(error_type: str | None) -> str:
    if error_type in ("UNKNOWN_AS_NEGATIVE", "NOT_ADDRESSED"):
        return "相关检查或治疗状态未提供，需医生确认。"
    if error_type == "DIAGNOSIS_CERTAINTY_UPGRADE":
        return "该诊断目前仅为初步判断或待排，需结合后续资料确认。"
    if error_type == "PLAN_AS_COMPLETED":
        return "相关检查或治疗尚处计划或结果待确认状态。"
    if error_type == "FACT_CONTRADICTION":
        return "该项患者事实存在冲突，需以原始资料复核结果为准。"
    if error_type == "CORRECTION_IGNORED":
        return "该项事实应以复核更正后的记录为准。"
    return "该项陈述依据不足，需医生结合原始资料确认。"


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
