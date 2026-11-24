"""Claim-addressed one-pass revision with exact, auditable patches."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

from .claim_extractor import extract_claims
from .errors import CaseError
from .model_client import Budget, REVISER_SCHEMA, parse_model_json
from .protocol import Case
from .validation import validate_record


BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_SECTIONS = {"case_features", "diagnostic_discussion", "plan"}
TECHNICAL_REVIEW_TERMS = ("JSON", "API", "模型", "解析", "自动核验", "程序错误", "调用失败")


def run_local_revision(case: Case, record: dict, clinical_plan: dict, issues: list[dict],
                       retriever, retrieval_top_k: int, client, response_format: str,
                       budget: Budget) -> tuple[dict, dict]:
    claim_map = {claim["claim_id"]: claim for claim in extract_claims(record)}
    target_ids = {issue["claim_id"] for issue in issues}
    if any(claim_id not in claim_map for claim_id in target_ids):
        raise CaseError("LOCAL_REVISION_LOCATOR_ERROR", "修订问题无法映射到当前病历 Claim。")
    retrieval = []
    retrieval_source_ids = set()
    if retriever is not None:
        for issue in issues:
            values = [candidate.to_dict() for candidate in
                      retriever.retrieve(issue["statement"], top_k=retrieval_top_k)]
            retrieval_source_ids.update(value.get("source_id") for value in values)
            retrieval.append({"claim_id": issue["claim_id"], "candidates": [
                {key: value.get(key) for key in
                 ("source_id", "similarity", "compatible", "compatibility_warnings")}
                for value in values
            ]})
    relevant_sources = _select_relevant_sources(case, issues, retrieval_source_ids)
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": relevant_sources,
        "issues": [{"claim_id": issue["claim_id"], "section": issue["section"],
                    "text": issue["statement"], "code": issue["error_type"],
                    "hint": issue["revision_hint"], "source_ids": issue["source_ids"]}
                   for issue in issues],
        "editable_claims": [{"claim_id": claim_id, "section": claim_map[claim_id]["section"],
                             "text": claim_map[claim_id]["statement"]}
                            for claim_id in sorted(target_ids)],
        "retrieval_candidates": retrieval,
    }
    raw = client.call(_messages("revise_system.txt", payload), response_format, budget,
                      stage="reviser", schema=REVISER_SCHEMA)
    revision = parse_model_json(raw, stage="reviser", client=client)
    try:
        revised, patch_meta = apply_revision(record, revision, issues)
    except CaseError as exc:
        _record_schema(client, False, exc.code)
        raise
    _record_schema(client, True)
    structure_errors = [item for item in validate_record(revised, case) if item["level"] == "error"]
    if structure_errors:
        raise CaseError("LOCAL_REVISION_VALIDATION_FAILED",
                        "局部修订破坏输出约束：" + "; ".join(item["message"] for item in structure_errors),
                        {"patch_applied": True, "validation_errors": structure_errors,
                         "revised_claim_ids": patch_meta["revised_claim_ids"]})
    # Import locally to avoid a module cycle at import time.
    from .claim_verifier import deterministic_claim_issues
    remaining = [item for item in deterministic_claim_issues(case, extract_claims(revised))
                 if item.get("severity") == "high"]
    if remaining:
        raise CaseError("LOCAL_REVISION_UNRESOLVED_RISK",
                        "局部修订后仍存在确定性高风险事实问题。",
                        {"patch_applied": True,
                         "remaining_claim_ids": [item["claim_id"] for item in remaining],
                         "revised_claim_ids": patch_meta["revised_claim_ids"]})
    target_sections = {issue["section"] for issue in issues}
    meta = {"strategy": "model_claim_patch",
            "target_sections": sorted(target_sections),
            "preserved_sections": sorted(section for section in TEXT_SECTIONS
                                         if revised[section] == record[section]),
            "revised_claim_ids": patch_meta["revised_claim_ids"],
            "unresolved_claim_ids": patch_meta["unresolved_claim_ids"],
            "review_items_added": patch_meta["review_items_added"],
            "source_count_sent": len(relevant_sources),
            "amplitude": revision_amplitude(record, revised, target_sections)}
    return revised, meta


def apply_revision(record: dict, revision, issues: list[dict]) -> tuple[dict, dict]:
    if (not isinstance(revision, dict) or set(revision) != {"edits", "review_items_to_add"}
            or not isinstance(revision.get("edits"), list)
            or not isinstance(revision.get("review_items_to_add"), list)):
        raise CaseError("INVALID_LOCAL_REVISION",
                        "局部修订必须仅包含 edits 和 review_items_to_add。")
    if any(not isinstance(note, str) or not note.strip() for note in revision["review_items_to_add"]):
        raise CaseError("INVALID_LOCAL_REVISION", "新增待核事项必须是非空字符串。")
    if any(_contains_technical_status(note) for note in revision["review_items_to_add"]):
        raise CaseError("LOCAL_REVISION_SCOPE_ERROR", "技术运行状态不得写入临床 review_items。")
    claim_map = {claim["claim_id"]: claim for claim in extract_claims(record)}
    issue_ids = {issue["claim_id"] for issue in issues}
    edits_by_id = {}
    for edit in revision["edits"]:
        expected = {"claim_id", "section", "action", "replacement"}
        if not isinstance(edit, dict) or set(edit) != expected:
            raise CaseError("INVALID_LOCAL_REVISION", "每个 edit 的字段无效。")
        claim_id = edit["claim_id"]
        if claim_id not in issue_ids or claim_id not in claim_map or claim_id in edits_by_id:
            raise CaseError("INVALID_LOCAL_REVISION", "edit 必须唯一对应已定位的高风险 Claim。")
        claim = claim_map[claim_id]
        if edit["section"] != claim["section"]:
            raise CaseError("LOCAL_REVISION_SCOPE_ERROR", "edit section 与 Claim 定位不一致。")
        if edit["action"] not in ("replace", "remove"):
            raise CaseError("INVALID_LOCAL_REVISION", "edit action 非法。")
        replacement = edit["replacement"]
        if not isinstance(replacement, str):
            raise CaseError("INVALID_LOCAL_REVISION", "replacement 必须是字符串。")
        if edit["action"] == "replace" and not replacement.strip():
            raise CaseError("INVALID_LOCAL_REVISION", "replace 必须提供非空 replacement。")
        if claim["section"] in TEXT_SECTIONS and edit["action"] != "replace":
            raise CaseError("LOCAL_REVISION_SCOPE_ERROR", "正文 Claim 只允许精确替换，不允许删除。")
        edits_by_id[claim_id] = edit
    result = deepcopy(record)
    text_edits = []
    list_edits = []
    for claim_id, edit in edits_by_id.items():
        claim = claim_map[claim_id]
        if claim["section"] in TEXT_SECTIONS:
            text_edits.append((claim, edit))
        else:
            list_edits.append((claim, edit))
    for claim, edit in sorted(list_edits, key=lambda item: (item[0]["section"],
                                                            item[0].get("item_index", -1)), reverse=True):
        _apply_list_edit(result, claim, edit)
    for claim, edit in sorted(text_edits, key=lambda item: _claim_start(record, item[0]), reverse=True):
        _apply_text_edit(result, claim, edit)
    result.setdefault("review_items", [])
    added = []
    for note in revision["review_items_to_add"]:
        if note not in result["review_items"]:
            result["review_items"].append(note)
            added.append(note)
    revised_ids = sorted(edits_by_id)
    return result, {"revised_claim_ids": revised_ids,
                    "unresolved_claim_ids": sorted(issue_ids - set(revised_ids)),
                    "review_items_added": len(added)}


def apply_conservative_fallback(record: dict, issues: list[dict]) -> tuple[dict, dict]:
    """Apply exact deterministic patches; never report this as a model revision."""
    issues = _coalesce_patch_targets(issues)
    claim_map = {claim["claim_id"]: claim for claim in extract_claims(record)}
    edits = []
    unresolved = []
    notes = []
    for issue in issues:
        claim = claim_map.get(issue.get("claim_id"))
        if claim is None:
            unresolved.append(issue.get("claim_id"))
            continue
        action = "remove" if claim["section"] == "evidence" else "replace"
        edits.append({"claim_id": claim["claim_id"], "section": claim["section"],
                      "action": action,
                      "replacement": "" if action == "remove" else _fallback_text(issue.get("error_type"))})
        hint = issue.get("revision_hint") or "相关事实需结合原始资料确认。"
        notes.append(f"相关事实需医生确认：{hint}")
    try:
        result, patch_meta = apply_revision(
            record, {"edits": edits, "review_items_to_add": notes}, issues,
        )
    except CaseError:
        return deepcopy(record), {"strategy": "deterministic_conservative_fallback",
                                  "applied": False, "revised_claim_ids": [],
                                  "unresolved_claim_ids": sorted({issue.get("claim_id") for issue in issues}),
                                  "amplitude": {}}
    unresolved.extend(patch_meta["unresolved_claim_ids"])
    target_sections = {issue.get("section") for issue in issues}
    return result, {"strategy": "deterministic_conservative_fallback", "applied": bool(edits),
                    "revised_claim_ids": patch_meta["revised_claim_ids"],
                    "unresolved_claim_ids": sorted(set(unresolved)),
                    "amplitude": revision_amplitude(record, result, target_sections)}


def _coalesce_patch_targets(issues: list[dict]) -> list[dict]:
    grouped = {}
    for issue in issues:
        claim_id = issue.get("claim_id")
        if claim_id not in grouped:
            grouped[claim_id] = dict(issue)
            continue
        current = grouped[claim_id]
        hints = [part for part in (current.get("revision_hint"), issue.get("revision_hint"))
                 if isinstance(part, str) and part]
        current["revision_hint"] = "；".join(dict.fromkeys(hints))
        current["source_ids"] = list(dict.fromkeys(
            list(current.get("source_ids", [])) + list(issue.get("source_ids", []))))
    return list(grouped.values())


def _contains_technical_status(note: str) -> bool:
    upper = note.upper()
    return ("JSON" in upper or "API" in upper
            or any(term in note for term in TECHNICAL_REVIEW_TERMS[2:]))


def revision_amplitude(before: dict, after: dict, target_sections: set[str]) -> dict:
    metrics = {}
    for section in sorted(TEXT_SECTIONS & target_sections):
        left, right = before.get(section, ""), after.get(section, "")
        metrics[section] = {"before_length": len(left), "after_length": len(right),
                            "changed_char_estimate": _changed_chars(left, right),
                            "sequence_similarity": round(SequenceMatcher(None, left, right).ratio(), 4),
                            "before_sha256": hashlib.sha256(left.encode()).hexdigest(),
                            "after_sha256": hashlib.sha256(right.encode()).hexdigest()}
    for section in ("evidence", "review_items"):
        if section in target_sections or before.get(section) != after.get(section):
            metrics[section] = {"before_count": len(before.get(section, [])),
                                "after_count": len(after.get(section, []))}
    return metrics


def _apply_text_edit(record: dict, claim: dict, edit: dict) -> None:
    section = claim["section"]; text = record[section]
    start, end = claim.get("span_start"), claim.get("span_end")
    if not isinstance(start, int) or not isinstance(end, int) or text[start:end] != claim["statement"]:
        occurrences = [match.span() for match in re.finditer(re.escape(claim["statement"]), text)]
        if len(occurrences) != 1:
            raise CaseError("LOCAL_REVISION_LOCATOR_ERROR",
                            "无法唯一定位正文 Claim，未应用修订。")
        start, end = occurrences[0]
    record[section] = text[:start] + edit["replacement"] + text[end:]


def _claim_start(record: dict, claim: dict) -> int:
    if isinstance(claim.get("span_start"), int):
        return claim["span_start"]
    return record[claim["section"]].find(claim["statement"])


def _apply_list_edit(record: dict, claim: dict, edit: dict) -> None:
    section = claim["section"]; index = claim.get("item_index")
    values = record.get(section)
    if not isinstance(values, list) or not isinstance(index, int) or not 0 <= index < len(values):
        raise CaseError("LOCAL_REVISION_LOCATOR_ERROR", "无法定位列表 Claim。")
    current = values[index]
    text = current.get("statement") if section == "evidence" and isinstance(current, dict) else current
    if text != claim["statement"]:
        raise CaseError("LOCAL_REVISION_LOCATOR_ERROR", "列表 Claim 已变化，未应用修订。")
    if edit["action"] == "remove":
        del values[index]
    elif section == "evidence":
        current["statement"] = edit["replacement"]
    else:
        values[index] = edit["replacement"]


def _select_relevant_sources(case: Case, issues: list[dict], retrieval_source_ids: set) -> list[dict]:
    selected_ids = {source_id for issue in issues for source_id in issue.get("source_ids", [])}
    selected_ids.update(source_id for source_id in retrieval_source_ids if isinstance(source_id, str))
    tokens = set()
    for issue in issues:
        tokens.update(re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%℃]{2,}",
                                 issue.get("statement", "") + issue.get("revision_hint", "")))
    ignored = {"患者", "当前", "相关", "检查", "治疗", "情况", "需要", "资料", "提供"}
    for source in case.sources:
        if any(token not in ignored and token in source.content for token in tokens):
            selected_ids.add(source.source_id)
    selected = [source for source in case.sources if source.source_id in selected_ids]
    if not selected:
        selected = list(case.sources)
    return [{"source_id": source.source_id, "field_path": source.field_path,
             "content": source.content} for source in selected]


def _changed_chars(left: str, right: str) -> int:
    matches = sum(block.size for block in SequenceMatcher(None, left, right).get_matching_blocks())
    return len(left) + len(right) - 2 * matches


def _fallback_text(error_type: str | None) -> str:
    if error_type in ("UNKNOWN_AS_NEGATIVE", "NOT_ADDRESSED", "UNSUPPORTED_NEGATIVE"):
        return "相关检查或治疗状态未提供，需医生确认。"
    if error_type == "DIAGNOSIS_CERTAINTY_UPGRADE":
        return "该诊断目前仅为初步判断或待排，需结合后续资料确认。"
    if error_type in ("PLAN_AS_COMPLETED", "RESULT_AS_AVAILABLE"):
        return "相关检查或治疗尚处计划或结果待确认状态。"
    if error_type in ("FACT_CONTRADICTION", "PATIENT_FACT_CONTRADICTION"):
        return "该项患者事实存在冲突，需以原始资料复核结果为准。"
    if error_type == "CORRECTION_IGNORED":
        return "该项事实应以复核更正后的记录为准。"
    if error_type in ("TIME_MISMATCH", "TEMPORAL_CONFLICT"):
        return "该项时间信息需结合原始资料重新确认。"
    if error_type in ("NUMERIC_VALUE_ERROR", "UNIT_ERROR"):
        return "该项数值或单位需以原始资料为准。"
    return "该项陈述依据不足，需医生结合原始资料确认。"


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _record_schema(client, success: bool, error_type=None) -> None:
    recorder = getattr(client, "record_schema_result", None)
    if recorder is not None:
        recorder("reviser", success, error_type)
