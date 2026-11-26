"""Deterministic final-output selection for clinical generation.

This module is the only component allowed to choose a degraded clinical record.
Verifier/Reviser findings are inputs to recovery, never whole-record verdicts.
No model call is made here.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from .claim_extractor import extract_claims
from .claim_verifier import deterministic_claim_issues
from .errors import CaseError
from .local_reviser import apply_conservative_fallback
from .protocol import Case
from .shared_clinical_memory import extract_explicit_plan_excerpt
from .validation import validate_record


TEXT_SECTIONS = ("case_features", "diagnostic_discussion", "plan")
_CATEGORY_ORDER = (
    "complaint", "history_present", "physical_exam", "auxiliary_exam",
    "past_history", "demographics", "diagnosis", "plan", "other",
)
_POSITIVE_CLINICAL_MARKERS = (
    "升高", "降低", "增高", "减低", "水肿", "肿胀", "压痛", "疼痛",
    "乏力", "发热", "咳嗽", "胸痛", "呼吸困难", "出血", "狭窄", "血栓", "囊肿",
    "阳性", "待排", "疑似", "可能", "↑", "↓",
)
_MEASUREMENT_HINT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|%|"
    r"×10[⁹¹²³⁴⁵⁶⁷⁸\d]+/L)", re.IGNORECASE,
)
_DIAGNOSIS_PREFIX = re.compile(
    r"^(?:入院)?(?:初步诊断|初步判断)(?:为)? *(?:[：:]|\S)|^诊断 *[：:]"
)


def guarantee_writer_output(case: Case, record: dict, unresolved_issues: list[dict],
                            *, verification_completed: bool,
                            revision_applied: bool,
                            internal_ids: set[str] | None = None) -> tuple[dict, dict]:
    """Return the safest usable Writer-derived record or a Sources fallback."""
    writer_hash = _record_hash(record)
    candidate, sanitization = sanitize_writer_record(
        case, record, unresolved_issues, internal_ids=internal_ids,
    )
    if sanitization["sources_fallback_used"]:
        level = 4
        reason = sanitization["fallback_reason"]
    elif sanitization["applied"]:
        level = 2 if verification_completed else 3
        reason = "localized_claim_sanitization"
    elif verification_completed:
        level = 1
        reason = "enhanced_record_complete"
    else:
        level = 3
        reason = "writer_preserved_after_incomplete_verification"
    audit = {
        "level": level,
        "level_name": {
            1: "complete_enhanced_output",
            2: "safe_local_revision_output",
            3: "conservatively_sanitized_writer_output",
            4: "deterministic_sources_fallback",
        }[level],
        "reason": reason,
        "writer_record_sha256": writer_hash,
        "writer_available": True,
        "safe_record_available": True,
        "verification_completed": verification_completed,
        "revision_applied": revision_applied,
        **sanitization,
    }
    return candidate, audit


def recover_from_stage_failure(case: Case, stage: str, error: CaseError) -> tuple[dict, dict]:
    """Recover a valid record when Planner/Writer or another generation stage fails."""
    record, fallback = build_sources_fallback(case)
    return record, {
        "level": 4,
        "level_name": "deterministic_sources_fallback",
        "reason": f"stage_failure:{stage}:{error.code}",
        "writer_record_sha256": None,
        "writer_available": False,
        "safe_record_available": True,
        "verification_completed": False,
        "revision_applied": False,
        "applied": True,
        "sources_fallback_used": True,
        "fallback_reason": f"stage_failure:{stage}:{error.code}",
        "sanitized_claim_ids": [],
        "removed_claim_ids": [],
        "unresolved_claim_ids": [],
        "manual_review_required": True,
        "fallback": fallback,
    }


def sanitize_writer_record(case: Case, record: dict, issues: list[dict],
                           *, internal_ids: set[str] | None = None) -> tuple[dict, dict]:
    """Isolate exact unsafe Claims, preserving unrelated Writer content.

    Known deterministic rewrites are attempted first. Remaining high-risk Claims
    are removed only at complete sentence/list-item boundaries. A precisely
    scoped field can then be rebuilt from Sources; unclear scope or failed
    revalidation still replaces the whole record with the Level 4 fallback.
    """
    if not isinstance(record, dict):
        fallback, meta = build_sources_fallback(case)
        return fallback, _fallback_audit("writer_not_object", meta)
    candidate = deepcopy(record)
    high = _coalesce_high_issues(issues)
    sanitized_ids: list[str] = []
    removed_ids: list[str] = []
    initially_rebuilt_fields: set[str] = set()
    fallback_meta = None
    field_recovery = _empty_field_recovery_audit(high)

    if high:
        candidate, deterministic_meta = apply_conservative_fallback(candidate, high)
        sanitized_ids.extend(deterministic_meta.get("revised_claim_ids", []))
        unresolved = set(deterministic_meta.get("unresolved_claim_ids", []))
        remaining = [item for item in high if item.get("claim_id") in unresolved]
    else:
        deterministic_meta = {"applied": False, "revised_claim_ids": [],
                              "unresolved_claim_ids": []}
        remaining = []

    # Sequential sentence deletion is appropriate for one isolated assertion,
    # but deleting several unsafe assertions from one prose field can leave an
    # unhelpful fragment (the CASE01 failure mode).  When two or more exact
    # high-risk Claims affect the same prose field, rebuild that field once
    # from Sources and preserve every unrelated field.
    grouped_remaining: dict[str, list[dict]] = {}
    for issue in remaining:
        section = issue.get("section")
        if section in TEXT_SECTIONS:
            grouped_remaining.setdefault(str(section), []).append(issue)
    pre_rebuilt_claim_ids: set[str] = set()
    for section, field_issues in grouped_remaining.items():
        if len(field_issues) < 2:
            continue
        if any(_locate_current_issue(candidate, issue) is None for issue in field_issues):
            fallback, fallback_meta = build_sources_fallback(case)
            return fallback, _fallback_audit(
                f"unsafe_claim_not_safely_locatable:{section}", fallback_meta,
                sanitized_ids=sanitized_ids, removed_ids=removed_ids,
                field_recovery=field_recovery,
            )
        if fallback_meta is None:
            fallback, fallback_meta = build_sources_fallback(case)
        candidate[section] = deepcopy(fallback[section])
        ids = {str(issue.get("claim_id")) for issue in field_issues}
        pre_rebuilt_claim_ids.update(ids)
        sanitized_ids.extend(ids)
        initially_rebuilt_fields.add(section)
        field_recovery["fragmentation_prevented_fields"].append(section)

    for issue in remaining:
        if str(issue.get("claim_id")) in pre_rebuilt_claim_ids:
            continue
        outcome = _isolate_issue(candidate, issue, case)
        if outcome == "removed":
            removed_ids.append(str(issue.get("claim_id")))
        elif outcome == "replaced":
            sanitized_ids.append(str(issue.get("claim_id")))
            initially_rebuilt_fields.add(str(issue.get("section")))
        elif outcome == "already_absent":
            continue
        else:
            fallback, fallback_meta = build_sources_fallback(case)
            return fallback, _fallback_audit(
                f"unsafe_claim_not_safely_locatable:{issue.get('claim_id')}", fallback_meta,
                sanitized_ids=sanitized_ids, removed_ids=removed_ids,
                field_recovery=field_recovery,
            )

    if initially_rebuilt_fields & set(TEXT_SECTIONS):
        if fallback_meta is None:
            fallback, fallback_meta = build_sources_fallback(case)
        candidate["evidence"] = deepcopy(fallback["evidence"])
        field_recovery.update({
            "attempted": True,
            "succeeded": True,
            "rebuilt_fields": sorted(initially_rebuilt_fields | {"evidence"}),
            "evidence_rebuilt_for_fields": sorted(initially_rebuilt_fields),
            "sanitized_claim_ids": sorted(set(sanitized_ids)),
            "sources_fallback_field_metadata": fallback_meta,
        })

    relocated_review_hashes = _remove_relocated_plan_review_items(candidate, case)
    if relocated_review_hashes:
        field_recovery["attempted"] = True
        field_recovery["succeeded"] = True
        field_recovery["localized_fields"] = sorted(set(
            field_recovery["localized_fields"] + ["review_items"]
        ))
        field_recovery["removed_source_backed_plan_review_item_sha256"].extend(
            relocated_review_hashes
        )

    final_high = [item for item in deterministic_claim_issues(
        case, extract_claims(candidate), internal_ids or set(),
    ) if item.get("severity") == "high"]
    structural_errors = [item for item in validate_record(candidate, case, "lite")
                         if item.get("level") == "error"]
    if final_high or structural_errors:
        previous_recovery = field_recovery
        candidate, recovery = _recover_affected_fields(
            case, candidate, final_high, structural_errors,
            internal_ids=internal_ids or set(),
        )
        field_recovery = _merge_field_recovery_audits(previous_recovery, recovery)
        sanitized_ids.extend(recovery["sanitized_claim_ids"])
        removed_ids.extend(recovery["removed_claim_ids"])
        if not recovery["succeeded"]:
            fallback, fallback_meta = build_sources_fallback(case)
            reason = recovery["failure_reason"] or (
                "post_sanitization_high_risk" if final_high
                else "post_sanitization_schema_or_fact_error"
            )
            return fallback, _fallback_audit(
                reason, fallback_meta, sanitized_ids=sanitized_ids,
                removed_ids=removed_ids, field_recovery=field_recovery,
            )

    return candidate, {
        "applied": bool(
            sanitized_ids or removed_ids or field_recovery["rebuilt_fields"]
            or field_recovery["removed_source_backed_plan_review_item_sha256"]
        ),
        "safe_record_available": True,
        "sources_fallback_used": False,
        "fallback_reason": None,
        "sanitized_claim_ids": sorted(set(sanitized_ids)),
        "removed_claim_ids": sorted(set(removed_ids)),
        "unresolved_claim_ids": [],
        "manual_review_required": bool(high or field_recovery["attempted"]),
        "deterministic_patch": deterministic_meta,
        "input_high_risk_audit": [_risk_audit(item) for item in high],
        "field_recovery": field_recovery,
        "fallback": fallback_meta,
    }


def _recover_affected_fields(case: Case, record: dict, high_issues: list[dict],
                             structural_errors: list[dict],
                             *, internal_ids: set[str]) -> tuple[dict, dict]:
    """Recover only fields whose unsafe scope is deterministic and isolated."""
    candidate = deepcopy(record)
    audit = _empty_field_recovery_audit(high_issues)
    audit["attempted"] = True
    audit["original_validation_error_audit"] = [
        {"code": item.get("code"), "field": _validation_error_field(item),
         "message_sha256": hashlib.sha256(
             str(item.get("message", "")).encode()
         ).hexdigest()}
        for item in structural_errors
    ]
    fallback = None
    fallback_meta = None
    fields_to_rebuild: set[str] = set()

    # These issues were produced from the current candidate. Both their field
    # and exact Claim text must still resolve before any mutation is accepted.
    for issue in high_issues:
        section = issue.get("section")
        if section not in (*TEXT_SECTIONS, "evidence", "review_items"):
            audit["failure_reason"] = "field_recovery_unknown_risk_scope"
            return record, audit
        if _locate_current_issue(candidate, issue) is None:
            audit["failure_reason"] = "field_recovery_ambiguous_risk_scope"
            return record, audit

    # A review-item deletion never suppresses body scanning: all prose Claims
    # are scanned again below before recovery can succeed.
    for issue in high_issues:
        section = str(issue.get("section"))
        outcome = _isolate_issue(candidate, issue, case)
        claim_id = str(issue.get("claim_id"))
        if outcome == "removed":
            audit["removed_claim_ids"].append(claim_id)
            audit["localized_fields"].append(section)
        elif outcome == "replaced":
            audit["sanitized_claim_ids"].append(claim_id)
            audit["rebuilt_fields"].append(section)
        elif outcome != "already_absent":
            fields_to_rebuild.add(section)

    for issue in structural_errors:
        section = _validation_error_field(issue)
        if section is None:
            audit["failure_reason"] = "field_recovery_unscoped_validation_error"
            return record, audit
        fields_to_rebuild.add(section)

    if fields_to_rebuild:
        fallback, fallback_meta = build_sources_fallback(case)
        for section in sorted(fields_to_rebuild):
            candidate[section] = deepcopy(fallback[section])
            audit["rebuilt_fields"].append(section)

    # Rebuilding prose invalidates prior implicit Evidence associations. Use
    # validated Sources-only Evidence instead of guessing which items still fit.
    rebuilt = set(audit["rebuilt_fields"])
    if rebuilt & set(TEXT_SECTIONS) and "evidence" not in rebuilt:
        if fallback is None:
            fallback, fallback_meta = build_sources_fallback(case)
        candidate["evidence"] = deepcopy(fallback["evidence"])
        audit["rebuilt_fields"].append("evidence")
        audit["evidence_rebuilt_for_fields"] = sorted(rebuilt & set(TEXT_SECTIONS))

    residual_high = [item for item in deterministic_claim_issues(
        case, extract_claims(candidate), internal_ids,
    ) if item.get("severity") == "high"]
    residual_errors = [item for item in validate_record(candidate, case, "lite")
                       if item.get("level") == "error"]
    audit["residual_high_risk_audit"] = [_risk_audit(item) for item in residual_high]
    audit["residual_validation_codes"] = [item.get("code") for item in residual_errors]
    audit["localized_fields"] = sorted(set(audit["localized_fields"]))
    audit["rebuilt_fields"] = sorted(set(audit["rebuilt_fields"]))
    audit["sanitized_claim_ids"] = sorted(set(audit["sanitized_claim_ids"]))
    audit["removed_claim_ids"] = sorted(set(audit["removed_claim_ids"]))
    audit["sources_fallback_field_metadata"] = fallback_meta
    if residual_high or residual_errors:
        audit["failure_reason"] = (
            "field_recovery_revalidation_high_risk" if residual_high
            else "field_recovery_revalidation_failed"
        )
        return record, audit
    audit["succeeded"] = True
    return candidate, audit


def _locate_current_issue(record: dict, issue: dict) -> dict | None:
    statement = issue.get("statement")
    section = issue.get("section")
    if not isinstance(statement, str) or not statement.strip():
        return None
    claims = extract_claims(record)
    by_id = [claim for claim in claims
             if claim.get("claim_id") == issue.get("claim_id")
             and claim.get("section") == section
             and claim.get("statement") == statement]
    if len(by_id) == 1:
        return by_id[0]
    exact = [claim for claim in claims
             if claim.get("section") == section and claim.get("statement") == statement]
    return exact[0] if len(exact) == 1 else None


def _validation_error_field(issue: dict) -> str | None:
    field = issue.get("field")
    if field in (*TEXT_SECTIONS, "evidence", "review_items"):
        return field
    message = issue.get("message", "")
    if not isinstance(message, str):
        return None
    if "evidence[" in message or issue.get("code") in {
            "EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND", "DATE_SUPPORT",
            "NUMERIC_SUPPORT", "PLAN_AS_DONE"}:
        return "evidence"
    if "review_items[" in message or issue.get("code") in {
            "NON_ACTIONABLE_REVIEW_ITEM", "TECHNICAL_REVIEW_ITEM"}:
        return "review_items"
    matched = [field for field in TEXT_SECTIONS if field in message]
    return matched[0] if len(matched) == 1 else None


def _empty_field_recovery_audit(issues: list[dict]) -> dict:
    return {
        "attempted": False,
        "succeeded": False,
        "localized_fields": [],
        "rebuilt_fields": [],
        "evidence_rebuilt_for_fields": [],
        "sanitized_claim_ids": [],
        "removed_claim_ids": [],
        "original_high_risk_audit": [_risk_audit(item) for item in issues],
        "original_validation_error_audit": [],
        "residual_high_risk_audit": [],
        "residual_validation_codes": [],
        "failure_reason": None,
        "sources_fallback_field_metadata": None,
        "fragmentation_prevented_fields": [],
        "removed_source_backed_plan_review_item_sha256": [],
    }


def _merge_field_recovery_audits(first: dict, second: dict) -> dict:
    result = deepcopy(second)
    for key in ("localized_fields", "rebuilt_fields", "evidence_rebuilt_for_fields",
                "sanitized_claim_ids", "removed_claim_ids",
                "fragmentation_prevented_fields",
                "removed_source_backed_plan_review_item_sha256"):
        result[key] = sorted(set(first.get(key, [])) | set(second.get(key, [])))
    original = list(first.get("original_high_risk_audit", []))
    for item in second.get("original_high_risk_audit", []):
        if item not in original:
            original.append(item)
    result["original_high_risk_audit"] = original
    validation = list(first.get("original_validation_error_audit", []))
    for item in second.get("original_validation_error_audit", []):
        if item not in validation:
            validation.append(item)
    result["original_validation_error_audit"] = validation
    result["attempted"] = bool(first.get("attempted") or second.get("attempted"))
    if result.get("sources_fallback_field_metadata") is None:
        result["sources_fallback_field_metadata"] = first.get(
            "sources_fallback_field_metadata"
        )
    return result


def _risk_audit(issue: dict) -> dict:
    statement = issue.get("statement")
    return {
        "claim_id": issue.get("claim_id"),
        "section": issue.get("section"),
        "code": issue.get("error_type") or issue.get("code"),
        "severity": issue.get("severity"),
        "source_ids": list(issue.get("source_ids", [])),
        "statement_sha256": (hashlib.sha256(statement.encode()).hexdigest()
                             if isinstance(statement, str) else None),
    }


def build_sources_fallback(case: Case) -> tuple[dict, dict]:
    """Build a clinically partitioned, Sources-only conservative record."""
    usable = [source for source in case.sources
              if isinstance(source.content, str) and source.content.strip()]
    if not usable:
        raise CaseError("OUTPUT_GUARANTEE_NO_RELIABLE_FACTS",
                        "输入中没有可用于构造保守病历草稿的非空原始资料。")

    buckets = {name: [] for name in _CATEGORY_ORDER}
    for source in usable:
        buckets[_source_category(source)].append(source)
    for category in ("physical_exam", "auxiliary_exam", "other"):
        buckets[category] = sorted(
            buckets[category], key=_clinical_source_score, reverse=True,
        )

    complaint = _take(buckets["complaint"], 1)
    present = _take(buckets["history_present"], 1)
    physical = _take(buckets["physical_exam"], 2)
    auxiliary = _take(buckets["auxiliary_exam"], 2)
    past = _take(buckets["past_history"], 1)
    demographics = _select_safe_demographics(buckets["demographics"])
    diagnosis_selected = _take(buckets["diagnosis"], 3)
    plan_selected = _take(buckets["plan"], 3)
    if not any((complaint, present, physical, auxiliary, past, demographics)):
        present = _take(buckets["other"], 2) or _take(usable, 2)

    feature_sections = []
    _append_feature_section(feature_sections, "主诉", complaint, max_parts=2, max_chars=320)
    _append_feature_section(feature_sections, "病程要点", present, max_parts=2, max_chars=520)
    _append_feature_section(feature_sections, "主要查体", physical, max_parts=2, max_chars=420)
    _append_feature_section(feature_sections, "重要检查", auxiliary, max_parts=5, max_chars=760)
    _append_feature_section(feature_sections, "既往史", past, max_parts=2, max_chars=360)
    _append_feature_section(feature_sections, "一般情况", demographics, max_parts=1, max_chars=120)
    case_features = "；".join(feature_sections).rstrip("。；; ") + "。"

    main_problem_sources = complaint or physical or present or auxiliary
    discussion_parts = []
    if main_problem_sources:
        discussion_parts.append(
            "当前主要临床问题："
            + _join_snippets(main_problem_sources, max_parts=2, max_chars=420)
        )
    course_basis = _join_snippets(present, max_parts=1, max_chars=360)
    exam_basis = _join_snippets(physical[:1], max_parts=2, max_chars=420)
    test_basis = _join_snippets(auxiliary[:1], max_parts=5, max_chars=700)
    if course_basis or exam_basis:
        discussion_parts.append(
            "病程与查体依据："
            + "；".join(value for value in (course_basis, exam_basis) if value)
        )
    if test_basis:
        discussion_parts.append("检查依据：" + test_basis)
    if diagnosis_selected:
        discussion_parts.append(
            "原始资料明确记载的初步诊断或鉴别方向："
            + _join_snippets(diagnosis_selected, max_parts=2, max_chars=650)
        )
    else:
        discussion_parts.append(
            "原始资料未明确记载初步诊断或鉴别方向，"
            "现阶段不宜增加确定性诊断。"
        )
    discussion_parts.append(
        "上述诊断的确定程度、诊断依据与尚缺信息仍需临床医师结合完整资料确认。"
    )
    diagnostic_discussion = "。".join(
        part.rstrip("。；; ") for part in discussion_parts if part
    ) + "。"

    if plan_selected:
        plan = (
            "原始资料明确记载的检查、监测、会诊或治疗安排包括："
            + _join_snippets(plan_selected, max_parts=3, max_chars=750)
            + "。具体执行需由临床医师结合当前病情确认。"
        )
    else:
        problem_text = _join_snippets(
            diagnosis_selected[:2] or main_problem_sources, max_parts=1, max_chars=360,
        )
        plan = "原始资料未明确记载后续检查、监测、会诊或治疗安排。"
        if problem_text:
            plan += f"需由临床医师围绕已记录的临床问题（{problem_text}）补充并确认相应诊疗计划。"
        else:
            plan += "需由临床医师结合原始资料补充并确认诊疗计划。"

    evidence_sources = _select_evidence_sources(
        complaint, present, physical, auxiliary, diagnosis_selected, plan_selected,
        past, demographics,
    )
    evidence = [{
        "statement": _source_snippet(source, max_parts=(5 if _source_category(source)
                                                         == "auxiliary_exam" else 2),
                                     max_chars=(760 if _source_category(source)
                                                == "auxiliary_exam" else 420)),
        "source_ids": [source.source_id],
        "kind": "fact",
    } for source in evidence_sources]
    review_items = []
    if not diagnosis_selected:
        review_items.append("需补充或确认当前初步诊断、诊断依据及其确定程度。")
    if not plan_selected:
        target = _join_snippets(
            diagnosis_selected[:2] or main_problem_sources, max_parts=1, max_chars=220,
        )
        review_items.append(
            (f"需补充并确认与{target}相对应的后续诊疗安排。"
             if target else "需结合当前临床问题补充并确认后续诊疗安排。")
        )
    record = {
        "case_features": case_features,
        "diagnostic_discussion": diagnostic_discussion,
        "plan": plan,
        "evidence": evidence,
        "review_items": review_items,
    }
    errors = [item for item in validate_record(record, case, "lite")
              if item.get("level") == "error"]
    if errors:
        raise CaseError(
            "OUTPUT_GUARANTEE_FALLBACK_INVALID",
            "原始资料保守草稿仍未通过必要结构与事实检查："
            + "; ".join(item.get("message", "") for item in errors),
        )
    return record, {
        "source_count": len(usable),
        "feature_source_ids": [item.source_id for item in _unique_sources(
            complaint + present + physical + auxiliary + past + demographics)],
        "diagnosis_source_ids": [item.source_id for item in diagnosis_selected],
        "plan_source_ids": [item.source_id for item in plan_selected],
        "evidence_source_ids": [item.source_id for item in evidence_sources],
        "source_categories": {
            category: [item.source_id for item in values]
            for category, values in buckets.items() if values
        },
        "contains_model_generated_patient_facts": False,
    }


def _isolate_issue(record: dict, issue: dict, case: Case) -> str:
    section = issue.get("section")
    statement = issue.get("statement")
    if section not in (*TEXT_SECTIONS, "evidence", "review_items"):
        return "unsafe_scope"
    if not isinstance(statement, str) or not statement.strip():
        return "unsafe_scope"
    claims = [claim for claim in extract_claims(record)
              if claim.get("section") == section
              and claim.get("statement") == statement]
    if not claims:
        # A preceding successful revision may already have removed the exact
        # unsafe assertion. Never search by partial text or guess a replacement.
        return "already_absent"
    if len(claims) != 1:
        return "unsafe_scope"
    claim = claims[0]
    if section in ("evidence", "review_items"):
        values = record.get(section)
        index = claim.get("item_index")
        if not isinstance(values, list) or not isinstance(index, int) or not 0 <= index < len(values):
            return "unsafe_scope"
        del values[index]
        return "removed"

    section_claims = [item for item in extract_claims(record) if item["section"] == section]
    if len(section_claims) > 1:
        start, end = claim.get("span_start"), claim.get("span_end")
        text = record.get(section)
        if (not isinstance(start, int) or not isinstance(end, int)
                or not isinstance(text, str) or text[start:end] != statement):
            return "unsafe_scope"
        record[section] = (text[:start] + text[end:]).strip()
        return "removed" if record[section] else "unsafe_scope"

    fallback, _ = build_sources_fallback(case)
    record[section] = fallback[section]
    return "replaced"


def _coalesce_high_issues(issues: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("severity") != "high":
            continue
        key = (issue.get("claim_id"), issue.get("section"),
               issue.get("error_type") or issue.get("code"))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _fallback_audit(reason: str, fallback: dict,
                    *, sanitized_ids: list[str] | None = None,
                    removed_ids: list[str] | None = None,
                    field_recovery: dict | None = None) -> dict:
    return {
        "applied": True,
        "safe_record_available": True,
        "sources_fallback_used": True,
        "fallback_reason": reason,
        "sanitized_claim_ids": sorted(set(sanitized_ids or [])),
        "removed_claim_ids": sorted(set(removed_ids or [])),
        "unresolved_claim_ids": [],
        "manual_review_required": True,
        "deterministic_patch": None,
        "field_recovery": field_recovery or _empty_field_recovery_audit([]),
        "fallback": fallback,
    }


def _source_category(source) -> str:
    path = source.field_path or ""
    text = _clean_source_text(source.content)
    if "初步诊断" in path or path.endswith("诊断"):
        return "diagnosis"
    if _source_has_explicit_plan(path, text):
        return "plan"
    if "主诉" in path:
        return "complaint"
    if "现病史" in path:
        return "history_present"
    if "体格检查" in path or "专科检查" in path:
        return "physical_exam"
    if any(marker in path for marker in ("辅助检查", "检验", "影像")):
        return "auxiliary_exam"
    if "既往史" in path:
        return "past_history"
    if "一般项目" in path:
        return "demographics"
    if not path:
        if _DIAGNOSIS_PREFIX.search(text):
            return "diagnosis"
        if text.startswith("主诉：") or text.startswith("主诉:"):
            return "complaint"
    return "other"


def _source_has_explicit_plan(path: str, text: str) -> bool:
    return extract_explicit_plan_excerpt(text, field_path=path) is not None


def _remove_relocated_plan_review_items(record: dict, case: Case) -> list[str]:
    """Remove only exact Sources-backed plan duplicates already present in plan.

    Partial matches and genuine result/conflict reminders remain untouched.
    Deterministic high-risk scanning still runs after this normalization.
    """
    plan = record.get("plan")
    review_items = record.get("review_items")
    if not isinstance(plan, str) or not isinstance(review_items, list):
        return []
    plan_normalized = _normalize_plan_duplicate(plan)
    plan_source_texts = {
        _normalize_plan_duplicate(excerpt)
        for source in case.sources
        if (excerpt := extract_explicit_plan_excerpt(
            source.content, field_path=source.field_path,
        )) is not None
    }
    kept = []
    removed_hashes = []
    for item in review_items:
        normalized = _normalize_plan_duplicate(item) if isinstance(item, str) else ""
        if (normalized and normalized in plan_source_texts
                and normalized in plan_normalized):
            removed_hashes.append(hashlib.sha256(item.encode()).hexdigest())
        else:
            kept.append(item)
    record["review_items"] = kept
    return sorted(set(removed_hashes))


def _normalize_plan_duplicate(text: str) -> str:
    return re.sub(r"[\s。；;]+", "", text).strip()


def _clinical_source_score(source) -> tuple[int, int]:
    text = _clean_source_text(source.content)
    score = 3 * len(_MEASUREMENT_HINT.findall(text))
    score += 12 * sum(marker in text for marker in _POSITIVE_CLINICAL_MARKERS)
    score += min(len(text), 400) // 80
    return score, -len(text)


def _take(sources: list, limit: int) -> list:
    return list(sources[:limit])


def _select_safe_demographics(sources: list) -> list:
    preferred = [source for source in sources
                 if any((source.field_path or "").endswith(marker)
                        for marker in ("性别", "年龄", "职业"))]
    return preferred[:3]


def _append_feature_section(parts: list[str], label: str, sources: list,
                            *, max_parts: int, max_chars: int) -> None:
    if sources:
        parts.append(f"{label}：{_join_snippets(sources, max_parts=max_parts, max_chars=max_chars)}")


def _select_evidence_sources(complaint: list, present: list, physical: list,
                             auxiliary: list, diagnosis: list, plan: list,
                             past: list, demographics: list) -> list:
    # One representative per clinically meaningful category prevents the
    # five-item protocol limit from being consumed by demographics alone.
    groups = [complaint, physical, auxiliary, diagnosis, plan]
    if not plan:
        groups = [complaint, present, physical, auxiliary, diagnosis]
    selected = [group[0] for group in groups if group]
    for group in (past, demographics, present, auxiliary, physical, diagnosis, plan):
        if len(selected) >= 5:
            break
        if group and group[0].source_id not in {item.source_id for item in selected}:
            selected.append(group[0])
    return _unique_sources(selected)[:5]


def _unique_sources(sources: list) -> list:
    result = []
    seen = set()
    for source in sources:
        if source.source_id not in seen:
            seen.add(source.source_id)
            result.append(source)
    return result


def _clean_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _source_snippet(source, *, max_parts: int, max_chars: int) -> str:
    """Return complete source fragments only; never truncate inside a fragment."""
    text = _clean_source_text(source.content)
    if _source_category(source) == "plan":
        text = extract_explicit_plan_excerpt(
            text, field_path=source.field_path,
        ) or ""
    fragments = [item.strip() for item in re.findall(r"[^。！？!?；;]+(?:[。！？!?；;]|$)", text)
                 if item.strip()]
    if not fragments:
        return text
    selected = []
    used = 0
    for fragment in fragments:
        if selected and (len(selected) >= max_parts or used + len(fragment) > max_chars):
            break
        selected.append(fragment)
        used += len(fragment)
        if len(selected) >= max_parts:
            break
    return "".join(selected).strip() or text


def _join_snippets(sources: list, *, max_parts: int, max_chars: int) -> str:
    remaining = max_chars
    values = []
    for source in sources:
        if remaining <= 0:
            break
        snippet = _source_snippet(source, max_parts=max_parts, max_chars=remaining)
        if not snippet:
            continue
        values.append(snippet.rstrip("。；; "))
        remaining -= len(snippet)
    return "；".join(values)


def _join_source_texts(sources: list) -> str:
    return "；".join(
        _clean_source_text(source.content).rstrip("。；; ") for source in sources
    )


def _record_hash(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
