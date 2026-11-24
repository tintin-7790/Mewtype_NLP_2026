import json
import time
from pathlib import Path

from .errors import CaseError
from .claim_extractor import extract_claims
from .claim_verifier import (deterministic_claim_issues, has_blocking_issues, issues_for_review,
                             issues_for_revision, run_claim_verifier)
from .clinical_planner import number_facts, retrieve_problem_evidence, run_planner
from .evidence_policy import enforce_evidence_policy
from .facts import organize_facts
from .local_reviser import apply_conservative_fallback, run_local_revision
from .model_client import Budget, MEDICAL_SCHEMA, parse_model_json
from .note_writer import run_note_writer
from .protocol import Case
from .retrieval import DenseEvidenceRetriever
from .validation import validate_record


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "2026-11-23.baseline-v3"
CLINICAL_PROMPT_VERSION = "2026-11-24.clinical-v4-contrastive"


def run_case(case: Case, client, pipeline: str, response_format: str, max_calls: int,
             retriever: DenseEvidenceRetriever | None = None, retrieval_top_k: int = 3,
             clinical_verify: bool = True, clinical_revise: bool = True,
             clinical_evidence_limit: int = 5,
             clinical_stage_formats: dict[str, str] | None = None,
             clinical_prompt_profile: str = "contrastive") -> tuple[dict, dict]:
    started = time.perf_counter(); budget = Budget(max_calls); checks = []
    final_evidence_limit = min(clinical_evidence_limit, 5) if pipeline == "clinical_pipeline" else 5
    raw = [{"source_id": s.source_id, "field_path": s.field_path, "content": s.content} for s in case.sources]
    if pipeline == "baseline":
        messages = _messages("baseline_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw})
        record = parse_model_json(client.call(messages, response_format, budget), stage="model", client=client)
    elif pipeline == "evidence_pipeline":
        deterministic = organize_facts(case)
        fact_messages = _messages("fact_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                       "deterministic_hints": deterministic})
        model_facts = parse_model_json(client.call(fact_messages, response_format, budget))
        if not isinstance(model_facts, dict) or not isinstance(model_facts.get("facts"), list):
            raise CaseError("INVALID_FACT_OUTPUT", "事实整理结果必须包含 facts 数组。")
        gen_messages = _messages("evidence_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                          "facts": model_facts["facts"]})
        record = parse_model_json(client.call(gen_messages, response_format, budget))
        pre_issues = validate_record(record, case); checks.extend(pre_issues)
        retrieval_context = _retrieval_context(record, retriever, retrieval_top_k)
        verify_messages = _messages("verify_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                            "facts": model_facts["facts"], "candidate": record,
                                                            "rule_issues": pre_issues,
                                                            "dense_retrieval_candidates": retrieval_context,
                                                            "retrieval_notice": (
                                                                "相似度仅用于定位候选原文，不代表事实正确或来源已验证；"
                                                                "必须结合原文、数值、否定、时间和状态核验。"
                                                                if retriever else "未启用 Dense 检索。")})
        verdict = parse_model_json(client.call(verify_messages, response_format, budget))
        if not isinstance(verdict, dict) or not isinstance(verdict.get("valid"), bool):
            raise CaseError("INVALID_VERIFIER_OUTPUT", "核验结果必须包含布尔值 valid。")
        if not verdict["valid"]:
            corrected = verdict.get("corrected_record")
            if not isinstance(corrected, dict):
                raise CaseError("VERIFICATION_FAILED", "模型核验未通过且未提供可校验的保守修正。")
            record = corrected
        _preserve_retrieval_review_items(record, retriever, retrieval_top_k)
    elif pipeline == "clinical_pipeline":
        record, clinical_meta, clinical_checks = _run_clinical(
            case, client, response_format, budget, retriever, retrieval_top_k,
            clinical_verify, clinical_revise, clinical_evidence_limit, clinical_stage_formats,
            clinical_prompt_profile,
        )
        checks.extend(clinical_checks)
    else:
        raise CaseError("PIPELINE_INVALID", f"未知 pipeline：{pipeline}")
    record, evidence_policy = enforce_evidence_policy(record, case, final_evidence_limit)
    final_issues = validate_record(record, case); checks.extend(final_issues)
    errors = [x for x in final_issues if x["level"] == "error"]
    if errors:
        raise CaseError("VALIDATION_FAILED", "; ".join(x["message"] for x in errors))
    prompt_version = CLINICAL_PROMPT_VERSION if pipeline == "clinical_pipeline" else PROMPT_VERSION
    meta = {"calls": budget.calls, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "checks": checks, "prompt_version": prompt_version, "evidence_policy": evidence_policy}
    if retriever:
        meta["retrieval"] = retriever.stats()
    if pipeline == "clinical_pipeline":
        meta["clinical"] = clinical_meta
    return record, meta


def _run_clinical(case: Case, client, response_format: str, budget: Budget,
                  retriever: DenseEvidenceRetriever | None, retrieval_top_k: int,
                  clinical_verify: bool, clinical_revise: bool,
                  clinical_evidence_limit: int,
                  stage_formats: dict[str, str] | None,
                  prompt_profile: str) -> tuple[dict, dict, list[dict]]:
    formats = {stage: response_format for stage in ("planner", "writer", "verifier", "reviser")}
    if stage_formats:
        formats.update(stage_formats)
    facts = number_facts(organize_facts(case))
    prompt_audit = {}
    algorithm_mode = prompt_profile
    planner_fallback = None
    clinical_plan = None; record = None; evidence_bundles = []
    try:
        clinical_plan = run_planner(case, facts, client, formats["planner"], budget,
                                    prompt_profile, prompt_audit)
    except CaseError as primary_error:
        fallback_eligible = {"INVALID_CLINICAL_PLAN", "MODEL_INVALID_JSON"}
        if (prompt_profile not in ("contrastive", "contrastive_fewshot")
                or primary_error.code not in fallback_eligible):
            raise
        planner_fallback = {"primary_error": primary_error.code,
                            "primary_message": primary_error.message, "attempts": []}
        # A legacy planner consumes one call and still needs one Writer call.
        if budget.remaining >= 2:
            legacy_audit = {}
            try:
                clinical_plan = run_planner(case, facts, client, formats["planner"], budget,
                                            "legacy", legacy_audit)
                prompt_audit["legacy_fallback_planner"] = legacy_audit.get("planner", {})
                algorithm_mode = "legacy_planner_fallback"
                planner_fallback["attempts"].append({"strategy": "legacy_planner", "status": "success"})
            except CaseError as legacy_error:
                planner_fallback["attempts"].append({"strategy": "legacy_planner", "status": "failed",
                                                     "error": legacy_error.code})
        if clinical_plan is None:
            if budget.remaining < 1:
                raise CaseError(
                    "CLINICAL_PLANNER_FALLBACK_BUDGET",
                    f"Contrastive Planner 无有效结果，剩余调用预算不足以执行安全回退；原错误：{primary_error.code}。",
                ) from primary_error
            record = _run_baseline_fallback(case, client, formats["writer"], budget)
            clinical_plan = {"problems": []}
            algorithm_mode = "baseline_fallback"
            planner_fallback["attempts"].append({"strategy": "baseline", "status": "success"})
    if record is None:
        evidence_bundles = retrieve_problem_evidence(clinical_plan, retriever, retrieval_top_k, facts)
        record = run_note_writer(case, facts, clinical_plan, evidence_bundles,
                                 client, formats["writer"], budget, clinical_evidence_limit,
                                 prompt_profile if algorithm_mode == prompt_profile else "legacy", prompt_audit)
    record, writer_evidence_policy = enforce_evidence_policy(record, case, clinical_evidence_limit)
    checks = validate_record(record, case)
    writer_structure_codes = {"FORMAT", "EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND"}
    format_errors = [issue for issue in checks
                     if issue["level"] == "error" and issue["code"] in writer_structure_codes]
    if format_errors:
        raise CaseError("INVALID_WRITER_OUTPUT",
                        "writer 阶段结构/引用校验失败：" + "; ".join(issue["message"] for issue in format_errors))
    claims = extract_claims(record)
    verification_issue_count = 0
    verification_risk_counts = {"high": 0, "medium": 0, "low": 0}
    revision_meta = None
    verification_status = "not_run"
    verification_error = None
    unresolved_risk_count = 0
    if clinical_verify:
        verification = None
        try:
            verification = run_claim_verifier(case, claims, record, clinical_plan, retriever,
                                               retrieval_top_k, client, formats["verifier"], budget)
        except CaseError as exc:
            verification_error = {"code": exc.code, "message": exc.message}
            verification_status = "incomplete_writer_preserved"
            deterministic = deterministic_claim_issues(case, claims)
            high_deterministic = [issue for issue in deterministic if issue.get("severity") == "high"]
            if high_deterministic:
                record, revision_meta = apply_conservative_fallback(record, high_deterministic)
                unresolved_risk_count = len(high_deterministic)
                verification_status = "incomplete_with_conservative_fallback"
            _append_review(record, "自动核验未完成，病历内容需由医生结合原始资料复核。")
        if verification is not None:
            verification_issue_count = len(verification["issues"])
            verification_risk_counts = {
                severity: sum(issue.get("severity") == severity for issue in verification["issues"])
                for severity in ("high", "medium", "low")
            }
            verification_status = "passed" if verification_issue_count == 0 else "risks_reported"
        if verification is not None and has_blocking_issues(verification):
            high_issues = issues_for_revision(verification)
            revision_error = None
            if clinical_revise:
                try:
                    record, revision_meta = run_local_revision(
                        case, record, clinical_plan, high_issues, retriever,
                        retrieval_top_k, client, formats["reviser"], budget,
                    )
                    verification_status = "revised_requires_review"
                except CaseError as exc:
                    revision_error = {"code": exc.code, "message": exc.message}
            if not clinical_revise or revision_error is not None:
                record, revision_meta = apply_conservative_fallback(record, high_issues)
                revision_meta["model_revision_error"] = revision_error
                verification_status = "fallback_disclosed"
            remaining = deterministic_claim_issues(case, extract_claims(record))
            if any(issue["severity"] == "high" for issue in remaining):
                unresolved_risk_count = sum(issue["severity"] == "high" for issue in remaining)
                for issue in remaining:
                    if issue["severity"] == "high":
                        note = f"尚未自动消除的高风险事实问题：{issue['revision_hint']}"
                        _append_review(record, note)
            else:
                unresolved_risk_count = len(high_issues)
                for issue in high_issues:
                    _append_review(record, f"相关事实已作保守修正，仍需医生确认：{issue['revision_hint']}")
        if verification is not None:
            for issue in issues_for_review(verification):
                _append_review(record, f"需医生复核：{issue['statement']}（{issue['revision_hint']}）")
    elif clinical_revise:
        raise CaseError("CLINICAL_CONFIG_INVALID", "启用局部修订前必须启用原子事实核验。")
    record, post_revision_evidence_policy = enforce_evidence_policy(record, case, clinical_evidence_limit)
    meta = {
        "problem_count": len(clinical_plan["problems"]),
        "fact_count": len(facts),
        "evidence_bundle_count": len(evidence_bundles),
        "claim_count": len(claims),
        "verification_enabled": clinical_verify,
        "verification_issue_count": verification_issue_count,
        "verification_risk_counts": verification_risk_counts,
        "verification_status": verification_status,
        "verification_error": verification_error,
        "unresolved_risk_count": unresolved_risk_count,
        "revision_enabled": clinical_revise,
        "revision_applied": revision_meta is not None,
        "revision": revision_meta,
        "evidence_limit": clinical_evidence_limit,
        "stage_response_formats": formats,
        "prompt_profile": prompt_profile,
        "algorithm_mode": algorithm_mode,
        "planner_fallback": planner_fallback,
        "prompt_audit": prompt_audit,
        "evidence_policy": {"writer": writer_evidence_policy, "post_revision": post_revision_evidence_policy},
    }
    return record, meta, checks


def _run_baseline_fallback(case: Case, client, response_format: str, budget: Budget) -> dict:
    raw_sources = [{"source_id": source.source_id, "field_path": source.field_path,
                    "content": source.content} for source in case.sources]
    messages = _messages("baseline_system.txt", {"record_cutoff": case.record_cutoff,
                                                   "sources": raw_sources})
    raw = client.call(messages, response_format, budget, stage="baseline_fallback", schema=MEDICAL_SCHEMA)
    return parse_model_json(raw, stage="baseline_fallback", client=client)


def _append_review(record: dict, note: str) -> None:
    record.setdefault("review_items", [])
    if note not in record["review_items"]:
        record["review_items"].append(note)


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _retrieval_context(record: dict, retriever: DenseEvidenceRetriever | None, top_k: int) -> list[dict]:
    if retriever is None or not isinstance(record, dict):
        return []
    context = []
    for index, item in enumerate(record.get("evidence", [])):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        candidates = retriever.retrieve(item["statement"], top_k=top_k)
        context.append({
            "evidence_index": index,
            "statement": item["statement"],
            "declared_source_ids": item.get("source_ids", []),
            "candidates": [candidate.to_dict() for candidate in candidates],
        })
    return context


def _preserve_retrieval_review_items(record: dict, retriever: DenseEvidenceRetriever | None, top_k: int) -> None:
    """Record insufficient retrieval support without changing declared sources."""
    if retriever is None or not isinstance(record, dict) or not isinstance(record.get("review_items"), list):
        return
    for item in record.get("evidence", []):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        declared = set(item.get("source_ids", []))
        candidates = retriever.retrieve(item["statement"], top_k=top_k)
        supported = any(candidate.compatible and candidate.unit.source_id in declared for candidate in candidates)
        if not supported:
            note = f"该陈述尚未找到与所引原始资料一致的支持内容，需医生复核：{item['statement']}"
            if note not in record["review_items"]:
                record["review_items"].append(note)
