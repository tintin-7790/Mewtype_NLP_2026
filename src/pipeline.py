import hashlib
import json
import time
from pathlib import Path

from .agent_orchestrator import assess_agent_need, run_counter_evidence_agent
from .clinical_graph import build_clinical_graph
from .errors import CaseError, TaskError
from .claim_extractor import extract_claims
from .claim_verifier import (deterministic_claim_issues, has_blocking_issues,
                             is_technical_review_item, issues_for_review,
                             issues_for_revision, run_claim_verifier)
from .clinical_planner import number_facts, retrieve_problem_evidence, run_planner
from .evidence_policy import enforce_evidence_policy
from .facts import organize_facts
from .graph_retrieval import GraphGuidedRetriever
from .local_reviser import apply_conservative_fallback, run_local_revision
from .model_client import Budget, MEDICAL_SCHEMA, parse_model_json
from .note_writer import run_note_writer
from .output_guarantee import guarantee_writer_output, recover_from_stage_failure
from .protocol import Case
from .retrieval import DenseEvidenceRetriever, LexicalEvidenceRetriever
from .shared_clinical_memory import SharedClinicalMemory
from .validation import (classify_claim_issue_for_final, classify_validation_issue, recover_evidence,
                         recover_plan_numeric_targets, validate_record)


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "2026-11-23.baseline-v3"
CLINICAL_PROMPT_VERSION = "2026-11-25.clinical-v4.8-full-record"
COUNTER_PROMPT_VERSION = "2026-11-24.compact-counter-evidence-v3-retrieval-aware"


def run_case(case: Case, client, pipeline: str, response_format: str, max_calls: int,
             retriever: DenseEvidenceRetriever | None = None, retrieval_top_k: int = 3,
             clinical_verify: bool = True, clinical_revise: bool = True,
             clinical_evidence_limit: int = 5,
             clinical_stage_formats: dict[str, str] | None = None,
             clinical_prompt_profile: str = "contrastive",
             graph_retrieval_mode: str = "off",
             clinical_agent_mode: str = "off",
             clinical_writing_guidance: str = "off",
             shared_clinical_memory: bool = False,
             retrieval_strategy: str = "no_graph",
             retrieval_fallback_enabled: bool = False,
             final_validation_mode: str = "strict") -> tuple[dict, dict]:
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
        try:
            record, clinical_meta, clinical_checks = _run_clinical(
                case, client, response_format, budget, retriever, retrieval_top_k,
                clinical_verify, clinical_revise, clinical_evidence_limit, clinical_stage_formats,
                clinical_prompt_profile, graph_retrieval_mode, clinical_agent_mode,
                clinical_writing_guidance, shared_clinical_memory,
                retrieval_strategy, retrieval_fallback_enabled, final_validation_mode,
            )
        except CaseError as exc:
            failed_stage = (getattr(exc, "details", {}) or {}).get("failed_stage") or _failure_stage(exc)
            record, guarantee_meta = recover_from_stage_failure(case, failed_stage, exc)
            clinical_checks = []
            clinical_meta = {
                "recovered_stage_error": {"stage": failed_stage, "code": exc.code,
                                            "message": exc.message},
                "verification_enabled": clinical_verify,
                "verification_attempted": failed_stage not in {"planner", "writer"},
                "verification_completed": False,
                "verification_status": "not_completed_stage_failure",
                "verification_error": None,
                "revision_enabled": clinical_revise,
                "revision_attempted": False,
                "revision_applied": False,
                "unresolved_risk_count": 0,
                "unresolved_claim_ids": [],
                "unresolved_hard_risk_count": 0,
                "unresolved_hard_claim_ids": [],
                "unresolved_soft_risk_count": 0,
                "unresolved_soft_claim_ids": [],
                "output_guarantee": guarantee_meta,
                "graph_retrieval": {"graph_requested": graph_retrieval_mode != "off",
                                    "graph_executed": False,
                                    "graph_fallback_reason": f"stage_failure:{failed_stage}"},
                "adaptive_agent": {"mode": clinical_agent_mode, "triggered": False,
                                   "completed": False, "error": None},
            }
        checks.extend(clinical_checks)
    else:
        raise CaseError("PIPELINE_INVALID", f"未知 pipeline：{pipeline}")
    recovery_initial = validate_record(record, case, final_validation_mode)
    record, final_recovery = recover_evidence(record, case)
    record, plan_numeric_recovery = recover_plan_numeric_targets(record, case)
    record, evidence_policy = enforce_evidence_policy(record, case, final_evidence_limit)
    final_issues = validate_record(record, case, final_validation_mode)
    if pipeline == "clinical_pipeline" and any(item["level"] == "error" for item in final_issues):
        failure = CaseError(
            "OUTPUT_GUARANTEE_POST_VALIDATION_RECOVERY",
            "增强输出仍存在必要结构或事实错误，改用原始 Sources 保守恢复。",
            {"failed_stage": "final_validation"},
        )
        record, guarantee_meta = recover_from_stage_failure(case, "final_validation", failure)
        record, final_recovery = recover_evidence(record, case)
        record, plan_numeric_recovery = recover_plan_numeric_targets(record, case)
        record, evidence_policy = enforce_evidence_policy(record, case, final_evidence_limit)
        final_issues = validate_record(record, case, final_validation_mode)
        clinical_meta["output_guarantee"] = guarantee_meta
    checks.extend(final_issues)
    prompt_version = CLINICAL_PROMPT_VERSION if pipeline == "clinical_pipeline" else PROMPT_VERSION
    meta = {"calls": budget.calls, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "checks": checks, "prompt_version": prompt_version, "evidence_policy": evidence_policy,
            "final_validation_recovery": {
                **final_recovery,
                "plan_numeric_recovery": plan_numeric_recovery,
                "validation_initial": _validation_risk_summary(recovery_initial),
                "validation_final": _validation_risk_summary(final_issues),
                "mode": final_validation_mode,
                "safe_content_reconstruction": False,
            }}
    if retriever:
        meta["retrieval"] = retriever.stats()
    if pipeline == "clinical_pipeline":
        meta["clinical"] = clinical_meta
    errors = [x for x in final_issues if x["level"] == "error"]
    if pipeline == "clinical_pipeline":
        errors.extend(x for x in clinical_checks
                      if x.get("level") == "error"
                      and x.get("code") in {"INTERNAL_AUDIT_LEAK", "TECHNICAL_REVIEW_ITEM"})
    if errors:
        raise CaseError("VALIDATION_FAILED", "; ".join(x["message"] for x in errors),
                        {"partial_meta": meta, "failed_stage": "final_validation",
                         "validation_errors": errors})
    return record, meta


def _run_clinical(case: Case, client, response_format: str, budget: Budget,
                  retriever: DenseEvidenceRetriever | None, retrieval_top_k: int,
                  clinical_verify: bool, clinical_revise: bool,
                  clinical_evidence_limit: int,
                  stage_formats: dict[str, str] | None,
                  prompt_profile: str, graph_retrieval_mode: str,
                  clinical_agent_mode: str, writing_guidance: str,
                  shared_memory_enabled: bool, retrieval_strategy: str,
                  retrieval_fallback_enabled: bool,
                  final_validation_mode: str) -> tuple[dict, dict, list[dict]]:
    formats = {stage: response_format for stage in
               ("planner", "counter_evidence", "writer", "verifier", "reviser")}
    if stage_formats:
        formats.update(stage_formats)
    facts = number_facts(organize_facts(case))
    shared_memory = SharedClinicalMemory(case, facts) if shared_memory_enabled else None
    prompt_audit = {}
    algorithm_mode = prompt_profile
    planner_fallback = None
    clinical_plan = None; record = None; evidence_bundles = []
    clinical_graph = None; active_retriever = retriever
    graph_setup_fallback_reason = None; graph_setup_elapsed_ms = 0.0
    actual_retrieval_strategy = "no_graph"
    retrieval_fallbacks = []
    agent_assessment = None; counter_review = None; agent_error = None
    writer_evidence_validation = None; final_evidence_validation = None
    try:
        clinical_plan = run_planner(case, facts, client, formats["planner"], budget,
                                    prompt_profile, prompt_audit)
    except CaseError as primary_error:
        fallback_eligible = {"INVALID_CLINICAL_PLAN", "MODEL_INVALID_JSON"}
        if (prompt_profile not in ("contrastive", "contrastive_fewshot", "contrastive_v2")
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
        if shared_memory is not None:
            shared_memory.ingest_plan(clinical_plan)
        (clinical_graph, active_retriever, evidence_bundles, actual_retrieval_strategy,
         retrieval_fallbacks, graph_setup_fallback_reason,
         graph_setup_elapsed_ms) = _prepare_clinical_retrieval(
            case, facts, clinical_plan, retriever, retrieval_top_k,
            graph_retrieval_mode, retrieval_strategy, retrieval_fallback_enabled,
        )
        if shared_memory is not None and graph_retrieval_mode != "off":
            shared_memory.annotate_retrieval(evidence_bundles)
        active_prompt_profile = prompt_profile if algorithm_mode == prompt_profile else "legacy"
        agent_assessment = (assess_agent_need(facts, clinical_plan)
                            if clinical_agent_mode != "off" else None)
        should_run_agent = (clinical_agent_mode == "always" or
                            (clinical_agent_mode == "adaptive" and agent_assessment
                             and agent_assessment["triggered"]))
        if should_run_agent:
            try:
                counter_review = run_counter_evidence_agent(
                    case, facts, clinical_plan, client, formats["counter_evidence"], budget,
                    evidence_bundles, active_prompt_profile, prompt_audit,
                    shared_memory.render("counter_evidence") if shared_memory is not None else None,
                )
            except CaseError as exc:
                agent_error = {"code": exc.code, "message": exc.message,
                               "details": getattr(exc, "details", {}),
                               **_stage_failure_detail(client, "counter_evidence")}
        if shared_memory is not None:
            shared_memory.annotate_counter_review(counter_review)
        record = run_note_writer(case, facts, clinical_plan, evidence_bundles,
                                 client, formats["writer"], budget, clinical_evidence_limit,
                                 active_prompt_profile, prompt_audit, counter_review,
                                 writing_guidance,
                                 shared_memory.render("writer") if shared_memory is not None else None)
    elif graph_retrieval_mode != "off" and graph_setup_fallback_reason is None:
        graph_setup_fallback_reason = "planner_baseline_fallback_bypassed_graph"
    record, writer_recovery = recover_evidence(record, case)
    record, writer_evidence_policy = enforce_evidence_policy(record, case, clinical_evidence_limit)
    checks = validate_record(record, case)
    writer_evidence_validation = _evidence_validation_summary(checks, writer_recovery)
    writer_structure_codes = {"FORMAT", "EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND"}
    format_errors = [issue for issue in checks
                     if issue["level"] == "error" and issue["code"] in writer_structure_codes]
    if format_errors:
        _record_stage_schema(client, "writer", False, "INVALID_WRITER_OUTPUT")
        raise CaseError("INVALID_WRITER_OUTPUT",
                        "writer 阶段结构/引用校验失败：" + "; ".join(issue["message"] for issue in format_errors))
    _record_stage_schema(client, "writer", True)
    claims = extract_claims(record)
    verification_issue_count = 0
    verification_risk_counts = {"high": 0, "medium": 0, "low": 0}
    revision_meta = None
    verification_status = "not_run"
    verification_error = None
    verification_source_guard = {"dismissed_count": 0, "dismissed": []}
    verification_issue_audit = []
    suppressed_technical_reviews = []
    verification_attempted = False; verification_completed = False
    verified_claim_count = 0; verification_coverage = 0.0
    revision_attempted = False; revision_applied = False
    revision_strategy = None; revision_error = None
    deterministic_fallback_applied = False
    revised_claim_ids = []; unresolved_claim_ids = []
    risk_issue_catalog = []
    writer_preserved = False
    partial_sanitization_applied = False
    removed_unsafe_claim_ids = []
    internal_ids = _runtime_internal_ids(facts, clinical_plan, claims, counter_review)
    if clinical_verify:
        verification = None
        verification_attempted = True
        try:
            verification = run_claim_verifier(case, claims, record, clinical_plan, active_retriever,
                                               retrieval_top_k, client, formats["verifier"], budget,
                                               (prompt_profile if algorithm_mode == prompt_profile else "legacy"),
                                               prompt_audit, internal_ids)
        except CaseError as exc:
            verification_error = {"code": exc.code, "message": exc.message,
                                  **_stage_failure_detail(client, "verifier")}
            verification_status = "incomplete_writer_preserved"
            writer_preserved = True
            deterministic = deterministic_claim_issues(case, claims, internal_ids)
            high_deterministic = [issue for issue in deterministic if issue.get("severity") == "high"]
            risk_issue_catalog.extend(high_deterministic)
            if high_deterministic:
                record, fallback_meta = apply_conservative_fallback(record, high_deterministic)
                deterministic_fallback_applied = bool(fallback_meta.get("applied"))
                partial_sanitization_applied = deterministic_fallback_applied
                revision_strategy = fallback_meta.get("strategy")
                revised_claim_ids = fallback_meta.get("revised_claim_ids", [])
                unresolved_claim_ids = fallback_meta.get("unresolved_claim_ids", [])
                revision_meta = fallback_meta
                verification_status = "incomplete_deterministic_fallback"
                remaining = [issue for issue in deterministic_claim_issues(
                    case, extract_claims(record), internal_ids,
                ) if issue.get("severity") == "high"]
                risk_issue_catalog.extend(remaining)
                unresolved_claim_ids = sorted({issue["claim_id"] for issue in remaining})
                # A malformed Verifier response must not prevent an otherwise
                # available bounded Reviser call.  The Reviser receives only
                # deterministically located, still-present Claims.
                if remaining and clinical_revise and budget.remaining > 0:
                    revision_attempted = True
                    try:
                        record, model_revision_meta = run_local_revision(
                            case, record, clinical_plan, remaining, active_retriever,
                            retrieval_top_k, client, formats["reviser"], budget,
                            (prompt_profile if algorithm_mode == prompt_profile else "legacy"),
                            prompt_audit, internal_ids,
                        )
                        revision_meta = model_revision_meta
                        revision_applied = bool(model_revision_meta.get("revised_claim_ids"))
                        revision_strategy = model_revision_meta.get("strategy")
                        revised_claim_ids = sorted(set(revised_claim_ids)
                                                   | set(model_revision_meta.get(
                                                       "revised_claim_ids", [])))
                        unresolved_claim_ids = model_revision_meta.get(
                            "unresolved_claim_ids", [])
                        verification_status = ("incomplete_model_revision"
                                               if revision_applied
                                               else "incomplete_revision_unresolved")
                    except CaseError as revision_exc:
                        revision_error = {
                            "code": revision_exc.code, "message": revision_exc.message,
                            "details": getattr(revision_exc, "details", {}),
                            **_stage_failure_detail(client, "reviser"),
                        }
                        record, second_fallback = apply_conservative_fallback(record, remaining)
                        deterministic_fallback_applied = (
                            deterministic_fallback_applied
                            or bool(second_fallback.get("applied"))
                        )
                        partial_sanitization_applied = (
                            partial_sanitization_applied
                            or bool(second_fallback.get("applied"))
                        )
                        revision_meta = second_fallback
                        revision_meta["model_revision_error"] = revision_error
                        revision_strategy = second_fallback.get("strategy")
                        revised_claim_ids = sorted(set(revised_claim_ids)
                                                   | set(second_fallback.get(
                                                       "revised_claim_ids", [])))
                        unresolved_claim_ids = second_fallback.get(
                            "unresolved_claim_ids", [])
                        verification_status = "incomplete_revision_failed_writer_preserved"
                elif remaining and clinical_revise and budget.remaining <= 0:
                    revision_error = {
                        "code": "CALL_BUDGET_EXHAUSTED",
                        "message": "Verifier 异常后的局部修订未执行：每例模型调用预算已用尽。",
                    }
        if verification is not None:
            risk_issue_catalog.extend(
                issue for issue in verification["issues"] if issue.get("severity") == "high"
            )
            verification_source_guard = verification.get("source_guard", verification_source_guard)
            verification_completed = True
            verified_claim_count = len(verification["checked_claim_ids"])
            verification_coverage = (verified_claim_count / len(claims)) if claims else 1.0
            verification_issue_count = len(verification["issues"])
            verification_risk_counts = {
                severity: sum(issue.get("severity") == severity for issue in verification["issues"])
                for severity in ("high", "medium", "low")
            }
            verification_issue_audit = [_issue_audit(item) for item in verification["issues"]]
            verification_status = ("completed_no_issues" if verification_issue_count == 0
                                   else "completed_issues_reported")
        if verification is not None and has_blocking_issues(verification):
            high_issues = issues_for_revision(verification)
            if clinical_revise:
                revision_attempted = True
                try:
                    record, revision_meta = run_local_revision(
                        case, record, clinical_plan, high_issues, active_retriever,
                        retrieval_top_k, client, formats["reviser"], budget,
                        (prompt_profile if algorithm_mode == prompt_profile else "legacy"),
                        prompt_audit, internal_ids,
                    )
                    revision_applied = bool(revision_meta.get("revised_claim_ids"))
                    revision_strategy = revision_meta.get("strategy")
                    revised_claim_ids = revision_meta.get("revised_claim_ids", [])
                    unresolved_claim_ids = revision_meta.get("unresolved_claim_ids", [])
                    verification_status = ("completed_model_revision"
                                           if revision_applied and not unresolved_claim_ids
                                           else "completed_revision_unresolved")
                except CaseError as exc:
                    revision_error = {"code": exc.code, "message": exc.message,
                                      "details": getattr(exc, "details", {}),
                                      **_stage_failure_detail(client, "reviser")}
                    writer_preserved = True
            if not clinical_revise or revision_error is not None:
                record, fallback_meta = apply_conservative_fallback(record, high_issues)
                deterministic_fallback_applied = bool(fallback_meta.get("applied"))
                partial_sanitization_applied = deterministic_fallback_applied
                revision_strategy = fallback_meta.get("strategy")
                revised_claim_ids = fallback_meta.get("revised_claim_ids", [])
                unresolved_claim_ids = fallback_meta.get("unresolved_claim_ids", [])
                fallback_meta["model_revision_error"] = revision_error
                revision_meta = fallback_meta
                verification_status = "completed_deterministic_fallback"
            remaining = deterministic_claim_issues(case, extract_claims(record), internal_ids)
            remaining_ids = {issue["claim_id"] for issue in remaining if issue["severity"] == "high"}
            unresolved_claim_ids = sorted(set(unresolved_claim_ids) | remaining_ids)
            # Verifier/reviser diagnostics belong to the internal audit log.  Do
            # not turn technical repair hints into clinician-facing review items.
        if verification is not None:
            for issue in issues_for_review(verification):
                note = f"需医生复核：{issue['statement']}（{issue['revision_hint']}）"
                if (is_technical_review_item(note)
                        or issue.get("error_type") in {"INTERNAL_AUDIT_LEAK", "TECHNICAL_REVIEW_ITEM"}):
                    suppressed_technical_reviews.append(_issue_audit(issue))
                else:
                    _append_review(record, note)
    elif clinical_revise:
        raise CaseError("CLINICAL_CONFIG_INVALID", "启用局部修订前必须启用原子事实核验。")
    record, final_recovery = recover_evidence(record, case)
    record, post_revision_evidence_policy = enforce_evidence_policy(record, case, clinical_evidence_limit)
    final_evidence_validation = _evidence_validation_summary(validate_record(record, case),
                                                             final_recovery)
    final_claims = extract_claims(record)
    final_quality_issues = deterministic_claim_issues(case, final_claims, internal_ids)
    final_high = [item for item in final_quality_issues if item.get("severity") == "high"]
    risk_issue_catalog.extend(final_high)
    # Last deterministic pass is bounded and model-free.  It is intentionally
    # limited to exact complete Claims; unsafe or ambiguous spans are preserved
    # and remain explicitly unresolved rather than being globally replaced.
    if final_high:
        record, final_fallback = apply_conservative_fallback(record, final_high)
        if final_fallback.get("applied"):
            deterministic_fallback_applied = True
            partial_sanitization_applied = True
            revision_strategy = final_fallback.get("strategy")
            revised_claim_ids = sorted(set(revised_claim_ids)
                                       | set(final_fallback.get("revised_claim_ids", [])))
            final_claims = extract_claims(record)
            final_quality_issues = deterministic_claim_issues(case, final_claims, internal_ids)
            final_high = [item for item in final_quality_issues
                          if item.get("severity") == "high"]
            risk_issue_catalog.extend(final_high)
        final_target_ids = {item["claim_id"] for item in final_quality_issues
                            if item.get("severity") == "high"}
        unresolved_claim_ids = sorted(
            (set(unresolved_claim_ids)
             - set(final_fallback.get("revised_claim_ids", [])))
            | set(final_fallback.get("unresolved_claim_ids", []))
            | final_target_ids
        )
    else:
        unresolved_claim_ids = sorted(set(unresolved_claim_ids))
    final_risk_assessments = _final_claim_risk_assessments(
        unresolved_claim_ids, risk_issue_catalog, final_validation_mode,
    )
    unresolved_hard_claim_ids = sorted({
        item["claim_id"] for item in final_risk_assessments
        if item["risk_level"] == "hard"
    })
    unresolved_soft_claim_ids = sorted({
        item["claim_id"] for item in final_risk_assessments
        if item["risk_level"] == "soft"
    } - set(unresolved_hard_claim_ids))
    detected_before_guarantee = {
        "unresolved_claim_ids": list(unresolved_claim_ids),
        "unresolved_hard_claim_ids": list(unresolved_hard_claim_ids),
        "unresolved_soft_claim_ids": list(unresolved_soft_claim_ids),
        "risk_assessments": list(final_risk_assessments),
    }
    unresolved_issue_candidates = [
        item for item in risk_issue_catalog
        if item.get("claim_id") in set(unresolved_claim_ids)
    ]
    record, output_guarantee = guarantee_writer_output(
        case, record, unresolved_issue_candidates,
        verification_completed=verification_completed,
        revision_applied=revision_applied,
        internal_ids=internal_ids,
    )
    record, guarantee_evidence_recovery = recover_evidence(record, case)
    record, guarantee_evidence_policy = enforce_evidence_policy(
        record, case, clinical_evidence_limit,
    )
    final_claims = extract_claims(record)
    final_quality_issues = deterministic_claim_issues(case, final_claims, internal_ids)
    final_high = [item for item in final_quality_issues if item.get("severity") == "high"]
    unresolved_claim_ids = sorted({item["claim_id"] for item in final_high})
    final_risk_assessments = _final_claim_risk_assessments(
        unresolved_claim_ids, final_high, final_validation_mode,
    )
    unresolved_hard_claim_ids = sorted({
        item["claim_id"] for item in final_risk_assessments
        if item["risk_level"] == "hard"
    })
    unresolved_soft_claim_ids = sorted({
        item["claim_id"] for item in final_risk_assessments
        if item["risk_level"] == "soft"
    } - set(unresolved_hard_claim_ids))
    output_blocking_risks = [
        item for item in final_risk_assessments
        if item["risk_level"] == "hard" and item.get("code") is not None
    ]
    final_evidence_validation = _evidence_validation_summary(
        validate_record(record, case), guarantee_evidence_recovery,
    )
    partial_sanitization_applied = (
        partial_sanitization_applied or bool(output_guarantee.get("applied"))
    )
    removed_unsafe_claim_ids = output_guarantee.get("removed_claim_ids", [])
    boundary_codes = {"INTERNAL_AUDIT_LEAK", "TECHNICAL_REVIEW_ITEM"}
    boundary_errors = [
        {"level": "error", "code": item["error_type"],
         "message": "正式输出仍含内部审计信息，未通过最终质量边界。"}
        for item in final_high if item.get("error_type") in boundary_codes
    ]
    checks.extend(boundary_errors)
    if verification_status.startswith("completed"):
        if unresolved_hard_claim_ids:
            verification_status = "completed_revision_unresolved"
        elif unresolved_soft_claim_ids:
            verification_status = "completed_soft_risks"
    meta = {
        "problem_count": len(clinical_plan["problems"]),
        "fact_count": len(facts),
        "evidence_bundle_count": len(evidence_bundles),
        "claim_count": len(claims),
        "verification_enabled": clinical_verify,
        "verification_attempted": verification_attempted,
        "verification_completed": verification_completed,
        "verification_issue_count": verification_issue_count,
        "issue_count": verification_issue_count,
        "verification_risk_counts": verification_risk_counts,
        "verification_status": verification_status,
        "verification_error": verification_error,
        "verification_source_guard": verification_source_guard,
        "verification_issue_audit": verification_issue_audit,
        "verified_claim_count": verified_claim_count,
        "total_claim_count": len(claims),
        "verification_coverage": round(verification_coverage, 6),
        "unresolved_risk_count": len(unresolved_claim_ids),
        "unresolved_claim_ids": unresolved_claim_ids,
        "unresolved_hard_risk_count": len(unresolved_hard_claim_ids),
        "unresolved_hard_claim_ids": unresolved_hard_claim_ids,
        "unresolved_soft_risk_count": len(unresolved_soft_claim_ids),
        "unresolved_soft_claim_ids": unresolved_soft_claim_ids,
        "final_validation_mode": final_validation_mode,
        "revision_enabled": clinical_revise,
        "revision_attempted": revision_attempted,
        "revision_applied": revision_applied,
        "revision_strategy": revision_strategy,
        "revised_claim_ids": revised_claim_ids,
        "deterministic_fallback_applied": deterministic_fallback_applied,
        "revision_error": revision_error,
        "revision": revision_meta,
        "writer_preserved": writer_preserved,
        "partial_sanitization_applied": partial_sanitization_applied,
        "removed_unsafe_claim_ids": removed_unsafe_claim_ids,
        "manual_review_required": bool(output_guarantee.get("manual_review_required")),
        "output_guarantee": output_guarantee,
        "detected_risks_before_output_guarantee": detected_before_guarantee,
        "final_quality_gate": {
            "runtime_internal_id_count": len(internal_ids),
            "post_revision_high_risk_count": len(final_high),
            "post_revision_high_risk_audit": [_issue_audit(item) for item in final_high],
            "lite_risk_assessments": final_risk_assessments,
            "boundary_error_count": len(boundary_errors),
            "suppressed_technical_review_count": len(suppressed_technical_reviews),
            "suppressed_technical_review_audit": suppressed_technical_reviews,
            "output_blocking_safety_failure": bool(output_blocking_risks),
            "output_blocking_risk_audit": output_blocking_risks,
        },
        "evidence_limit": clinical_evidence_limit,
        "stage_response_formats": formats,
        "prompt_profile": prompt_profile,
        "algorithm_mode": algorithm_mode,
        "planner_fallback": planner_fallback,
        "graph_retrieval": _graph_retrieval_meta(
            graph_retrieval_mode, retriever, active_retriever, clinical_graph,
            graph_setup_fallback_reason, graph_setup_elapsed_ms,
            retrieval_strategy, actual_retrieval_strategy, retrieval_fallbacks,
        ),
        "adaptive_agent": {
            "mode": clinical_agent_mode,
            "prompt_version": COUNTER_PROMPT_VERSION,
            "prompt_profile": (prompt_profile if algorithm_mode == prompt_profile else "legacy"),
            "assessment": agent_assessment,
            "triggered": counter_review is not None or agent_error is not None,
            "completed": counter_review is not None,
            "error": agent_error,
            "reviewed_problem_count": (len(counter_review.get("reviewed_problem_ids", []))
                                       if counter_review else 0),
            "issue_count": len(counter_review.get("issues", [])) if counter_review else 0,
            "retrieval_bundle_count": (len(evidence_bundles)
                                       if counter_review is not None or agent_error is not None else 0),
            "retrieval_candidate_count": (
                sum(len(bundle.get("candidates", [])) for bundle in evidence_bundles)
                if counter_review is not None or agent_error is not None else 0
            ),
        },
        "writer_evidence_validation": writer_evidence_validation,
        "final_evidence_validation": final_evidence_validation,
        "prompt_audit": prompt_audit,
        "writing_guidance": {"mode": writing_guidance,
                             "enabled": writing_guidance != "off"},
        "shared_clinical_memory": (shared_memory.audit() if shared_memory is not None else
                                   {"enabled": False}),
        "evidence_policy": {"writer": writer_evidence_policy,
                            "post_revision": post_revision_evidence_policy,
                            "output_guarantee": guarantee_evidence_policy},
        "output_guarantee_evidence_recovery": guarantee_evidence_recovery,
    }
    return record, meta, checks


def _run_baseline_fallback(case: Case, client, response_format: str, budget: Budget) -> dict:
    raw_sources = [{"source_id": source.source_id, "field_path": source.field_path,
                    "content": source.content} for source in case.sources]
    messages = _messages("baseline_system.txt", {"record_cutoff": case.record_cutoff,
                                                   "sources": raw_sources})
    raw = client.call(messages, response_format, budget, stage="baseline_fallback", schema=MEDICAL_SCHEMA)
    return parse_model_json(raw, stage="baseline_fallback", client=client)


def _prepare_clinical_retrieval(case: Case, facts: list[dict], clinical_plan: dict,
                                dense_retriever: DenseEvidenceRetriever | None,
                                top_k: int, graph_mode: str, requested_strategy: str,
                                fallback_enabled: bool):
    """Run one bounded retrieval chain without changing the clinical graph.

    Explicit dense_graph may fall back once to lexical_graph and then no_graph.
    Explicit lexical_graph may fall back once to no_graph.  Legacy ``auto`` CLI
    calls pass ``fallback_enabled=False`` and retain their previous behavior.
    """
    started = time.perf_counter()
    if graph_mode == "off" and requested_strategy == "dense_only":
        if dense_retriever is None:
            raise CaseError("GRAPH_RETRIEVAL_CONFIG_MISSING",
                            "兼容 Dense-only 检索缺少检索器。")
        bundles = retrieve_problem_evidence(clinical_plan, dense_retriever, top_k, facts)
        elapsed = (time.perf_counter() - started) * 1000
        return None, dense_retriever, bundles, "dense_only", [], None, elapsed
    if graph_mode == "off" or requested_strategy == "no_graph":
        return None, None, [], "no_graph", [], None, 0.0
    fallbacks = []
    try:
        clinical_graph = build_clinical_graph(case, facts, clinical_plan)
    except Exception as exc:
        reason = f"graph_setup_failed:{type(exc).__name__}"
        fallbacks.append({"from": requested_strategy, "to": "no_graph", "reason": reason})
        elapsed = (time.perf_counter() - started) * 1000
        return None, None, [], "no_graph", fallbacks, reason, elapsed

    attempts = []
    if requested_strategy == "dense_graph":
        if dense_retriever is not None:
            attempts.append(("dense_graph", dense_retriever))
        else:
            reason = "graph_retrieval_failed:DENSE_RETRIEVER_UNAVAILABLE"
            next_strategy = "lexical_graph" if fallback_enabled else "no_graph"
            fallbacks.append({"from": "dense_graph", "to": next_strategy, "reason": reason})
        if fallback_enabled:
            attempts.append(("lexical_graph", LexicalEvidenceRetriever(
                case, max_chunk_chars=(dense_retriever.max_chunk_chars
                                       if dense_retriever is not None else 240),
            )))
    elif requested_strategy == "lexical_graph":
        attempts.append(("lexical_graph", LexicalEvidenceRetriever(case)))
    else:
        reason = f"graph_retrieval_failed:UNKNOWN_STRATEGY:{requested_strategy}"
        elapsed = (time.perf_counter() - started) * 1000
        return clinical_graph, None, [], "no_graph", [
            {"from": requested_strategy, "to": "no_graph", "reason": reason}
        ], reason, elapsed

    last_reason = None
    for index, (strategy, seed_retriever) in enumerate(attempts):
        graph_retriever = GraphGuidedRetriever(
            seed_retriever, clinical_graph, strategy=graph_mode,
        )
        try:
            bundles = retrieve_problem_evidence(clinical_plan, graph_retriever, top_k, facts)
            stats = graph_retriever.stats()
            if fallback_enabled and not stats.get("graph_executed"):
                reason = stats.get("graph_fallback_reason") or "graph_not_executed"
                raise TaskError("GRAPH_NOT_EXECUTED", reason)
            elapsed = (time.perf_counter() - started) * 1000
            setup_reason = ";".join(item["reason"] for item in fallbacks) or None
            return (clinical_graph, graph_retriever, bundles, strategy, fallbacks,
                    setup_reason, elapsed)
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
            last_reason = f"graph_retrieval_failed:{code}"
            next_strategy = (attempts[index + 1][0]
                             if index + 1 < len(attempts) else "no_graph")
            fallbacks.append({"from": strategy, "to": next_strategy, "reason": last_reason})

    elapsed = (time.perf_counter() - started) * 1000
    setup_reason = ";".join(item["reason"] for item in fallbacks) or last_reason
    return clinical_graph, None, [], "no_graph", fallbacks, setup_reason, elapsed


def _graph_retrieval_meta(mode: str, dense_retriever, active_retriever,
                          clinical_graph, setup_fallback_reason: str | None,
                          setup_elapsed_ms: float, requested_strategy: str,
                          actual_strategy: str, strategy_fallbacks: list[dict]) -> dict:
    graph_stats = (active_retriever.stats()
                   if isinstance(active_retriever, GraphGuidedRetriever) else None)
    runtime_reason = graph_stats.get("graph_fallback_reason") if graph_stats else None
    fallback_reason = setup_fallback_reason or runtime_reason
    embedding_provider = (dense_retriever.provider_name if dense_retriever is not None else None)
    seed_provider = (graph_stats.get("seed_provider") if graph_stats else None)
    return {
        "mode": mode,
        "enabled": mode != "off",
        "retrieval_strategy_requested": requested_strategy,
        "retrieval_strategy": actual_strategy,
        "retrieval_fallbacks": strategy_fallbacks,
        "graph_requested": mode != "off",
        "graph_executed": bool(graph_stats and graph_stats.get("graph_executed")),
        "graph_mode": mode,
        "graph_fallback_reason": fallback_reason,
        "fallback_status": fallback_reason is not None,
        "embedding_provider": embedding_provider,
        "seed_provider": seed_provider,
        "seed_fact_ids": graph_stats.get("seed_fact_ids", []) if graph_stats else [],
        "graph_edge_type_counts": (clinical_graph.stats().get("relations", {})
                                   if clinical_graph else {}),
        "candidate_fact_ids": graph_stats.get("candidate_fact_ids", []) if graph_stats else [],
        "reranked_fact_ids": graph_stats.get("reranked_fact_ids", []) if graph_stats else [],
        "selected_source_ids": graph_stats.get("selected_source_ids", []) if graph_stats else [],
        "graph_elapsed_ms": round(setup_elapsed_ms, 3),
        "ranking_changed": bool(graph_stats and graph_stats.get("ranking_changed")),
        "graph": clinical_graph.stats() if clinical_graph else None,
        "retriever": graph_stats,
    }


def _append_review(record: dict, note: str) -> None:
    record.setdefault("review_items", [])
    if note not in record["review_items"]:
        record["review_items"].append(note)


def _runtime_internal_ids(facts: list[dict], clinical_plan: dict, claims: list[dict],
                          counter_review: dict | None = None) -> set[str]:
    """Return exact identifiers created for this case and this run only."""
    values = {item.get("fact_id") for item in facts}
    values.update(problem.get("problem_id") for problem in clinical_plan.get("problems", []))
    values.update(item.get("claim_id") for item in claims)
    if counter_review:
        values.update(f"CI{index:03d}"
                      for index, _ in enumerate(counter_review.get("issues", []), start=1))
    return {value for value in values if isinstance(value, str) and value}


def _issue_audit(issue: dict) -> dict:
    statement = issue.get("statement", "")
    return {
        "claim_id": issue.get("claim_id"),
        "section": issue.get("section"),
        "code": issue.get("error_type"),
        "severity": issue.get("severity"),
        "source_ids": list(issue.get("source_ids", [])),
        "statement_sha256": hashlib.sha256(str(statement).encode("utf-8")).hexdigest(),
    }


def _final_claim_risk_assessments(unresolved_claim_ids: list[str], issues: list[dict],
                                  validation_mode: str) -> list[dict]:
    unresolved = set(unresolved_claim_ids)
    result = []
    seen = set()
    represented = set()
    for issue in issues:
        claim_id = issue.get("claim_id")
        if claim_id not in unresolved:
            continue
        code = issue.get("error_type") or issue.get("code")
        key = (claim_id, code)
        if key in seen:
            continue
        seen.add(key)
        represented.add(claim_id)
        classification = classify_claim_issue_for_final(issue, validation_mode)
        result.append({**_issue_audit(issue), **classification})
    for claim_id in sorted(unresolved - represented):
        classification = classify_claim_issue_for_final(
            {"claim_id": claim_id, "severity": "high", "statement": None},
            validation_mode,
        )
        result.append({
            "claim_id": claim_id, "section": None, "code": None, "severity": "high",
            "source_ids": [], "statement_sha256": None, **classification,
        })
    return result


def _stage_failure_detail(client, stage: str) -> dict:
    events = [event for event in getattr(client, "call_events", []) if event.get("stage") == stage]
    if not events:
        return {"finish_reason": None, "max_tokens": None, "output_truncation_suspected": False}
    event = events[-1]
    usage = event.get("token_usage") or {}
    maximum = event.get("max_tokens")
    finish_reason = event.get("finish_reason")
    suspected = (finish_reason in ("length", "max_tokens")
                 or (isinstance(maximum, int) and usage.get("completion_tokens") == maximum))
    return {"finish_reason": finish_reason, "max_tokens": maximum,
            "output_truncation_suspected": suspected}


def _failure_stage(error: CaseError) -> str:
    code = error.code.upper()
    if "PLANNER" in code or "CLINICAL_PLAN" in code:
        return "planner"
    if "WRITER" in code:
        return "writer"
    if "VERIF" in code or "CLAIM" in code:
        return "verifier"
    if "REVIS" in code:
        return "reviser"
    if code in {"MODEL_TIMEOUT", "MODEL_ERROR", "MODEL_HTTP_ERROR", "MODEL_INVALID_JSON",
                "INCOMPLETE_JSON", "JSON_TRAILING_TEXT"}:
        return "model_stage"
    return "clinical_pipeline"


def _evidence_validation_summary(issues: list[dict], repair: dict) -> dict:
    codes = {"EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND", "DATE_SUPPORT", "NUMERIC_SUPPORT",
             "NEGATION_SUPPORT", "PLAN_AS_DONE", "FACT_SUPPORT_LIMITED"}
    evidence_issues = [issue for issue in issues if issue.get("code") in codes]
    return {"repair": repair,
            "error_count": sum(issue.get("level") == "error" for issue in evidence_issues),
            "warning_count": sum(issue.get("level") == "warning" for issue in evidence_issues),
            "codes": [issue.get("code") for issue in evidence_issues]}


def _validation_risk_summary(issues: list[dict]) -> dict:
    levels = {"hard": 0, "repairable": 0, "soft": 0}
    for issue in issues:
        levels[classify_validation_issue(issue)] += 1
    return {"error_count": sum(issue.get("level") == "error" for issue in issues),
            "warning_count": sum(issue.get("level") == "warning" for issue in issues),
            "risk_counts": levels,
            "codes": [issue.get("code") for issue in issues]}


def _record_stage_schema(client, stage: str, success: bool, error_type=None) -> None:
    recorder = getattr(client, "record_schema_result", None)
    if recorder is not None:
        recorder(stage, success, error_type)


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
