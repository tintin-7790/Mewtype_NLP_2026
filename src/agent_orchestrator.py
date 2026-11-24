"""Bounded adaptive orchestration for one optional counter-evidence review.

This is deliberately not a free-form agent conversation. A deterministic
gate decides whether one extra, schema-constrained model call is warranted.
The review may annotate uncertainty for the Writer but cannot mutate sources,
facts or the clinical plan.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .errors import CaseError
from .model_client import (Budget, COUNTER_EVIDENCE_CODES, COUNTER_EVIDENCE_SCHEMA,
                           parse_model_json)
from .protocol import Case
from .prompt_profiles import load_clinical_prompt


BASE_DIR = Path(__file__).resolve().parent.parent


def assess_agent_need(facts: list[dict], clinical_plan: dict) -> dict:
    """Return a transparent complexity score; no medical conclusion is made."""
    problems = clinical_plan.get("problems", [])
    differentials = [d for problem in problems for d in problem.get("differential_diagnoses", [])]
    missing_count = sum(len(item.get("missing_evidence", [])) for item in differentials)
    uncertain_count = sum(fact.get("assertion") in ("uncertain", "not_provided") for fact in facts)
    correction_count = sum(bool(fact.get("correction")) for fact in facts)
    no_counter_count = sum(not item.get("contradicting_facts") for item in differentials)
    reasons = []
    score = 0
    for condition, points, reason in (
        (len(problems) >= 3, 2, "three_or_more_problems"),
        (len(differentials) >= 4, 2, "four_or_more_differentials"),
        (missing_count >= 3, 1, "multiple_missing_evidence_items"),
        (uncertain_count >= 3, 1, "multiple_uncertain_or_missing_facts"),
        (correction_count > 0, 1, "corrected_source_record_present"),
        (no_counter_count >= 2, 1, "differentials_without_counter_evidence"),
    ):
        if condition:
            score += points
            reasons.append(reason)
    return {"score": score, "threshold": 3, "triggered": score >= 3, "reasons": reasons,
            "problem_count": len(problems), "differential_count": len(differentials),
            "missing_evidence_count": missing_count, "uncertain_fact_count": uncertain_count,
            "correction_count": correction_count}


def run_counter_evidence_agent(case: Case, facts: list[dict], clinical_plan: dict,
                               client, response_format: str, budget: Budget,
                               evidence_bundles: list[dict] | None = None,
                               prompt_profile: str = "contrastive",
                               prompt_audit: dict | None = None,
                               shared_memory_context: dict | None = None) -> dict:
    preparation_started = time.perf_counter()
    payload = {
        "record_cutoff": case.record_cutoff,
        "source_catalog": [{"source_id": source.source_id, "field_path": source.field_path}
                           for source in case.sources],
        "fact_catalog_notice": (
            "fact_text 是从当前病例 source_id 确定性提取的原文片段；只能引用已有 Fact ID，"
            "不得据此创造新患者事实。"
        ),
        "facts": [{key: fact.get(key) for key in
                   ("fact_id", "source_id", "field_path", "fact_text", "time", "value", "unit",
                    "assertion", "status", "correction")} for fact in facts],
        "problems": [_compact_problem(problem) for problem in clinical_plan.get("problems", [])],
        "retrieval_evidence": _compact_retrieval_evidence(evidence_bundles or [], facts),
        "retrieval_notice": (
            "检索候选只用于定位反证或缺失证据；Dense 相似度、Graph/PPR 分数和排序均不证明事实成立，"
            "必须回到候选的 source_id 与已有 Fact ID。"
        ),
    }
    if shared_memory_context is not None:
        payload["shared_decision_cards"] = shared_memory_context
    system, _ = load_clinical_prompt("counter_evidence", prompt_profile, "")
    if prompt_audit is not None:
        prompt_audit["counter_evidence"] = {
            "profile": prompt_profile,
            "system_prompt_chars": len(system),
            "payload_chars": len(json.dumps(payload, ensure_ascii=False)),
            "shared_memory_injected": shared_memory_context is not None,
            "preparation_elapsed_ms": round((time.perf_counter() - preparation_started) * 1000, 3),
        }
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    raw = client.call(messages, response_format, budget, stage="counter_evidence",
                      schema=COUNTER_EVIDENCE_SCHEMA)
    review = parse_model_json(raw, stage="counter_evidence", client=client)
    try:
        validate_counter_evidence_review(review, clinical_plan, {fact["fact_id"] for fact in facts})
    except CaseError as exc:
        _record_schema(client, False, exc.code)
        raise
    _record_schema(client, True)
    return review


def validate_counter_evidence_review(review, clinical_plan: dict, fact_ids: set[str]) -> None:
    if (not isinstance(review, dict)
            or set(review) != {"reviewed_problem_ids", "issues"}
            or not isinstance(review.get("reviewed_problem_ids"), list)
            or not isinstance(review.get("issues"), list)):
        raise CaseError("INVALID_COUNTER_EVIDENCE",
                        "反证代理输出必须仅包含 reviewed_problem_ids 和 issues 数组。")
    problem_ids = {problem["problem_id"] for problem in clinical_plan.get("problems", [])}
    reviewed = review["reviewed_problem_ids"]
    if (any(not isinstance(problem_id, str) for problem_id in reviewed)
            or len(reviewed) != len(set(reviewed)) or set(reviewed) != problem_ids):
        raise CaseError("INCOMPLETE_COUNTER_EVIDENCE_COVERAGE",
                        "reviewed_problem_ids 必须无重复地完整覆盖全部临床问题。")
    seen = set()
    required = {"problem_id", "code", "fact_ids", "note"}
    for index, item in enumerate(review["issues"]):
        if not isinstance(item, dict) or set(item) != required:
            raise CaseError("INVALID_COUNTER_EVIDENCE", f"issues[{index}] 字段无效。")
        problem_id = item["problem_id"]
        key = (problem_id, item.get("code"))
        if problem_id not in problem_ids or key in seen:
            raise CaseError("INVALID_COUNTER_EVIDENCE", "反证代理引用了未知问题或重复 issue。")
        seen.add(key)
        if item["code"] not in COUNTER_EVIDENCE_CODES:
            raise CaseError("INVALID_COUNTER_EVIDENCE", f"{problem_id}.code 非法。")
        refs = item["fact_ids"]
        if (not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in fact_ids
                                              for ref in refs)):
            raise CaseError("INVALID_COUNTER_EVIDENCE", f"{problem_id}.fact_ids 引用了未知 Fact ID。")
        if item["code"] in ("COUNTER_EVIDENCE", "AMBIGUOUS_FACT_DIRECTION") and not refs:
            raise CaseError("INVALID_COUNTER_EVIDENCE", f"{problem_id} 的事实型 issue 必须引用 Fact ID。")
        if not isinstance(item["note"], str) or not item["note"].strip():
            raise CaseError("INVALID_COUNTER_EVIDENCE", f"{problem_id}.note 不能为空。")


def attach_counter_review(context: dict, review: dict | None) -> dict:
    """Attach compact reviews by problem without duplicating patient source text."""
    if not review:
        return context
    by_problem = {}
    for item in review.get("issues", []):
        by_problem.setdefault(item["problem_id"], []).append(item)
    for problem in context.get("problems", []):
        issues = by_problem.get(problem.get("problem_id"), [])
        if issues:
            problem["counter_evidence_review"] = {"issues": issues}
    context["counter_evidence_notice"] = (
        "该内容是辅助审查意见，不是患者事实；仅 Fact ID 可回溯至原始 sources，"
        "不得据此升级诊断或把建议写成已执行事实。"
    )
    return context


def _compact_problem(problem: dict) -> dict:
    return {
        "problem_id": problem.get("problem_id"),
        "name": problem.get("name"),
        "supporting_facts": problem.get("supporting_facts", []),
        "contradicting_facts": problem.get("contradicting_facts", []),
        "differentials": [{key: differential.get(key) for key in
                           ("name", "certainty", "supporting_facts",
                            "contradicting_facts", "missing_evidence")}
                          for differential in problem.get("differential_diagnoses", [])],
    }


def _compact_retrieval_evidence(evidence_bundles: list[dict], facts: list[dict]) -> list[dict]:
    source_facts: dict[str, list[str]] = {}
    for fact in facts:
        source_id = fact.get("source_id")
        fact_id = fact.get("fact_id")
        if isinstance(source_id, str) and isinstance(fact_id, str):
            source_facts.setdefault(source_id, []).append(fact_id)
    compact = []
    for bundle in evidence_bundles:
        candidates = []
        for candidate in bundle.get("candidates", [])[:3]:
            source_id = candidate.get("source_id")
            candidates.append({
                "source_id": source_id,
                "field_path": candidate.get("field_path", ""),
                "text": str(candidate.get("text", ""))[:160],
                "fact_ids": source_facts.get(source_id, []),
                "dense_similarity": candidate.get("similarity"),
                "ranking_score": candidate.get("ranking_score"),
                "graph_relevance": candidate.get("graph_relevance"),
                "graph_strategy": candidate.get("graph_strategy"),
                "compatible": candidate.get("compatible"),
            })
        compact.append({
            "problem_id": bundle.get("problem_id"),
            "planner_query": bundle.get("planner_query"),
            "candidates": candidates,
        })
    return compact


def _record_schema(client, success: bool, error_type=None) -> None:
    recorder = getattr(client, "record_schema_result", None)
    if recorder is not None:
        recorder("counter_evidence", success, error_type)
