"""Clinical problem planning for the optional clinical pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import CaseError
from .model_client import Budget, PLANNER_SCHEMA, parse_model_json
from .protocol import Case


BASE_DIR = Path(__file__).resolve().parent.parent
PRIORITIES = {"high", "medium", "low"}
ORIGINS = {"documented_problem", "model_inference", "mixed"}
CERTAINTIES = {"suspected", "rule_out", "possible", "unlikely"}


def number_facts(facts: list[dict]) -> list[dict]:
    """Attach deterministic IDs without discarding the existing fact structure."""
    return [{"fact_id": f"F{index:03d}", **fact} for index, fact in enumerate(facts, 1)]


def run_planner(case: Case, facts: list[dict], client, response_format: str, budget: Budget) -> dict:
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "facts": facts,
    }
    messages = _messages("planner_system.txt", payload)
    raw = client.call(messages, response_format, budget, stage="planner", schema=PLANNER_SCHEMA)
    plan = parse_model_json(raw, stage="planner", client=client)
    validate_clinical_plan(plan, {fact["fact_id"] for fact in facts})
    return plan


def validate_clinical_plan(plan, fact_ids: set[str]) -> None:
    if not isinstance(plan, dict) or not isinstance(plan.get("problems"), list) or not plan["problems"]:
        raise CaseError("INVALID_CLINICAL_PLAN", "临床规划必须包含非空 problems 数组。")
    if set(plan) != {"problems"}:
        raise CaseError("INVALID_CLINICAL_PLAN", "临床规划顶层只能包含 problems。")
    seen = set()
    for index, problem in enumerate(plan["problems"]):
        if not isinstance(problem, dict):
            raise CaseError("INVALID_CLINICAL_PLAN", f"problems[{index}] 必须是对象。")
        allowed_problem = {"problem_id", "name", "priority", "origin", "supporting_facts",
                           "contradicting_facts", "differential_diagnoses", "plan_goals", "retrieval_queries"}
        if set(problem) != allowed_problem:
            raise CaseError("INVALID_CLINICAL_PLAN", f"problems[{index}] 字段不完整或含未知字段。")
        problem_id = problem.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id or problem_id in seen:
            raise CaseError("INVALID_CLINICAL_PLAN", "problem_id 必须非空且唯一。")
        seen.add(problem_id)
        if not isinstance(problem.get("name"), str) or not problem["name"].strip():
            raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.name 必须是非空字符串。")
        if problem.get("priority") not in PRIORITIES or problem.get("origin") not in ORIGINS:
            raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的 priority/origin 非法。")
        _fact_references(problem.get("supporting_facts"), fact_ids, problem_id, "supporting_facts")
        _fact_references(problem.get("contradicting_facts"), fact_ids, problem_id, "contradicting_facts")
        for field in ("plan_goals", "retrieval_queries"):
            values = problem.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.{field} 必须是非空字符串数组。")
        differentials = problem.get("differential_diagnoses")
        if not isinstance(differentials, list):
            raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.differential_diagnoses 必须是数组。")
        for differential in differentials:
            if not isinstance(differential, dict) or not isinstance(differential.get("name"), str):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的鉴别诊断结构无效。")
            allowed_differential = {"name", "certainty", "origin", "supporting_facts",
                                    "contradicting_facts", "missing_evidence"}
            if set(differential) != allowed_differential:
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的鉴别诊断字段不完整或含未知字段。")
            if differential.get("certainty") not in CERTAINTIES or differential.get("origin") not in ORIGINS:
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的鉴别诊断 certainty/origin 非法。")
            _fact_references(differential.get("supporting_facts"), fact_ids, problem_id, "differential.supporting_facts")
            _fact_references(differential.get("contradicting_facts"), fact_ids, problem_id,
                             "differential.contradicting_facts")
            missing = differential.get("missing_evidence")
            if not isinstance(missing, list) or any(not isinstance(value, str) for value in missing):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的 missing_evidence 必须是字符串数组。")


def retrieve_problem_evidence(plan: dict, retriever, top_k: int) -> list[dict]:
    if retriever is None:
        return []
    bundles = []
    for problem in plan["problems"]:
        queries = problem.get("retrieval_queries") or [problem["name"]]
        for query in queries:
            bundles.append({
                "problem_id": problem["problem_id"],
                "query": query,
                "candidates": [candidate.to_dict() for candidate in retriever.retrieve(query, top_k=top_k)],
            })
    return bundles


def _fact_references(values, fact_ids: set[str], problem_id: str, field: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, str) or value not in fact_ids for value in values):
        raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.{field} 引用了不存在的事实 ID。")


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
