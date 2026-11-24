"""Clinical problem planning for the optional clinical pipeline."""

from __future__ import annotations

import json
from copy import deepcopy

from .errors import CaseError
from .model_client import Budget, PLANNER_SCHEMA, parse_model_json
from .prompt_profiles import load_clinical_prompt
from .protocol import Case


PRIORITIES = {"high", "medium", "low"}
ORIGINS = {"documented_problem", "model_inference", "mixed"}
CERTAINTIES = {"suspected", "rule_out", "possible", "unlikely"}


def number_facts(facts: list[dict]) -> list[dict]:
    """Attach deterministic IDs without discarding the existing fact structure."""
    return [{"fact_id": f"F{index:03d}", **fact} for index, fact in enumerate(facts, 1)]


def compact_facts_for_prompt(facts: list[dict]) -> list[dict]:
    """Keep one concise Fact-ID mapping; raw source text is already sent once."""
    fields = ("fact_id", "source_id", "field_path", "fact_text", "time", "value", "unit",
              "assertion", "status", "correction")
    return [{key: fact.get(key) for key in fields} for fact in facts]


def run_planner(case: Case, facts: list[dict], client, response_format: str, budget: Budget,
                prompt_profile: str = "contrastive", prompt_audit: dict | None = None) -> dict:
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        # The archived Legacy prompt expects the original fact payload.  Only
        # Contrastive profiles use the de-duplicated representation.
        "facts": facts if prompt_profile == "legacy" else compact_facts_for_prompt(facts),
    }
    case_text = "\n".join(source.content for source in case.sources)
    system, example_id = load_clinical_prompt("planner", prompt_profile, case_text)
    if prompt_audit is not None:
        prompt_audit["planner"] = {"profile": prompt_profile, "fewshot_example_id": example_id,
                                   "system_prompt_chars": len(system)}
    messages = _messages(system, payload)
    raw = client.call(messages, response_format, budget, stage="planner", schema=PLANNER_SCHEMA)
    plan = parse_model_json(raw, stage="planner", client=client)
    fact_ids = {fact["fact_id"] for fact in facts}
    validate_plan_fact_references(plan, fact_ids)
    if prompt_profile != "legacy":
        plan, policy = apply_contrastive_planning_policy(plan)
        if prompt_audit is not None:
            prompt_audit["planner"]["planning_policy"] = policy
    validate_clinical_plan(plan, fact_ids)
    return plan


def apply_contrastive_planning_policy(plan) -> tuple[object, dict]:
    """Deduplicate and cap differentials without inventing new candidates."""
    if not isinstance(plan, dict) or not isinstance(plan.get("problems"), list):
        return plan, {"applied": False, "reason": "invalid_shape"}
    result = deepcopy(plan)
    removed_duplicates = 0; removed_overflow = 0; conflicts = []
    for problem in result["problems"]:
        if not isinstance(problem, dict) or not isinstance(problem.get("differential_diagnoses"), list):
            continue
        unique = []; names = set()
        for differential in problem["differential_diagnoses"]:
            name = differential.get("name") if isinstance(differential, dict) else None
            key = name.strip().lower() if isinstance(name, str) else None
            if key and key in names:
                removed_duplicates += 1
                continue
            if key:
                names.add(key)
            unique.append(differential)
        removed_overflow += max(0, len(unique) - 3)
        problem["differential_diagnoses"] = unique[:3]
        for differential in problem["differential_diagnoses"]:
            if not isinstance(differential, dict):
                continue
            supporting = differential.get("supporting_facts")
            contradicting = differential.get("contradicting_facts")
            if not isinstance(supporting, list) or not isinstance(contradicting, list):
                continue
            ambiguous = sorted(set(supporting) & set(contradicting))
            if not ambiguous:
                continue
            differential["supporting_facts"] = [fact_id for fact_id in supporting if fact_id not in ambiguous]
            differential["contradicting_facts"] = [fact_id for fact_id in contradicting if fact_id not in ambiguous]
            missing = differential.get("missing_evidence")
            if isinstance(missing, list):
                note = "以下事实方向需复核，不作为支持或反对依据：" + "、".join(ambiguous)
                if note not in missing:
                    missing.append(note)
            conflicts.append({"problem_id": problem.get("problem_id"),
                              "differential_name": differential.get("name"),
                              "fact_ids": ambiguous})
        for field in ("plan_goals", "retrieval_queries"):
            values = problem.get(field)
            if isinstance(values, list):
                problem[field] = list(dict.fromkeys(values))
    return result, {"applied": True, "max_differentials": 3,
                    "duplicates_removed": removed_duplicates, "overflow_removed": removed_overflow,
                    "fact_conflict_normalizations": len(conflicts),
                    "ambiguous_fact_ids": sorted({fact_id for item in conflicts for fact_id in item["fact_ids"]}),
                    "fact_conflicts": conflicts}


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
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value.strip() for value in values)):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.{field} 必须是非空字符串数组。")
        differentials = problem.get("differential_diagnoses")
        if not isinstance(differentials, list):
            raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.differential_diagnoses 必须是数组。")
        if len(differentials) > 3:
            raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的鉴别诊断不得超过3项。")
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
            if set(differential["supporting_facts"]) & set(differential["contradicting_facts"]):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 同一事实不能同时支持并反对同一鉴别诊断。")
            missing = differential.get("missing_evidence")
            if not isinstance(missing, list) or any(not isinstance(value, str) for value in missing):
                raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id} 的 missing_evidence 必须是字符串数组。")


def validate_plan_fact_references(plan, fact_ids: set[str]) -> None:
    """Reject unknown Fact IDs before ambiguity normalization can remove them."""
    if not isinstance(plan, dict) or not isinstance(plan.get("problems"), list):
        return
    for index, problem in enumerate(plan["problems"]):
        if not isinstance(problem, dict):
            continue
        problem_id = problem.get("problem_id") or f"problems[{index}]"
        for field in ("supporting_facts", "contradicting_facts"):
            if field in problem:
                _fact_references(problem[field], fact_ids, problem_id, field)
        differentials = problem.get("differential_diagnoses")
        if not isinstance(differentials, list):
            continue
        for differential in differentials:
            if not isinstance(differential, dict):
                continue
            for field in ("supporting_facts", "contradicting_facts"):
                if field in differential:
                    _fact_references(differential[field], fact_ids, problem_id, f"differential.{field}")


def retrieve_problem_evidence(plan: dict, retriever, top_k: int, facts: list[dict] | None = None) -> list[dict]:
    if retriever is None:
        return []
    bundles = []
    fact_map = {fact["fact_id"]: fact for fact in (facts or [])}
    for problem in plan["problems"]:
        queries = problem.get("retrieval_queries") or [problem["name"]]
        for query in queries:
            enriched_query = _contrastive_query(problem, query, fact_map) if fact_map else query
            bundles.append({
                "problem_id": problem["problem_id"],
                "query": enriched_query,
                "planner_query": query,
                "candidates": [candidate.to_dict() for candidate in
                               retriever.retrieve(enriched_query, top_k=top_k)],
            })
    return bundles


def _fact_references(values, fact_ids: set[str], problem_id: str, field: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, str) or value not in fact_ids for value in values):
        raise CaseError("INVALID_CLINICAL_PLAN", f"{problem_id}.{field} 引用了不存在的事实 ID。")


def _contrastive_query(problem: dict, query: str, fact_map: dict[str, dict]) -> str:
    """Add patient-specific support/limitation context without asserting truth."""
    support = [fact_map[fact_id].get("fact_text", "") for fact_id in problem["supporting_facts"]
               if fact_id in fact_map][:3]
    limiting = [fact_map[fact_id].get("fact_text", "") for fact_id in problem["contradicting_facts"]
                if fact_id in fact_map][:2]
    missing = []
    for differential in problem["differential_diagnoses"][:3]:
        missing.extend(differential.get("missing_evidence", []))
    parts = [query]
    if support:
        parts.append("相关患者事实 " + "；".join(text[:80] for text in support))
    if limiting:
        parts.append("限制信息 " + "；".join(text[:80] for text in limiting))
    if missing:
        parts.append("待区分资料 " + "；".join(dict.fromkeys(missing))[:160])
    return "｜".join(parts)


def _messages(system: str, payload: dict) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
