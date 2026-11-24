"""Deterministic problem-conditioned evidence organization for the Writer.

This module ranks and groups existing facts.  It never changes fact text and
does not decide whether a diagnosis is medically correct.
"""

from __future__ import annotations

import re
from collections import defaultdict


_ABNORMAL_TERMS = (
    "升高", "降低", "异常", "阳性", "增高", "减少", "增大", "空洞", "积液",
    "水肿", "出血", "梗死", "狭窄", "肿大", "湿啰音", "压痛", "发热",
)
_CLINICAL_PATH_TERMS = ("现病史", "体格检查", "辅助检查", "初步诊断", "既往史", "治疗")
_DEMOGRAPHIC_PATH_TERMS = ("一般项目", "姓名", "民族", "婚姻", "籍贯", "职业", "住址")
_IMPORTANT_EXPOSURES = ("接触", "暴露", "吸烟", "饮酒", "家族史", "过敏", "手术", "用药")


def evidence_salience_score(fact: dict, *, role: str = "unlinked", priority: str = "low") -> dict:
    """Return an auditable ranking score without modifying the fact."""
    text = str(fact.get("fact_text", ""))
    path = str(fact.get("field_path", ""))
    score = 0
    reasons: list[str] = []

    role_points = {"contradicting": 44, "supporting": 40, "differential": 36, "unlinked": 0}
    if role in role_points and role_points[role]:
        score += role_points[role]; reasons.append(f"problem_{role}")
    priority_points = {"high": 12, "medium": 6, "low": 0}
    score += priority_points.get(priority, 0)
    if priority_points.get(priority, 0): reasons.append(f"priority_{priority}")
    if fact.get("correction"):
        score += 28; reasons.append("correction")
    if fact.get("value") is not None and fact.get("unit"):
        score += 18; reasons.append("numeric_measurement")
    if fact.get("assertion") == "negative":
        score += 13; reasons.append("explicit_negative")
    elif fact.get("assertion") in ("uncertain", "not_provided"):
        score += 12; reasons.append("uncertainty_or_missing")
    if fact.get("status") in ("planned", "ongoing", "completed"):
        score += 10; reasons.append(f"status_{fact['status']}")
    if fact.get("time"):
        score += 8; reasons.append("time_anchored")
    if any(term in text for term in _ABNORMAL_TERMS):
        score += 12; reasons.append("abnormal_finding")
    if any(term in text for term in _IMPORTANT_EXPOSURES):
        score += 7; reasons.append("exposure_or_history")
    if any(term in path for term in _CLINICAL_PATH_TERMS):
        score += 6; reasons.append("clinical_field")
    if any(term in path for term in _DEMOGRAPHIC_PATH_TERMS):
        score -= 20; reasons.append("demographic_downrank")
    if re.search(r"(?:否认|未见|无明显|不伴)", text):
        score += 5; reasons.append("explicit_scope")
    return {"score": score, "reasons": reasons}


def build_problem_evidence_context(facts: list[dict], clinical_plan: dict,
                                   evidence_bundles: list[dict], *, max_unlinked: int = 8) -> dict:
    """Resolve Fact IDs and retrieval bundles into a Writer-oriented context."""
    fact_map = {fact["fact_id"]: fact for fact in facts}
    bundles_by_problem: dict[str, list[dict]] = defaultdict(list)
    for bundle in evidence_bundles:
        problem_id = bundle.get("problem_id")
        if isinstance(problem_id, str):
            bundles_by_problem[problem_id].append(bundle)

    linked_ids = set()
    problems = []
    for problem in clinical_plan["problems"]:
        problem_id = problem["problem_id"]
        supporting = _rank_references(problem["supporting_facts"], fact_map, "supporting", problem["priority"])
        contradicting = _rank_references(problem["contradicting_facts"], fact_map, "contradicting", problem["priority"])
        linked_ids.update(item["fact_id"] for item in supporting + contradicting)
        differentials = []
        for differential in problem["differential_diagnoses"]:
            diff_supporting = _rank_references(
                differential["supporting_facts"], fact_map, "differential", problem["priority"])
            diff_contradicting = _rank_references(
                differential["contradicting_facts"], fact_map, "contradicting", problem["priority"])
            linked_ids.update(item["fact_id"] for item in diff_supporting + diff_contradicting)
            differentials.append({
                "name": differential["name"],
                "certainty": differential["certainty"],
                "origin": differential["origin"],
                "supporting_evidence": diff_supporting,
                "limiting_evidence": diff_contradicting,
                "discriminating_evidence_needed": list(differential["missing_evidence"]),
            })
        problems.append({
            "problem_id": problem_id,
            "problem_name": problem["name"],
            "priority": problem["priority"],
            "origin": problem["origin"],
            "supporting_evidence": supporting,
            "limiting_evidence": contradicting,
            "differential_diagnoses": differentials,
            "plan_goals": list(problem["plan_goals"]),
            "retrieval_evidence": bundles_by_problem.get(problem_id, []),
        })

    unlinked = []
    for fact in facts:
        if fact["fact_id"] in linked_ids:
            continue
        ranking = evidence_salience_score(fact)
        unlinked.append(_fact_view(fact, ranking))
    unlinked.sort(key=lambda item: (-item["salience_score"], item["fact_id"]))
    return {
        "fact_catalog": {fact_id: _fact_catalog_view(fact) for fact_id, fact in fact_map.items()},
        "problems": problems,
        "salient_unlinked_facts": unlinked[:max_unlinked],
        "ranking_notice": (
            "salience仅用于组织已有事实；不得改变事实内容、诊断确定性或把检索相似度当作支持关系。"
        ),
    }


def _rank_references(fact_ids: list[str], fact_map: dict[str, dict], role: str, priority: str) -> list[dict]:
    ranked = []
    for fact_id in fact_ids:
        fact = fact_map[fact_id]
        ranked.append(_fact_view(fact, evidence_salience_score(fact, role=role, priority=priority)))
    ranked.sort(key=lambda item: (-item["salience_score"], item["fact_id"]))
    return ranked


def _fact_view(fact: dict, ranking: dict) -> dict:
    """Relationship view: text lives once in fact_catalog, not in every role."""
    return {
        "fact_id": fact["fact_id"],
        "salience_score": ranking["score"],
        "salience_reasons": ranking["reasons"],
    }


def _fact_catalog_view(fact: dict) -> dict:
    return {
        "source_id": fact.get("source_id"),
        "field_path": fact.get("field_path", ""),
        "fact_text": fact.get("fact_text", ""),
        "time": fact.get("time"),
        "value": fact.get("value"),
        "unit": fact.get("unit"),
        "assertion": fact.get("assertion"),
        "status": fact.get("status"),
        "correction": fact.get("correction"),
    }
