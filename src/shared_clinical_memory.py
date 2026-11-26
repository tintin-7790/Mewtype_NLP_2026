"""Case-local structured coordination memory for the clinical pipeline.

This store is never persisted and never becomes patient evidence.  It carries
current-case IDs, source-backed navigation references and compact stage
decisions between Planner, retrieval, Counter-Evidence Agent and Writer.
Verifier/Reviser deliberately do not read it.
"""

from __future__ import annotations

import re
import time
from copy import deepcopy

from .protocol import Case


_PLAN_PATH_TERMS = (
    "诊疗计划", "治疗计划", "处理计划", "检查计划", "后续计划", "出院医嘱", "医嘱",
)
_DIAGNOSIS_PATH_TERMS = ("初步诊断", "入院诊断", "诊断")
_HISTORY_PATH_TERMS = ("既往史", "个人史", "家族史", "手术史", "用药史", "过敏史")
_TREATMENT_TEXT_TERMS = (
    "治疗", "服用", "口服", "静滴", "输注", "泵入", "予以", "用药", "手术",
    "支架", "PCI", "透析", "吸氧", "镇痛", "镇静",
)
_DIRECT_PLAN_MARKERS = ("入院后待", "待完善", "待复查", "拟行", "拟予")
_ADDITIONAL_PLAN_RE = re.compile(
    r"(?:拟(?:复诊|复查|完善|检查|治疗|手术|监测|评估)|"
    r"(?<![无不])需(?:进一步)?(?:复诊|复查|完善|检查|治疗|手术|监测|评估)|"
    r"嘱.{0,30}(?:复诊|复查|检查|监测|随访))"
)
_PLAN_ACTION_RE = re.compile(
    r"(?<![无不])计划(?!生育).{0,80}"
    r"(?:复诊|复查|检查|监测|评估|会诊|治疗|手术|随访|用药|处置)"
)
_FOLLOWUP_RECOMMENDATION_RE = re.compile(
    r"(?:(?<!不)建议|必要时).{0,100}(?:复诊|复查|排查|检查|监测|评估|治疗|随访)"
)
_TIMED_FOLLOWUP_RE = re.compile(
    r"(?:出院后|后续|定期|\d+(?:天|日|周|月)后).{0,60}"
    r"(?:复诊|复查|随访|监测|检查)"
)
_FOLLOWUP_RESULT_RE = re.compile(
    r"(?:复诊|复查|检查|监测).{0,100}(?:结果|回报|提示|显示|"
    r"为(?:正常|异常|阴性|阳性|反应型))"
)
_ADMISSION_FOR_PLAN_RE = re.compile(
    r"为(?:进一步)?行.{0,80}(?:术|检查|治疗).{0,30}(?:收入|入院)"
)
_HISTORICAL_PLAN_RE = re.compile(
    r"(?:\d+(?:天|日|周|月|年)前|既往|曾经|曾拟|原计划|当时)"
)
_CURRENT_PLAN_TERMS = ("本次", "此次", "目前", "现拟", "入院后", "出院后", "后续")
_PENDING_PLAN_TERMS = ("尚未", "未复查", "未完成", "未予重视", "仍待")
_PENDING_OVERRIDABLE_RESOLVED_TERMS = ("未实施", "未执行", "未予实施", "未予执行")
_PLAN_RESOLVED_TERMS = (
    "已取消", "已经取消", "取消计划", "不再实施", "已完成", "已经完成",
    "已复查", "已经复查", "已实施", "已经实施", "已行", "术后出院",
    "患者拒绝", "家属拒绝", "拒绝实施", "拒绝执行", "已放弃", "放弃计划",
    "暂缓", "未实施", "未执行", "未予实施", "未予执行",
)
_FOCUS_REF_LIMITS = {
    "documented_diagnosis_fact_refs": 32,
    "explicit_plan_fact_refs": 24,
    "ongoing_treatment_fact_refs": 24,
    "completed_treatment_fact_refs": 24,
    "treatment_history_fact_refs": 32,
    "key_measurement_fact_refs": 32,
    "explicit_negative_fact_refs": 24,
    "correction_fact_refs": 16,
    "missing_or_uncertain_fact_refs": 24,
    "important_history_fact_refs": 24,
}


class SharedClinicalMemory:
    def __init__(self, case: Case, facts: list[dict]):
        self.case_id = case.case_id
        self._source_ids = {source.source_id for source in case.sources}
        self._fact_catalog = {
            fact["fact_id"]: {
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
            for fact in facts
        }
        self._cards: dict[str, dict] = {}
        self._updates = 0
        self._render_count = 0
        self._elapsed_ms = 0.0

    def ingest_plan(self, clinical_plan: dict) -> None:
        started = time.perf_counter()
        for problem in clinical_plan.get("problems", []):
            problem_id = problem["problem_id"]
            self._cards[problem_id] = {
                "problem_id": problem_id,
                "problem_name": problem.get("name"),
                "priority": problem.get("priority"),
                "origin": problem.get("origin"),
                "supporting_fact_ids": list(problem.get("supporting_facts", [])),
                "contradicting_fact_ids": list(problem.get("contradicting_facts", [])),
                "differential_diagnoses": [{
                    "name": item.get("name"),
                    "certainty": item.get("certainty"),
                    "origin": item.get("origin"),
                    "supporting_fact_ids": list(item.get("supporting_facts", [])),
                    "contradicting_fact_ids": list(item.get("contradicting_facts", [])),
                    "missing_evidence": list(item.get("missing_evidence", [])),
                } for item in problem.get("differential_diagnoses", [])],
                "differential_labels": [item.get("name") for item in
                                        problem.get("differential_diagnoses", [])],
                "missing_evidence": list(dict.fromkeys(
                    value for item in problem.get("differential_diagnoses", [])
                    for value in item.get("missing_evidence", [])
                )),
                "plan_goals": list(problem.get("plan_goals", [])),
                "graph_candidate_ids": [],
                "counter_issue_ids": [],
                "counter_evidence_review": [],
                "produced_by": ["planner"],
            }
            self._refresh_card_fact_groups(self._cards[problem_id])
        self._updates += 1
        self._elapsed_ms += (time.perf_counter() - started) * 1000

    def annotate_retrieval(self, evidence_bundles: list[dict]) -> None:
        started = time.perf_counter()
        source_to_facts: dict[str, list[str]] = {}
        for fact_id, fact in self._fact_catalog.items():
            source_to_facts.setdefault(str(fact.get("source_id")), []).append(fact_id)
        for bundle in evidence_bundles:
            card = self._cards.get(bundle.get("problem_id"))
            if card is None:
                continue
            candidate_ids = []
            for candidate in bundle.get("candidates", []):
                source_id = candidate.get("source_id")
                candidate_ids.extend(source_to_facts.get(str(source_id), []))
            card["graph_candidate_ids"] = list(dict.fromkeys(
                card["graph_candidate_ids"] + candidate_ids
            ))
            card["retrieval_candidate_fact_refs"] = self._fact_refs(
                card["graph_candidate_ids"]
            )
            if "graph" not in card["produced_by"]:
                card["produced_by"].append("graph")
        self._updates += 1
        self._elapsed_ms += (time.perf_counter() - started) * 1000

    def annotate_counter_review(self, review: dict | None) -> None:
        if not review:
            return
        started = time.perf_counter()
        for index, issue in enumerate(review.get("issues", []), start=1):
            card = self._cards.get(issue.get("problem_id"))
            if card is None:
                continue
            card["counter_issue_ids"].append(f"CI{index:03d}")
            compact_issue = {
                "counter_issue_id": f"CI{index:03d}",
                "code": issue.get("code"),
                "fact_refs": self._fact_refs(issue.get("fact_ids", [])),
                "note": issue.get("note"),
            }
            card["counter_evidence_review"].append(compact_issue)
            if "counter_evidence_agent" not in card["produced_by"]:
                card["produced_by"].append("counter_evidence_agent")
        self._updates += 1
        self._elapsed_ms += (time.perf_counter() - started) * 1000

    def render(self, consumer: str) -> dict:
        started = time.perf_counter()
        self._render_count += 1
        writer_view = consumer == "writer"
        result = {
            "notice": (
                "CASE_LOCAL_DECISION_CARDS_NOT_PATIENT_EVIDENCE；仅用于阶段协调，"
                "任何患者事实必须回到当前病例 sources 核验。"
            ),
            "case_id": self.case_id,
            "consumer": consumer,
            "decision_cards": [
                deepcopy(self._cards[key]) if writer_view
                else self._compact_coordination_card(self._cards[key])
                for key in sorted(self._cards)
            ],
        }
        if writer_view:
            result["case_clinical_focus"] = self._case_clinical_focus()
        self._elapsed_ms += (time.perf_counter() - started) * 1000
        return result

    def audit(self) -> dict:
        clinical_focus = self._case_clinical_focus()
        return {
            "enabled": True,
            "case_id": self.case_id,
            "fact_index_count": len(self._fact_catalog),
            "source_index_count": len(self._source_ids),
            "decision_card_count": len(self._cards),
            "problem_ids": sorted(self._cards),
            "clinical_focus_counts": {
                key: len(value) for key, value in clinical_focus.items()
                if isinstance(value, list)
            },
            "clinical_focus_truncated_reference_counts":
                clinical_focus["truncated_reference_counts"],
            "update_count": self._updates,
            "render_count": self._render_count,
            "elapsed_ms": round(self._elapsed_ms, 3),
            "persisted": False,
            "cross_case_scope": False,
        }

    def _refresh_card_fact_groups(self, card: dict) -> None:
        related = list(card.get("supporting_fact_ids", []))
        related.extend(card.get("contradicting_fact_ids", []))
        for differential in card.get("differential_diagnoses", []):
            related.extend(differential.get("supporting_fact_ids", []))
            related.extend(differential.get("contradicting_fact_ids", []))
        related = list(dict.fromkeys(related))
        card["supporting_fact_refs"] = self._fact_refs(card["supporting_fact_ids"])
        card["contradicting_fact_refs"] = self._fact_refs(card["contradicting_fact_ids"])
        card["key_measurement_fact_refs"] = self._fact_refs([
            fact_id for fact_id in related if self._is_measurement(fact_id)
        ])
        card["explicit_negative_fact_refs"] = self._fact_refs([
            fact_id for fact_id in related
            if self._fact_catalog.get(fact_id, {}).get("assertion") == "negative"
        ])
        card["ongoing_treatment_fact_refs"] = self._fact_refs([
            fact_id for fact_id in related if self._is_treatment(fact_id, "ongoing")
        ])
        card["completed_treatment_fact_refs"] = self._fact_refs([
            fact_id for fact_id in related if self._is_treatment(fact_id, "completed")
        ])
        card["treatment_history_fact_refs"] = self._fact_refs([
            fact_id for fact_id in related if self._has_treatment_cue(fact_id)
        ])
        card["planned_care_fact_refs"] = self._plan_refs([
            fact_id for fact_id in related if self._is_explicit_plan(fact_id)
        ])
        card["retrieval_candidate_fact_refs"] = []

    @staticmethod
    def _compact_coordination_card(card: dict) -> dict:
        """Keep the Counter-Agent view at the pre-existing compact contract."""
        keys = (
            "problem_id", "priority", "supporting_fact_ids", "contradicting_fact_ids",
            "differential_labels", "missing_evidence", "plan_goals",
            "graph_candidate_ids", "counter_issue_ids", "produced_by",
        )
        return {key: deepcopy(card.get(key)) for key in keys}

    def _case_clinical_focus(self) -> dict:
        fact_ids = list(self._fact_catalog)
        groups = {
            "documented_diagnosis_fact_refs": [
                fact_id for fact_id in fact_ids
                if any(term in str(self._fact_catalog[fact_id].get("field_path", ""))
                       for term in _DIAGNOSIS_PATH_TERMS)
            ],
            "explicit_plan_fact_refs": [
                fact_id for fact_id in fact_ids if self._is_explicit_plan(fact_id)
            ],
            "ongoing_treatment_fact_refs": [
                fact_id for fact_id in fact_ids if self._is_treatment(fact_id, "ongoing")
            ],
            "completed_treatment_fact_refs": [
                fact_id for fact_id in fact_ids if self._is_treatment(fact_id, "completed")
            ],
            "treatment_history_fact_refs": [
                fact_id for fact_id in fact_ids if self._has_treatment_cue(fact_id)
            ],
            "key_measurement_fact_refs": [
                fact_id for fact_id in fact_ids if self._is_measurement(fact_id)
            ],
            "explicit_negative_fact_refs": [
                fact_id for fact_id in fact_ids
                if self._fact_catalog[fact_id].get("assertion") == "negative"
            ],
            "correction_fact_refs": [
                fact_id for fact_id in fact_ids if self._fact_catalog[fact_id].get("correction")
            ],
            "missing_or_uncertain_fact_refs": [
                fact_id for fact_id in fact_ids
                if self._fact_catalog[fact_id].get("assertion") in ("not_provided", "uncertain")
            ],
            "important_history_fact_refs": [
                fact_id for fact_id in fact_ids
                if any(term in str(self._fact_catalog[fact_id].get("field_path", ""))
                       for term in _HISTORY_PATH_TERMS)
            ],
        }
        result = {}
        truncated = {}
        for key, group_ids in groups.items():
            limit = _FOCUS_REF_LIMITS[key]
            selected = group_ids[:limit]
            omitted = max(0, len(group_ids) - len(selected))
            if omitted:
                truncated[key] = omitted
            if key == "explicit_plan_fact_refs":
                result[key] = self._plan_refs(selected)
            else:
                result[key] = self._fact_refs(
                    selected,
                    include_text=key in {
                        "documented_diagnosis_fact_refs", "correction_fact_refs",
                    },
                )
        result["truncated_reference_counts"] = truncated
        return result

    def _fact_refs(self, fact_ids, *, include_text: bool = False) -> list[dict]:
        result = []
        for fact_id in dict.fromkeys(fact_ids or []):
            fact = self._fact_catalog.get(fact_id)
            if fact is None or fact.get("source_id") not in self._source_ids:
                continue
            ref = {"fact_id": fact_id, "source_id": fact.get("source_id")}
            if include_text:
                ref["fact_text"] = fact.get("fact_text", "")
                ref["assertion"] = fact.get("assertion")
                ref["status"] = fact.get("status")
            result.append(ref)
        return result

    def _plan_refs(self, fact_ids) -> list[dict]:
        result = []
        for fact_id in dict.fromkeys(fact_ids or []):
            fact = self._fact_catalog.get(fact_id)
            excerpt = self._explicit_plan_excerpt(fact_id)
            if (fact is None or fact.get("source_id") not in self._source_ids
                    or excerpt is None):
                continue
            result.append({
                "fact_id": fact_id,
                "source_id": fact.get("source_id"),
                "fact_text": excerpt,
                "role": "documented_plan_or_followup",
            })
        return result

    def _is_measurement(self, fact_id: str) -> bool:
        fact = self._fact_catalog.get(fact_id, {})
        return fact.get("value") is not None and bool(fact.get("unit"))

    def _is_explicit_plan(self, fact_id: str) -> bool:
        return self._explicit_plan_excerpt(fact_id) is not None

    def _explicit_plan_excerpt(self, fact_id: str) -> str | None:
        fact = self._fact_catalog.get(fact_id, {})
        return extract_explicit_plan_excerpt(
            str(fact.get("fact_text", "")),
            field_path=str(fact.get("field_path", "")),
            time=fact.get("time"),
        )

    @classmethod
    def _unresolved_plan_fragment(cls, sentence: str, *,
                                  allow_path_default: bool) -> str | None:
        starts = []
        for marker in _DIRECT_PLAN_MARKERS:
            offset = 0
            while True:
                start = sentence.find(marker, offset)
                if start < 0:
                    break
                starts.append(start)
                offset = start + len(marker)
        for pattern in (_ADDITIONAL_PLAN_RE, _PLAN_ACTION_RE,
                        _FOLLOWUP_RECOMMENDATION_RE, _TIMED_FOLLOWUP_RE,
                        _ADMISSION_FOR_PLAN_RE):
            for match in pattern.finditer(sentence):
                starts.append(match.start())
        if not starts:
            if allow_path_default and cls._plan_tail_is_unresolved(sentence):
                return sentence
            return None
        valid_starts = [start for start in starts
                        if cls._plan_tail_is_unresolved(sentence[start:])]
        if not valid_starts:
            return None
        plan_start = min(valid_starts)
        clause_start = max(
            (sentence.rfind(separator, 0, plan_start) for separator in "，,：:"),
            default=-1,
        ) + 1
        return sentence[clause_start:].strip()

    @staticmethod
    def _plan_tail_is_unresolved(tail: str) -> bool:
        if _FOLLOWUP_RESULT_RE.search(tail):
            return False
        resolved_terms = [term for term in _PLAN_RESOLVED_TERMS if term in tail]
        if any(term in tail for term in _PENDING_PLAN_TERMS):
            resolved_terms = [
                term for term in resolved_terms
                if term not in _PENDING_OVERRIDABLE_RESOLVED_TERMS
            ]
        return not resolved_terms

    @staticmethod
    def _is_stale_historical_plan(fact: dict, path: str, sentence: str,
                                  fragment: str) -> bool:
        current_or_pending = any(term in fragment for term in _CURRENT_PLAN_TERMS)
        current_or_pending = current_or_pending or any(
            term in sentence for term in _PENDING_PLAN_TERMS
        )
        if current_or_pending:
            return False
        history_path = any(term in path for term in _HISTORY_PATH_TERMS)
        historical_text = bool(_HISTORICAL_PLAN_RE.search(sentence))
        historical_time = str(fact.get("time") or "").endswith("前")
        return history_path or historical_text or historical_time

    def _is_treatment(self, fact_id: str, status: str) -> bool:
        fact = self._fact_catalog.get(fact_id, {})
        return (fact.get("status") == status
                and self._has_treatment_cue(fact_id))

    def _has_treatment_cue(self, fact_id: str) -> bool:
        fact = self._fact_catalog.get(fact_id, {})
        text = str(fact.get("fact_text", ""))
        path = str(fact.get("field_path", ""))
        return (any(term in text for term in _TREATMENT_TEXT_TERMS)
                or any(term in path for term in ("治疗", "用药", "医嘱")))


def extract_explicit_plan_excerpt(text: str, *, field_path: str = "",
                                  time=None) -> str | None:
    """Return only unresolved, current/future plan fragments from one fact.

    Shared Memory and deterministic Sources fallback deliberately share this
    implementation so completed treatment cannot become a future plan and a
    mixed historical/current sentence does not lose its pending clause.
    """
    cleaned = str(text).strip()
    if not cleaned:
        return None
    path = str(field_path or "")
    fact = {"time": time}
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", cleaned)
                 if part.strip()]
    plan_path = any(term in path for term in _PLAN_PATH_TERMS)
    matches = []
    for sentence in sentences:
        fragment = SharedClinicalMemory._unresolved_plan_fragment(
            sentence, allow_path_default=plan_path,
        )
        if fragment is None:
            continue
        if SharedClinicalMemory._is_stale_historical_plan(
                fact, path, sentence, fragment):
            continue
        matches.append(fragment)
    return "".join(matches) if matches else None
