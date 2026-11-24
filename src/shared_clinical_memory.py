"""Case-local structured coordination memory for optional clinical experiments.

This store is never persisted and never becomes patient evidence.  It carries
only current-case IDs and compact stage decisions between Planner, retrieval,
Counter-Evidence Agent and Writer.  Verifier/Reviser deliberately do not read it.
"""

from __future__ import annotations

import time
from copy import deepcopy

from .protocol import Case


class SharedClinicalMemory:
    def __init__(self, case: Case, facts: list[dict]):
        self.case_id = case.case_id
        self._source_ids = {source.source_id for source in case.sources}
        self._fact_catalog = {
            fact["fact_id"]: {
                "source_id": fact.get("source_id"),
                "field_path": fact.get("field_path", ""),
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
                "priority": problem.get("priority"),
                "supporting_fact_ids": list(problem.get("supporting_facts", [])),
                "contradicting_fact_ids": list(problem.get("contradicting_facts", [])),
                "differential_labels": [item.get("name") for item in
                                        problem.get("differential_diagnoses", [])],
                "missing_evidence": list(dict.fromkeys(
                    value for item in problem.get("differential_diagnoses", [])
                    for value in item.get("missing_evidence", [])
                )),
                "plan_goals": list(problem.get("plan_goals", [])),
                "graph_candidate_ids": [],
                "counter_issue_ids": [],
                "produced_by": ["planner"],
            }
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
            if "counter_evidence_agent" not in card["produced_by"]:
                card["produced_by"].append("counter_evidence_agent")
        self._updates += 1
        self._elapsed_ms += (time.perf_counter() - started) * 1000

    def render(self, consumer: str) -> dict:
        started = time.perf_counter()
        self._render_count += 1
        result = {
            "notice": (
                "CASE_LOCAL_DECISION_CARDS_NOT_PATIENT_EVIDENCE；仅用于阶段协调，"
                "任何患者事实必须回到当前病例 sources 核验。"
            ),
            "case_id": self.case_id,
            "consumer": consumer,
            "decision_cards": [deepcopy(self._cards[key]) for key in sorted(self._cards)],
        }
        self._elapsed_ms += (time.perf_counter() - started) * 1000
        return result

    def audit(self) -> dict:
        return {
            "enabled": True,
            "case_id": self.case_id,
            "fact_index_count": len(self._fact_catalog),
            "source_index_count": len(self._source_ids),
            "decision_card_count": len(self._cards),
            "update_count": self._updates,
            "render_count": self._render_count,
            "elapsed_ms": round(self._elapsed_ms, 3),
            "persisted": False,
            "cross_case_scope": False,
        }
