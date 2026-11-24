"""Typed, case-scoped clinical evidence graph built without external services.

Patient evidence and planner hypotheses share one in-memory heterogeneous graph,
but remain distinguishable through node types, edge provenance and identifiers.
No graph relation is treated as proof that a diagnosis is true.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from itertools import combinations

from .problem_context import evidence_salience_score
from .protocol import Case


@dataclass(frozen=True)
class ClinicalNode:
    node_id: str
    node_type: str
    label: str
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ClinicalEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    provenance: str
    evidence_ids: tuple[str, ...]
    confidence: float


class ClinicalEvidenceGraph:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.nodes: dict[str, ClinicalNode] = {}
        self.edges: list[ClinicalEdge] = []
        self._adjacency: dict[str, list[ClinicalEdge]] = {}
        self.ambiguities: list[dict] = []

    def add_node(self, node: ClinicalNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting node: {node.node_id}")
        self.nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])

    def add_edge(self, source: str, target: str, relation: str, *, provenance: str,
                 evidence_ids: list[str] | tuple[str, ...], confidence: float) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("edge endpoints must exist")
        identity = f"{source}|{relation}|{target}|{len(self.edges)}"
        edge = ClinicalEdge("E:" + hashlib.sha256(identity.encode()).hexdigest()[:12],
                            source, target, relation, provenance,
                            tuple(dict.fromkeys(evidence_ids)), max(0.0, min(float(confidence), 1.0)))
        self.edges.append(edge)
        self._adjacency[source].append(edge)
        self._adjacency[target].append(edge)

    def neighbors(self, node_id: str) -> list[tuple[ClinicalNode, ClinicalEdge]]:
        result = []
        for edge in self._adjacency.get(node_id, []):
            other = edge.target if edge.source == node_id else edge.source
            result.append((self.nodes[other], edge))
        return result

    def source_node_id(self, source_id: str) -> str:
        return f"source:{source_id}"

    def fact_node_id(self, fact_id: str) -> str:
        return f"fact:{fact_id}"

    def stats(self) -> dict:
        node_types: dict[str, int] = {}
        relations: dict[str, int] = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        for edge in self.edges:
            relations[edge.relation] = relations.get(edge.relation, 0) + 1
        return {"case_id": self.case_id, "node_count": len(self.nodes),
                "edge_count": len(self.edges), "node_types": node_types,
                "relations": relations, "ambiguity_count": len(self.ambiguities)}

    def to_dict(self) -> dict:
        return {"case_id": self.case_id,
                "nodes": [asdict(node) for node in self.nodes.values()],
                "edges": [{**asdict(edge), "evidence_ids": list(edge.evidence_ids)} for edge in self.edges],
                "ambiguities": list(self.ambiguities), "stats": self.stats()}


def build_clinical_graph(case: Case, facts: list[dict], clinical_plan: dict | None = None) -> ClinicalEvidenceGraph:
    graph = ClinicalEvidenceGraph(case.case_id)
    for source in case.sources:
        graph.add_node(ClinicalNode(
            graph.source_node_id(source.source_id), "Source", source.content,
            {"source_id": source.source_id, "field_path": source.field_path,
             "source_id_origin": source.source_id_origin},
        ))
    for fact in facts:
        fact_id = fact["fact_id"]
        graph.add_node(ClinicalNode(
            graph.fact_node_id(fact_id), _fact_node_type(fact), fact.get("fact_text", ""),
            {"fact_id": fact_id, "source_id": fact.get("source_id"),
             "field_path": fact.get("field_path", ""), "time": fact.get("time"),
             "value": fact.get("value"), "unit": fact.get("unit"),
             "assertion": fact.get("assertion"), "status": fact.get("status"),
             "correction": fact.get("correction"),
             "salience": evidence_salience_score(fact)["score"]},
        ))
        source_node = graph.source_node_id(fact["source_id"])
        graph.add_edge(graph.fact_node_id(fact_id), source_node, "derived_from",
                       provenance="deterministic", evidence_ids=[fact["source_id"]], confidence=1.0)
        if fact.get("time"):
            time_id = "time:" + hashlib.sha256(str(fact["time"]).encode()).hexdigest()[:10]
            graph.add_node(ClinicalNode(time_id, "TimePoint", str(fact["time"]), {"time": fact["time"]}))
            graph.add_edge(graph.fact_node_id(fact_id), time_id, "observed_at",
                           provenance="deterministic", evidence_ids=[fact["source_id"]], confidence=0.95)
        if fact.get("correction"):
            graph.add_edge(graph.fact_node_id(fact_id), source_node, "corrects_record",
                           provenance="deterministic", evidence_ids=[fact["source_id"]], confidence=1.0)
    if clinical_plan:
        _add_problem_layer(graph, clinical_plan, facts)
    return graph


def _add_problem_layer(graph: ClinicalEvidenceGraph, plan: dict, facts: list[dict]) -> None:
    fact_map = {fact["fact_id"]: fact for fact in facts}
    for problem in plan.get("problems", []):
        problem_id = problem["problem_id"]
        problem_node = f"problem:{problem_id}"
        graph.add_node(ClinicalNode(problem_node, "ClinicalProblem", problem["name"],
                                    {"problem_id": problem_id, "priority": problem["priority"],
                                     "origin": problem["origin"]}))
        _add_directional_fact_edges(graph, problem_node, problem.get("supporting_facts", []),
                                    problem.get("contradicting_facts", []), problem_id, fact_map)
        for index, differential in enumerate(problem.get("differential_diagnoses", []), 1):
            diff_node = f"differential:{problem_id}:{index}"
            graph.add_node(ClinicalNode(diff_node, "DifferentialDiagnosis", differential["name"],
                                        {"certainty": differential["certainty"],
                                         "origin": differential["origin"]}))
            graph.add_edge(problem_node, diff_node, "considers", provenance="planner_inference",
                           evidence_ids=[], confidence=0.6)
            _add_directional_fact_edges(graph, diff_node, differential.get("supporting_facts", []),
                                        differential.get("contradicting_facts", []), problem_id, fact_map)
            for missing_index, missing in enumerate(differential.get("missing_evidence", []), 1):
                missing_node = f"missing:{problem_id}:{index}:{missing_index}"
                graph.add_node(ClinicalNode(missing_node, "MissingEvidence", missing, {}))
                graph.add_edge(missing_node, diff_node, "discriminates", provenance="planner_inference",
                               evidence_ids=[], confidence=0.5)
        for goal_index, goal in enumerate(problem.get("plan_goals", []), 1):
            goal_node = f"goal:{problem_id}:{goal_index}"
            graph.add_node(ClinicalNode(goal_node, "PlanGoal", goal, {}))
            graph.add_edge(problem_node, goal_node, "motivates_plan", provenance="planner_inference",
                           evidence_ids=[], confidence=0.6)
        _connect_problem_sources(graph, problem, fact_map, problem_id)


def _add_directional_fact_edges(graph: ClinicalEvidenceGraph, target: str, supporting: list[str],
                                contradicting: list[str], problem_id: str, fact_map: dict[str, dict]) -> None:
    ambiguous = sorted(set(supporting) & set(contradicting))
    if ambiguous:
        graph.ambiguities.append({"problem_id": problem_id, "target_node": target,
                                  "fact_ids": ambiguous, "action": "omitted_both_directions"})
    for relation, values in (("supports", supporting), ("contradicts", contradicting)):
        for fact_id in values:
            if fact_id in ambiguous or fact_id not in fact_map:
                continue
            fact = fact_map[fact_id]
            graph.add_edge(graph.fact_node_id(fact_id), target, relation,
                           provenance="planner_inference", evidence_ids=[fact["source_id"]], confidence=0.65)


def _connect_problem_sources(graph: ClinicalEvidenceGraph, problem: dict, fact_map: dict[str, dict],
                             problem_id: str) -> None:
    fact_ids = list(problem.get("supporting_facts", [])) + list(problem.get("contradicting_facts", []))
    for differential in problem.get("differential_diagnoses", []):
        fact_ids.extend(differential.get("supporting_facts", []))
        fact_ids.extend(differential.get("contradicting_facts", []))
    source_ids = sorted({fact_map[fact_id]["source_id"] for fact_id in fact_ids if fact_id in fact_map})
    for left, right in combinations(source_ids, 2):
        graph.add_edge(graph.source_node_id(left), graph.source_node_id(right), "co_evidence",
                       provenance="planner_inference", evidence_ids=[left, right], confidence=0.45)


def _fact_node_type(fact: dict) -> str:
    path = str(fact.get("field_path", ""))
    text = str(fact.get("fact_text", ""))
    if any(term in path for term in ("血常规", "生化", "CRP", "实验室", "检验")):
        return "LaboratoryResult"
    if any(term in path for term in ("辅助检查", "体格检查", "CT", "MRI", "超声", "生命体征")):
        return "Examination"
    if any(term in text for term in ("接触", "暴露", "吸烟", "饮酒", "鸽")):
        return "Exposure"
    if any(term in path for term in ("主诉", "现病史")):
        return "Symptom"
    return "Fact"
