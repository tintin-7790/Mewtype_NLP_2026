"""Graph-guided re-ranking over the existing case-scoped dense retriever."""

from __future__ import annotations

from dataclasses import dataclass

from .clinical_graph import ClinicalEvidenceGraph
from .retrieval import DenseEvidenceRetriever, EvidenceCandidate


RELATION_WEIGHT = {
    "derived_from": 1.0, "observed_at": 0.35, "corrects_record": 1.0,
    "supports": 0.8, "contradicts": 0.9, "considers": 0.45,
    "discriminates": 0.4, "motivates_plan": 0.35, "co_evidence": 0.65,
}


@dataclass(frozen=True)
class GraphEvidenceCandidate:
    dense: EvidenceCandidate
    ranking_score: float
    graph_relevance: float
    clinical_salience: float
    source_priority: float
    strategy: str

    @property
    def unit(self):
        return self.dense.unit

    @property
    def compatible(self):
        return self.dense.compatible

    def to_dict(self) -> dict:
        return {**self.dense.to_dict(), "ranking_score": round(self.ranking_score, 6),
                "graph_relevance": round(self.graph_relevance, 6),
                "clinical_salience": round(self.clinical_salience, 6),
                "source_priority": round(self.source_priority, 6),
                "graph_strategy": self.strategy}


class GraphGuidedRetriever:
    """Dense seeds followed by one-hop or Personalized PageRank re-ranking.

    The graph never crosses cases because both the dense retriever and graph are
    constructed for exactly one Case instance.
    """

    def __init__(self, dense: DenseEvidenceRetriever, graph: ClinicalEvidenceGraph, *,
                 strategy: str = "one_hop", seed_k: int = 5, ppr_alpha: float = 0.2,
                 ppr_iterations: int = 20):
        if strategy not in ("one_hop", "ppr"):
            raise ValueError("strategy must be one_hop or ppr")
        if graph.case_id != dense.case.case_id:
            raise ValueError("graph and retriever must belong to the same case")
        self.dense = dense; self.graph = graph; self.strategy = strategy
        self.seed_k = max(1, seed_k); self.ppr_alpha = ppr_alpha
        self.ppr_iterations = max(1, ppr_iterations); self.query_count = 0

    def retrieve(self, statement: str, top_k: int = 3) -> list[GraphEvidenceCandidate]:
        if top_k <= 0:
            return []
        self.query_count += 1
        dense_all = self.dense.retrieve(statement, top_k=max(top_k, len(self.dense.units)))
        if not dense_all:
            return []
        seeds = dense_all[:min(self.seed_k, len(dense_all))]
        seed_scores = _seed_scores(seeds)
        graph_scores = (self._one_hop(seed_scores) if self.strategy == "one_hop"
                        else self._personalized_page_rank(seed_scores))
        candidates = []
        for item in dense_all:
            source_node = self.graph.source_node_id(item.unit.source_id)
            dense_score = max(0.0, min(1.0, (item.similarity + 1.0) / 2.0))
            graph_score = graph_scores.get(source_node, 0.0)
            salience = _source_salience(self.graph, item.unit.source_id)
            priority = _source_priority(item.unit.field_path)
            penalty = 0.2 * len(item.compatibility_warnings)
            score = 0.50 * dense_score + 0.25 * graph_score + 0.15 * salience + 0.10 * priority - penalty
            candidates.append(GraphEvidenceCandidate(item, score, graph_score, salience, priority, self.strategy))
        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            used_sources = {item.unit.source_id for item in selected}
            best = max(remaining, key=lambda item: (
                item.ranking_score - (0.08 if item.unit.source_id in used_sources else 0.0),
                item.dense.similarity, -item.unit.ordinal,
            ))
            selected.append(best); remaining.remove(best)
        return selected

    def stats(self) -> dict:
        return {"strategy": self.strategy, "queries": self.query_count,
                "seed_k": self.seed_k, "ppr_alpha": self.ppr_alpha,
                "ppr_iterations": self.ppr_iterations, "graph": self.graph.stats()}

    def _one_hop(self, seeds: dict[str, float]) -> dict[str, float]:
        scores = dict(seeds)
        for source_node, seed_score in seeds.items():
            for neighbor, edge in self.graph.neighbors(source_node):
                if neighbor.node_type == "Source":
                    scores[neighbor.node_id] = max(scores.get(neighbor.node_id, 0.0),
                                                   seed_score * RELATION_WEIGHT.get(edge.relation, 0.2))
        return _normalise(scores)

    def _personalized_page_rank(self, seeds: dict[str, float]) -> dict[str, float]:
        if not seeds:
            return {}
        personalization = _normalise_sum(seeds)
        scores = dict(personalization)
        for _ in range(self.ppr_iterations):
            updated = {node_id: self.ppr_alpha * personalization.get(node_id, 0.0)
                       for node_id in self.graph.nodes}
            for node_id, score in scores.items():
                neighbors = self.graph.neighbors(node_id)
                weights = [RELATION_WEIGHT.get(edge.relation, 0.2) * edge.confidence
                           for _, edge in neighbors]
                total = sum(weights)
                if not total:
                    updated[node_id] = updated.get(node_id, 0.0) + (1 - self.ppr_alpha) * score
                    continue
                for (neighbor, _), weight in zip(neighbors, weights):
                    updated[neighbor.node_id] = updated.get(neighbor.node_id, 0.0) + \
                        (1 - self.ppr_alpha) * score * weight / total
            scores = updated
        return _normalise({node_id: score for node_id, score in scores.items()
                           if self.graph.nodes[node_id].node_type == "Source"})


def _seed_scores(candidates: list[EvidenceCandidate]) -> dict[str, float]:
    result = {}
    for item in candidates:
        node_id = f"source:{item.unit.source_id}"
        result[node_id] = max(result.get(node_id, 0.0), max(0.0, (item.similarity + 1.0) / 2.0))
    return result


def _source_salience(graph: ClinicalEvidenceGraph, source_id: str) -> float:
    values = [float(node.attributes.get("salience", 0)) for node in graph.nodes.values()
              if node.node_id.startswith("fact:") and node.attributes.get("source_id") == source_id]
    return max(0.0, min(1.0, max(values, default=0.0) / 100.0))


def _source_priority(field_path: str) -> float:
    if any(term in field_path for term in ("辅助检查", "体格检查", "现病史", "主诉", "初步诊断")):
        return 1.0
    if any(term in field_path for term in ("既往史", "个人史", "家族史")):
        return 0.7
    if any(term in field_path for term in ("姓名", "婚姻", "民族", "住址", "职业")):
        return 0.1
    return 0.4


def _normalise(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {key: (value / maximum if maximum else 0.0) for key, value in values.items()}


def _normalise_sum(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total else {}
