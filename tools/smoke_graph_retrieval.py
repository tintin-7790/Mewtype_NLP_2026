#!/usr/bin/env python3
"""Offline PRACTICE03 Graph/PPR integration smoke test.

This tool uses the production Case/Facts/Graph/Dense-retrieval/PPR code with a
deterministic hash-vector fixture.  It never calls Qwen, BGE, or the network and
must not be reported as a BGE-M3 quality experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_graph import build_clinical_graph
from src.clinical_planner import number_facts, retrieve_problem_evidence, validate_clinical_plan
from src.embedding_client import MockEmbeddingClient
from src.facts import organize_facts
from src.graph_retrieval import CLINICAL_RELATIONS, GraphGuidedRetriever
from src.input_adapter import load_input_cases
from src.io_utils import atomic_write_json
from src.protocol import CaseError
from src.retrieval import DenseEvidenceRetriever


DEFAULT_INPUT = (ROOT / "information" / "临床智录_练习病历数据集JSON"
                 / "03_老年病科_多病共存TIA待排.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="离线验收 PRACTICE03 的 Clinical Graph 与 PPR 重排")
    value.add_argument("--input", default=str(DEFAULT_INPUT), help="PRACTICE03 原始或正式协议 JSON")
    value.add_argument("--output", help="可选 JSON 报告；不提供时仅打印摘要")
    value.add_argument("--top-k", type=int, default=5)
    value.add_argument("--seed-k", type=int, default=5)
    value.add_argument("--ppr-iterations", type=int, default=20)
    return value


def practice03_mock_plan() -> dict:
    """Fixed audit plan.  It is a test fixture, not a medical conclusion."""
    return {"problems": [
        {
            "problem_id": "PR001", "name": "发作性右侧肢体感觉异常，TIA/脑梗死待排",
            "priority": "high", "origin": "mixed",
            "supporting_facts": ["F014", "F015", "F046", "F050", "F051"],
            "contradicting_facts": [],
            "differential_diagnoses": [
                {"name": "短暂性脑缺血发作", "certainty": "suspected",
                 "origin": "documented_problem", "supporting_facts": ["F014", "F015", "F051"],
                 "contradicting_facts": [], "missing_evidence": ["需结合神经影像及血管评估区分"]},
                {"name": "脑梗死", "certainty": "rule_out", "origin": "documented_problem",
                 "supporting_facts": ["F014", "F046", "F050", "F051"],
                 "contradicting_facts": [], "missing_evidence": ["当前资料未提供确证性影像结果"]},
                {"name": "其他发作性神经系统原因", "certainty": "possible",
                 "origin": "model_inference", "supporting_facts": ["F015"],
                 "contradicting_facts": [], "missing_evidence": ["发作期观察及针对性检查资料不足"]},
            ],
            "plan_goals": ["明确发作性质与神经系统病因"],
            "retrieval_queries": ["右侧肢体间断麻木 TIA 脑梗死待排 神经系统检查"],
        },
        {
            "problem_id": "PR002", "name": "糖代谢异常及低血糖反应待核",
            "priority": "high", "origin": "mixed",
            "supporting_facts": ["F018", "F049", "F054"], "contradicting_facts": [],
            "differential_diagnoses": [
                {"name": "2型糖尿病血糖控制不佳", "certainty": "suspected",
                 "origin": "documented_problem", "supporting_facts": ["F018", "F054"],
                 "contradicting_facts": [], "missing_evidence": ["连续血糖监测资料未提供"]},
                {"name": "低血糖反应", "certainty": "possible", "origin": "documented_problem",
                 "supporting_facts": ["F054"], "contradicting_facts": [],
                 "missing_evidence": ["症状发作时血糖值未提供"]},
            ],
            "plan_goals": ["核对血糖波动与现用降糖方案"],
            "retrieval_queries": ["糖尿病 血糖控制 低血糖 降糖药 胰岛素"],
        },
        {
            "problem_id": "PR003", "name": "高血压及慢性心功能不全",
            "priority": "medium", "origin": "documented_problem",
            "supporting_facts": ["F016", "F017", "F053"], "contradicting_facts": [],
            "differential_diagnoses": [
                {"name": "既往高血压及慢性心功能不全状态", "certainty": "suspected",
                 "origin": "documented_problem", "supporting_facts": ["F016", "F017", "F053"],
                 "contradicting_facts": [], "missing_evidence": ["近期家庭血压及心功能评估资料不足"]},
            ],
            "plan_goals": ["核对当前血压和容量状态"],
            "retrieval_queries": ["高血压 慢性心功能不全 当前用药 血压记录"],
        },
    ]}


def build_smoke_report(input_path: Path, *, top_k: int = 5, seed_k: int = 5,
                       ppr_iterations: int = 20) -> dict:
    loaded = load_input_cases(input_path)
    valid = [case for case in loaded.cases if not isinstance(case, CaseError)]
    if len(valid) != 1:
        raise RuntimeError("smoke test requires exactly one valid case")
    case = valid[0]
    facts = number_facts(organize_facts(case))
    plan = practice03_mock_plan()
    validate_clinical_plan(plan, {fact["fact_id"] for fact in facts})
    graph = build_clinical_graph(case, facts, plan)
    graph_stats = graph.stats()
    clinical_relation_count = sum(
        count for relation, count in graph_stats["relations"].items()
        if relation in CLINICAL_RELATIONS
    )
    dense = DenseEvidenceRetriever(
        case, MockEmbeddingClient(dimension=128), min_similarity=-1.0, max_chunk_chars=240,
    )
    retriever = GraphGuidedRetriever(
        dense, graph, strategy="ppr", seed_k=seed_k, ppr_iterations=ppr_iterations,
    )
    bundles = retrieve_problem_evidence(plan, retriever, top_k, facts)
    audit = retriever.stats()
    failures = []
    if clinical_relation_count == 0:
        failures.append("no_clinical_problem_fact_relations")
    if not audit["graph_executed"]:
        failures.append(audit["graph_fallback_reason"] or "ppr_not_executed")
    if not audit["selected_source_ids"]:
        failures.append("no_selected_sources")
    return {
        "status": "failed" if failures else "success",
        "case_id": case.case_id,
        "notice": (
            "离线 Mock-Dense+Graph 控制流验收；哈希向量不是 BGE-M3，也不代表语义质量或医学正确性。"
        ),
        "source_count": len(case.sources),
        "fact_count": len(facts),
        "clinical_problem_count": len(plan["problems"]),
        "node_count": graph_stats["node_count"],
        "edge_count": graph_stats["edge_count"],
        "edge_type_distribution": graph_stats["relations"],
        "clinical_relation_edge_count": clinical_relation_count,
        "embedding_provider": "MockEmbeddingClient",
        "embedding_experiment_label": "offline_mock_dense_control_not_bge_m3",
        "dense_seed_count": audit["dense_seed_count"],
        "graph_candidate_count": audit["graph_candidate_count"],
        "ppr_iterations": audit["ppr_iterations"],
        "ppr_scores": [item["ppr_scores"] for item in audit["query_audits"]],
        "ranked_fact_ids": audit["reranked_fact_ids"],
        "selected_source_ids": audit["selected_source_ids"],
        "ranking_changed": audit["ranking_changed"],
        "graph_elapsed_ms": audit["graph_elapsed_ms"],
        "fallback_status": audit["fallback_status"],
        "fallback_reason": audit["graph_fallback_reason"],
        "query_count": len(bundles),
        "query_audits": audit["query_audits"],
        "failures": failures,
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    report = build_smoke_report(
        Path(args.input), top_k=args.top_k, seed_k=args.seed_k,
        ppr_iterations=args.ppr_iterations,
    )
    if args.output:
        atomic_write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
