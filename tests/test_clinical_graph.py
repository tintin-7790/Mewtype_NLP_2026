import copy
import unittest

from src.clinical_graph import build_clinical_graph
from src.clinical_planner import number_facts
from src.embedding_client import MockEmbeddingClient
from src.facts import organize_facts
from src.graph_retrieval import GraphGuidedRetriever
from src.protocol import Case, Source
from src.retrieval import DenseEvidenceRetriever


def plan():
    return {"problems":[{"problem_id":"PR001","name":"发热伴咳嗽","priority":"high",
        "origin":"mixed","supporting_facts":["F001","F002"],"contradicting_facts":[],
        "differential_diagnoses":[{"name":"肺部感染","certainty":"suspected",
            "origin":"model_inference","supporting_facts":["F001"],
            "contradicting_facts":["F003"],"missing_evidence":["培养结果未知"]}],
        "plan_goals":["明确病原"],"retrieval_queries":["发热 咳嗽 病原"]}]}


class ClinicalGraphTests(unittest.TestCase):
    def setUp(self):
        self.case=Case("C1","T0",[
            Source("S1","发热、咳嗽3天。","现病史"),
            Source("S2","胸部CT示右肺斑片影。","辅助检查.CT"),
            Source("S3","否认胸痛。","现病史"),
        ])
        self.facts=number_facts(organize_facts(self.case))

    def test_graph_preserves_provenance_and_case_boundary(self):
        graph=build_clinical_graph(self.case,self.facts,plan())
        self.assertEqual(graph.case_id,"C1")
        self.assertIn("source:S1",graph.nodes)
        self.assertTrue(any(edge.relation=="derived_from" and edge.evidence_ids==("S1",)
                            for edge in graph.edges))
        self.assertTrue(any(edge.provenance=="planner_inference" for edge in graph.edges))

    def test_conflicting_direction_is_omitted_not_decided(self):
        conflicted=copy.deepcopy(plan())
        diff=conflicted["problems"][0]["differential_diagnoses"][0]
        diff["contradicting_facts"].append("F001")
        graph=build_clinical_graph(self.case,self.facts,conflicted)
        target="differential:PR001:1"
        relations={(edge.source,edge.target,edge.relation) for edge in graph.edges}
        self.assertNotIn(("fact:F001",target,"supports"),relations)
        self.assertNotIn(("fact:F001",target,"contradicts"),relations)
        self.assertEqual(graph.ambiguities[0]["fact_ids"],["F001"])

    def test_one_hop_and_ppr_return_explainable_candidates(self):
        graph=build_clinical_graph(self.case,self.facts,plan())
        query="发热咳嗽"
        vectors={query:[1,0],"发热、咳嗽3天。":[1,0],
                 "胸部CT示右肺斑片影。":[0.8,0.2],"否认胸痛。":[0,1]}
        for strategy in ("one_hop","ppr"):
            dense=DenseEvidenceRetriever(self.case,MockEmbeddingClient(vectors),min_similarity=-1)
            retriever=GraphGuidedRetriever(dense,graph,strategy=strategy,seed_k=2)
            results=retriever.retrieve(query,top_k=2)
            self.assertEqual(results[0].unit.case_id,"C1")
            payload=results[0].to_dict()
            self.assertIn("graph_relevance",payload)
            self.assertEqual(payload["graph_strategy"],strategy)
            stats=retriever.stats()
            self.assertTrue(stats["graph_executed"])
            self.assertEqual(stats["graph_mode"],strategy)
            self.assertTrue(stats["seed_fact_ids"])
            self.assertTrue(stats["selected_source_ids"])

    def test_missing_clinical_relations_uses_disclosed_dense_fallback(self):
        graph=build_clinical_graph(self.case,self.facts,None)
        dense=DenseEvidenceRetriever(self.case,MockEmbeddingClient(dimension=16),min_similarity=-1)
        retriever=GraphGuidedRetriever(dense,graph,strategy="ppr",seed_k=2)
        results=retriever.retrieve("发热咳嗽",top_k=2)
        stats=retriever.stats()
        self.assertEqual(len(results),2)
        self.assertFalse(stats["graph_executed"])
        self.assertTrue(stats["fallback_status"])
        self.assertEqual(stats["graph_fallback_reason"],"no_clinical_problem_fact_relations")
        self.assertTrue(all(item.strategy=="dense_fallback" for item in results))


if __name__=="__main__": unittest.main()
