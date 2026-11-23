import unittest

from src.clinical_planner import number_facts, retrieve_problem_evidence, validate_clinical_plan
from src.embedding_client import MockEmbeddingClient
from src.errors import CaseError
from src.protocol import Case, Source
from src.retrieval import DenseEvidenceRetriever


def valid_plan():
    return {"problems": [{"problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
        "origin": "mixed", "supporting_facts": ["F001"], "contradicting_facts": [],
        "differential_diagnoses": [{"name": "肺部感染性病变", "certainty": "suspected",
            "origin": "model_inference", "supporting_facts": ["F001"], "contradicting_facts": [],
            "missing_evidence": ["胸部影像结果未提供"]}],
        "plan_goals": ["明确感染部位"], "retrieval_queries": ["发热 咳嗽 炎症指标"]}]}


class ClinicalPlannerTests(unittest.TestCase):
    def test_number_facts_and_validate_references(self):
        facts = number_facts([{"source_id": "S1", "fact_text": "发热"},
                              {"source_id": "S2", "fact_text": "咳嗽"}])
        self.assertEqual([item["fact_id"] for item in facts], ["F001", "F002"])
        validate_clinical_plan(valid_plan(), {"F001", "F002"})
        broken = valid_plan(); broken["problems"][0]["supporting_facts"] = ["F999"]
        with self.assertRaises(CaseError): validate_clinical_plan(broken, {"F001"})

    def test_problem_queries_retrieve_case_scoped_evidence(self):
        query = "发热 咳嗽 炎症指标"; source = "发热、咳嗽3天，白细胞12.5×10^9/L。"
        case = Case("A", "T0", [Source("S1", source, "现病史")])
        retriever = DenseEvidenceRetriever(case, MockEmbeddingClient({query: [1, 0], source: [1, 0]}))
        bundles = retrieve_problem_evidence(valid_plan(), retriever, 1)
        self.assertEqual(bundles[0]["problem_id"], "PR001")
        self.assertEqual(bundles[0]["candidates"][0]["case_id"], "A")
        self.assertEqual(bundles[0]["candidates"][0]["source_id"], "S1")

    def test_writer_shape_is_rejected_as_planner_output(self):
        writer = {"case_features":"x", "diagnostic_discussion":"y", "plan":"z",
                  "evidence":[], "review_items":[]}
        with self.assertRaises(CaseError):
            validate_clinical_plan(writer, {"F001"})


if __name__ == "__main__": unittest.main()
