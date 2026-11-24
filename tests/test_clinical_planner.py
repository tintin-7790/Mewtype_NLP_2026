import copy
import json
import unittest

from src.clinical_planner import (apply_contrastive_planning_policy, number_facts,
                                  retrieve_problem_evidence, run_planner, validate_clinical_plan,
                                  validate_plan_fact_references)
from src.embedding_client import MockEmbeddingClient
from src.errors import CaseError
from src.model_client import Budget
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

    def test_patient_facts_enrich_retrieval_query_without_changing_planner_query(self):
        source = "发热、咳嗽3天，白细胞升高。"
        case = Case("A", "T0", [Source("S1", source, "现病史")])
        client = MockEmbeddingClient(dimension=8)
        retriever = DenseEvidenceRetriever(case, client, min_similarity=-1)
        facts = [{"fact_id":"F001", "fact_text":"发热、咳嗽3天", "source_id":"S1"}]
        bundles = retrieve_problem_evidence(valid_plan(), retriever, 1, facts)
        self.assertEqual(bundles[0]["planner_query"], "发热 咳嗽 炎症指标")
        self.assertIn("相关患者事实", bundles[0]["query"])
        self.assertIn("发热、咳嗽3天", bundles[0]["query"])

    def test_writer_shape_is_rejected_as_planner_output(self):
        writer = {"case_features":"x", "diagnostic_discussion":"y", "plan":"z",
                  "evidence":[], "review_items":[]}
        with self.assertRaises(CaseError):
            validate_clinical_plan(writer, {"F001"})

    def test_contrastive_policy_deduplicates_and_caps_differentials(self):
        plan = valid_plan(); base = plan["problems"][0]["differential_diagnoses"][0]
        plan["problems"][0]["differential_diagnoses"] = [
            dict(base), dict(base), {**base, "name":"结核"}, {**base, "name":"真菌感染"},
            {**base, "name":"罕见感染"},
        ]
        normalized, report = apply_contrastive_planning_policy(plan)
        names = [item["name"] for item in normalized["problems"][0]["differential_diagnoses"]]
        self.assertEqual(names, ["肺部感染性病变", "结核", "真菌感染"])
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["overflow_removed"], 1)

    def test_ambiguous_fact_is_removed_from_both_directions_and_reported(self):
        original = valid_plan()
        original["problems"][0]["differential_diagnoses"][0]["contradicting_facts"] = ["F001"]
        untouched = copy.deepcopy(original)
        normalized, report = apply_contrastive_planning_policy(original)
        differential = normalized["problems"][0]["differential_diagnoses"][0]
        self.assertEqual(differential["supporting_facts"], [])
        self.assertEqual(differential["contradicting_facts"], [])
        self.assertIn("F001", differential["missing_evidence"][-1])
        self.assertEqual(report["fact_conflict_normalizations"], 1)
        self.assertEqual(report["ambiguous_fact_ids"], ["F001"])
        self.assertEqual(original, untouched)
        validate_clinical_plan(normalized, {"F001"})

    def test_empty_contradicting_facts_is_valid(self):
        validate_clinical_plan(valid_plan(), {"F001"})

    def test_unknown_fact_id_is_rejected_before_conflict_normalization(self):
        plan = valid_plan()
        differential = plan["problems"][0]["differential_diagnoses"][0]
        differential["supporting_facts"] = ["F999"]
        differential["contradicting_facts"] = ["F999"]
        with self.assertRaises(CaseError):
            validate_plan_fact_references(plan, {"F001"})

    def test_legacy_planner_keeps_raw_fact_payload(self):
        class CapturePlannerClient:
            def call(inner_self, messages, response_format, budget, **kwargs):
                budget.take(); inner_self.messages = messages
                return json.dumps(valid_plan(), ensure_ascii=False)

        client = CapturePlannerClient()
        case = Case("A", "T0", [Source("S1", "发热", "现病史")])
        facts = [{"fact_id":"F001", "source_id":"S1", "field_path":"现病史",
                  "fact_text":"发热", "raw_text":"原始长文本"}]
        run_planner(case, facts, client, "text", Budget(1), prompt_profile="legacy")
        payload = json.loads(client.messages[1]["content"])
        self.assertEqual(payload["facts"][0]["raw_text"], "原始长文本")


if __name__ == "__main__": unittest.main()
