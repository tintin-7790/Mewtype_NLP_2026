import json
import unittest

from src.model_client import Budget
from src.note_writer import run_note_writer
from src.problem_context import build_problem_evidence_context, evidence_salience_score
from src.prompt_profiles import load_clinical_prompt
from src.protocol import Case, Source


class CaptureClient:
    def call(self, messages, response_format, budget, **kwargs):
        budget.take(); self.messages = messages
        return json.dumps({"case_features":"发热。", "diagnostic_discussion":"感染原因待查。",
                           "plan":"建议结合病情进一步检查。", "evidence":[], "review_items":[]},
                          ensure_ascii=False)


class ProblemContextTests(unittest.TestCase):
    def setUp(self):
        self.facts = [
            {"fact_id":"F001", "source_id":"S1", "field_path":"入院记录.一般项目.民族",
             "fact_text":"汉族", "time":None, "value":None, "unit":None,
             "assertion":"positive", "status":"unknown", "correction":None},
            {"fact_id":"F002", "source_id":"S2", "field_path":"入院记录.辅助检查.血常规",
             "fact_text":"白细胞升高至12.5×10⁹/L", "time":"入院时", "value":"12.5", "unit":"×10⁹/L",
             "assertion":"positive", "status":"completed", "correction":None},
            {"fact_id":"F003", "source_id":"S3", "field_path":"入院记录.现病史",
             "fact_text":"否认胸痛", "time":None, "value":None, "unit":None,
             "assertion":"negative", "status":"unknown", "correction":None},
        ]
        self.plan = {"problems":[{
            "problem_id":"PR001", "name":"感染性病变", "priority":"high", "origin":"mixed",
            "supporting_facts":["F002"], "contradicting_facts":["F003"],
            "differential_diagnoses":[{"name":"细菌感染", "certainty":"suspected",
                "origin":"model_inference", "supporting_facts":["F002"],
                "contradicting_facts":[], "missing_evidence":["培养结果未提供"]}],
            "plan_goals":["明确病原并动态评估"], "retrieval_queries":["感染 炎症指标 病原"]
        }]}

    def test_problem_context_resolves_and_ranks_evidence(self):
        context = build_problem_evidence_context(self.facts, self.plan, [])
        problem = context["problems"][0]
        self.assertEqual(problem["supporting_evidence"][0]["fact_id"], "F002")
        self.assertEqual(problem["limiting_evidence"][0]["fact_id"], "F003")
        self.assertEqual(problem["differential_diagnoses"][0]["discriminating_evidence_needed"],
                         ["培养结果未提供"])
        self.assertEqual(context["salient_unlinked_facts"][0]["fact_id"], "F001")

    def test_salience_downranks_demographics_and_preserves_text(self):
        demographic = evidence_salience_score(self.facts[0])
        abnormal = evidence_salience_score(self.facts[1])
        self.assertGreater(abnormal["score"], demographic["score"])
        self.assertEqual(self.facts[1]["fact_text"], "白细胞升高至12.5×10⁹/L")

    def test_fewshot_profile_injects_only_one_matching_example(self):
        prompt, example_id = load_clinical_prompt("writer", "contrastive_fewshot", "发热咳嗽，肺部影像异常")
        self.assertEqual(example_id, "synthetic_respiratory_note")
        self.assertIn("合成病例写作示意", prompt)
        self.assertNotIn("当前短暂神经症状是高优先级问题", prompt)

    def test_writer_receives_resolved_problem_evidence_context(self):
        client = CaptureClient()
        case = Case("C", "T0", [Source("S2", "白细胞升高至12.5×10⁹/L", "辅助检查")])
        run_note_writer(case, self.facts, self.plan, [], client, "text", Budget(1), 5)
        payload = json.loads(client.messages[1]["content"])
        context = payload["problem_evidence_context"]
        self.assertEqual(context["fact_catalog"]["F002"]["source_id"], "S2")
        self.assertEqual(context["problems"][0]["supporting_evidence"][0]["fact_id"], "F002")
        self.assertEqual(context["problems"][0]["plan_goals"], ["明确病原并动态评估"])
        self.assertNotIn("facts", payload)
        self.assertNotIn("evidence_bundles", payload)

    def test_problem_context_does_not_repeat_same_long_fact_text(self):
        long_text = "重要影像异常" * 80
        self.facts[1]["fact_text"] = long_text
        context = build_problem_evidence_context(self.facts, self.plan, [])
        serialized = json.dumps(context, ensure_ascii=False)
        self.assertEqual(serialized.count(long_text), 1)

    def test_legacy_writer_keeps_original_payload_contract(self):
        client = CaptureClient()
        case = Case("C", "T0", [Source("S2", "白细胞升高至12.5×10⁹/L", "辅助检查")])
        run_note_writer(case, self.facts, self.plan, [], client, "text", Budget(1), 5,
                        prompt_profile="legacy")
        payload = json.loads(client.messages[1]["content"])
        self.assertIn("facts", payload)
        self.assertIn("evidence_bundles", payload)
        self.assertNotIn("problem_evidence_context", payload)


if __name__ == "__main__":
    unittest.main()
