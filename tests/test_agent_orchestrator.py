import json
import tempfile
import unittest
from pathlib import Path

from src.agent_orchestrator import (assess_agent_need, run_counter_evidence_agent,
                                    validate_counter_evidence_review)
from src.app import main
from src.errors import CaseError
from src.model_client import Budget
from src.protocol import Case, Source


def _planner():
    return {"problems": [{"problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
        "origin": "mixed", "supporting_facts": ["F001"], "contradicting_facts": [],
        "differential_diagnoses": [{"name": "感染性病变", "certainty": "suspected",
            "origin": "model_inference", "supporting_facts": ["F001"],
            "contradicting_facts": [], "missing_evidence": ["病原学结果"]}],
        "plan_goals": ["明确病因"], "retrieval_queries": ["发热 咳嗽"]}]}


def _draft():
    return {"case_features": "患者发热伴咳嗽。",
            "diagnostic_discussion": "现有资料提示感染性病变可能，病因待查。",
            "plan": "建议结合临床情况完善病原学检查，需医生确认。",
            "evidence": [{"statement": "患者发热伴咳嗽。", "source_ids": ["S1"], "kind": "fact"}],
            "review_items": ["病原学结果未提供。"]}


def _review():
    return {"reviewed_problem_ids":["PR001"], "issues":[{
        "problem_id":"PR001", "code":"MISSING_DISCRIMINATING_EVIDENCE",
        "fact_ids":[], "note":"病原学证据有限"}]}


def _verification():
    return {"checked_claim_ids":["C001","C002","C003","C004","C005"], "issues":[]}


class AgentOrchestratorTests(unittest.TestCase):
    def test_adaptive_gate_skips_simple_case_and_triggers_complex_case(self):
        facts = [{"fact_id": "F001", "assertion": "positive", "correction": None}]
        self.assertFalse(assess_agent_need(facts, _planner())["triggered"])
        complex_plan = _planner()
        complex_plan["problems"][0]["differential_diagnoses"] *= 4
        self.assertTrue(assess_agent_need(facts, complex_plan)["triggered"])

    def test_unknown_fact_reference_is_rejected(self):
        review = {"reviewed_problem_ids":["PR001"], "issues": [{"problem_id": "PR001",
            "code":"COUNTER_EVIDENCE", "fact_ids": ["F999"], "note": "需复核"}]}
        with self.assertRaises(CaseError):
            validate_counter_evidence_review(review, _planner(), {"F001"})

    def test_incomplete_problem_coverage_is_rejected(self):
        with self.assertRaises(CaseError) as raised:
            validate_counter_evidence_review({"reviewed_problem_ids":[], "issues":[]},
                                             _planner(), {"F001"})
        self.assertEqual(raised.exception.code, "INCOMPLETE_COUNTER_EVIDENCE_COVERAGE")

    def test_agent_payload_uses_compact_fact_catalog_without_duplicate_source_text(self):
        class CaptureClient:
            def __init__(self): self.messages=None; self.call_events=[]
            def call(self,messages,response_format,budget,*,stage="model",schema=None):
                budget.take(); self.messages=messages
                self.call_events.append({"stage":stage,"call_success":True,
                    "json_parse_success":None,"schema_validation_success":None})
                return json.dumps({"reviewed_problem_ids":["PR001"],"issues":[]},ensure_ascii=False)
            def record_parse_result(self,stage,success,error_type=None,position=None):
                self.call_events[-1]["json_parse_success"]=success
            def record_schema_result(self,stage,success,error_type=None):
                self.call_events[-1]["schema_validation_success"]=success
        client=CaptureClient(); case=Case("C","T0",[Source("S1","患者发热伴咳嗽。","现病史")])
        facts=[{"fact_id":"F001","source_id":"S1","field_path":"现病史",
                "fact_text":"患者发热伴咳嗽。","time":None,"value":None,"unit":None,
                "assertion":"positive","status":"unknown","correction":None}]
        review=run_counter_evidence_agent(case,facts,_planner(),client,"json_schema",Budget(1))
        payload=json.loads(client.messages[1]["content"])
        self.assertEqual(review["issues"],[])
        self.assertNotIn("content",payload["source_catalog"][0])
        self.assertEqual(payload["facts"][0]["fact_text"],"患者发热伴咳嗽。")
        self.assertNotIn("clinical_plan",payload)
        self.assertTrue(client.call_events[-1]["schema_validation_success"])

    def test_always_agent_and_graph_retrieval_are_logged(self):
        review = {"reviewed_problem_ids":["PR001"], "issues": [{"problem_id": "PR001",
            "code":"MISSING_DISCRIMINATING_EVIDENCE", "fact_ids": [], "note": "病因证据有限"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"; output = root / "output.json"
            log = root / "log.json"; responses = root / "responses.json"
            retrieval = root / "retrieval.json"
            input_path.write_text(json.dumps([{"case_id": "C1", "record_cutoff": "T0",
                "sources": [{"source_id": "S1", "content": "患者发热伴咳嗽。"}]}],
                ensure_ascii=False), encoding="utf-8")
            responses.write_text(json.dumps([{"content": json.dumps(value, ensure_ascii=False)}
                for value in (_planner(), review, _draft())], ensure_ascii=False), encoding="utf-8")
            retrieval.write_text(json.dumps({"provider": "mock", "dimension": 16, "top_k": 1,
                                             "min_similarity": -1, "max_chunk_chars": 240}),
                                 encoding="utf-8")
            code = main(["--input", str(input_path), "--output", str(output), "--log", str(log),
                         "--mode", "mock", "--mock-responses", str(responses),
                         "--pipeline", "clinical_pipeline", "--clinical-verification", "off",
                         "--clinical-revision", "off", "--clinical-agents", "always",
                         "--counter-evidence-response-format", "json_schema",
                         "--retrieval-config", str(retrieval), "--graph-retrieval", "one_hop"])
            saved = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())[0]["status"], "success")
            stages = [event["stage"] for event in saved["cases"][0]["stage_calls"]]
            self.assertEqual(stages, ["planner", "counter_evidence", "writer"])
            self.assertEqual(saved["cases"][0]["stage_calls"][1]["response_format"], "json_schema")
            self.assertTrue(saved["cases"][0]["stage_calls"][1]["schema_validation_success"])
            clinical = saved["cases"][0]["clinical"]
            self.assertTrue(clinical["adaptive_agent"]["completed"])
            self.assertEqual(clinical["adaptive_agent"]["reviewed_problem_count"], 1)
            self.assertEqual(clinical["adaptive_agent"]["issue_count"], 1)
            self.assertEqual(clinical["graph_retrieval"]["mode"], "one_hop")
            self.assertGreater(clinical["graph_retrieval"]["graph"]["node_count"], 0)

    def test_invalid_optional_agent_review_does_not_block_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_path = root / "input.json"; output = root / "output.json"
            log = root / "log.json"; responses = root / "responses.json"
            input_path.write_text(json.dumps([{"case_id": "C1", "record_cutoff": "T0",
                "sources": [{"source_id": "S1", "content": "患者发热伴咳嗽。"}]}],
                ensure_ascii=False), encoding="utf-8")
            responses.write_text(json.dumps([{"content": json.dumps(value, ensure_ascii=False)}
                for value in (_planner(), {"reviewed_problem_ids":["UNKNOWN"], "issues":[]}, _draft())],
                ensure_ascii=False), encoding="utf-8")
            code = main(["--input", str(input_path), "--output", str(output), "--log", str(log),
                         "--mode", "mock", "--mock-responses", str(responses),
                         "--pipeline", "clinical_pipeline", "--clinical-verification", "off",
                         "--clinical-revision", "off", "--clinical-agents", "always"])
            saved = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())[0]["status"], "success")
            agent = saved["cases"][0]["clinical"]["adaptive_agent"]
            self.assertFalse(agent["completed"])
            self.assertEqual(agent["error"]["code"], "INCOMPLETE_COUNTER_EVIDENCE_COVERAGE")

    def test_truncated_agent_is_disclosed_and_writer_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_path = root/"input.json"; output = root/"output.json"
            log = root/"log.json"; responses = root/"responses.json"
            input_path.write_text(json.dumps([{"case_id":"C1", "record_cutoff":"T0",
                "sources":[{"source_id":"S1", "content":"患者发热伴咳嗽。"}]}],
                ensure_ascii=False), encoding="utf-8")
            entries = [
                {"content":json.dumps(_planner(), ensure_ascii=False)},
                {"content":'{"reviewed_problem_ids":["PR001"],"issues":[{"problem_id":"PR001"',
                 "finish_reason":"length", "max_tokens":2048,
                 "usage":{"prompt_tokens":100, "completion_tokens":2048, "total_tokens":2148}},
                {"content":json.dumps(_draft(), ensure_ascii=False)},
            ]
            responses.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            code = main(["--input",str(input_path),"--output",str(output),"--log",str(log),
                         "--mode","mock","--mock-responses",str(responses),
                         "--pipeline","clinical_pipeline","--clinical-verification","off",
                         "--clinical-revision","off","--clinical-agents","always"])
            saved=json.loads(log.read_text()); agent=saved["cases"][0]["clinical"]["adaptive_agent"]
            self.assertEqual(code,0); self.assertTrue(agent["triggered"]); self.assertFalse(agent["completed"])
            self.assertTrue(agent["error"]["output_truncation_suspected"])
            event=saved["cases"][0]["stage_calls"][1]
            self.assertEqual(event["parse_error_type"],"INCOMPLETE_JSON")
            self.assertEqual(event["finish_reason"],"length")

    def test_full_d_agent_compact_control_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); input_path=root/"input.json"; output=root/"output.json"
            log=root/"log.json"; responses=root/"responses.json"
            input_path.write_text(json.dumps([{"case_id":"C1","record_cutoff":"T0",
                "sources":[{"source_id":"S1","content":"患者发热伴咳嗽。"}]}],ensure_ascii=False),encoding="utf-8")
            responses.write_text(json.dumps([{"content":json.dumps(value,ensure_ascii=False)}
                for value in (_planner(),_review(),_draft(),_verification())],ensure_ascii=False),encoding="utf-8")
            code=main(["--input",str(input_path),"--output",str(output),"--log",str(log),
                "--mode","mock","--mock-responses",str(responses),"--pipeline","clinical_pipeline",
                "--clinical-agents","always","--clinical-verification","on","--clinical-revision","on"])
            saved=json.loads(log.read_text()); clinical=saved["cases"][0]["clinical"]
            self.assertEqual(code,0); self.assertTrue(clinical["adaptive_agent"]["completed"])
            self.assertTrue(clinical["verification_completed"])
            self.assertEqual(clinical["verification_coverage"],1.0)

    def test_final_evidence_failure_keeps_clinical_progress_log(self):
        bad=_draft(); bad["evidence"]=[{"statement":"诊断阈值为10μmol/L。",
            "source_ids":["S1"],"kind":"inference"}]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); input_path=root/"input.json"; output=root/"output.json"
            log=root/"log.json"; responses=root/"responses.json"
            input_path.write_text(json.dumps([{"case_id":"C1","record_cutoff":"T0",
                "sources":[{"source_id":"S1","content":"总胆汁酸14.2μmol/L。"}]}],ensure_ascii=False),encoding="utf-8")
            responses.write_text(json.dumps([{"content":json.dumps(value,ensure_ascii=False)}
                for value in (_planner(),_review(),bad,_verification())],ensure_ascii=False),encoding="utf-8")
            code=main(["--input",str(input_path),"--output",str(output),"--log",str(log),
                "--mode","mock","--mock-responses",str(responses),"--pipeline","clinical_pipeline",
                "--clinical-agents","always","--clinical-verification","on","--clinical-revision","on"])
            result=json.loads(output.read_text())[0]; case_log=json.loads(log.read_text())["cases"][0]
            self.assertEqual(code,0); self.assertEqual(result["status"],"failed")
            self.assertEqual(case_log["failed_stage"],"final_validation")
            self.assertIsInstance(case_log["elapsed_ms"],float)
            self.assertTrue(case_log["clinical"]["adaptive_agent"]["completed"])
            self.assertTrue(case_log["clinical"]["verification_completed"])
            self.assertEqual(case_log["clinical"]["final_evidence_validation"]["error_count"],1)


if __name__ == "__main__":
    unittest.main()
