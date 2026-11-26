import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.claim_extractor import extract_claims
from src.competition_profile import ACTIVE_COMPETITION_PROFILE, PROFILES


ROOT = Path(__file__).resolve().parent.parent


def _planner(fact_ids, problem="主要临床问题"):
    return {"problems": [{
        "problem_id": "PR001", "name": problem, "priority": "high",
        "origin": "mixed", "supporting_facts": list(fact_ids),
        "contradicting_facts": [], "differential_diagnoses": [],
        "plan_goals": ["明确病情并制定安全方案"],
        "retrieval_queries": [problem],
    }]}


def _agent():
    return {"reviewed_problem_ids": ["PR001"], "issues": []}


def _verifier(record, issues=None):
    return {"checked_claim_ids": [claim["claim_id"] for claim in extract_claims(record)],
            "issues": issues or []}


class FullCompetitionProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        self.package.mkdir()
        for filename in ("main.py", "requires.txt", "requirements.txt", ".env.example",
                         "README.md", "resource_manifest.json"):
            shutil.copy2(ROOT / filename, self.package / filename)
        for directory in ("src", "prompts", "resources"):
            shutil.copytree(ROOT / directory, self.package / directory)
        self.requests = []
        self.responses = []

        requests = self.requests
        responses = self.responses

        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                size = int(inner.headers.get("Content-Length", "0"))
                body = json.loads(inner.rfile.read(size).decode("utf-8"))
                requests.append({"path": inner.path,
                                 "authorization": inner.headers.get("Authorization"),
                                 "body": body})
                if not responses:
                    inner.send_error(500, "mock response queue exhausted")
                    return
                scenario = responses.pop(0)
                payload = {
                    "choices": [{"message": {"content": scenario}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                              "total_tokens": 150},
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                inner.send_response(200)
                inner.send_header("Content-Type", "application/json")
                inner.send_header("Content-Length", str(len(encoded)))
                inner.end_headers()
                inner.wfile.write(encoded)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        (self.package / ".env").write_text(
            f"MODEL_NAME=official-mock-qwen\nENDPOINT={endpoint}\n", encoding="utf-8")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def _run(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.package)
        environment["MEWTYPE_DEV_MODE"] = "1"
        environment["DASHSCOPE_API_KEY"] = "must-not-be-used"
        environment["QWEN_BASE_URL"] = "https://external.invalid"
        environment["QWEN_MODEL"] = "must-not-be-used"
        return subprocess.run([sys.executable, "main.py"], cwd=self.package,
                              capture_output=True, text=True, timeout=30, env=environment)

    def _queue(self, *values):
        self.responses.extend(json.dumps(value, ensure_ascii=False) if not isinstance(value, str)
                              else value for value in values)

    def _read(self):
        output = json.loads((self.package / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.package / "run_log.json").read_text(encoding="utf-8"))
        return output, log

    def test_profile_contract(self):
        self.assertEqual(ACTIVE_COMPETITION_PROFILE, "competition_full_v2")
        profile = PROFILES["competition_full_v2"]
        self.assertEqual(profile.pipeline, "clinical_pipeline")
        self.assertEqual((profile.clinical_prompt_profile, profile.retrieval_strategy,
                          profile.graph_retrieval), ("contrastive_v2", "lexical_graph", "ppr"))
        self.assertEqual((profile.clinical_agents, profile.clinical_writing_guidance,
                          profile.shared_clinical_memory), ("always", "rules_examples", True))
        self.assertTrue(profile.clinical_verification)
        self.assertTrue(profile.clinical_revision)
        self.assertEqual(profile.clinical_response_format, "text")
        self.assertFalse(profile.requires_mock_embedding)
        self.assertFalse(profile.requires_official_embedding)
        self.assertEqual(profile.qwen_calls_reserved, 5)
        self.assertEqual(profile.final_validation_mode, "lite")

    def test_demo001_runs_all_five_qwen_stages_with_lexical_ppr_and_revision(self):
        case = [{
            "case_id": "DEMO001", "record_cutoff": "2026-11-25 08:00",
            "sources": [
                {"source_id": "S01", "field_path": "入院记录.主诉",
                 "content": "患者突发胸痛5小时。"},
                {"source_id": "S02", "field_path": "入院记录.辅助检查",
                 "content": "CTA提示主动脉壁间血肿可能。"},
                {"source_id": "S03", "field_path": "入院记录.初步诊断",
                 "content": "初步诊断：主动脉壁内血肿待进一步评估。"},
                {"source_id": "S04", "field_path": "入院记录.诊疗计划",
                 "content": "计划持续监测生命体征。"},
            ],
        }]
        (self.package / "input.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者突发胸痛5小时。",
            "diagnostic_discussion": "CTA提示主动脉壁间血肿可能，当前诊断仍需进一步评估。",
            "plan": "建议立即开展未经评估的高风险操作。",
            "evidence": [
                {"statement": "患者突发胸痛5小时。", "source_ids": ["S01"], "kind": "fact"},
                {"statement": "CTA提示主动脉壁间血肿可能。", "source_ids": ["S02"], "kind": "fact"},
            ],
            "review_items": ["主动脉壁内血肿诊断及后续方案需结合完整影像确认。"],
        }
        verifier = _verifier(writer, [{
            "claim_id": "C003", "code": "UNSAFE_RECOMMENDATION", "severity": "high",
            "hint": "改为由医生结合病情决定的条件性建议。", "source_ids": [],
        }])
        reviser = {"edits": [{
            "claim_id": "C003", "section": "plan", "action": "replace",
            "replacement": "建议由医生结合病情及完整影像审慎制定后续方案。",
        }], "review_items_to_add": []}
        self._queue(_planner(["F001", "F002"], "突发胸痛与主动脉壁间血肿可能"),
                    _agent(), writer, verifier, reviser)

        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(output[0]["medical_record"]["plan"], reviser["edits"][0]["replacement"])
        self.assertEqual(log["algorithm_profile"], "competition_full_v2")
        self.assertEqual(log["environment"], "competition")
        self.assertEqual(log["total_calls"], 5)
        case_log = log["cases"][0]
        self.assertEqual([event["stage"] for event in case_log["stage_calls"]],
                         ["planner", "counter_evidence", "writer", "verifier", "reviser"])
        self.assertEqual(case_log["token_usage"]["total_tokens"], 750)
        clinical = case_log["clinical"]
        graph = clinical["graph_retrieval"]
        self.assertTrue(graph["graph_executed"])
        self.assertEqual((graph["graph_mode"], graph["retrieval_strategy"], graph["seed_provider"]),
                         ("ppr", "lexical_graph", "lexical_bm25_cpu"))
        self.assertTrue(graph["reranked_fact_ids"])
        self.assertTrue(clinical["adaptive_agent"]["completed"])
        self.assertTrue(clinical["shared_clinical_memory"]["enabled"])
        self.assertEqual(clinical["writing_guidance"]["mode"], "rules_examples")
        self.assertTrue(clinical["verification_completed"])
        self.assertTrue(clinical["revision_attempted"])
        self.assertTrue(clinical["revision_applied"])
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)
        self.assertTrue(all(request["authorization"] is None for request in self.requests))
        self.assertTrue(all(request["body"]["model"] == "official-mock-qwen"
                            for request in self.requests))
        self.assertTrue(all("response_format" not in request["body"] for request in self.requests))

    def test_practice05_runs_real_adapter_and_four_stage_safe_path(self):
        source = next((ROOT / "information" / "临床智录_练习病历数据集JSON").glob("05_*.json"))
        shutil.copy2(source, self.package / "input.json")
        writer = {
            "case_features": "45岁男性，突发持续性胸痛5小时，疼痛向后背肩胛间区放射。",
            "diagnostic_discussion": "大血管CTA提示主动脉壁内血肿可能，初步诊断为Stanford B型可能，仍需结合完整影像审慎评估。",
            "plan": "继续监测生命体征、心率及血压，后续治疗方案由医生结合病情和完整影像审慎确定。",
            "evidence": [
                {"statement": "突发胸痛5小时。", "source_ids": ["P0013"], "kind": "fact"},
                {"statement": "主动脉壁内血肿（Stanford B型可能）。",
                 "source_ids": ["P0045"], "kind": "fact"},
                {"statement": "主动脉管壁最厚处约16mm。",
                 "source_ids": ["P0043"], "kind": "fact"},
            ],
            "review_items": ["主动脉壁内血肿的分型及后续治疗方案需医生结合完整影像确认。"],
        }
        self._queue(_planner(["F014", "F046", "F052"], "突发胸痛与主动脉壁内血肿可能"),
                    _agent(), writer, _verifier(writer))
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertRegex(output[0]["case_id"], r"^PRACTICE_AUTO_[0-9A-F]{12}$")
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(log["input_adapter"]["detected_format"], "practice")
        self.assertEqual(log["input_adapter"]["case_mappings"][0]["case_id_origin"],
                         "redacted_content_fingerprint")
        case_log = log["cases"][0]
        self.assertEqual([event["stage"] for event in case_log["stage_calls"]],
                         ["planner", "counter_evidence", "writer", "verifier"])
        self.assertEqual(case_log["token_usage"]["total_tokens"], 600)
        clinical = case_log["clinical"]
        self.assertTrue(clinical["graph_retrieval"]["graph_executed"])
        self.assertEqual(clinical["graph_retrieval"]["retrieval_strategy"], "lexical_graph")
        self.assertTrue(clinical["adaptive_agent"]["completed"])
        self.assertTrue(clinical["verification_completed"])
        self.assertFalse(clinical["revision_attempted"])
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)

    def test_invalid_verifier_schema_is_audited_and_writer_is_not_silently_rewritten(self):
        case = [{"case_id": "DEMO001", "record_cutoff": "T0",
                 "sources": [{"source_id": "S01", "content": "患者发热。"}]}]
        (self.package / "input.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者发热。", "diagnostic_discussion": "发热原因待查。",
            "plan": "建议由医生结合病情进一步评估。",
            "evidence": [{"statement": "患者发热。", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }
        self._queue(_planner(["F001"], "发热"), _agent(), writer,
                    {"checked_claim_ids": []})
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(output[0]["medical_record"], writer)
        clinical = log["cases"][0]["clinical"]
        self.assertFalse(clinical["verification_completed"])
        self.assertEqual(clinical["verification_error"]["code"], "INVALID_CLAIM_VERIFICATION")
        self.assertEqual(clinical["verification_status"], "incomplete_writer_preserved")
        verifier_event = log["cases"][0]["stage_calls"][-1]
        self.assertEqual(verifier_event["stage"], "verifier")
        self.assertFalse(verifier_event["schema_validation_success"])
        self.assertEqual(verifier_event["schema_error_type"], "INVALID_CLAIM_VERIFICATION")

    def test_invalid_reviser_schema_uses_safe_local_sanitization_and_preserves_record(self):
        case = [{"case_id": "DEMO-RISK", "record_cutoff": "T0",
                 "sources": [{"source_id": "S01", "content": "患者突发胸痛。"}]}]
        (self.package / "input.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者突发胸痛。", "diagnostic_discussion": "胸痛原因待查。",
            "plan": "建议立即开展未经评估的高风险操作。",
            "evidence": [{"statement": "患者突发胸痛。", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }
        verifier = _verifier(writer, [{
            "claim_id": "C003", "code": "UNSAFE_RECOMMENDATION", "severity": "high",
            "hint": "改为由医生审慎决定的条件性建议。", "source_ids": [],
        }])
        self._queue(_planner(["F001"], "突发胸痛"), _agent(), writer, verifier,
                    {"edits": []})
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        self.assertIn("适应证及风险审慎评估", output[0]["medical_record"]["plan"])
        self.assertEqual(output[0]["medical_record"]["case_features"], writer["case_features"])
        case_log = log["cases"][0]
        self.assertEqual([event["stage"] for event in case_log["stage_calls"]],
                         ["planner", "counter_evidence", "writer", "verifier", "reviser"])
        reviser_event = case_log["stage_calls"][-1]
        self.assertFalse(reviser_event["schema_validation_success"])
        self.assertEqual(reviser_event["schema_error_type"], "INVALID_LOCAL_REVISION")
        clinical = case_log["clinical"]
        self.assertTrue(clinical["partial_sanitization_applied"])
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)
        self.assertFalse(clinical["final_quality_gate"]["output_blocking_safety_failure"])

    def test_invalid_verifier_schema_repairs_practice001_style_claims_without_null_output(self):
        case = [{"case_id": "PRACTICE001", "record_cutoff": "T0", "sources": [
            {"source_id": "S01", "content": "影像未见明确感染征象。"},
            {"source_id": "S02", "content": "初步诊断：肺炎待排。"},
        ]}]
        (self.package / "input.json").write_text(
            json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者影像未见明确感染征象。",
            "diagnostic_discussion": "影像未见明确感染征象，排除肺炎。患者诊断为肺炎。",
            "plan": "建议由医生结合病情进一步评估。",
            "evidence": [
                {"statement": "影像未见明确感染征象。", "source_ids": ["S01"], "kind": "fact"},
                {"statement": "初步诊断为肺炎待排。", "source_ids": ["S02"], "kind": "fact"},
            ],
            "review_items": [],
        }
        self._queue(_planner(["F001", "F002"], "肺炎待排"), _agent(), writer,
                    {"checked_claim_ids": []})
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        discussion = output[0]["medical_record"]["diagnostic_discussion"]
        self.assertIn("尚不能排除肺炎", discussion)
        self.assertIn("考虑肺炎，尚待进一步检查确认", discussion)
        clinical = log["cases"][0]["clinical"]
        self.assertFalse(clinical["verification_completed"])
        self.assertEqual(clinical["verification_error"]["code"],
                         "INVALID_CLAIM_VERIFICATION")
        self.assertTrue(clinical["writer_preserved"])
        self.assertTrue(clinical["partial_sanitization_applied"])
        self.assertFalse(clinical["revision_attempted"])
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)
        self.assertEqual([event["stage"] for event in log["cases"][0]["stage_calls"]],
                         ["planner", "counter_evidence", "writer", "verifier"])

    def test_unlocatable_severe_recommendation_isolated_without_erasing_record(self):
        case = [{"case_id": "DEMO-UNSAFE", "record_cutoff": "T0",
                 "sources": [{"source_id": "S01", "content": "患者胸痛。"}]}]
        (self.package / "input.json").write_text(
            json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者胸痛。", "diagnostic_discussion": "胸痛原因待查。",
            "plan": "必须实施高风险治疗。",
            "evidence": [{"statement": "患者胸痛。", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }
        verifier = _verifier(writer, [{
            "claim_id": "C003", "code": "UNSAFE_RECOMMENDATION", "severity": "high",
            "hint": "移除无适应证的强制治疗建议。", "source_ids": [],
        }])
        self._queue(_planner(["F001"], "胸痛"), _agent(), writer, verifier,
                    {"edits": []})
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        self.assertNotIn("必须实施高风险治疗", output[0]["medical_record"]["plan"])
        self.assertEqual(output[0]["medical_record"]["case_features"], writer["case_features"])
        clinical = log["cases"][0]["clinical"]
        self.assertFalse(clinical["revision_applied"])
        self.assertTrue(clinical["manual_review_required"])
        self.assertEqual(clinical["output_guarantee"]["level"], 2)
        self.assertFalse(clinical["final_quality_gate"]["output_blocking_safety_failure"])

    def test_fifth_call_revision_with_one_residual_high_keeps_safe_record(self):
        case = [{"case_id": "PRACTICE04-MOCK", "record_cutoff": "T0", "sources": [
            {"source_id": "S01", "content": "患者腹痛3天。"},
            {"source_id": "S02", "content": "初步诊断：腹痛原因待排。"},
            {"source_id": "S03", "content": "计划结合病情完善相关检查。"},
        ]}]
        (self.package / "input.json").write_text(
            json.dumps(case, ensure_ascii=False), encoding="utf-8")
        writer = {
            "case_features": "患者腹痛3天。",
            "diagnostic_discussion": "腹痛原因待排。",
            "plan": "建议立即开展高风险操作甲。必须实施高风险治疗乙。",
            "evidence": [{"statement": "患者腹痛3天。", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": [],
        }
        verifier = _verifier(writer, [
            {"claim_id": "C003", "code": "UNSAFE_RECOMMENDATION", "severity": "high",
             "hint": "删除强制建议甲。", "source_ids": []},
            {"claim_id": "C004", "code": "UNSAFE_RECOMMENDATION", "severity": "high",
             "hint": "删除强制建议乙。", "source_ids": []},
        ])
        reviser = {"edits": [{
            "claim_id": "C003", "section": "plan", "action": "replace",
            "replacement": "后续操作需由医生结合病情、适应证及风险审慎评估。",
        }], "review_items_to_add": []}
        self._queue(_planner(["F001", "F002"], "腹痛原因待排"),
                    _agent(), writer, verifier, reviser)
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output, log = self._read()
        self.assertEqual(output[0]["status"], "success")
        final = output[0]["medical_record"]
        self.assertIn("后续操作需由医生", final["plan"])
        self.assertNotIn("高风险治疗乙", final["plan"])
        clinical = log["cases"][0]["clinical"]
        self.assertEqual(log["total_calls"], 5)
        self.assertTrue(clinical["revision_applied"])
        self.assertIn("C004", clinical["detected_risks_before_output_guarantee"]
                      ["unresolved_claim_ids"])
        self.assertEqual(clinical["output_guarantee"]["level"], 2)
        self.assertIn("C004", clinical["output_guarantee"]["removed_claim_ids"])
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)


if __name__ == "__main__":
    unittest.main()
