import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.app import _enforce_profile_safety
from src.config import load_config
from src.competition_profile import LOCAL_DEVELOPMENT_PROFILE, PROFILES, get_active_competition_profile
from src.errors import CaseError, TaskError


def _record(source_id="S001"):
    return {
        "case_features": "患者发热、咳嗽3天，最高体温38.8℃，否认胸痛。",
        "diagnostic_discussion": "发热伴咳嗽，病因尚待结合检查判断。",
        "plan": "原资料记载计划完善胸部CT，目前尚无结果。",
        "evidence": [{"statement": "最高体温38.8℃，否认胸痛。",
                      "source_ids": [source_id], "kind": "fact"}],
        "review_items": ["胸部CT结果待确认。"],
    }


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


class OfficialRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "package"
        self.root.mkdir()
        project = Path(__file__).resolve().parent.parent
        self.project = project
        shutil.copy2(project / "main.py", self.root / "main.py")
        shutil.copytree(project / "src", self.root / "src")
        shutil.copytree(project / "prompts", self.root / "prompts")
        self.requests = []

    def tearDown(self):
        self.temp.cleanup()

    def _start_server(self, scenarios):
        requests = self.requests
        queue = list(scenarios)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                size = int(inner.headers.get("Content-Length", "0"))
                body = inner.rfile.read(size)
                requests.append({"path": inner.path,
                                 "authorization": inner.headers.get("Authorization"),
                                 "body": json.loads(body.decode("utf-8"))})
                scenario = queue.pop(0)
                delay = scenario.get("delay", 0)
                if delay:
                    time.sleep(delay)
                status = scenario.get("status", 200)
                if status != 200:
                    payload = {"error": {"message": "mock failure"}}
                else:
                    payload = {
                        "choices": [{"message": {"content": scenario["content"]},
                                     "finish_reason": scenario.get("finish_reason", "stop")}],
                        "usage": scenario.get("usage", {"prompt_tokens": 10,
                                                          "completion_tokens": 20,
                                                          "total_tokens": 30}),
                    }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                inner.send_response(status)
                inner.send_header("Content-Type", "application/json")
                inner.send_header("Content-Length", str(len(encoded)))
                inner.send_header("Connection", "close")
                inner.end_headers()
                try:
                    inner.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        server = _QuietServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/official/full/chat/completions"

    def _write_runtime(self, endpoint, cases=None, extra_env=""):
        (self.root / ".env").write_text(
            f"MODEL_NAME=official-qwen\nENDPOINT={endpoint}\n{extra_env}", encoding="utf-8")
        cases = cases or [{"case_id": "C1", "record_cutoff": "2026-11-24 12:00",
                           "sources": [{"source_id": "S001", "content": "最高体温38.8℃，否认胸痛。"}]}]
        (self.root / "input.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    def _run(self, arguments=None, extra_env=None):
        environment = os.environ.copy()
        environment.pop("MEWTYPE_DEV_MODE", None)
        environment.update(extra_env or {})
        return subprocess.run([sys.executable, "main.py", *(arguments or [])], cwd=self.root,
                              capture_output=True, text=True, timeout=15, env=environment)

    def test_two_arg_competition_profile_writes_valid_output_without_auth(self):
        endpoint = self._start_server([{"content": json.dumps(_record(), ensure_ascii=False)}])
        self._write_runtime(endpoint)
        completed = self._run(
            ["--input", "input.json", "--output", "output.json"],
            {"DASHSCOPE_API_KEY": "must-not-be-used", "QWEN_BASE_URL": "https://external.invalid",
             "QWEN_MODEL": "must-not-be-used"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.root / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(log["runtime_profile"], "competition_profile")
        self.assertEqual(log["algorithm_profile"], "baseline_safe")
        self.assertEqual(log["total_calls"], 1)
        event = log["cases"][0]["stage_calls"][0]
        self.assertEqual((event["stage"], event["finish_reason"]), ("model", "stop"))
        self.assertEqual(event["token_usage"]["total_tokens"], 30)
        self.assertEqual(self.requests[0]["path"], "/official/full/chat/completions")
        self.assertIsNone(self.requests[0]["authorization"])
        self.assertEqual(set(self.requests[0]["body"]), {"model", "messages", "stream"})

    def test_two_arg_development_profile_adapts_root_practice_json(self):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "发热", "priority": "high",
            "origin": "model_inference", "supporting_facts": ["F002"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["明确发热原因"], "retrieval_queries": ["发热"],
        }]}
        agent = {"reviewed_problem_ids": ["PR001"], "issues": []}
        writer = {"case_features": "患者诉发热。", "diagnostic_discussion": "发热原因待查。",
                  "plan": "建议由医生结合临床情况进一步评估。",
                  "evidence": [{"statement": "患者诉发热。", "source_ids": ["P0002"],
                                "kind": "fact"}], "review_items": []}
        verifier = {"checked_claim_ids": ["C001", "C002", "C003", "C004"], "issues": []}
        endpoint = self._start_server([
            {"content": json.dumps(value, ensure_ascii=False)}
            for value in (planner, agent, writer, verifier)
        ])
        practice = {"入院记录": {"一般项目": {"记录时间": "2026‑05‑08 14:25"},
                                   "主诉": "患者诉发热。"}}
        (self.root / "input.json").write_text(json.dumps(practice, ensure_ascii=False), encoding="utf-8")
        (self.root / "output.json").write_text('[{"historical":true}]\n', encoding="utf-8")
        (self.root / "run_log.json").write_text('{"historical":true}\n', encoding="utf-8")
        completed = self._run(
            ["--input", "input.json", "--output", "output.json"],
            {"MEWTYPE_DEV_MODE": "1", "DASHSCOPE_API_KEY": "test-only-not-secret",
             "QWEN_BASE_URL": endpoint, "QWEN_MODEL": "qwen-plus-test"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.root / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")
        self.assertRegex(output[0]["case_id"], r"^PRACTICE_AUTO_[0-9A-F]{12}$")
        self.assertEqual(log["input_adapter"]["detected_format"], "practice")
        self.assertEqual(log["input_adapter"]["case_mappings"][0]["case_id_origin"],
                         "redacted_content_fingerprint")
        self.assertTrue(log["cases"][0]["clinical"]["graph_retrieval"]["graph_executed"])
        self.assertTrue(log["cases"][0]["clinical"]["adaptive_agent"]["completed"])
        self.assertEqual(len(log["artifact_backups"]), 2)
        self.assertTrue(all((self.root / path).is_file() for path in log["artifact_backups"]))
        self.assertTrue(all(request["body"]["response_format"]["type"] == "json_schema"
                            for request in self.requests))

    def test_practice07_invalid_json_is_reported_without_repair_or_model_call(self):
        source = (self.project / "information" / "临床智录_练习病历数据集JSON" /
                  "07_心血管内科_多次PCI术后再发胸痛.json")
        shutil.copy2(source, self.root / "input.json")
        completed = self._run(
            ["--input", "input.json", "--output", "output.json"],
            {"MEWTYPE_DEV_MODE": "1", "DASHSCOPE_API_KEY": "test-only-not-secret",
             "QWEN_BASE_URL": "http://127.0.0.1:9/chat/completions", "QWEN_MODEL": "qwen-plus-test"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("INPUT_INVALID_JSON", completed.stderr)
        self.assertIn("Extra data: line 67 column 1 (char 3660)", completed.stderr)
        self.assertFalse((self.root / "output.json").exists())

    def test_multiple_cases_preserve_count_order_and_case_ids(self):
        contents = [{"content": json.dumps(_record(), ensure_ascii=False)} for _ in range(2)]
        endpoint = self._start_server(contents)
        cases = [{"case_id": case_id, "record_cutoff": "T0",
                  "sources": [{"source_id": "S001", "content": "最高体温38.8℃，否认胸痛。"}]}
                 for case_id in ("FIRST", "SECOND")]
        self._write_runtime(endpoint, cases)
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        self.assertEqual([item["case_id"] for item in output], ["FIRST", "SECOND"])
        self.assertEqual(len(self.requests), 2)

    def test_truncated_model_json_is_a_per_case_failure(self):
        endpoint = self._start_server([{"content": '{"case_features": "不完整"'}])
        self._write_runtime(endpoint)
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.root / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["error"]["code"], "MODEL_INVALID_JSON")
        self.assertEqual(log["cases"][0]["stage_calls"][0]["parse_error_type"], "INCOMPLETE_JSON")

    def test_http_error_is_explicit_and_not_mocked(self):
        endpoint = self._start_server([{"status": 503}])
        self._write_runtime(endpoint)
        completed = self._run()
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(output[0]["error"]["code"], "MODEL_HTTP_ERROR")

    def test_timeout_is_explicit(self):
        endpoint = self._start_server([{"content": json.dumps(_record()), "delay": 0.2}])
        self._write_runtime(endpoint, extra_env="TIMEOUT_SECONDS=0.05\n")
        completed = self._run()
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(output[0]["error"]["code"], "MODEL_TIMEOUT")

    def test_competition_env_requires_only_published_identity_fields(self):
        env_file = self.root / ".env"
        env_file.write_text("MODEL_NAME=qwen-official\nENDPOINT=http://127.0.0.1/full\n", encoding="utf-8")
        config = load_config(None, "competition", "baseline", official_env_path=env_file)
        self.assertEqual(config.endpoint, "http://127.0.0.1/full")
        self.assertFalse(config.send_generation_parameters)
        self.assertEqual(config.max_calls_per_case, 1)

    def test_missing_env_is_a_task_error(self):
        with self.assertRaises(TaskError) as caught:
            load_config(None, "competition", "baseline", official_env_path=self.root / "missing.env")
        self.assertEqual(caught.exception.code, "COMPETITION_ENV_MISSING")

    def test_build_time_profiles_have_explicit_bounded_budgets(self):
        self.assertEqual(get_active_competition_profile().name, "baseline_safe")
        self.assertEqual(PROFILES["clinical_d"].qwen_calls_reserved, 4)
        self.assertEqual(PROFILES["clinical_d_agent"].qwen_calls_reserved, 5)
        advanced = PROFILES["clinical_d_graph_agent"]
        self.assertEqual((advanced.pipeline, advanced.graph_retrieval, advanced.clinical_agents),
                         ("clinical_pipeline", "ppr", "always"))
        self.assertTrue(advanced.requires_official_embedding)
        self.assertTrue(advanced.block_unresolved_high_risk)
        self.assertEqual((LOCAL_DEVELOPMENT_PROFILE.environment,
                          LOCAL_DEVELOPMENT_PROFILE.clinical_response_format,
                          LOCAL_DEVELOPMENT_PROFILE.graph_retrieval),
                         ("dev_api", "json_schema", "ppr"))
        self.assertTrue(LOCAL_DEVELOPMENT_PROFILE.requires_mock_embedding)

    def test_advanced_profile_blocks_unresolved_high_risk(self):
        meta = {"clinical": {"unresolved_risk_count": 1}}
        with self.assertRaises(CaseError) as caught:
            _enforce_profile_safety(PROFILES["clinical_d_graph_agent"], meta)
        self.assertEqual(caught.exception.code, "UNRESOLVED_HIGH_RISK")
        self.assertEqual(caught.exception.details["failed_stage"], "clinical_safety_gate")

    def test_two_arg_development_profile_executes_ppr_agent_verifier_and_reviser(self):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "再发胸痛", "priority": "high",
            "origin": "mixed", "supporting_facts": ["F001", "F003"],
            "contradicting_facts": [],
            "differential_diagnoses": [{
                "name": "不稳定性心绞痛待排", "certainty": "rule_out",
                "origin": "mixed", "supporting_facts": ["F001", "F003"],
                "contradicting_facts": [], "missing_evidence": ["心电图及心肌损伤标志物"],
            }],
            "plan_goals": ["明确胸痛原因并核对抗血小板用药"],
            "retrieval_queries": ["胸痛 PCI 抗血小板 漏服"],
        }]}
        agent = {"reviewed_problem_ids": ["PR001"], "issues": [{
            "problem_id": "PR001", "code": "MISSING_DISCRIMINATING_EVIDENCE",
            "fact_ids": [], "note": "检查结果未完整提供，诊断确定性需保守。",
        }]}
        writer = {
            "case_features": "患者多次PCI术后再发胸痛，近月存在抗血小板药漏服。",
            "diagnostic_discussion": "不稳定性心绞痛待排，仍需结合检查判断。",
            "plan": "建议立即开展有创检查。",
            "evidence": [{"statement": "患者多次PCI术后再发胸痛，近月存在抗血小板药漏服。",
                          "source_ids": ["S01"], "kind": "fact"}],
            "review_items": ["相关检查结果待确认。"],
        }
        verifier = {
            "checked_claim_ids": ["C001", "C002", "C003", "C004", "C005"],
            "issues": [{"claim_id": "C003", "code": "UNSAFE_RECOMMENDATION",
                        "severity": "high", "hint": "改为由医生审慎决定的条件性建议。",
                        "source_ids": []}],
        }
        reviser = {"edits": [{"claim_id": "C003", "section": "plan", "action": "replace",
                                "replacement": "建议由医生结合临床情况决定后续检查。"}],
                   "review_items_to_add": []}
        endpoint = self._start_server([
            {"content": json.dumps(value, ensure_ascii=False)}
            for value in (planner, agent, writer, verifier, reviser)
        ])
        cases = [{"case_id": "DEMO001", "record_cutoff": "2026-11-24 12:00",
                  "sources": [
                      {"source_id": "S01", "field_path": "入院记录.现病史",
                       "content": "患者多次PCI术后再发胸痛，近月存在抗血小板药漏服。"},
                      {"source_id": "S02", "field_path": "入院记录.既往史",
                       "content": "2018年及2023年先后接受PCI治疗。"},
                      {"source_id": "S03", "field_path": "入院记录.初步诊断",
                       "content": "初步诊断：不稳定性心绞痛待排。"},
                      {"source_id": "S04", "field_path": "入院记录.诊疗计划",
                       "content": "计划完善心电图及心肌损伤标志物，目前结果未提供。"},
                  ]}]
        self._write_runtime(endpoint, cases)
        completed = self._run(
            ["--input", "input.json", "--output", "output.json"],
            {"MEWTYPE_DEV_MODE": "1", "DASHSCOPE_API_KEY": "test-only-not-secret",
             "QWEN_BASE_URL": endpoint, "QWEN_MODEL": "qwen-plus-test"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.root / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(output[0]["medical_record"]["plan"],
                         "建议由医生结合临床情况决定后续检查。")
        self.assertEqual(log["runtime_profile"], "development_profile")
        self.assertEqual(log["algorithm_profile"], "development_clinical_d_graph_agent")
        self.assertEqual(log["environment"], "dev_api")
        self.assertEqual(log["retrieval"]["provider"], "mock_development")
        self.assertEqual(set(log["prompt_manifest"]), {"hash_algorithm", "files"})
        self.assertTrue(all(len(value) == 64 for value in log["prompt_manifest"]["files"].values()))
        self.assertEqual(log["total_calls"], 5)
        stages = [event["stage"] for event in log["cases"][0]["stage_calls"]]
        self.assertEqual(stages, ["planner", "counter_evidence", "writer", "verifier", "reviser"])
        clinical = log["cases"][0]["clinical"]
        graph = clinical["graph_retrieval"]
        self.assertTrue(graph["graph_executed"])
        self.assertEqual(graph["graph_mode"], "ppr")
        self.assertEqual(graph["retriever"]["ppr_iterations"], 20)
        self.assertTrue(graph["selected_source_ids"])
        self.assertTrue(clinical["adaptive_agent"]["completed"])
        self.assertGreater(clinical["adaptive_agent"]["retrieval_candidate_count"], 0)
        self.assertTrue(clinical["revision_attempted"])
        self.assertTrue(clinical["revision_applied"])
        self.assertEqual(clinical["unresolved_risk_count"], 0)
        agent_payload = json.loads(self.requests[1]["body"]["messages"][-1]["content"])
        writer_payload = json.loads(self.requests[2]["body"]["messages"][-1]["content"])
        self.assertEqual(agent_payload["retrieval_evidence"][0]["candidates"][0]["graph_strategy"], "ppr")
        self.assertIn('"graph_strategy": "ppr"', json.dumps(writer_payload, ensure_ascii=False))

    def test_advanced_profile_discloses_embedding_failure_and_keeps_legal_d_agent_output(self):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "发热", "priority": "high",
            "origin": "model_inference", "supporting_facts": ["F001"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["明确原因"], "retrieval_queries": ["发热"],
        }]}
        agent = {"reviewed_problem_ids": ["PR001"], "issues": []}
        writer = {
            "case_features": "患者发热。", "diagnostic_discussion": "发热原因待查。",
            "plan": "建议由医生结合临床情况决定后续检查。",
            "evidence": [{"statement": "患者发热。", "source_ids": ["S001"], "kind": "fact"}],
            "review_items": [],
        }
        verifier = {"checked_claim_ids": ["C001", "C002", "C003", "C004"], "issues": []}
        endpoint = self._start_server([
            {"content": json.dumps(value, ensure_ascii=False)}
            for value in (planner, agent, writer, verifier)
        ])
        self._write_runtime(endpoint, [{"case_id": "DEMO001", "record_cutoff": "T0",
                                       "sources": [{"source_id": "S001", "content": "患者发热。"}]}])
        (self.root / "sitecustomize.py").write_text(
            "import src.competition_profile as cp\n"
            "cp.ACTIVE_COMPETITION_PROFILE = 'clinical_d_graph_agent'\n"
            "from src.errors import TaskError\n"
            "import src.app as app\n"
            "class FailingEmbedding:\n"
            "    total_calls=0; total_texts=0; elapsed_ms=0.0\n"
            "    def embed(self, texts):\n"
            "        raise TaskError('OFFICIAL_EMBEDDING_UNAVAILABLE','test-only failure')\n"
            "def _failing_embedding(profile):\n"
            "    return FailingEmbedding(), {'provider':'official-test-failure','model':None,"
            "'top_k':3,'min_similarity':0.0,'max_chunk_chars':240,'batch_size':None,"
            "'max_length':None,'use_fp16':None}\n"
            "app._official_profile_embedding = _failing_embedding\n",
            encoding="utf-8",
        )
        completed = self._run(extra_env={"PYTHONPATH": str(self.root)})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads((self.root / "output.json").read_text(encoding="utf-8"))
        log = json.loads((self.root / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(output[0]["status"], "success")
        self.assertEqual(log["total_calls"], 4)
        clinical = log["cases"][0]["clinical"]
        self.assertFalse(clinical["graph_retrieval"]["graph_executed"])
        self.assertEqual(clinical["graph_retrieval"]["graph_fallback_reason"],
                         "graph_retrieval_failed:OFFICIAL_EMBEDDING_UNAVAILABLE")
        self.assertTrue(clinical["adaptive_agent"]["completed"])


if __name__ == "__main__":
    unittest.main()
