import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.build_package import INCLUDE


ROOT = Path(__file__).resolve().parent.parent


class SubmissionLayoutTests(unittest.TestCase):
    def test_dependency_aliases_are_identical_and_have_no_packages(self):
        required = (ROOT / "requires.txt").read_text(encoding="utf-8")
        compatible = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertEqual(required, compatible)
        package_lines = [line for line in required.splitlines()
                         if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(package_lines, [])

    def test_env_example_contains_only_published_keys(self):
        lines = [line.strip() for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(lines, [
            "MODEL_NAME=replace_with_model_name",
            "ENDPOINT=http://127.0.0.1:8000/v1/chat/completions",
        ])

    def test_build_whitelist_contains_both_dependency_names_and_no_real_env(self):
        self.assertIn("requires.txt", INCLUDE)
        self.assertIn("requirements.txt", INCLUDE)
        self.assertIn(".env.example", INCLUDE)
        self.assertNotIn(".env", INCLUDE)
        runtime_paths = [ROOT / "main.py", ROOT / "src",
                         ROOT / "config" / "competition.example.json",
                         ROOT / "config" / "local.mlx.example.json",
                         ROOT / "config" / "retrieval.local.example.json"]
        runtime_text = []
        for path in runtime_paths:
            files = [path] if path.is_file() else sorted(item for item in path.rglob("*.py"))
            runtime_text.extend(item.read_text(encoding="utf-8") for item in files)
        combined = "\n".join(runtime_text)
        self.assertNotIn("/Users/", combined)
        self.assertNotIn("tintin7790", combined)

    def test_minimal_archive_contains_required_root_files(self):
        required = {"main.py", "requires.txt", "requirements.txt", ".env.example",
                    "README.md", "resource_manifest.json"}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "submission.zip"
            prefix = "北京语言大学Mewtype_v1"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                for name in INCLUDE:
                    path = ROOT / name
                    paths = ([path] if path.is_file() else
                             sorted(item for item in path.rglob("*")
                                    if item.is_file() and "__pycache__" not in item.parts))
                    for item in paths:
                        output.write(item, Path(prefix) / item.relative_to(ROOT))
            with zipfile.ZipFile(archive) as packaged:
                root_files = {Path(name).relative_to(prefix).as_posix()
                              for name in packaged.namelist()
                              if len(Path(name).parts) == 2}
                names = packaged.namelist()
            self.assertTrue(required <= root_files)
            self.assertFalse(any(Path(name).name == ".env" for name in names))

    def test_built_archive_installs_and_runs_zero_argument_official_protocol(self):
        requests = []
        record = {
            "case_features": "患者发热。",
            "diagnostic_discussion": "发热原因待查。",
            "plan": "建议由医生结合临床情况进一步评估。",
            "evidence": [{"statement": "患者发热。", "source_ids": ["S01"],
                          "kind": "fact"}],
            "review_items": [],
        }
        responses = [
            {"problems": [{
                "problem_id": "PR001", "name": "发热", "priority": "high",
                "origin": "model_inference", "supporting_facts": ["F001"],
                "contradicting_facts": [], "differential_diagnoses": [],
                "plan_goals": ["明确发热原因"], "retrieval_queries": ["发热"],
            }]},
            {"reviewed_problem_ids": ["PR001"], "issues": []},
            record,
            {"checked_claim_ids": ["C001", "C002", "C003", "C004"], "issues": []},
        ]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                requests.append({"path": self.path,
                                 "authorization": self.headers.get("Authorization"),
                                 "body": body})
                response = responses.pop(0)
                payload = {"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)},
                                         "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                     "total_tokens": 2}}
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            archive = temporary / "submission.zip"
            built = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "build_package.py"),
                 "--team", "北京语言大学Mewtype", "--version", "v1",
                 "--output", str(archive)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            with zipfile.ZipFile(archive) as packaged:
                packaged.extractall(temporary / "extracted")
            package_root = temporary / "extracted" / "北京语言大学Mewtype_v1"
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            (package_root / ".env").write_text(
                f"MODEL_NAME=official-mock-qwen\nENDPOINT={endpoint}\n", encoding="utf-8")
            (package_root / "input.json").write_text(json.dumps([{
                "case_id": "DEMO001", "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "患者发热。"}],
            }], ensure_ascii=False), encoding="utf-8")

            virtual = temporary / "venv"
            created = subprocess.run(
                [sys.executable, "-m", "venv", str(virtual)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            python = virtual / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            installed = subprocess.run(
                [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=package_root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            environment = os.environ.copy()
            environment.update({
                "MEWTYPE_DEV_MODE": "1",
                "DASHSCOPE_API_KEY": "must-not-be-used",
                "QWEN_BASE_URL": "https://external.invalid",
                "QWEN_MODEL": "must-not-be-used",
            })
            completed = subprocess.run(
                [str(python), "main.py"], cwd=package_root,
                capture_output=True, text=True, timeout=30, env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads((package_root / "output.json").read_text(encoding="utf-8"))
            log = json.loads((package_root / "run_log.json").read_text(encoding="utf-8"))
            self.assertEqual([(item["case_id"], item["status"]) for item in output],
                             [("DEMO001", "success")])
            self.assertEqual(log["algorithm_profile"], "competition_full_v2")
            self.assertEqual([event["stage"] for event in log["cases"][0]["stage_calls"]],
                             ["planner", "counter_evidence", "writer", "verifier"])
            self.assertTrue(log["cases"][0]["clinical"]["graph_retrieval"]["graph_executed"])
            self.assertEqual(log["cases"][0]["clinical"]["graph_retrieval"]["retrieval_strategy"],
                             "lexical_graph")
            self.assertTrue(log["cases"][0]["clinical"]["adaptive_agent"]["completed"])
            self.assertTrue(log["cases"][0]["clinical"]["shared_clinical_memory"]["enabled"])
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[0]["path"], "/v1/chat/completions")
            self.assertTrue(all(request["body"]["model"] == "official-mock-qwen"
                                for request in requests))
            self.assertTrue(all(request["authorization"] is None for request in requests))


if __name__ == "__main__":
    unittest.main()
