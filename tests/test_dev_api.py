import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app import main
from src.config import ModelConfig, load_config
from src.errors import TaskError
from src.model_client import Budget, ChatCompletionsClient, DevApiChatCompletionsClient


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def valid_record():
    return {"case_features": "入院前体温最高38.5℃。", "diagnostic_discussion": "现有资料支持发热，原因待查。",
            "plan": "拟继续观察体温变化。",
            "evidence": [{"statement": "入院前体温最高38.5℃。", "source_ids": ["S01"], "kind": "fact"}],
            "review_items": []}


class DevApiTests(unittest.TestCase):
    ENV = {"DASHSCOPE_API_KEY": "test-secret-not-real", "QWEN_BASE_URL": "https://example.invalid/compatible-mode/v1",
           "QWEN_MODEL": "qwen-plus"}

    def test_dev_api_requires_all_environment_variables(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TaskError): load_config(None, "dev_api", "baseline")

    def test_dev_api_config_uses_base_url_and_model(self):
        with patch.dict(os.environ, self.ENV, clear=True):
            config = load_config(None, "dev_api", "baseline")
        self.assertEqual(config.model, "qwen-plus")
        self.assertEqual(config.endpoint, "https://example.invalid/compatible-mode/v1/chat/completions")
        self.assertEqual(config.max_calls_per_case, 1)

    def test_dev_api_client_records_usage_and_auth_is_isolated(self):
        captured = {}
        payload = {"choices": [{"message": {"content": json.dumps(valid_record(), ensure_ascii=False)}}],
                   "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}}
        def fake_open(request, timeout): captured["request"] = request; return FakeResponse(payload)
        with patch.dict(os.environ, self.ENV, clear=True):
            config = load_config(None, "dev_api", "baseline")
            client = DevApiChatCompletionsClient(config)
            with patch("urllib.request.urlopen", side_effect=fake_open):
                client.call([{"role": "user", "content": "test"}], "text", Budget(1))
        self.assertTrue(captured["request"].get_header("Authorization").startswith("Bearer "))
        self.assertEqual(client.total_usage, {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26})

    def test_competition_client_still_sends_no_authorization(self):
        captured = {}
        payload = {"choices": [{"message": {"content": "{}"}}]}
        def fake_open(request, timeout): captured["request"] = request; return FakeResponse(payload)
        config = ModelConfig(environment="competition", adapter="chat_completions", model="official",
                             endpoint="https://example.invalid/chat/completions", max_calls_per_case=1)
        with patch("urllib.request.urlopen", side_effect=fake_open):
            ChatCompletionsClient(config).call([], "text", Budget(1))
        self.assertIsNone(captured["request"].get_header("Authorization"))

    def test_dev_api_end_to_end_log_contains_usage(self):
        class FakeClient:
            def __init__(self, config):
                self.total_calls = 0
                self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            def call(self, messages, response_format, budget):
                budget.take(); self.total_calls += 1
                self.total_usage = {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
                return json.dumps(valid_record(), ensure_ascii=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inp = root / "input.json"; out = root / "output.json"; log = root / "log.json"
            inp.write_text(json.dumps([{"case_id": "D1", "record_cutoff": "T0",
                                        "sources": [{"source_id": "S01", "content": "入院前体温最高38.5℃。"}]}], ensure_ascii=False), encoding="utf-8")
            with patch.dict(os.environ, self.ENV, clear=True), patch("src.app.DevApiChatCompletionsClient", FakeClient):
                code = main(["--input", str(inp), "--output", str(out), "--log", str(log),
                             "--mode", "dev_api", "--pipeline", "baseline"])
            saved = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(code, 0); self.assertEqual(saved["model"], "qwen-plus")
            self.assertEqual(saved["total_calls"], 1); self.assertEqual(saved["token_usage"]["total_tokens"], 50)
            self.assertEqual(saved["cases"][0]["token_usage"]["total_tokens"], 50)


if __name__ == "__main__": unittest.main()
