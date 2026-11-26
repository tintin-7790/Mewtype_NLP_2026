import json
import io
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.app import main
from src.config import ModelConfig, load_config
from src.errors import CaseError, TaskError
from src.model_client import (Budget, ChatCompletionsClient, DevApiChatCompletionsClient,
                              MEDICAL_SCHEMA, MlxDirectClient, PLANNER_SCHEMA)


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class FakeStreamResponse:
    def __init__(self, chunks): self.chunks = chunks
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self):
        for chunk in self.chunks:
            if chunk == "[DONE]":
                yield b"data: [DONE]\n\n"
            else:
                yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()


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

    def test_json_schema_request_matches_medical_schema(self):
        captured = {}
        payload = {"choices": [{"message": {"content": json.dumps(valid_record(), ensure_ascii=False)}}]}
        def fake_open(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8")); return FakeResponse(payload)
        with patch.dict(os.environ, self.ENV, clear=True):
            client = DevApiChatCompletionsClient(load_config(None, "dev_api", "baseline"))
            with patch("urllib.request.urlopen", side_effect=fake_open):
                client.call([{"role": "system", "content": "JSON"}], "json_schema", Budget(1))
        self.assertEqual(captured["body"]["response_format"],
                         {"type": "json_schema", "json_schema": MEDICAL_SCHEMA})
        self.assertEqual(client.format_attempts, ["json_schema"])
        self.assertEqual(client.last_response_format, "json_schema")

    def test_stage_json_schema_request_uses_explicit_stage_schema(self):
        captured = {}
        payload = {"choices": [{"message": {"content": '{"problems":[]}'}}]}
        def fake_open(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8")); return FakeResponse(payload)
        with patch.dict(os.environ, self.ENV, clear=True):
            client = DevApiChatCompletionsClient(load_config(None, "dev_api", "baseline"))
            with patch("urllib.request.urlopen", side_effect=fake_open):
                client.call([], "json_schema", Budget(1), stage="planner", schema=PLANNER_SCHEMA)
        self.assertEqual(captured["body"]["response_format"]["json_schema"], PLANNER_SCHEMA)
        self.assertEqual(client.call_events[0]["stage"], "planner")
        self.assertEqual(client.call_events[0]["response_format"], "json_schema")

    def test_json_schema_http_rejection_is_explicit_without_fallback(self):
        error = urllib.error.HTTPError("https://example.invalid", 400, "bad request", {},
                                      io.BytesIO(b'{"message":"json_schema unsupported"}'))
        with patch.dict(os.environ, self.ENV, clear=True):
            client = DevApiChatCompletionsClient(load_config(None, "dev_api", "baseline"))
            with patch("urllib.request.urlopen", side_effect=error), self.assertRaises(CaseError) as caught:
                client.call([{"role": "system", "content": "JSON"}], "json_schema", Budget(1))
        self.assertEqual(caught.exception.code, "RESPONSE_FORMAT_REJECTED")
        self.assertEqual(client.total_calls, 1)
        self.assertEqual(client.format_attempts, ["json_schema"])
        self.assertFalse(client.call_events[0]["call_success"])

    def test_competition_client_still_sends_no_authorization(self):
        captured = {}
        payload = {"choices": [{"message": {"content": "{}"}}]}
        def fake_open(request, timeout): captured["request"] = request; return FakeResponse(payload)
        config = ModelConfig(environment="competition", adapter="chat_completions", model="official",
                             endpoint="https://example.invalid/chat/completions", max_calls_per_case=1)
        with patch("urllib.request.urlopen", side_effect=fake_open):
            ChatCompletionsClient(config).call([], "text", Budget(1))
        self.assertIsNone(captured["request"].get_header("Authorization"))

    def test_stage_policy_sends_thinking_and_timeout_then_ignores_reasoning_text(self):
        captured = {}
        payload = {"choices": [{"message": {"reasoning_content": "private reasoning",
                                                "content": "{}"},
                                 "finish_reason": "stop"}]}
        def fake_open(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(payload)
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions", timeout_seconds=901,
            stage_generation_parameters={
                "writer": {"max_tokens": 12288, "enable_thinking": True},
            },
        )
        client = ChatCompletionsClient(config)
        debug_events = []
        client.debug_hook = debug_events.append
        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = client.call([], "text", Budget(1), stage="writer")
        self.assertEqual(result, "{}")
        self.assertEqual(captured["timeout"], 901)
        self.assertEqual(captured["body"]["max_tokens"], 12288)
        self.assertEqual(captured["body"]["chat_template_kwargs"],
                         {"enable_thinking": True})
        self.assertTrue(client.call_events[0]["reasoning_content_present"])
        self.assertEqual(client.call_events[0]["reasoning_content_chars"], 17)
        self.assertNotIn("private reasoning", json.dumps(debug_events))

    def test_missing_final_content_is_invalid_even_when_reasoning_is_present(self):
        payload = {"choices": [{"message": {"reasoning_content": "reasoning only"}}]}
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions",
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            with self.assertRaises(CaseError) as caught:
                ChatCompletionsClient(config).call([], "text", Budget(1), stage="writer")
        self.assertEqual(caught.exception.code, "MODEL_INVALID_RESPONSE")

    def test_streaming_transport_accumulates_content_without_reasoning_leak(self):
        captured = {}
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "private"},
                            "finish_reason": None}]},
            {"choices": [{"delta": {"content": "{"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "}"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3,
                                        "total_tokens": 5}},
            "[DONE]",
        ]
        def fake_open(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeStreamResponse(chunks)
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions",
            stage_generation_parameters={"writer": {"stream": True,
                                                        "enable_thinking": True}},
        )
        client = ChatCompletionsClient(config)
        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = client.call([], "text", Budget(1), stage="writer")
        self.assertEqual(result, "{}")
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(client.total_usage["total_tokens"], 5)
        self.assertTrue(client.call_events[0]["reasoning_content_present"])
        self.assertEqual(client.call_events[0]["reasoning_content_chars"], 7)

    def test_thinking_parameter_rejection_has_one_budgeted_compatibility_retry(self):
        requests = []
        payload = {"choices": [{"message": {"content": "{}"}}]}
        def fake_open(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            if len(requests) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, 400, "bad request", {},
                    io.BytesIO(b'{"error":"chat_template_kwargs.enable_thinking unsupported"}'),
                )
            return FakeResponse(payload)
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions",
            stage_generation_parameters={
                "planner": {"max_tokens": 8192, "enable_thinking": True},
            },
            compatibility_retry_without_thinking=True,
        )
        client = ChatCompletionsClient(config)
        with patch("urllib.request.urlopen", side_effect=fake_open):
            self.assertEqual(client.call([], "text", Budget(2), stage="planner"), "{}")
        self.assertEqual(client.total_calls, 2)
        self.assertIn("chat_template_kwargs", requests[0])
        self.assertNotIn("chat_template_kwargs", requests[1])
        self.assertTrue(client.call_events[0]["compatibility_downgrade_triggered"])
        self.assertTrue(client.call_events[1]["compatibility_retry"])

    def test_thinking_compatibility_retry_never_exceeds_remaining_budget(self):
        def reject(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 400, "bad request", {},
                io.BytesIO(b'{"error":"enable_thinking unsupported"}'),
            )
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions",
            stage_generation_parameters={"writer": {"enable_thinking": True}},
            compatibility_retry_without_thinking=True,
        )
        client = ChatCompletionsClient(config)
        budget = Budget(1)
        with patch("urllib.request.urlopen", side_effect=reject):
            with self.assertRaises(CaseError) as caught:
                client.call([], "text", budget, stage="writer")
        self.assertEqual(caught.exception.code, "MODEL_HTTP_ERROR")
        self.assertEqual(client.total_calls, 1)
        self.assertEqual(budget.remaining, 0)
        self.assertFalse(client.call_events[0]["compatibility_downgrade_triggered"])
        self.assertTrue(
            client.call_events[0]["compatibility_downgrade_blocked_by_budget"]
        )

    def test_thinking_compatibility_downgrade_never_loops(self):
        def reject(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 400, "bad request", {},
                io.BytesIO(b'{"error":"enable_thinking unsupported"}'),
            )
        config = ModelConfig(
            environment="competition", adapter="chat_completions", model="official",
            endpoint="https://example.invalid/chat/completions",
            stage_generation_parameters={"planner": {"enable_thinking": True}},
            compatibility_retry_without_thinking=True,
        )
        client = ChatCompletionsClient(config)
        with patch("urllib.request.urlopen", side_effect=reject):
            with self.assertRaises(CaseError) as caught:
                client.call([], "text", Budget(5), stage="planner")
        self.assertEqual(caught.exception.code, "MODEL_HTTP_ERROR")
        self.assertEqual(client.total_calls, 2)

    def test_mlx_uses_stage_max_tokens_without_requiring_thinking_support(self):
        captured = {}
        module = types.ModuleType("mlx_lm")
        module.load = lambda path: ("model", "tokenizer")
        def generate(model, tokenizer, *, prompt, max_tokens, verbose):
            captured["max_tokens"] = max_tokens
            return "{}"
        module.generate = generate
        tokenizer = type("Tokenizer", (), {
            "apply_chat_template": lambda self, messages, tokenize, add_generation_prompt: "prompt",
        })()
        config = ModelConfig(
            environment="local", adapter="mlx_direct", model="local-qwen2.5",
            stage_generation_parameters={"writer": {"max_tokens": 12288}},
        )
        client = MlxDirectClient(config)
        client._loaded = ("model", tokenizer)
        with patch.dict(sys.modules, {"mlx_lm": module}):
            self.assertEqual(client.call([], "text", Budget(1), stage="writer"), "{}")
        self.assertEqual(captured["max_tokens"], 12288)
        self.assertIsNone(client.call_events[0]["generation_parameters"]["enable_thinking"])

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

    def test_raw_response_debug_requires_confirmation_and_stays_out_of_normal_log(self):
        payload = {"choices": [{"message": {"content": json.dumps(valid_record(), ensure_ascii=False)}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inp = root/"input.json"; out = root/"output.json"
            log = root/"log.json"; debug = root/"raw.json"
            inp.write_text(json.dumps([{"case_id":"F1","record_cutoff":"T0","sources":[
                {"source_id":"S01","content":"入院前体温最高38.5℃。"}]}], ensure_ascii=False), encoding="utf-8")
            base = ["--input",str(inp),"--output",str(out),"--log",str(log),"--mode","dev_api",
                    "--pipeline","baseline","--dev-response-format","json_schema",
                    "--debug-raw-response",str(debug)]
            with patch.dict(os.environ, self.ENV, clear=True):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(main(base), 2)
            self.assertFalse(debug.exists())
            with patch.dict(os.environ, self.ENV, clear=True), patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
                self.assertEqual(main(base+["--confirm-fictional-input"]), 0)
            raw = json.loads(debug.read_text(encoding="utf-8"))
            self.assertEqual(raw["events"][0]["requested_response_format"], "json_schema")
            self.assertIn("38.5℃", raw["events"][0]["content"])
            normal_log = log.read_text(encoding="utf-8")
            self.assertNotIn("入院前体温最高38.5℃", normal_log)
            saved = json.loads(normal_log)
            self.assertEqual(saved["cases"][0]["response_format_used"], "json_schema")


if __name__ == "__main__": unittest.main()
