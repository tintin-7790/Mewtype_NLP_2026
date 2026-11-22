import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ModelConfig
from .errors import CaseError, TaskError


MEDICAL_SCHEMA = {"name": "medical_record", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["case_features", "diagnostic_discussion", "plan", "evidence", "review_items"],
    "properties": {
        "case_features": {"type": "string", "minLength": 1},
        "diagnostic_discussion": {"type": "string", "minLength": 1},
        "plan": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["statement", "source_ids", "kind"], "properties": {
                "statement": {"type": "string", "minLength": 1},
                "source_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "kind": {"enum": ["fact", "inference"]}}}},
        "review_items": {"type": "array", "items": {"type": "string"}}
    }}}


class Budget:
    def __init__(self, maximum: int): self.maximum, self.calls = maximum, 0
    def take(self):
        if self.calls >= self.maximum: raise CaseError("CALL_BUDGET_EXCEEDED", "已达到显式配置的每例调用预算。")
        self.calls += 1


class MockClient:
    def __init__(self, script: Path):
        self.entries = json.loads(script.read_text(encoding="utf-8")); self.index = 0; self.total_calls = 0
        self.total_usage = _empty_usage()
    def call(self, messages, response_format, budget: Budget):
        budget.take(); self.total_calls += 1
        if self.index >= len(self.entries): raise CaseError("MOCK_EXHAUSTED", "mock 响应不足。")
        entry = self.entries[self.index]; self.index += 1
        if entry.get("exception") == "timeout": raise CaseError("MODEL_TIMEOUT", "mock 超时。")
        if entry.get("exception"): raise CaseError("MODEL_ERROR", str(entry["exception"]))
        return entry.get("content", "")


class ChatCompletionsClient:
    def __init__(self, config: ModelConfig):
        self.config = config; self.total_calls = 0; self.total_usage = _empty_usage()
    def call(self, messages, response_format, budget: Budget):
        formats = [response_format] + list(self.config.fallback_response_formats)
        last = None
        for fmt in formats:
            try: return self._request(messages, fmt, budget)
            except CaseError as exc: last = exc
        raise last or CaseError("MODEL_ERROR", "模型调用失败。")
    def _request(self, messages, fmt, budget):
        body = {"model": self.config.model, "messages": messages, "stream": False,
                "temperature": self.config.temperature, "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens, "seed": self.config.seed, **self.config.extra_body}
        if fmt == "json_object": body["response_format"] = {"type": "json_object"}
        elif fmt == "json_schema": body["response_format"] = {"type": "json_schema", "json_schema": MEDICAL_SCHEMA}
        headers = self._headers()
        request = urllib.request.Request(self.config.endpoint, json.dumps(body).encode(), headers, method="POST")
        for attempt in range(self.config.max_retries + 1):
            budget.take(); self.total_calls += 1
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._record_usage(payload.get("usage"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, socket.timeout, KeyError, ValueError) as exc:
                if attempt == self.config.max_retries:
                    code = "MODEL_TIMEOUT" if isinstance(exc, (socket.timeout, TimeoutError)) else "MODEL_ERROR"
                    raise CaseError(code, f"模型接口失败：{type(exc).__name__}") from exc

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _record_usage(self, usage):
        if not isinstance(usage, dict):
            return
        for key in self.total_usage:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                self.total_usage[key] += value


class DevApiChatCompletionsClient(ChatCompletionsClient):
    """阿里云百炼开发适配器；与 competition 的无认证客户端隔离。"""

    def __init__(self, config: ModelConfig):
        if config.environment != "dev_api" or config.adapter != "dev_api_chat_completions":
            raise TaskError("CONFIG_INVALID", "开发 API 客户端只允许 dev_api 环境。")
        api_key = os.environ.get(config.credential_env, "")
        if not api_key:
            raise TaskError("DEV_API_CONFIG_MISSING", f"缺少环境变量：{config.credential_env}")
        self._api_key = api_key
        super().__init__(config)

    def _headers(self):
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}


class MlxDirectClient:
    def __init__(self, config: ModelConfig):
        if config.environment != "local": raise TaskError("CONFIG_INVALID", "MLX 适配器仅允许 local 环境。")
        self.config = config; self._loaded = None; self.total_calls = 0; self.total_usage = _empty_usage()
    def call(self, messages, response_format, budget: Budget):
        if response_format != "text": raise CaseError("UNSUPPORTED_RESPONSE_FORMAT", "MLX 直接适配器当前仅验证 text。")
        budget.take(); self.total_calls += 1
        try:
            from mlx_lm import generate, load
        except ImportError as exc: raise TaskError("LOCAL_DEPENDENCY_MISSING", "local 模式缺少 mlx_lm；它不是正式提交依赖。") from exc
        if self._loaded is None:
            try:
                self._loaded = load(self.config.model)
            except Exception as exc:
                raise CaseError("LOCAL_MODEL_LOAD_ERROR", f"本地 MLX 模型加载失败：{type(exc).__name__}") from exc
        model, tokenizer = self._loaded
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            return generate(model, tokenizer, prompt=prompt, max_tokens=self.config.max_tokens, verbose=False)
        except Exception as exc:
            raise CaseError("LOCAL_MODEL_ERROR", f"本地 MLX 推理失败：{type(exc).__name__}") from exc


def parse_model_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines(); text = "\n".join(lines[1:-1]); text = text[4:].lstrip() if text.startswith("json") else text
    try: return json.loads(text)
    except json.JSONDecodeError as exc: raise CaseError("MODEL_INVALID_JSON", f"模型未返回有效 JSON：{exc.msg}") from exc


def _empty_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
