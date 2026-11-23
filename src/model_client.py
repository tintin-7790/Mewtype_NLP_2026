import json
import os
import re
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
                "kind": {"type": "string", "enum": ["fact", "inference"]}}}},
        "review_items": {"type": "array", "items": {"type": "string"}}
    }}}

PLANNER_SCHEMA = {"name": "clinical_plan", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["problems"],
    "properties": {"problems": {"type": "array", "minItems": 1, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["problem_id", "name", "priority", "origin", "supporting_facts",
                     "contradicting_facts", "differential_diagnoses", "plan_goals", "retrieval_queries"],
        "properties": {
            "problem_id": {"type": "string", "minLength": 1},
            "name": {"type": "string", "minLength": 1},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "origin": {"type": "string", "enum": ["documented_problem", "model_inference", "mixed"]},
            "supporting_facts": {"type": "array", "items": {"type": "string"}},
            "contradicting_facts": {"type": "array", "items": {"type": "string"}},
            "differential_diagnoses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "certainty", "origin", "supporting_facts",
                             "contradicting_facts", "missing_evidence"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "certainty": {"type": "string", "enum": ["suspected", "rule_out", "possible", "unlikely"]},
                    "origin": {"type": "string", "enum": ["documented_problem", "model_inference", "mixed"]},
                    "supporting_facts": {"type": "array", "items": {"type": "string"}},
                    "contradicting_facts": {"type": "array", "items": {"type": "string"}},
                    "missing_evidence": {"type": "array", "items": {"type": "string"}},
                }}},
            "plan_goals": {"type": "array", "items": {"type": "string"}},
            "retrieval_queries": {"type": "array", "items": {"type": "string"}},
        }}}}}}

_VERIFIER_CLAIM = {"type": "object", "additionalProperties": False,
    "required": ["claim_id", "verdict", "rationale", "source_ids"], "properties": {
        "claim_id": {"type": "string", "minLength": 1},
        "verdict": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "NOT_ADDRESSED", "GROUNDED",
                                                     "PARTIALLY_GROUNDED", "INSUFFICIENT_EVIDENCE",
                                                     "REQUIRES_CLINICIAN_REVIEW"]},
        "rationale": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    }}
_VERIFIER_ISSUE = {"type": "object", "additionalProperties": False,
    "required": ["claim_id", "section", "statement", "error_type", "severity", "revision_hint", "source_ids"],
    "properties": {
        "claim_id": {"type": "string", "minLength": 1},
        "section": {"type": "string", "enum": ["case_features", "diagnostic_discussion", "plan",
                                                   "evidence", "review_items"]},
        "statement": {"type": "string"}, "error_type": {"type": "string"},
        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        "revision_hint": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    }}
VERIFIER_SCHEMA = {"name": "claim_verification", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["claims", "issues"],
    "properties": {"claims": {"type": "array", "items": _VERIFIER_CLAIM},
                   "issues": {"type": "array", "items": _VERIFIER_ISSUE}}}}

REVISER_SCHEMA = {"name": "localized_revision", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["section_updates", "evidence", "review_items"],
    "properties": {
        "section_updates": {"type": "object", "additionalProperties": False,
                            "properties": {key: {"type": "string", "minLength": 1}
                                           for key in ("case_features", "diagnostic_discussion", "plan")}},
        "evidence": MEDICAL_SCHEMA["schema"]["properties"]["evidence"],
        "review_items": MEDICAL_SCHEMA["schema"]["properties"]["review_items"],
    }}}


class Budget:
    def __init__(self, maximum: int): self.maximum, self.calls = maximum, 0
    def take(self):
        if self.calls >= self.maximum: raise CaseError("CALL_BUDGET_EXCEEDED", "已达到显式配置的每例调用预算。")
        self.calls += 1


class MockClient:
    def __init__(self, script: Path):
        self.entries = json.loads(script.read_text(encoding="utf-8")); self.index = 0; self.total_calls = 0
        self.total_usage = _empty_usage(); self.call_events = []
    def call(self, messages, response_format, budget: Budget, *, stage="model", schema=None):
        budget.take(); self.total_calls += 1
        if self.index >= len(self.entries):
            self._event(stage, response_format, False, "MOCK_EXHAUSTED"); raise CaseError("MOCK_EXHAUSTED", "mock 响应不足。")
        entry = self.entries[self.index]; self.index += 1
        if entry.get("exception") == "timeout":
            self._event(stage, response_format, False, "MODEL_TIMEOUT"); raise CaseError("MODEL_TIMEOUT", "mock 超时。")
        if entry.get("exception"):
            self._event(stage, response_format, False, "MODEL_ERROR"); raise CaseError("MODEL_ERROR", str(entry["exception"]))
        self._event(stage, response_format, True, None)
        return entry.get("content", "")
    def _event(self, stage, response_format, success, error_type):
        self.call_events.append({"stage": stage, "model": "mock", "response_format": response_format,
                                 "call_success": success, "json_parse_success": None,
                                 "parse_error_type": None, "parse_error_position": None,
                                 "token_usage": _empty_usage(), "elapsed_ms": 0.0,
                                 "call_error_type": error_type})
    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)


class ChatCompletionsClient:
    def __init__(self, config: ModelConfig):
        self.config = config; self.total_calls = 0; self.total_usage = _empty_usage()
        self.format_attempts: list[str] = []; self.last_response_format: str | None = None
        self.debug_hook = None; self.call_events: list[dict] = []
    def call(self, messages, response_format, budget: Budget, *, stage="model", schema=None):
        formats = [response_format] + list(self.config.fallback_response_formats)
        last = None
        for fmt in formats:
            try: return self._request(messages, fmt, budget, stage, schema)
            except CaseError as exc: last = exc
        raise last or CaseError("MODEL_ERROR", "模型调用失败。")
    def _request(self, messages, fmt, budget, stage, schema):
        body = {"model": self.config.model, "messages": messages, "stream": False,
                "temperature": self.config.temperature, "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens, "seed": self.config.seed, **self.config.extra_body}
        if fmt == "json_object": body["response_format"] = {"type": "json_object"}
        elif fmt == "json_schema":
            selected_schema = schema or MEDICAL_SCHEMA
            body["response_format"] = {"type": "json_schema", "json_schema": selected_schema}
        headers = self._headers()
        request = urllib.request.Request(self.config.endpoint, json.dumps(body).encode(), headers, method="POST")
        for attempt in range(self.config.max_retries + 1):
            budget.take(); self.total_calls += 1; self.format_attempts.append(fmt)
            request_started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._record_usage(payload.get("usage"))
                content = payload["choices"][0]["message"]["content"]
                self.last_response_format = fmt
                self._record_call_event(stage, fmt, True, payload.get("usage"), request_started)
                self._debug({"event": "response", "model": self.config.model,
                             "stage": stage, "requested_response_format": fmt, "content": content,
                             "usage": payload.get("usage", {})})
                return content
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")[:4000]
                self._record_call_event(stage, fmt, False, None, request_started, type(exc).__name__)
                self._debug({"event": "http_error", "model": self.config.model, "stage": stage,
                             "requested_response_format": fmt, "http_status": exc.code,
                             "service_error": body_text})
                if 400 <= exc.code < 500:
                    code = "RESPONSE_FORMAT_REJECTED" if fmt in ("json_object", "json_schema") else "MODEL_HTTP_ERROR"
                    raise CaseError(code, f"模型接口 HTTP {exc.code}，请求格式={fmt}；未执行格式回退。") from exc
                if attempt == self.config.max_retries:
                    raise CaseError("MODEL_HTTP_ERROR", f"模型接口 HTTP {exc.code}，请求格式={fmt}。") from exc
            except (urllib.error.URLError, socket.timeout, KeyError, ValueError) as exc:
                self._record_call_event(stage, fmt, False, None, request_started, type(exc).__name__)
                self._debug({"event": "client_error", "model": self.config.model, "stage": stage,
                             "requested_response_format": fmt, "error_type": type(exc).__name__})
                if attempt == self.config.max_retries:
                    code = "MODEL_TIMEOUT" if isinstance(exc, (socket.timeout, TimeoutError)) else "MODEL_ERROR"
                    raise CaseError(code, f"模型接口失败：{type(exc).__name__}") from exc

    def _record_call_event(self, stage, fmt, success, usage, started, error_type=None):
        self.call_events.append({"stage": stage, "model": self.config.model, "response_format": fmt,
                                 "call_success": success, "json_parse_success": None,
                                 "parse_error_type": None, "parse_error_position": None,
                                 "token_usage": _safe_usage(usage),
                                 "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                                 "call_error_type": error_type})

    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _record_usage(self, usage):
        if not isinstance(usage, dict):
            return
        for key in self.total_usage:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                self.total_usage[key] += value

    def _debug(self, event: dict) -> None:
        hook = getattr(self, "debug_hook", None)
        if hook is not None:
            hook(event)


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
        self.config = config; self._loaded = None; self.total_calls = 0; self.total_usage = _empty_usage(); self.call_events = []
    def call(self, messages, response_format, budget: Budget, *, stage="model", schema=None):
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
            result = generate(model, tokenizer, prompt=prompt, max_tokens=self.config.max_tokens, verbose=False)
            self.call_events.append({"stage": stage, "model": self.config.model, "response_format": response_format,
                                     "call_success": True, "json_parse_success": None,
                                     "parse_error_type": None, "parse_error_position": None,
                                     "token_usage": _empty_usage(), "elapsed_ms": 0.0, "call_error_type": None})
            return result
        except Exception as exc:
            raise CaseError("LOCAL_MODEL_ERROR", f"本地 MLX 推理失败：{type(exc).__name__}") from exc
    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)


_COMPLETE_JSON_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z", re.IGNORECASE)


def parse_model_json(text: str, *, stage: str = "model", client=None):
    """Strictly parse one JSON value; only a single whole-response JSON fence is normalized."""
    if not isinstance(text, str):
        _record_parse(client, stage, False, "NON_TEXT_RESPONSE", None)
        raise CaseError("MODEL_INVALID_JSON", f"{stage} 阶段模型响应不是文本。")
    normalized = text.strip()
    fence = _COMPLETE_JSON_FENCE.fullmatch(normalized)
    if fence:
        normalized = fence.group("body").strip()
    elif normalized.startswith("```") or normalized.endswith("```"):
        _record_parse(client, stage, False, "MALFORMED_JSON_FENCE", 0)
        raise CaseError("MODEL_INVALID_JSON", f"{stage} 阶段 JSON 代码围栏不完整或围栏外含额外内容。")
    leading = len(normalized) - len(normalized.lstrip())
    candidate = normalized.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        error_type = _decode_error_type(exc, len(candidate))
        position = leading + exc.pos
        _record_parse(client, stage, False, error_type, position)
        raise CaseError("MODEL_INVALID_JSON",
                        f"{stage} 阶段模型未返回有效 JSON：{error_type}；{exc.msg} "
                        f"(line {exc.lineno}, column {exc.colno}, char {position})") from exc
    tail = candidate[end:]
    if tail.strip():
        relative = len(tail) - len(tail.lstrip())
        position = leading + end + relative
        marker = tail.lstrip()[0]
        error_type = "MULTIPLE_JSON_VALUES" if marker in "[{" else "JSON_TRAILING_TEXT"
        _record_parse(client, stage, False, error_type, position)
        raise CaseError("MODEL_INVALID_JSON",
                        f"{stage} 阶段 JSON 后存在额外内容：{error_type} (char {position})。")
    _record_parse(client, stage, True)
    return value


def _empty_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _safe_usage(usage):
    result = _empty_usage()
    if isinstance(usage, dict):
        for key in result:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                result[key] = value
    return result


def _record_parse(client, stage, success, error_type=None, position=None):
    recorder = getattr(client, "record_parse_result", None)
    if recorder is not None:
        recorder(stage, success, error_type, position)


def _update_parse_event(events, stage, success, error_type, position):
    for event in reversed(events):
        if event.get("stage") == stage and event.get("json_parse_success") is None and event.get("call_success"):
            event["json_parse_success"] = success
            event["parse_error_type"] = error_type
            event["parse_error_position"] = position
            return


def _decode_error_type(exc: json.JSONDecodeError, length: int) -> str:
    if exc.pos >= max(0, length - 1) or exc.msg.startswith(("Unterminated", "Expecting ',' delimiter",
                                                            "Expecting value", "Expecting property name")):
        return "INCOMPLETE_JSON"
    return "INVALID_JSON_SYNTAX"
