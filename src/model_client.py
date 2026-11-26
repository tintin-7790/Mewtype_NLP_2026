import json
import http.client
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
            "differential_diagnoses": {"type": "array", "maxItems": 3, "items": {
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
            "plan_goals": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "retrieval_queries": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        }}}}}}

COUNTER_EVIDENCE_CODES = [
    "COUNTER_EVIDENCE", "AMBIGUOUS_FACT_DIRECTION",
    "MISSING_DISCRIMINATING_EVIDENCE", "OVERSTATED_CERTAINTY",
]
COUNTER_EVIDENCE_SCHEMA = {"name": "compact_counter_evidence_review", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["reviewed_problem_ids", "issues"],
    "properties": {
        "reviewed_problem_ids": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["problem_id", "code", "fact_ids", "note"],
            "properties": {
                "problem_id": {"type": "string", "minLength": 1},
                "code": {"type": "string", "enum": COUNTER_EVIDENCE_CODES},
                "fact_ids": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string", "minLength": 1},
            }},
        },
    }}}

VERIFIER_ISSUE_CODES = [
    "PATIENT_FACT_CONTRADICTION", "NUMERIC_VALUE_ERROR", "UNIT_ERROR",
    "TEMPORAL_CONFLICT", "CORRECTION_IGNORED", "UNSUPPORTED_NEGATIVE",
    "PLAN_AS_COMPLETED", "RESULT_AS_AVAILABLE", "DIAGNOSIS_CERTAINTY_UPGRADE",
    "INFERENCE_UNDER_SUPPORTED", "IMPORTANT_EVIDENCE_OMITTED", "PLAN_ALIGNMENT_WEAK",
    "UNREFERENCED_INFERENCE", "UNSAFE_RECOMMENDATION", "SOURCE_MISMATCH",
]
_VERIFIER_ISSUE = {"type": "object", "additionalProperties": False,
    "required": ["claim_id", "code", "severity", "hint", "source_ids"],
    "properties": {
        "claim_id": {"type": "string", "minLength": 1},
        "code": {"type": "string", "enum": VERIFIER_ISSUE_CODES},
        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        "hint": {"type": "string", "minLength": 1},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    }}
VERIFIER_SCHEMA = {"name": "claim_verification", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["checked_claim_ids", "issues"],
    "properties": {"checked_claim_ids": {"type": "array", "items": {"type": "string"}},
                   "issues": {"type": "array", "items": _VERIFIER_ISSUE}}}}

REVISER_SCHEMA = {"name": "localized_claim_patch", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["edits", "review_items_to_add"],
    "properties": {
        "edits": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["claim_id", "section", "action", "replacement"], "properties": {
                "claim_id": {"type": "string", "minLength": 1},
                "section": {"type": "string", "enum": ["case_features", "diagnostic_discussion", "plan",
                                                           "evidence", "review_items"]},
                "action": {"type": "string", "enum": ["replace", "remove"]},
                "replacement": {"type": "string"},
            }}},
        "review_items_to_add": {"type": "array", "items": {"type": "string"}},
    }}}


class _ChatResponseError(ValueError):
    pass


class Budget:
    def __init__(self, maximum: int): self.maximum, self.calls = maximum, 0
    @property
    def remaining(self): return max(0, self.maximum - self.calls)
    def take(self):
        if self.calls >= self.maximum: raise CaseError("CALL_BUDGET_EXCEEDED", "已达到显式配置的每例调用预算。")
        self.calls += 1


class MockClient:
    def __init__(self, script: Path):
        self.entries = json.loads(script.read_text(encoding="utf-8")); self.index = 0; self.total_calls = 0
        self.total_usage = _empty_usage(); self.call_events = []
    def call(self, messages, response_format, budget: Budget, *, stage="model", schema=None):
        budget.take(); self.total_calls += 1
        prompt_metrics = _message_metrics(messages)
        if self.index >= len(self.entries):
            self._event(stage, response_format, False, "MOCK_EXHAUSTED", prompt_metrics); raise CaseError("MOCK_EXHAUSTED", "mock 响应不足。")
        entry = self.entries[self.index]; self.index += 1
        if entry.get("exception") == "timeout":
            self._event(stage, response_format, False, "MODEL_TIMEOUT", prompt_metrics); raise CaseError("MODEL_TIMEOUT", "mock 超时。")
        if entry.get("exception"):
            self._event(stage, response_format, False, "MODEL_ERROR", prompt_metrics); raise CaseError("MODEL_ERROR", str(entry["exception"]))
        usage = _safe_usage(entry.get("usage"))
        for key in self.total_usage:
            self.total_usage[key] += usage[key]
        self._event(stage, response_format, True, None, prompt_metrics,
                    entry.get("finish_reason"), usage, entry.get("max_tokens"))
        return entry.get("content", "")
    def _event(self, stage, response_format, success, error_type, prompt_metrics,
               finish_reason=None, usage=None, max_tokens=None):
        self.call_events.append({"stage": stage, "model": "mock", "response_format": response_format,
                                 "call_success": success, "json_parse_success": None,
                                 "parse_error_type": None, "parse_error_position": None,
                                 "schema_validation_success": None, "schema_error_type": None,
                                 "token_usage": usage or _empty_usage(), "elapsed_ms": 0.0,
                                 "call_error_type": error_type, "finish_reason": finish_reason,
                                 "max_tokens": max_tokens, **prompt_metrics})
    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)
    def record_schema_result(self, stage, success, error_type=None):
        _update_schema_event(self.call_events, stage, success, error_type)


class ChatCompletionsClient:
    def __init__(self, config: ModelConfig):
        self.config = config; self.total_calls = 0; self.total_usage = _empty_usage()
        self.format_attempts: list[str] = []; self.last_response_format: str | None = None
        self.debug_hook = None; self.call_events: list[dict] = []
        self._thinking_disabled_by_compatibility = False
    def call(self, messages, response_format, budget: Budget, *, stage="model", schema=None):
        formats = list(dict.fromkeys(
            [response_format] + list(self.config.fallback_response_formats)
        ))
        last = None
        for fmt in formats:
            try:
                return self._request(messages, fmt, budget, stage, schema)
            except CaseError as exc:
                last = exc
                # Alternate response formats are only meaningful when the
                # service explicitly rejected that format.  A timeout or
                # malformed response must not silently spend another call.
                if exc.code != "RESPONSE_FORMAT_REJECTED":
                    raise
        raise last or CaseError("MODEL_ERROR", "模型调用失败。")

    def _request(self, messages, fmt, budget, stage, schema):
        generation = self.config.generation_for(stage)
        headers = self._headers()
        prompt_metrics = _message_metrics(messages)
        transport_attempt = 0
        compatibility_retry = False
        while True:
            body = self._request_body(messages, fmt, schema, generation)
            effective_generation = _effective_generation_audit(
                body, generation, self.config.send_generation_parameters,
            )
            request = urllib.request.Request(
                self.config.endpoint, json.dumps(body).encode(), headers, method="POST",
            )
            budget.take()
            self.total_calls += 1
            self.format_attempts.append(fmt)
            request_started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    if generation["stream"]:
                        parsed = _extract_streaming_chat_completion(response)
                    else:
                        payload = json.loads(response.read().decode("utf-8"))
                        parsed = _extract_chat_completion(payload)
                self._record_usage(parsed["usage"])
                self.last_response_format = fmt
                self._record_call_event(
                    stage, fmt, True, parsed["usage"], request_started,
                    prompt_metrics=prompt_metrics,
                    finish_reason=parsed["finish_reason"],
                    generation=effective_generation,
                    compatibility_retry=compatibility_retry,
                    reasoning_content_present=parsed["reasoning_content_present"],
                    reasoning_content_chars=parsed["reasoning_content_chars"],
                )
                self._debug({"event": "response", "model": self.config.model,
                             "stage": stage, "requested_response_format": fmt,
                             "content": parsed["content"], "usage": parsed["usage"],
                             "reasoning_content_present": parsed["reasoning_content_present"]})
                # reasoning_content is intentionally never used as the final
                # JSON and is never copied into the normal or raw-response log.
                return parsed["content"]
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")[:4000]
                downgrade_eligible = self._can_downgrade_thinking(exc, body_text, body)
                can_downgrade = downgrade_eligible and budget.remaining > 0
                self._record_call_event(
                    stage, fmt, False, None, request_started, type(exc).__name__,
                    prompt_metrics, generation=effective_generation,
                    compatibility_retry=compatibility_retry,
                    compatibility_downgrade_triggered=can_downgrade,
                    compatibility_downgrade_blocked_by_budget=(
                        downgrade_eligible and not can_downgrade
                    ),
                )
                self._debug({"event": "http_error", "model": self.config.model, "stage": stage,
                             "requested_response_format": fmt, "http_status": exc.code,
                             "service_error": body_text})
                if can_downgrade:
                    # One physical retry is allowed, counted against the same
                    # per-case budget.  The capability decision is cached so
                    # later stages do not repeat the rejected parameter.
                    self._thinking_disabled_by_compatibility = True
                    compatibility_retry = True
                    continue
                if 400 <= exc.code < 500:
                    code = "RESPONSE_FORMAT_REJECTED" if fmt in ("json_object", "json_schema") else "MODEL_HTTP_ERROR"
                    raise CaseError(code, f"模型接口 HTTP {exc.code}，请求格式={fmt}；未执行格式回退。") from exc
                if transport_attempt >= self.config.max_retries:
                    raise CaseError("MODEL_HTTP_ERROR", f"模型接口 HTTP {exc.code}，请求格式={fmt}。") from exc
                transport_attempt += 1
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    http.client.HTTPException, ConnectionError) as exc:
                self._record_call_event(
                    stage, fmt, False, None, request_started, type(exc).__name__,
                    prompt_metrics, generation=effective_generation,
                    compatibility_retry=compatibility_retry,
                )
                self._debug({"event": "client_error", "model": self.config.model, "stage": stage,
                             "requested_response_format": fmt, "error_type": type(exc).__name__})
                if transport_attempt >= self.config.max_retries:
                    code = "MODEL_TIMEOUT" if _is_timeout_error(exc) else "MODEL_ERROR"
                    raise CaseError(code, f"模型接口失败：{type(exc).__name__}") from exc
                transport_attempt += 1
            except (json.JSONDecodeError, UnicodeError, _ChatResponseError) as exc:
                self._record_call_event(
                    stage, fmt, False, None, request_started, type(exc).__name__,
                    prompt_metrics, generation=effective_generation,
                    compatibility_retry=compatibility_retry,
                )
                self._debug({"event": "invalid_response", "model": self.config.model,
                             "stage": stage, "requested_response_format": fmt,
                             "error_type": type(exc).__name__})
                raise CaseError(
                    "MODEL_INVALID_RESPONSE",
                    f"模型接口返回无效 Chat Completions 响应：{type(exc).__name__}",
                ) from exc

    def _request_body(self, messages, fmt, schema, generation):
        if not isinstance(self.config.extra_body, dict):
            raise TaskError("CONFIG_INVALID", "extra_body 必须是对象。")
        body = dict(self.config.extra_body)
        body.update({"model": self.config.model, "messages": messages,
                     "stream": generation["stream"]})
        if self.config.send_generation_parameters:
            body.update({"temperature": generation["temperature"],
                         "top_p": generation["top_p"],
                         "max_tokens": generation["max_tokens"],
                         "seed": generation["seed"]})
            template_kwargs = body.get("chat_template_kwargs", {})
            if not isinstance(template_kwargs, dict):
                raise TaskError("CONFIG_INVALID", "chat_template_kwargs 必须是对象。")
            template_kwargs = dict(template_kwargs)
            if self._thinking_disabled_by_compatibility:
                template_kwargs.pop("enable_thinking", None)
            elif generation["enable_thinking"] is not None:
                template_kwargs["enable_thinking"] = generation["enable_thinking"]
            if template_kwargs:
                body["chat_template_kwargs"] = template_kwargs
            else:
                body.pop("chat_template_kwargs", None)
        if fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif fmt == "json_schema":
            selected_schema = schema or MEDICAL_SCHEMA
            body["response_format"] = {"type": "json_schema", "json_schema": selected_schema}
        else:
            body.pop("response_format", None)
        return body

    def _can_downgrade_thinking(self, exc, body_text, body) -> bool:
        if (not self.config.compatibility_retry_without_thinking
                or self._thinking_disabled_by_compatibility
                or exc.code not in (400, 422)
                or "chat_template_kwargs" not in body):
            return False
        lowered = body_text.lower()
        return "enable_thinking" in lowered or "chat_template_kwargs" in lowered

    def _record_call_event(self, stage, fmt, success, usage, started, error_type=None,
                           prompt_metrics=None, finish_reason=None, generation=None,
                           compatibility_retry=False,
                           compatibility_downgrade_triggered=False,
                           compatibility_downgrade_blocked_by_budget=False,
                           reasoning_content_present=False,
                           reasoning_content_chars=0):
        generation = generation or {}
        self.call_events.append({"stage": stage, "model": self.config.model, "response_format": fmt,
                                 "call_success": success, "json_parse_success": None,
                                 "parse_error_type": None, "parse_error_position": None,
                                 "schema_validation_success": None, "schema_error_type": None,
                                 "token_usage": _safe_usage(usage),
                                 "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                                 "call_error_type": error_type, "finish_reason": finish_reason,
                                 "max_tokens": generation.get("max_tokens"),
                                 "generation_parameters": generation,
                                 "timeout_seconds": self.config.timeout_seconds,
                                 "compatibility_retry": compatibility_retry,
                                 "compatibility_downgrade_triggered":
                                     compatibility_downgrade_triggered,
                                 "compatibility_downgrade_blocked_by_budget":
                                     compatibility_downgrade_blocked_by_budget,
                                 "reasoning_content_present": reasoning_content_present,
                                 "reasoning_content_chars": reasoning_content_chars,
                                 **(prompt_metrics or {})})

    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)

    def record_schema_result(self, stage, success, error_type=None):
        _update_schema_event(self.call_events, stage, success, error_type)

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
        generation = self.config.generation_for(stage)
        if generation["stream"]:
            raise CaseError("UNSUPPORTED_STREAM", "MLX 直接适配器当前仅支持非流式生成。")
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
        prompt_metrics = _message_metrics(messages)
        try:
            result = generate(model, tokenizer, prompt=prompt,
                              max_tokens=generation["max_tokens"], verbose=False)
            self.call_events.append({"stage": stage, "model": self.config.model, "response_format": response_format,
                                     "call_success": True, "json_parse_success": None,
                                     "parse_error_type": None, "parse_error_position": None,
                                     "schema_validation_success": None, "schema_error_type": None,
                                     "token_usage": _empty_usage(), "elapsed_ms": 0.0, "call_error_type": None,
                                     "finish_reason": None, "max_tokens": generation["max_tokens"],
                                     "generation_parameters": {
                                         "max_tokens": generation["max_tokens"],
                                         "stream": False,
                                         "enable_thinking": None,
                                     },
                                     "timeout_seconds": None,
                                     "reasoning_content_present": False,
                                     "reasoning_content_chars": 0})
            self.call_events[-1].update(prompt_metrics)
            return result
        except Exception as exc:
            raise CaseError("LOCAL_MODEL_ERROR", f"本地 MLX 推理失败：{type(exc).__name__}") from exc
    def record_parse_result(self, stage, success, error_type=None, position=None):
        _update_parse_event(self.call_events, stage, success, error_type, position)

    def record_schema_result(self, stage, success, error_type=None):
        _update_schema_event(self.call_events, stage, success, error_type)


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


def _extract_chat_completion(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise _ChatResponseError("response root is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _ChatResponseError("choices[0] is missing")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _ChatResponseError("choices[0].message is missing")
    content = message.get("content")
    if not isinstance(content, str):
        raise _ChatResponseError("message.content is not text")
    reasoning = message.get("reasoning_content")
    return {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
        "reasoning_content_present": reasoning is not None,
        "reasoning_content_chars": len(reasoning) if isinstance(reasoning, str) else 0,
    }


def _extract_streaming_chat_completion(response) -> dict:
    content_parts: list[str] = []
    content_seen = False
    reasoning_present = False
    reasoning_chars = 0
    finish_reason = None
    usage = {}
    data_seen = False
    for raw_line in response:
        if not isinstance(raw_line, (bytes, bytearray)):
            raise _ChatResponseError("stream line is not bytes")
        line = bytes(raw_line).decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise _ChatResponseError("stream chunk is not an object")
        if isinstance(chunk.get("error"), dict):
            raise _ChatResponseError("stream returned an error object")
        data_seen = True
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not choices:
            continue
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise _ChatResponseError("stream choices[0] is invalid")
        choice = choices[0]
        delta = choice.get("delta")
        if delta is None:
            delta = choice.get("message")
        if not isinstance(delta, dict):
            raise _ChatResponseError("stream delta is invalid")
        content = delta.get("content")
        if isinstance(content, str):
            content_seen = True
            content_parts.append(content)
        elif content is not None:
            raise _ChatResponseError("stream content is not text")
        reasoning = delta.get("reasoning_content")
        if reasoning is not None:
            reasoning_present = True
            if isinstance(reasoning, str):
                reasoning_chars += len(reasoning)
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
    if not data_seen or not content_seen:
        raise _ChatResponseError("stream contains no final content")
    return {
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "reasoning_content_present": reasoning_present,
        "reasoning_content_chars": reasoning_chars,
    }


def _effective_generation_audit(body: dict, configured: dict,
                                parameters_sent: bool) -> dict:
    if not parameters_sent:
        return {"stream": bool(body.get("stream")), "parameters_sent": False,
                "max_tokens": None, "enable_thinking": None}
    template_kwargs = body.get("chat_template_kwargs")
    enable_thinking = (template_kwargs.get("enable_thinking")
                       if isinstance(template_kwargs, dict) else None)
    return {
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_tokens": body.get("max_tokens"),
        "seed": body.get("seed"),
        "stream": bool(body.get("stream")),
        "enable_thinking": enable_thinking,
        "parameters_sent": True,
        "configured_enable_thinking": configured.get("enable_thinking"),
    }


def _message_metrics(messages) -> dict:
    contents = [message.get("content", "") if isinstance(message, dict) else "" for message in messages]
    return {
        "message_count": len(contents),
        "input_text_chars": sum(len(content) for content in contents),
        "system_prompt_chars": len(contents[0]) if contents else 0,
        "user_payload_chars": len(contents[-1]) if len(contents) > 1 else 0,
    }


def _safe_usage(usage):
    result = _empty_usage()
    if isinstance(usage, dict):
        for key in result:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                result[key] = value
    return result


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (socket.timeout, TimeoutError))
    return False


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


def _update_schema_event(events, stage, success, error_type):
    for event in reversed(events):
        if event.get("stage") == stage and event.get("call_success"):
            event["schema_validation_success"] = success
            event["schema_error_type"] = error_type
            return


def _decode_error_type(exc: json.JSONDecodeError, length: int) -> str:
    if exc.pos >= max(0, length - 1) or exc.msg.startswith(("Unterminated", "Expecting ',' delimiter",
                                                            "Expecting value", "Expecting property name")):
        return "INCOMPLETE_JSON"
    return "INVALID_JSON_SYNTAX"
