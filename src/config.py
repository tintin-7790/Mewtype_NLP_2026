import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import TaskError


@dataclass
class ModelConfig:
    environment: str
    adapter: str
    model: str
    endpoint: str = ""
    response_format: str = "text"
    fallback_response_formats: list[str] = field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 0
    max_calls_per_case: int = 1
    seed: int = 20261125
    extra_body: dict = field(default_factory=dict)
    credential_env: str = ""


def load_config(path: Path | None, mode: str, pipeline: str, minimum_calls: int | None = None) -> ModelConfig | None:
    minimum = minimum_calls if minimum_calls is not None else _minimum_calls(pipeline)
    if mode == "mock":
        return None
    if mode == "dev_api":
        values = {name: os.environ.get(name, "").strip()
                  for name in ("DASHSCOPE_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL")}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise TaskError("DEV_API_CONFIG_MISSING",
                            f"dev_api 缺少环境变量：{', '.join(missing)}")
        base_url = values["QWEN_BASE_URL"].rstrip("/")
        endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        return ModelConfig(environment="dev_api", adapter="dev_api_chat_completions",
                           model=values["QWEN_MODEL"], endpoint=endpoint,
                           response_format="text", max_calls_per_case=minimum,
                           credential_env="DASHSCOPE_API_KEY")
    if path is None:
        raise TaskError("CONFIG_MISSING", "真实运行必须通过 --config 提供模型配置。")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError("CONFIG_INVALID", f"无法读取模型配置：{exc}") from exc
    required = ("environment", "adapter", "model", "max_calls_per_case")
    missing = [key for key in required if key not in raw]
    if missing:
        raise TaskError("CONFIG_INVALID", f"模型配置缺少：{', '.join(missing)}")
    if raw["environment"] != mode:
        raise TaskError("CONFIG_INVALID", "配置 environment 必须与 --mode 一致。")
    if mode == "competition" and raw["adapter"] != "chat_completions":
        raise TaskError("CONFIG_INVALID", "competition 模式仅允许赛方确认的 Chat Completions 适配器。")
    if raw["adapter"] == "chat_completions" and not raw.get("endpoint"):
        raise TaskError("CONFIG_INVALID", "Chat Completions 配置必须提供完整 endpoint。")
    allowed = {"text", "json_object", "json_schema"}
    if raw.get("response_format", "text") not in allowed:
        raise TaskError("CONFIG_INVALID", "response_format 必须为 text/json_object/json_schema。")
    if int(raw["max_calls_per_case"]) < minimum:
        raise TaskError("CONFIG_INVALID", f"{pipeline} 至少需要每例 {minimum} 次调用预算。")
    fields = ModelConfig.__dataclass_fields__
    return ModelConfig(**{key: value for key, value in raw.items() if key in fields})


def _minimum_calls(pipeline: str) -> int:
    if pipeline == "evidence_pipeline":
        return 3
    if pipeline == "clinical_pipeline":
        return 4
    return 1
