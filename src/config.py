import json
import os
import re
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
    send_generation_parameters: bool = True


def load_config(path: Path | None, mode: str, pipeline: str, minimum_calls: int | None = None,
                *, official_env_path: Path | None = None) -> ModelConfig | None:
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
    if mode == "competition" and path is None and official_env_path is not None:
        return _load_official_env(official_env_path, minimum)
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


_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _load_official_env(path: Path, minimum_calls: int) -> ModelConfig:
    """Load the zero-argument competition profile without importing dotenv.

    ENDPOINT is already the complete Chat Completions URL and is deliberately
    preserved verbatim.  Optional timeout/retry controls are engineering
    safeguards, not assumptions about the unpublished service quota.
    """
    values = _read_env_file(path)
    missing = [name for name in ("MODEL_NAME", "ENDPOINT") if not values.get(name, "").strip()]
    if missing:
        raise TaskError("COMPETITION_ENV_MISSING", f".env 缺少：{', '.join(missing)}")
    timeout = _positive_float(values.get("TIMEOUT_SECONDS", "120"), "TIMEOUT_SECONDS")
    retries = _bounded_int(values.get("MAX_RETRIES", "0"), "MAX_RETRIES", 0, 10)
    maximum = minimum_calls + retries
    return ModelConfig(
        environment="competition",
        adapter="chat_completions",
        model=values["MODEL_NAME"].strip(),
        endpoint=values["ENDPOINT"].strip(),
        response_format="text",
        timeout_seconds=timeout,
        max_retries=retries,
        max_calls_per_case=maximum,
        send_generation_parameters=False,
    )


def _read_env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TaskError("COMPETITION_ENV_MISSING", f"无法读取正式运行配置 {path.name}：{exc}") from exc
    values: dict[str, str] = {}
    for line_number, original in enumerate(lines, 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise TaskError("COMPETITION_ENV_INVALID", f".env 第 {line_number} 行不是 KEY=VALUE。")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not _ENV_KEY.fullmatch(key):
            raise TaskError("COMPETITION_ENV_INVALID", f".env 第 {line_number} 行变量名无效。")
        if key in values:
            raise TaskError("COMPETITION_ENV_INVALID", f".env 变量 {key} 重复定义。")
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise TaskError("COMPETITION_ENV_INVALID", f".env 第 {line_number} 行引号不完整。")
            value = value[1:-1]
        values[key] = value
    return values


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise TaskError("COMPETITION_ENV_INVALID", f"{name} 必须为正数。") from exc
    if parsed <= 0:
        raise TaskError("COMPETITION_ENV_INVALID", f"{name} 必须为正数。")
    return parsed


def _bounded_int(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TaskError("COMPETITION_ENV_INVALID", f"{name} 必须为整数。") from exc
    if not minimum <= parsed <= maximum:
        raise TaskError("COMPETITION_ENV_INVALID", f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return parsed


def _minimum_calls(pipeline: str) -> int:
    if pipeline == "evidence_pipeline":
        return 3
    if pipeline == "clinical_pipeline":
        return 4
    return 1
