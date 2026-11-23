import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .config import load_config
from .errors import CaseError, TaskError
from .embedding_client import load_embedding_client
from .io_utils import atomic_write_json
from .model_client import ChatCompletionsClient, DevApiChatCompletionsClient, MlxDirectClient, MockClient
from .pipeline import PROMPT_VERSION, run_case
from .protocol import CaseError as ProtocolCaseError
from .protocol import failed_result, load_standard_cases, success_result
from .retrieval import DenseEvidenceRetriever


VERSION = "0.3.1"


def parser():
    p = argparse.ArgumentParser(description="循证智录首次病程记录生成程序")
    p.add_argument("--input", required=True, help="标准 JSON 数组；相对路径按当前工作目录解析")
    p.add_argument("--output", required=True, help="输出 JSON；相对路径按当前工作目录解析")
    p.add_argument("--mode", choices=("mock", "local", "dev_api", "competition"), default="competition")
    p.add_argument("--pipeline", choices=("baseline", "evidence_pipeline"), default="baseline")
    p.add_argument("--config", help="local/competition 模型配置 JSON；dev_api 仅读取指定环境变量")
    p.add_argument("--mock-responses", help="mock 模式响应脚本 JSON")
    p.add_argument("--log", help="运行日志路径；默认与输出同目录的 run_log.json")
    p.add_argument("--retrieval-config", help="可选 Dense 检索配置 JSON；仅用于 evidence_pipeline")
    p.add_argument("--dev-response-format", choices=("text", "json_object", "json_schema"),
                   help="仅 dev_api Baseline：显式测试服务端输出格式；不自动回退")
    p.add_argument("--debug-raw-response", help="仅虚构 dev_api Baseline：将原始响应写入独立调试文件")
    p.add_argument("--confirm-fictional-input", action="store_true",
                   help="确认输入为允许保存原始响应的虚构开发病例")
    return p


def main(argv=None):
    args = parser().parse_args(argv); started = time.time(); run_id = str(uuid.uuid4())
    output = Path(args.output); log_path = Path(args.log) if args.log else output.parent / "run_log.json"
    log = {"run_id": run_id, "version": VERSION, "prompt_version": PROMPT_VERSION,
           "environment": args.mode, "pipeline": args.pipeline, "development_label": args.mode != "competition",
           "started_at_epoch": started, "model": None, "parameters": {}, "cases": [], "status": "running"}
    try:
        config = load_config(Path(args.config) if args.config else None, args.mode, args.pipeline)
        if args.dev_response_format:
            if args.mode != "dev_api" or args.pipeline != "baseline":
                raise TaskError("DEV_FORMAT_SCOPE_INVALID", "--dev-response-format 仅用于 dev_api Baseline。")
            config.response_format = args.dev_response_format
            config.fallback_response_formats = []
        recorder = _debug_recorder(args, run_id)
        client, maximum, response_format = _client(args, config)
        if recorder:
            client.debug_hook = recorder.record
        embedding_client = None; retrieval_options = None
        if args.retrieval_config:
            if args.pipeline != "evidence_pipeline":
                raise TaskError("RETRIEVAL_PIPELINE_INVALID", "Dense 检索只允许用于 evidence_pipeline。")
            embedding_client, retrieval_options = load_embedding_client(Path(args.retrieval_config), args.mode)
            log["retrieval"] = {key: value for key, value in retrieval_options.items() if key != "vectors"}
        if config:
            log["model"] = config.model
            log["parameters"] = {"adapter": config.adapter, "response_format": config.response_format,
                                 "temperature": config.temperature, "top_p": config.top_p,
                                 "max_tokens": config.max_tokens, "seed": config.seed,
                                 "max_calls_per_case": config.max_calls_per_case,
                                 "max_retries": config.max_retries}
        else:
            log["model"] = "mock"; log["parameters"] = {"max_calls_per_case": maximum}
        cases = load_standard_cases(Path(args.input)); results = []
        for item in cases:
            if isinstance(item, ProtocolCaseError):
                case_id = getattr(item, "case_id", "")
                results.append(failed_result(case_id, item.code, item.message))
                log["cases"].append({"case_id": case_id, "status": "failed", "calls": 0, "error": item.code})
                continue
            try:
                calls_before = getattr(client, "total_calls", 0)
                usage_before = dict(getattr(client, "total_usage", {}))
                formats_before = len(getattr(client, "format_attempts", []))
                if recorder:
                    recorder.current_case_id = item.case_id
                retriever = (DenseEvidenceRetriever(
                    item, embedding_client,
                    max_chunk_chars=retrieval_options["max_chunk_chars"],
                    min_similarity=retrieval_options["min_similarity"],
                ) if embedding_client else None)
                record, meta = run_case(item, client, args.pipeline, response_format, maximum,
                                        retriever, retrieval_options["top_k"] if retrieval_options else 3)
                meta["token_usage"] = _usage_delta(usage_before, getattr(client, "total_usage", {}))
                if args.pipeline == "baseline":
                    attempts = list(getattr(client, "format_attempts", []))[formats_before:]
                    meta["response_formats_attempted"] = attempts
                    meta["response_format_used"] = getattr(client, "last_response_format", None)
                results.append(success_result(item.case_id, record))
                log["cases"].append({"case_id": item.case_id, "status": "success", **meta})
            except CaseError as exc:
                attempts = list(getattr(client, "format_attempts", []))[formats_before:]
                results.append(failed_result(item.case_id, exc.code, exc.message))
                log["cases"].append({"case_id": item.case_id, "status": "failed",
                                     "calls": getattr(client, "total_calls", 0) - calls_before,
                                     "token_usage": _usage_delta(usage_before, getattr(client, "total_usage", {})),
                                     "response_formats_attempted": attempts if args.pipeline == "baseline" else [],
                                     "response_format_used": (getattr(client, "last_response_format", None)
                                                              if args.pipeline == "baseline" and attempts else None),
                                     "error": exc.code, "error_message": exc.message})
        log["token_usage"] = dict(getattr(client, "total_usage", {}))
        log["total_calls"] = getattr(client, "total_calls", 0)
        atomic_write_json(output, results); log["status"] = "completed"; log["elapsed_ms"] = round((time.time()-started)*1000, 2)
        atomic_write_json(log_path, log); return 0
    except TaskError as exc:
        log["status"] = "failed"; log["error"] = {"code": exc.code, "message": exc.message}
        try: atomic_write_json(log_path, log)
        except OSError: pass
        print(f"{exc.code}: {exc.message}", file=sys.stderr); return exc.exit_code
    except Exception as exc:
        log["status"] = "failed"; log["error"] = {"code": "TASK_RUNTIME_ERROR", "message": type(exc).__name__}
        try: atomic_write_json(log_path, log)
        except OSError: pass
        print(f"TASK_RUNTIME_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 1


def _client(args, config):
    if args.mode == "mock":
        if not args.mock_responses: raise TaskError("MOCK_CONFIG_MISSING", "mock 模式必须显式提供 --mock-responses。")
        maximum = 3 if args.pipeline == "evidence_pipeline" else 1
        return MockClient(Path(args.mock_responses)), maximum, "text"
    if config.adapter == "dev_api_chat_completions":
        return DevApiChatCompletionsClient(config), config.max_calls_per_case, config.response_format
    if config.adapter == "chat_completions": return ChatCompletionsClient(config), config.max_calls_per_case, config.response_format
    if config.adapter == "mlx_direct": return MlxDirectClient(config), config.max_calls_per_case, config.response_format
    raise TaskError("CONFIG_INVALID", f"未知模型适配器：{config.adapter}")


def _usage_delta(before, after):
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


class _RawResponseRecorder:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.payload = {
            "warning": "仅限虚构开发病例；文件含模型原始输出，不得用于真实患者资料。",
            "run_id": run_id,
            "events": [],
        }
        self.current_case_id = None

    def record(self, event: dict) -> None:
        self.payload["events"].append({"case_id": self.current_case_id, **event})
        atomic_write_json(self.path, self.payload)


def _debug_recorder(args, run_id: str):
    if not args.debug_raw_response:
        if args.confirm_fictional_input:
            raise TaskError("DEBUG_CONFIRMATION_UNUSED", "未提供 --debug-raw-response，无需确认虚构输入。")
        return None
    if args.mode != "dev_api" or args.pipeline != "baseline":
        raise TaskError("DEBUG_SCOPE_INVALID", "原始响应调试仅允许 dev_api Baseline。")
    if not args.confirm_fictional_input:
        raise TaskError("DEBUG_CONFIRMATION_REQUIRED", "保存原始响应前必须显式确认输入为虚构开发病例。")
    return _RawResponseRecorder(Path(args.debug_raw_response), run_id)
