import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .config import load_config
from .competition_profile import (LOCAL_DEVELOPMENT_PROFILE, get_active_competition_profile,
                                  profile_cli_arguments)
from .errors import CaseError, TaskError
from .embedding_client import MockEmbeddingClient, OfficialEmbeddingClient, load_embedding_client
from .input_adapter import load_input_cases
from .io_utils import atomic_write_json, backup_existing_file
from .model_client import ChatCompletionsClient, DevApiChatCompletionsClient, MlxDirectClient, MockClient
from .pipeline import CLINICAL_PROMPT_VERSION, PROMPT_VERSION, run_case
from .prompt_profiles import resolve_clinical_prompt_path
from .protocol import CaseError as ProtocolCaseError
from .protocol import failed_result, success_result
from .retrieval import DenseEvidenceRetriever


VERSION = "0.4.9-prompt-v2-research-modules"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def parser():
    p = argparse.ArgumentParser(description="循证智录首次病程记录生成程序")
    p.add_argument("--input", required=True, help="正式 JSON 数组或受支持的练习嵌套 JSON；相对路径按当前工作目录解析")
    p.add_argument("--input-format", choices=("auto", "competition", "practice"), default="auto",
                   help="输入协议；默认严格自动识别")
    p.add_argument("--practice-case-id", help="可选：覆盖练习病例的文件序号或脱敏稳定 case_id")
    p.add_argument("--practice-record-cutoff", help="练习文件无可靠记录时间时显式指定截止时间")
    p.add_argument("--output", required=True, help="输出 JSON；相对路径按当前工作目录解析")
    p.add_argument("--mode", choices=("mock", "local", "dev_api", "competition"), default="competition")
    p.add_argument("--pipeline", choices=("baseline", "evidence_pipeline", "clinical_pipeline"), default="baseline")
    p.add_argument("--config", help="local/competition 模型配置 JSON；dev_api 仅读取指定环境变量")
    p.add_argument("--mock-responses", help="mock 模式响应脚本 JSON")
    p.add_argument("--log", help="运行日志路径；默认与输出同目录的 run_log.json")
    p.add_argument("--retrieval-config", help="可选 Dense 检索配置 JSON；用于 evidence/clinical pipeline")
    p.add_argument("--graph-retrieval", choices=("off", "one_hop", "ppr"), default="off",
                   help="clinical_pipeline 可选病例内图引导重排；默认关闭")
    p.add_argument("--clinical-agents", choices=("off", "adaptive", "always"), default="off",
                   help="可选反证代理：关闭、复杂病例触发或每例触发；最多一次额外调用")
    p.add_argument("--clinical-verification", choices=("on", "off"), default="on",
                   help="clinical_pipeline 是否调用原子事实核验器")
    p.add_argument("--clinical-revision", choices=("on", "off"), default="on",
                   help="clinical_pipeline 核验失败时是否最多局部修订一次")
    p.add_argument("--clinical-evidence-limit", type=int, default=5,
                   help="clinical_pipeline 最终 evidence 条数上限，默认5")
    p.add_argument("--clinical-prompt-profile",
                   choices=("legacy", "contrastive", "contrastive_fewshot", "contrastive_v2"),
                   default="contrastive",
                   help="Clinical Planner/Writer 提示词档位；fewshot 每阶段最多注入一个合成示例")
    p.add_argument("--clinical-writing-guidance",
                   choices=("off", "rules", "rules_examples"), default="off",
                   help="可选 Writer 写作规则/合成示例风格提要；不作为患者事实")
    p.add_argument("--shared-clinical-memory", choices=("off", "on"), default="off",
                   help="可选单病例结构化决策卡；默认关闭且不提供给 Verifier")
    p.add_argument("--dev-response-format", choices=("text", "json_object", "json_schema"),
                   help="仅 dev_api Baseline：显式测试服务端输出格式；不自动回退")
    p.add_argument("--clinical-response-format", choices=("text", "json_object", "json_schema"),
                   help="clinical_pipeline 所有阶段的显式默认格式；不支持时直接失败")
    for stage in ("planner", "counter_evidence", "writer", "verifier", "reviser"):
        option = stage.replace("_", "-")
        p.add_argument(f"--{option}-response-format", dest=f"{stage}_response_format",
                       choices=("text", "json_object", "json_schema"),
                       help=f"仅覆盖 clinical_pipeline 的 {stage} 阶段格式")
    p.add_argument("--debug-raw-response", help="仅虚构 dev_api 病例：将原始响应写入独立调试文件")
    p.add_argument("--confirm-fictional-input", action="store_true",
                   help="确认输入为允许保存原始响应的虚构开发病例")
    return p


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    official_zero_arg = not raw_argv
    profile_driven = official_zero_arg or not _has_explicit_algorithm_controls(raw_argv)
    development_requested = os.environ.get("MEWTYPE_DEV_MODE", "").strip() == "1"
    selected_profile = (LOCAL_DEVELOPMENT_PROFILE if development_requested
                        else get_active_competition_profile()) if profile_driven else None
    if official_zero_arg:
        raw_argv = ["--input", str(PACKAGE_ROOT / "input.json"),
                    "--output", str(PACKAGE_ROOT / "output.json"),
                    "--log", str(PACKAGE_ROOT / "run_log.json")]
    if selected_profile:
        raw_argv.extend(profile_cli_arguments(selected_profile))
    args = parser().parse_args(raw_argv); started = time.time(); run_id = str(uuid.uuid4())
    output = Path(args.output); log_path = Path(args.log) if args.log else output.parent / "run_log.json"
    prompt_version = CLINICAL_PROMPT_VERSION if args.pipeline == "clinical_pipeline" else PROMPT_VERSION
    backup_paths = []
    if selected_profile and selected_profile.environment == "dev_api":
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))
        for existing in (output, log_path):
            backup = backup_existing_file(existing, run_id, timestamp)
            if backup:
                backup_paths.append(str(backup))
    log = {"run_id": run_id, "version": VERSION, "prompt_version": prompt_version,
           "environment": args.mode, "pipeline": args.pipeline, "development_label": args.mode != "competition",
           "runtime_profile": ("development_profile" if selected_profile and selected_profile.environment == "dev_api"
                               else "competition_profile" if selected_profile else "explicit_cli"),
           "algorithm_profile": (selected_profile.name if selected_profile else "explicit_cli"),
           "prompt_manifest": _prompt_manifest(args.pipeline, args.clinical_prompt_profile),
           "artifact_backups": backup_paths,
           "started_at_epoch": started, "model": None, "parameters": {}, "cases": [], "status": "running"}
    try:
        clinical_verify = args.clinical_verification == "on"
        clinical_revise = args.clinical_revision == "on"
        if args.pipeline == "clinical_pipeline" and clinical_revise and not clinical_verify:
            raise TaskError("CLINICAL_CONFIG_INVALID", "局部修订依赖原子事实核验，不能 verification=off/revision=on。")
        if args.pipeline == "clinical_pipeline" and not 1 <= args.clinical_evidence_limit <= 5:
            raise TaskError("CLINICAL_CONFIG_INVALID", "--clinical-evidence-limit 必须在1到5之间。")
        if args.graph_retrieval != "off" and args.pipeline != "clinical_pipeline":
            raise TaskError("GRAPH_RETRIEVAL_SCOPE_INVALID", "图引导检索只允许 clinical_pipeline。")
        internal_official_retrieval = bool(
            selected_profile and selected_profile.requires_official_embedding
        )
        internal_mock_retrieval = bool(selected_profile and selected_profile.requires_mock_embedding)
        if (args.graph_retrieval != "off" and not args.retrieval_config
                and not internal_official_retrieval and not internal_mock_retrieval):
            raise TaskError("GRAPH_RETRIEVAL_CONFIG_MISSING", "图引导检索必须显式提供 --retrieval-config。")
        if args.clinical_agents != "off" and args.pipeline != "clinical_pipeline":
            raise TaskError("CLINICAL_AGENT_SCOPE_INVALID", "反证代理只允许 clinical_pipeline。")
        if args.clinical_writing_guidance != "off" and args.pipeline != "clinical_pipeline":
            raise TaskError("WRITING_GUIDANCE_SCOPE_INVALID", "写作知识只允许 clinical_pipeline。")
        if args.shared_clinical_memory != "off" and args.pipeline != "clinical_pipeline":
            raise TaskError("SHARED_MEMORY_SCOPE_INVALID", "共享临床记忆只允许 clinical_pipeline。")
        minimum_calls = _minimum_calls(args.pipeline, clinical_verify, clinical_revise,
                                       args.clinical_agents)
        if selected_profile and minimum_calls != selected_profile.qwen_calls_reserved:
            raise TaskError("COMPETITION_PROFILE_BUDGET_MISMATCH",
                            "正式 Profile 的调用预算与实际阶段配置不一致。")
        config = load_config(Path(args.config) if args.config else None, args.mode, args.pipeline, minimum_calls,
                             official_env_path=(PACKAGE_ROOT / ".env"
                                                if selected_profile and selected_profile.environment == "competition"
                                                else None))
        if args.dev_response_format:
            if args.mode != "dev_api" or args.pipeline != "baseline":
                raise TaskError("DEV_FORMAT_SCOPE_INVALID", "--dev-response-format 仅用于 dev_api Baseline。")
            config.response_format = args.dev_response_format
            config.fallback_response_formats = []
        clinical_stage_formats = _clinical_stage_formats(args, config.response_format if config else "text")
        if args.pipeline == "clinical_pipeline":
            log["clinical_config"] = {"verification": clinical_verify, "localized_revision": clinical_revise,
                                      "maximum_revision_rounds": 1, "minimum_call_budget": minimum_calls,
                                      "evidence_limit": args.clinical_evidence_limit,
                                      "prompt_profile": args.clinical_prompt_profile,
                                      "writing_guidance": args.clinical_writing_guidance,
                                      "shared_clinical_memory": args.shared_clinical_memory,
                                      "graph_retrieval": args.graph_retrieval,
                                      "clinical_agents": args.clinical_agents,
                                      "stage_response_formats": clinical_stage_formats}
        recorder = _debug_recorder(args, run_id)
        client, maximum, response_format = _client(args, config)
        if recorder:
            client.debug_hook = recorder.record
        embedding_client = None; retrieval_options = None
        if args.retrieval_config:
            if args.pipeline not in ("evidence_pipeline", "clinical_pipeline"):
                raise TaskError("RETRIEVAL_PIPELINE_INVALID", "Dense 检索只允许用于 evidence/clinical pipeline。")
            embedding_client, retrieval_options = load_embedding_client(Path(args.retrieval_config), args.mode)
            log["retrieval"] = {key: value for key, value in retrieval_options.items() if key != "vectors"}
        elif internal_official_retrieval:
            embedding_client, retrieval_options = _official_profile_embedding(selected_profile)
            log["retrieval"] = {**retrieval_options, "protocol_status": "awaiting_official_contract"}
        elif internal_mock_retrieval:
            embedding_client, retrieval_options = _development_profile_embedding(selected_profile)
            log["retrieval"] = {**retrieval_options, "protocol_status": "development_mock_only"}
        if config:
            log["model"] = config.model
            log["parameters"] = {"adapter": config.adapter, "response_format": config.response_format,
                                 "max_calls_per_case": config.max_calls_per_case,
                                 "max_retries": config.max_retries,
                                 "generation_parameters_sent": config.send_generation_parameters,
                                 "authorization_header_sent": config.adapter == "dev_api_chat_completions"}
            if config.send_generation_parameters:
                log["parameters"].update({"temperature": config.temperature, "top_p": config.top_p,
                                          "max_tokens": config.max_tokens, "seed": config.seed})
        else:
            log["model"] = "mock"; log["parameters"] = {"max_calls_per_case": maximum}
        loaded_input = load_input_cases(
            Path(args.input), args.input_format,
            practice_case_id=args.practice_case_id,
            practice_record_cutoff=args.practice_record_cutoff,
        )
        cases = loaded_input.cases
        log["input_adapter"] = loaded_input.audit
        results = []
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
                events_before = len(getattr(client, "call_events", []))
                if recorder:
                    recorder.current_case_id = item.case_id
                retriever = (DenseEvidenceRetriever(
                    item, embedding_client,
                    max_chunk_chars=retrieval_options["max_chunk_chars"],
                    min_similarity=retrieval_options["min_similarity"],
                ) if embedding_client else None)
                record, meta = run_case(item, client, args.pipeline, response_format, maximum,
                                        retriever, retrieval_options["top_k"] if retrieval_options else 3,
                                        clinical_verify, clinical_revise, args.clinical_evidence_limit,
                                        clinical_stage_formats, args.clinical_prompt_profile,
                                        args.graph_retrieval, args.clinical_agents,
                                        args.clinical_writing_guidance,
                                        args.shared_clinical_memory == "on")
                _enforce_profile_safety(selected_profile, meta)
                meta["token_usage"] = _usage_delta(usage_before, getattr(client, "total_usage", {}))
                meta["stage_calls"] = list(getattr(client, "call_events", []))[events_before:]
                if args.pipeline == "baseline":
                    attempts = list(getattr(client, "format_attempts", []))[formats_before:]
                    meta["response_formats_attempted"] = attempts
                    meta["response_format_used"] = getattr(client, "last_response_format", None)
                results.append(success_result(item.case_id, record))
                log["cases"].append({"case_id": item.case_id, "status": "success", **meta})
            except CaseError as exc:
                attempts = list(getattr(client, "format_attempts", []))[formats_before:]
                results.append(failed_result(item.case_id, exc.code, exc.message))
                details = getattr(exc, "details", {}) or {}
                partial = details.get("partial_meta", {})
                failed_case = {"case_id": item.case_id, "status": "failed", **partial,
                               "calls": getattr(client, "total_calls", 0) - calls_before,
                               "token_usage": _usage_delta(usage_before, getattr(client, "total_usage", {})),
                               "stage_calls": list(getattr(client, "call_events", []))[events_before:],
                               "response_formats_attempted": attempts if args.pipeline == "baseline" else [],
                               "response_format_used": (getattr(client, "last_response_format", None)
                                                        if args.pipeline == "baseline" and attempts else None),
                               "failed_stage": details.get("failed_stage", "case_processing"),
                               "validation_errors": details.get("validation_errors", []),
                               "error": exc.code, "error_message": exc.message}
                log["cases"].append(failed_case)
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
        maximum = _minimum_calls(args.pipeline, args.clinical_verification == "on",
                                 args.clinical_revision == "on", args.clinical_agents)
        return MockClient(Path(args.mock_responses)), maximum, "text"
    if config.adapter == "dev_api_chat_completions":
        return DevApiChatCompletionsClient(config), config.max_calls_per_case, config.response_format
    if config.adapter == "chat_completions": return ChatCompletionsClient(config), config.max_calls_per_case, config.response_format
    if config.adapter == "mlx_direct": return MlxDirectClient(config), config.max_calls_per_case, config.response_format
    raise TaskError("CONFIG_INVALID", f"未知模型适配器：{config.adapter}")


_ALGORITHM_FLAGS = frozenset({
    "--mode", "--pipeline", "--config", "--mock-responses", "--retrieval-config",
    "--graph-retrieval", "--clinical-agents", "--clinical-verification",
    "--clinical-revision", "--clinical-evidence-limit", "--clinical-prompt-profile",
    "--dev-response-format", "--clinical-response-format", "--planner-response-format",
    "--counter-evidence-response-format", "--writer-response-format",
    "--verifier-response-format", "--reviser-response-format",
    "--clinical-writing-guidance", "--shared-clinical-memory",
})


def _has_explicit_algorithm_controls(argv: list[str]) -> bool:
    return any(argument in _ALGORITHM_FLAGS
               or any(argument.startswith(flag + "=") for flag in _ALGORITHM_FLAGS)
               for argument in argv)


def _official_profile_embedding(profile):
    """Construct the unpublished official boundary without guessing HTTP fields.

    Tests replace this function inside an isolated package with a deterministic
    MockEmbeddingClient.  Production code always returns the explicit pending
    official adapter until the organiser publishes its contract.
    """
    if not profile.requires_official_embedding:
        raise TaskError("COMPETITION_PROFILE_INVALID", "当前 Profile 不需要官方 Embedding。")
    return OfficialEmbeddingClient(), {
        "provider": "official",
        "model": None,
        "top_k": 3,
        "min_similarity": 0.0,
        "max_chunk_chars": 240,
        "batch_size": None,
        "max_length": None,
        "use_fp16": None,
    }


def _development_profile_embedding(profile):
    if not profile.requires_mock_embedding or profile.environment != "dev_api":
        raise TaskError("DEVELOPMENT_PROFILE_INVALID", "当前 Profile 不允许开发 Mock Embedding。")
    return MockEmbeddingClient(dimension=64), {
        "provider": "mock_development",
        "model": "deterministic-hash-test-only",
        "top_k": 3,
        "min_similarity": -1.0,
        "max_chunk_chars": 240,
        "batch_size": None,
        "max_length": None,
        "use_fp16": None,
    }


def _prompt_manifest(pipeline: str, prompt_profile: str = "contrastive") -> dict:
    names = (["baseline_system.txt"] if pipeline == "baseline" else
             ["fact_system.txt", "evidence_system.txt", "verify_system.txt"]
             if pipeline == "evidence_pipeline" else
             ["planner_system.txt", "counter_evidence_system.txt", "writer_system.txt",
              "claim_verify_system.txt", "revise_system.txt", "baseline_system.txt"])
    files = {}
    stage_by_name = {"planner_system.txt": "planner", "counter_evidence_system.txt": "counter_evidence",
                     "writer_system.txt": "writer", "claim_verify_system.txt": "verifier",
                     "revise_system.txt": "reviser"}
    for name in names:
        path = (resolve_clinical_prompt_path(stage_by_name[name], prompt_profile)
                if name in stage_by_name else PACKAGE_ROOT / "prompts" / name)
        files[str(path.relative_to(PACKAGE_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"hash_algorithm": "sha256", "files": files}


def _enforce_profile_safety(profile, meta: dict) -> None:
    if (profile and profile.block_unresolved_high_risk
            and meta.get("clinical", {}).get("unresolved_risk_count", 0)):
        raise CaseError(
            "UNRESOLVED_HIGH_RISK",
            "正式 Clinical Profile 仍有未修复的高风险患者事实，禁止作为成功结果输出。",
            {"partial_meta": meta, "failed_stage": "clinical_safety_gate"},
        )


def _usage_delta(before, after):
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _minimum_calls(pipeline: str, clinical_verify: bool, clinical_revise: bool,
                   clinical_agent_mode: str = "off") -> int:
    if pipeline == "evidence_pipeline":
        return 3
    if pipeline == "clinical_pipeline":
        return 2 + int(clinical_verify) + int(clinical_revise) + int(clinical_agent_mode != "off")
    return 1


def _clinical_stage_formats(args, configured: str) -> dict[str, str]:
    stage_args = {stage: getattr(args, f"{stage}_response_format")
                  for stage in ("planner", "counter_evidence", "writer", "verifier", "reviser")}
    any_clinical_flag = args.clinical_response_format or any(stage_args.values())
    if any_clinical_flag and args.pipeline != "clinical_pipeline":
        raise TaskError("CLINICAL_FORMAT_SCOPE_INVALID", "分阶段 response_format 只允许 clinical_pipeline。")
    common = args.clinical_response_format or configured
    return {stage: stage_args[stage] or common for stage in stage_args}


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
    if args.mode != "dev_api" or args.pipeline not in ("baseline", "clinical_pipeline"):
        raise TaskError("DEBUG_SCOPE_INVALID", "原始响应调试仅允许 dev_api Baseline/Clinical Pipeline。")
    if not args.confirm_fictional_input:
        raise TaskError("DEBUG_CONFIRMATION_REQUIRED", "保存原始响应前必须显式确认输入为虚构开发病例。")
    return _RawResponseRecorder(Path(args.debug_raw_response), run_id)
