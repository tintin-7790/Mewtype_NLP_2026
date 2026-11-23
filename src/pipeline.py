import json
import time
from pathlib import Path

from .errors import CaseError
from .facts import organize_facts
from .model_client import Budget, parse_model_json
from .protocol import Case
from .retrieval import DenseEvidenceRetriever
from .validation import validate_record


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "2026-11-23.baseline-v3"


def run_case(case: Case, client, pipeline: str, response_format: str, max_calls: int,
             retriever: DenseEvidenceRetriever | None = None, retrieval_top_k: int = 3) -> tuple[dict, dict]:
    started = time.perf_counter(); budget = Budget(max_calls); checks = []
    raw = [{"source_id": s.source_id, "field_path": s.field_path, "content": s.content} for s in case.sources]
    if pipeline == "baseline":
        messages = _messages("baseline_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw})
        record = parse_model_json(client.call(messages, response_format, budget))
    else:
        deterministic = organize_facts(case)
        fact_messages = _messages("fact_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                       "deterministic_hints": deterministic})
        model_facts = parse_model_json(client.call(fact_messages, response_format, budget))
        if not isinstance(model_facts, dict) or not isinstance(model_facts.get("facts"), list):
            raise CaseError("INVALID_FACT_OUTPUT", "事实整理结果必须包含 facts 数组。")
        gen_messages = _messages("evidence_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                          "facts": model_facts["facts"]})
        record = parse_model_json(client.call(gen_messages, response_format, budget))
        pre_issues = validate_record(record, case); checks.extend(pre_issues)
        retrieval_context = _retrieval_context(record, retriever, retrieval_top_k)
        verify_messages = _messages("verify_system.txt", {"record_cutoff": case.record_cutoff, "sources": raw,
                                                            "facts": model_facts["facts"], "candidate": record,
                                                            "rule_issues": pre_issues,
                                                            "dense_retrieval_candidates": retrieval_context,
                                                            "retrieval_notice": (
                                                                "相似度仅用于定位候选原文，不代表事实正确或来源已验证；"
                                                                "必须结合原文、数值、否定、时间和状态核验。"
                                                                if retriever else "未启用 Dense 检索。")})
        verdict = parse_model_json(client.call(verify_messages, response_format, budget))
        if not isinstance(verdict, dict) or not isinstance(verdict.get("valid"), bool):
            raise CaseError("INVALID_VERIFIER_OUTPUT", "核验结果必须包含布尔值 valid。")
        if not verdict["valid"]:
            corrected = verdict.get("corrected_record")
            if not isinstance(corrected, dict):
                raise CaseError("VERIFICATION_FAILED", "模型核验未通过且未提供可校验的保守修正。")
            record = corrected
        _preserve_retrieval_review_items(record, retriever, retrieval_top_k)
    final_issues = validate_record(record, case); checks.extend(final_issues)
    errors = [x for x in final_issues if x["level"] == "error"]
    if errors:
        raise CaseError("VALIDATION_FAILED", "; ".join(x["message"] for x in errors))
    meta = {"calls": budget.calls, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "checks": checks, "prompt_version": PROMPT_VERSION}
    if retriever:
        meta["retrieval"] = retriever.stats()
    return record, meta


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _retrieval_context(record: dict, retriever: DenseEvidenceRetriever | None, top_k: int) -> list[dict]:
    if retriever is None or not isinstance(record, dict):
        return []
    context = []
    for index, item in enumerate(record.get("evidence", [])):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        candidates = retriever.retrieve(item["statement"], top_k=top_k)
        context.append({
            "evidence_index": index,
            "statement": item["statement"],
            "declared_source_ids": item.get("source_ids", []),
            "candidates": [candidate.to_dict() for candidate in candidates],
        })
    return context


def _preserve_retrieval_review_items(record: dict, retriever: DenseEvidenceRetriever | None, top_k: int) -> None:
    """Record insufficient retrieval support without changing declared sources."""
    if retriever is None or not isinstance(record, dict) or not isinstance(record.get("review_items"), list):
        return
    for item in record.get("evidence", []):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        declared = set(item.get("source_ids", []))
        candidates = retriever.retrieve(item["statement"], top_k=top_k)
        supported = any(candidate.compatible and candidate.unit.source_id in declared for candidate in candidates)
        if not supported:
            note = f"Dense 检索未找到与已声明来源一致且无明显冲突的候选：{item['statement']}"
            if note not in record["review_items"]:
                record["review_items"].append(note)
