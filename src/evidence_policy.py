"""Deterministic competition-facing evidence selection and safe normalization."""

from __future__ import annotations

import re
from copy import deepcopy

from .protocol import Case


MAX_EVIDENCE = 5


def enforce_evidence_policy(record: dict, case: Case, limit: int = MAX_EVIDENCE) -> tuple[dict, dict]:
    """Keep at most five valid, distinct, high-value evidence items.

    Invalid evidence is never repaired by inventing a source. It is removed from the
    valid evidence list and disclosed in review_items. Fewer than three valid items
    are kept as-is because the protocol forbids padding with fabricated evidence.
    """
    if not isinstance(record, dict) or not isinstance(record.get("evidence"), list):
        return record, {"applied": False, "reason": "evidence_not_array"}
    maximum = max(1, min(int(limit), MAX_EVIDENCE))
    known = {source.source_id for source in case.sources}
    valid = []
    invalid_count = 0
    duplicate_count = 0
    by_key: dict[tuple[str, str], dict] = {}
    for index, item in enumerate(record["evidence"]):
        if not _valid_item(item, known):
            invalid_count += 1
            continue
        key = (_normalized_statement(item["statement"]), item["kind"])
        if key in by_key:
            duplicate_count += 1
            existing = by_key[key]["item"]
            existing["source_ids"] = list(dict.fromkeys(existing["source_ids"] + item["source_ids"]))
            continue
        copied = deepcopy(item)
        entry = {"item": copied, "index": index, "score": _importance_score(copied)}
        by_key[key] = entry
        valid.append(entry)
    selected = []
    covered_sources = set()
    remaining = list(valid)
    while remaining and len(selected) < maximum:
        best = max(remaining, key=lambda entry: (
            entry["score"] + 2 * len(set(entry["item"]["source_ids"]) - covered_sources),
            -entry["index"],
        ))
        selected.append(best)
        covered_sources.update(best["item"]["source_ids"])
        remaining.remove(best)
    selected.sort(key=lambda entry: entry["index"])
    result = deepcopy(record)
    result["evidence"] = [entry["item"] for entry in selected]
    if not isinstance(result.get("review_items"), list):
        result["review_items"] = []
    if invalid_count:
        note = f"{invalid_count}条证据因结构或来源编号无效未作为有效证据保留，需人工核对。"
        if note not in result["review_items"]:
            result["review_items"].append(note)
    return result, {
        "applied": True,
        "input_count": len(record["evidence"]),
        "valid_distinct_count": len(valid),
        "selected_count": len(result["evidence"]),
        "invalid_removed": invalid_count,
        "duplicates_merged": duplicate_count,
        "limit_removed": max(0, len(valid) - len(selected)),
        "below_recommended_minimum": len(result["evidence"]) < 3,
    }


def _valid_item(item, known: set[str]) -> bool:
    return (isinstance(item, dict)
            and set(item) == {"statement", "source_ids", "kind"}
            and isinstance(item.get("statement"), str) and bool(item["statement"].strip())
            and item.get("kind") in ("fact", "inference")
            and isinstance(item.get("source_ids"), list) and bool(item["source_ids"])
            and all(isinstance(source_id, str) and source_id in known for source_id in item["source_ids"]))


def _normalized_statement(statement: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ]+", "", statement).lower()


def _importance_score(item: dict) -> int:
    statement = item["statement"]
    score = 1 if item["kind"] == "fact" else 0
    score += 3 if re.search(r"\d+(?:\.\d+)?\s*(?:℃|mmHg|次/分|%|mg/L|g/L|mmol/L|×10[⁹9]/L)", statement) else 0
    score += 3 if any(word in statement for word in ("否认", "无", "未见", "复核", "更正")) else 0
    score += 2 if any(word in statement for word in ("初步诊断", "待排", "疑似", "考虑")) else 0
    score += 2 if any(word in statement for word in ("计划", "拟", "尚未取得", "未提供")) else 0
    score += 1 if any(word in statement for word in ("入院前", "入院后", "既往", "当前")) else 0
    return score
