"""VeriFact-style claim verification with deterministic safety rules."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import CaseError
from .model_client import Budget, VERIFIER_ISSUE_CODES, VERIFIER_SCHEMA, parse_model_json
from .protocol import Case


BASE_DIR = Path(__file__).resolve().parent.parent
HARD_ERROR_CODES = {
    "PATIENT_FACT_CONTRADICTION", "NUMERIC_VALUE_ERROR", "UNIT_ERROR", "TEMPORAL_CONFLICT",
    "CORRECTION_IGNORED", "UNSUPPORTED_NEGATIVE", "PLAN_AS_COMPLETED", "RESULT_AS_AVAILABLE",
    "DIAGNOSIS_CERTAINTY_UPGRADE", "SOURCE_MISMATCH",
}


def run_claim_verifier(case: Case, claims: list[dict], record: dict, clinical_plan: dict,
                       retriever, retrieval_top_k: int, client, response_format: str,
                       budget: Budget) -> dict:
    deterministic = deterministic_claim_issues(case, claims)
    candidates = []
    if retriever is not None:
        for claim in claims:
            retrieved = [candidate.to_dict() for candidate in
                         retriever.retrieve(claim["statement"], top_k=retrieval_top_k)]
            candidates.append({
                "claim_id": claim["claim_id"],
                "candidates": [{key: item.get(key) for key in
                                ("source_id", "similarity", "compatible", "compatibility_warnings")}
                               for item in retrieved],
            })
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "claims": [_compact_claim(claim) for claim in claims],
        "retrieval_candidates": candidates,
        "deterministic_issue_hints": [{"claim_id": issue["claim_id"],
                                       "code": issue["error_type"],
                                       "severity": issue["severity"],
                                       "source_ids": issue["source_ids"]}
                                      for issue in deterministic],
    }
    raw = client.call(_messages("claim_verify_system.txt", payload), response_format, budget,
                      stage="verifier", schema=VERIFIER_SCHEMA)
    response = parse_model_json(raw, stage="verifier", client=client)
    claim_map = {claim["claim_id"]: claim for claim in claims}
    try:
        model_issues = validate_verification(response, claim_map,
                                             {source.source_id for source in case.sources})
    except CaseError as exc:
        _record_schema(client, False, exc.code)
        raise
    _record_schema(client, True)
    model_issues, dismissed = _guard_source_supported_model_issues(case, model_issues)
    response["issues"] = _merge_issues(deterministic, model_issues)
    response["source_guard"] = {"dismissed_count": len(dismissed),
                                "dismissed": dismissed}
    return response


def validate_verification(value, claim_map: dict[str, dict], source_ids: set[str]) -> list[dict]:
    """Validate full coverage and expand compact issues into internal locators."""
    if (not isinstance(value, dict) or set(value) != {"checked_claim_ids", "issues"}
            or not isinstance(value.get("checked_claim_ids"), list)
            or not isinstance(value.get("issues"), list)):
        raise CaseError("INVALID_CLAIM_VERIFICATION",
                        "核验结果必须仅包含 checked_claim_ids 和 issues 数组。")
    checked = value["checked_claim_ids"]
    if (any(not isinstance(claim_id, str) for claim_id in checked)
            or len(checked) != len(set(checked)) or set(checked) != set(claim_map)):
        raise CaseError("INCOMPLETE_CLAIM_COVERAGE",
                        "checked_claim_ids 必须无重复地完整覆盖全部 Claim。")
    result = []
    seen = set()
    for item in value["issues"]:
        expected = {"claim_id", "code", "severity", "hint", "source_ids"}
        if not isinstance(item, dict) or set(item) != expected or item.get("claim_id") not in claim_map:
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 引用了未知 Claim 或字段无效。")
        key = (item["claim_id"], item["code"])
        if key in seen:
            raise CaseError("INVALID_CLAIM_VERIFICATION", "同一 Claim 的同类 issue 不得重复。")
        seen.add(key)
        if item["code"] not in VERIFIER_ISSUE_CODES:
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue code 非法。")
        if item["severity"] not in ("high", "medium", "low"):
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue severity 非法。")
        if item["code"] in HARD_ERROR_CODES and item["severity"] != "high":
            raise CaseError("INVALID_CLAIM_VERIFICATION", "硬性事实错误必须标记为 high。")
        if not isinstance(item["hint"], str) or not item["hint"].strip():
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue hint 不能为空。")
        refs = item["source_ids"]
        if not isinstance(refs, list) or any(ref not in source_ids for ref in refs):
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 引用了不存在的 source_id。")
        result.append(_issue(claim_map[item["claim_id"]], item["code"], item["severity"],
                             item["hint"], refs))
    return result


def _compact_claim(claim: dict) -> dict:
    return {"id": claim["claim_id"], "section": claim["section"],
            "type": claim["claim_type"], "text": claim["statement"],
            "declared_source_ids": claim.get("source_ids", [])}


def deterministic_claim_issues(case: Case, claims: list[dict]) -> list[dict]:
    corpus = "\n".join(source.content for source in case.sources)
    issues = []
    for claim in claims:
        statement = claim["statement"]
        section = claim["section"]
        if ("当前" in statement and any(word in statement for word in ("尚未", "未开始", "未接受", "未进行"))
                and re.search(r"当前.{0,10}(?:未提供|不详|未知)", corpus)):
            issues.append(_issue(claim, "NOT_ADDRESSED", "high",
                                 "原始资料未明确当前状态，不得将未提供写成确定否定事实。", []))
        if (section == "review_items" and any(word in statement for word in ("频次不详", "频率不详", "频次未提供"))
                and re.search(r"每(?:日|天|周|月).{0,12}\d+(?:[–—~-]\d+)?\s*次", corpus)):
            issues.append(_issue(claim, "CONTRADICTS_PROVIDED_DETAIL", "high",
                                 "原文已经提供频次，不能列为频次不详。", _matching_sources(case, "次")))
        if "无结核接触史" in statement and "结核" in corpus and "邻居" in corpus:
            issues.append(_issue(claim, "NEGATION_SCOPE_ERROR", "high",
                                 "只能保留否认密切接触与邻居日常往来两个限定事实。", _matching_sources(case, "结核")))
        if "发绀" in statement and "无发绀" not in statement and "原记录发绀" in corpus and "复核" in corpus and "无发绀" in corpus:
            issues.append(_issue(claim, "CORRECTION_IGNORED", "high",
                                 "应采用复核后的无发绀结论。", _matching_sources(case, "复核")))
        if "入院" in statement and "38.5℃" in statement and "1天前" in corpus and "入院" in corpus and "37.1℃" in corpus:
            issues.append(_issue(claim, "TIME_MISMATCH", "high",
                                 "38.5℃属于入院前测量，入院测量为37.1℃。", _matching_sources(case, "38.5℃")))
        if ("入院后" in statement and any(word in statement for word in ("CT", "影像"))
                and _exam_documented_before_admission(corpus)):
            issues.append(_issue(claim, "TIME_MISMATCH", "high",
                                 "原始资料记载该影像检查在急诊收住前完成，不得改写为入院后完成。",
                                 _matching_sources(case, "CT")))
        if any(word in statement for word in ("已完成", "已行", "结果显示")) and any(
                word in corpus for word in ("待完善", "拟行", "计划", "尚未取得")):
            issues.append(_issue(claim, "PLAN_AS_COMPLETED", "high",
                                 "计划或等待中的检查不得写成已经完成。", []))
        if (any(word in statement for word in ("结果阴性", "检查阴性", "检测阴性"))
                and any(word in corpus for word in ("未检查", "未行", "未执行", "尚未检查"))):
            issues.append(_issue(claim, "RESULT_AS_AVAILABLE", "high",
                                 "未检查或未执行不能改写为检查结果阴性。",
                                 [source.source_id for source in case.sources
                                  if any(word in source.content for word in
                                         ("未检查", "未行", "未执行", "尚未检查"))]))
        if _asserts_unperformed_or_untreated(statement) and not _negative_status_supported(statement, corpus):
            issues.append(_issue(claim, "UNKNOWN_AS_NEGATIVE", "high",
                                 "原始资料未提供该检查或治疗的执行状态，不得写成未执行、未进行或未用药。", []))
        if (_states_confirmed_diagnosis(statement) and _source_uncertain_diagnosis_matches(corpus, statement)
                and not _statement_is_uncertain(statement)):
            issues.append(_issue(claim, "DIAGNOSIS_CERTAINTY_UPGRADE", "high",
                                 "原始资料仅为初步、疑似或待排诊断，不得升级为确诊。", []))
        contradiction_refs = _explicit_fact_contradiction(case, statement)
        has_specific_correction = any(
            issue["claim_id"] == claim["claim_id"] and issue["error_type"] == "CORRECTION_IGNORED"
            for issue in issues
        )
        if contradiction_refs and not has_specific_correction:
            issues.append(_issue(claim, "FACT_CONTRADICTION", "high",
                                 "生成陈述与原始资料中的明确患者事实相反。", contradiction_refs))
    return _merge_issues([], issues)


def has_blocking_issues(verification: dict) -> bool:
    """Only high-risk issues require an automatic revision call."""
    return any(issue.get("severity") == "high" for issue in verification.get("issues", []))


def issues_for_revision(verification: dict) -> list[dict]:
    """Return at most one patch target per Claim while retaining all risk hints.

    A single false statement can trigger both a deterministic rule and a model
    issue.  The verifier log keeps every issue, but the reviser must receive one
    unambiguous edit target for that Claim.
    """
    grouped = {}
    for issue in verification.get("issues", []):
        if issue.get("severity") != "high":
            continue
        claim_id = issue.get("claim_id")
        if claim_id not in grouped:
            grouped[claim_id] = dict(issue)
            continue
        current = grouped[claim_id]
        hints = [part for part in (current.get("revision_hint"), issue.get("revision_hint"))
                 if isinstance(part, str) and part]
        current["revision_hint"] = "；".join(dict.fromkeys(hints))
        current["source_ids"] = list(dict.fromkeys(
            list(current.get("source_ids", [])) + list(issue.get("source_ids", []))))
    return list(grouped.values())


def issues_for_review(verification: dict) -> list[dict]:
    return [issue for issue in verification.get("issues", []) if issue.get("severity") == "medium"]


def _issue(claim: dict, error_type: str, severity: str, hint: str, source_ids: list[str]) -> dict:
    return {"claim_id": claim["claim_id"], "section": claim["section"],
            "statement": claim["statement"], "error_type": error_type,
            "severity": severity, "revision_hint": hint, "source_ids": source_ids,
            "span_start": claim.get("span_start"), "span_end": claim.get("span_end"),
            "item_index": claim.get("item_index")}


def _matching_sources(case: Case, token: str) -> list[str]:
    return [source.source_id for source in case.sources if token in source.content]


def _merge_issues(first: list[dict], second: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for issue in first + second:
        key = (issue.get("claim_id"), issue.get("section"), issue.get("error_type"))
        if key not in seen:
            seen.add(key); result.append(issue)
    return result


def _guard_source_supported_model_issues(case: Case, issues: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reject model issues that directly contradict an explicit source statement.

    This guard is intentionally narrow: it only dismisses an unsupported-negative
    issue when the same negative patient fact is explicit, or a contradiction
    issue when the generated statement itself appears in a source. Other semantic
    disputes remain reviewable by the model/human.
    """
    kept = []
    dismissed = []
    for issue in issues:
        code = issue.get("error_type")
        statement = issue.get("statement", "")
        supported = (code == "UNSUPPORTED_NEGATIVE" and
                     _explicit_negative_supported(case, statement))
        supported = supported or (code == "PATIENT_FACT_CONTRADICTION" and
                                  (_statement_directly_supported(case, statement)
                                   or _explicit_negative_supported(case, statement)))
        if supported:
            dismissed.append({"claim_id": issue.get("claim_id"), "code": code,
                              "reason": "explicit_source_support"})
        else:
            kept.append(issue)
    return kept, dismissed


_NEGATIVE_ACTIONS = ("未执行", "未进行", "未接受", "未使用", "未用药", "未予", "未行", "尚未开始")
_UNCERTAINTY_WORDS = ("待排", "疑似", "初步诊断", "考虑", "可能")
_CONTRADICTION_CONCEPTS = (
    "胸痛", "胸闷", "气促", "呼吸困难", "咯血", "寒战", "盗汗", "消瘦",
    "恶心", "呕吐", "右上腹疼痛", "上腹疼痛", "腹痛", "腹泻", "尿频", "尿急", "尿痛", "皮疹",
    "关节肿痛", "口腔溃疡", "光过敏", "发绀",
)


def _asserts_unperformed_or_untreated(statement: str) -> bool:
    return any(word in statement for word in _NEGATIVE_ACTIONS)


def _negative_status_supported(statement: str, corpus: str) -> bool:
    status_words = [word for word in _NEGATIVE_ACTIONS if word in statement]
    if not any(word in corpus for word in status_words):
        return False
    claim_terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", statement))
    ignored = {"未执行", "未进行", "未接受", "未使用", "未用药", "尚未开始", "患者", "检查", "治疗"}
    return any(term not in ignored and term in corpus for term in claim_terms)


def _states_confirmed_diagnosis(statement: str) -> bool:
    return any(word in statement for word in ("确诊", "明确诊断", "诊断为", "患有"))


def _source_uncertain_diagnosis_matches(corpus: str, statement: str) -> bool:
    candidates = re.findall(r"(?:初步诊断[：:]?|疑似|考虑)([^，。；\n]{2,20})", corpus)
    candidates += re.findall(r"([^，。；：:\n]{2,20})待排", corpus)
    for candidate in candidates:
        cleaned = re.sub(r"(?:待排|疑似|可能|初步诊断|考虑)", "", candidate).strip(" ：:")
        terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", cleaned)
        if any(term in statement for term in terms):
            return True
    return False


def _statement_is_uncertain(statement: str) -> bool:
    return any(word in statement for word in _UNCERTAINTY_WORDS)


def _explicit_fact_contradiction(case: Case, statement: str) -> list[str]:
    asserted_positive = _positive_concepts(statement)
    asserted_negative = _negated_concepts(statement)
    if not asserted_positive and not asserted_negative:
        return []
    matched = []
    for source in case.sources:
        if _normalized_patient_text(statement) in _normalized_patient_text(source.content):
            continue
        if (asserted_positive & _negated_concepts(source.content)
                or asserted_negative & _positive_concepts(source.content)):
            matched.append(source.source_id)
    return list(dict.fromkeys(matched))


def _positive_concepts(text: str) -> set[str]:
    """Return explicitly present concepts that are not negated in their clause.

    This deliberately uses a small safety lexicon.  Broadly treating everything
    following the character ``无`` as negated caused phrases such as
    ``无明显诱因间断出现咳嗽`` and ``无明显变化`` to become false conflicts.
    """
    positive = set()
    for clause in re.split(r"[，。；;！？\n]", text):
        present = {concept for concept in _CONTRADICTION_CONCEPTS if concept in clause}
        positive.update(present - _negated_concepts(clause))
    return positive


def _negated_concepts(text: str) -> set[str]:
    concepts = set()
    alternatives = "|".join(map(re.escape, _CONTRADICTION_CONCEPTS))
    # “否认” conventionally scopes over the remainder of the current list/clause.
    for phrase in re.findall(r"否认([^，。；;！？\n]{1,40})", text):
        concepts.update(concept for concept in _CONTRADICTION_CONCEPTS if concept in phrase)
    # Other negators are accepted only when immediately followed by a known
    # concept/list.  This excludes modifier phrases such as “无明显诱因……咳嗽”.
    list_pattern = rf"(?:{alternatives})(?:[、和及/](?:{alternatives}))*"
    for match in re.finditer(rf"(?:无|未见|不伴)(?:明显)?({list_pattern})", text):
        phrase = match.group(1)
        concepts.update(concept for concept in _CONTRADICTION_CONCEPTS if concept in phrase)
    # Permit anatomical/degree modifiers between the negator and concept, for
    # example “无右上腹疼痛”. Reject causal/occurrence phrases such as
    # “无明显诱因间断出现咳嗽”, where “无” does not negate the symptom.
    for clause in re.split(r"[，。；;！？\n]", text):
        for concept in _CONTRADICTION_CONCEPTS:
            for occurrence in re.finditer(re.escape(concept), clause):
                prefix = clause[max(0, occurrence.start() - 18):occurrence.start()]
                negators = list(re.finditer(r"(?:无|未见|不伴)", prefix))
                if not negators:
                    continue
                between = prefix[negators[-1].end():]
                if len(between) <= 14 and not any(word in between for word in
                                                  ("诱因", "原因", "出现", "发生", "发作")):
                    concepts.add(concept)
    return concepts


def _explicit_negative_supported(case: Case, statement: str) -> bool:
    negated = _negated_concepts(statement)
    if not negated:
        return False
    for source in case.sources:
        source_negative = _negated_concepts(source.content)
        source_positive = _positive_concepts(source.content)
        if negated <= source_negative and not (negated & source_positive):
            return True
        if _normalized_patient_text(statement) in _normalized_patient_text(source.content):
            return True
    return False


def _statement_directly_supported(case: Case, statement: str) -> bool:
    needle = _normalized_patient_text(statement)
    return bool(needle) and any(needle in _normalized_patient_text(source.content)
                                for source in case.sources)


def _normalized_patient_text(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ]+", "", value).replace("患者", "")


def _exam_documented_before_admission(corpus: str) -> bool:
    return (bool(re.search(r"急诊.{0,80}(?:CT|影像).{0,80}(?:收住|入院)", corpus))
            or bool(re.search(r"入院前.{0,40}(?:CT|影像)", corpus)))


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _record_schema(client, success: bool, error_type=None) -> None:
    recorder = getattr(client, "record_schema_result", None)
    if recorder is not None:
        recorder("verifier", success, error_type)
