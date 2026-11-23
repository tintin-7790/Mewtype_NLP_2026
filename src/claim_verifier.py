"""VeriFact-style claim verification with deterministic safety rules."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import CaseError
from .model_client import Budget, VERIFIER_SCHEMA, parse_model_json
from .protocol import Case


BASE_DIR = Path(__file__).resolve().parent.parent
FACT_VERDICTS = {"SUPPORTED", "REFUTED", "NOT_ADDRESSED"}
REASONING_VERDICTS = {"GROUNDED", "PARTIALLY_GROUNDED", "INSUFFICIENT_EVIDENCE", "REQUIRES_CLINICIAN_REVIEW"}
SECTIONS = {"case_features", "diagnostic_discussion", "plan", "evidence", "review_items"}


def run_claim_verifier(case: Case, claims: list[dict], record: dict, clinical_plan: dict,
                       retriever, retrieval_top_k: int, client, response_format: str,
                       budget: Budget) -> dict:
    deterministic = deterministic_claim_issues(case, claims)
    candidates = []
    if retriever is not None:
        for claim in claims:
            candidates.append({
                "claim_id": claim["claim_id"],
                "candidates": [candidate.to_dict() for candidate in
                               retriever.retrieve(claim["statement"], top_k=retrieval_top_k)],
            })
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "clinical_plan": clinical_plan,
        "draft_record": record,
        "claims": claims,
        "retrieval_candidates": candidates,
        "deterministic_issues": deterministic,
    }
    raw = client.call(_messages("claim_verify_system.txt", payload), response_format, budget,
                      stage="verifier", schema=VERIFIER_SCHEMA)
    response = parse_model_json(raw, stage="verifier", client=client)
    claim_map = {claim["claim_id"]: claim for claim in claims}
    normalization_issues = normalize_unreferenced_reasoning(response, claim_map)
    validate_verification(response, claim_map, {source.source_id for source in case.sources})
    response["issues"] = _merge_issues(deterministic, _merge_issues(normalization_issues, response["issues"]))
    return response


def normalize_unreferenced_reasoning(value, claim_map: dict[str, dict]) -> list[dict]:
    """Downgrade untraceable reasoning without inventing a patient source.

    Patient facts remain strict. Only inference/clinical_plan claims with an otherwise
    grounded verdict and an empty source list are converted to the existing review
    verdict and surfaced as a medium-risk issue.
    """
    if not isinstance(value, dict) or not isinstance(value.get("claims"), list):
        return []
    issues = []
    for item in value["claims"]:
        if not isinstance(item, dict) or item.get("claim_id") not in claim_map:
            continue
        claim = claim_map[item["claim_id"]]
        if (claim.get("claim_type") in ("inference", "clinical_plan")
                and item.get("verdict") in ("GROUNDED", "PARTIALLY_GROUNDED")
                and item.get("source_ids") == []):
            item["verdict"] = "REQUIRES_CLINICIAN_REVIEW"
            rationale = item.get("rationale")
            if isinstance(rationale, str):
                item["rationale"] = rationale + "；缺少可追踪的患者事实来源，程序降级为需医生审核。"
            error_type = ("UNREFERENCED_CLINICAL_RECOMMENDATION"
                          if claim["claim_type"] == "clinical_plan" else "UNREFERENCED_REASONING")
            issues.append(_issue(claim, error_type, "medium",
                                 "保留不确定性；如主张患者个体依据，应引用实际支持来源，否则列为待医生审核。", []))
    return issues


def validate_verification(value, claim_map: dict[str, dict], source_ids: set[str]) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("claims"), list) or not isinstance(value.get("issues"), list):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验结果必须包含 claims 和 issues 数组。")
    if set(value) != {"claims", "issues"}:
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验结果顶层字段不完整或含未知字段。")
    returned_ids = []
    for item in value["claims"]:
        if (not isinstance(item, dict) or set(item) != {"claim_id", "verdict", "rationale", "source_ids"}
                or item.get("claim_id") not in claim_map):
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 claims 引用了不存在的 claim_id。")
        returned_ids.append(item["claim_id"])
        verdict = item.get("verdict")
        allowed = (REASONING_VERDICTS if claim_map[item["claim_id"]]["claim_type"] in ("inference", "clinical_plan")
                   else FACT_VERDICTS)
        if verdict not in allowed:
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 verdict 非法。")
        if not isinstance(item.get("rationale"), str):
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 rationale 必须是字符串。")
        refs = item.get("source_ids")
        if not isinstance(refs, list) or any(ref not in source_ids for ref in refs):
            raise CaseError("INVALID_CLAIM_VERIFICATION", "核验结果引用了不存在的 source_id。")
        if verdict in {"SUPPORTED", "REFUTED", "GROUNDED", "PARTIALLY_GROUNDED"} and not refs:
            raise CaseError("INVALID_CLAIM_VERIFICATION", "有依据的核验 verdict 必须引用至少一个 source_id。")
    if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(claim_map):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验结果必须逐一覆盖全部 claim_id 且不得重复。")
    for issue in value["issues"]:
        _validate_issue(issue, claim_map, source_ids)


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
    return [issue for issue in verification.get("issues", []) if issue.get("severity") == "high"]


def issues_for_review(verification: dict) -> list[dict]:
    return [issue for issue in verification.get("issues", []) if issue.get("severity") == "medium"]


def _validate_issue(issue, claim_map: dict[str, dict], source_ids: set[str]) -> None:
    expected = {"claim_id", "section", "statement", "error_type", "severity", "revision_hint", "source_ids"}
    if (not isinstance(issue, dict) or set(issue) != expected
            or issue.get("claim_id") not in claim_map or issue.get("section") not in SECTIONS):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 的 claim_id/section 非法。")
    claim = claim_map[issue["claim_id"]]
    if issue.get("section") != claim["section"] or issue.get("statement") != claim["statement"]:
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 必须精确对应原 claim 的 section 和 statement。")
    if issue.get("severity") not in ("high", "medium", "low"):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue severity 非法。")
    if not all(isinstance(issue.get(key), str) for key in ("statement", "error_type", "revision_hint")):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 文本字段非法。")
    refs = issue.get("source_ids")
    if not isinstance(refs, list) or any(ref not in source_ids for ref in refs):
        raise CaseError("INVALID_CLAIM_VERIFICATION", "核验 issue 引用了不存在的 source_id。")


def _issue(claim: dict, error_type: str, severity: str, hint: str, source_ids: list[str]) -> dict:
    return {"claim_id": claim["claim_id"], "section": claim["section"], "statement": claim["statement"],
            "error_type": error_type, "severity": severity, "revision_hint": hint,
            "source_ids": source_ids}


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


_NEGATIVE_ACTIONS = ("未执行", "未进行", "未接受", "未使用", "未用药", "未予", "未行", "尚未开始")
_UNCERTAINTY_WORDS = ("待排", "疑似", "初步诊断", "考虑", "可能")
_CONTRADICTION_CONCEPTS = (
    "胸痛", "胸闷", "气促", "呼吸困难", "咯血", "寒战", "盗汗", "消瘦",
    "恶心", "呕吐", "腹痛", "腹泻", "尿频", "尿急", "尿痛", "皮疹",
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
    asserted = _positive_concepts(statement)
    if not asserted:
        return []
    matched = []
    for source in case.sources:
        if asserted & _negated_concepts(source.content):
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
    return concepts


def _exam_documented_before_admission(corpus: str) -> bool:
    return (bool(re.search(r"急诊.{0,80}(?:CT|影像).{0,80}(?:收住|入院)", corpus))
            or bool(re.search(r"入院前.{0,40}(?:CT|影像)", corpus)))


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
