import hashlib
import re
import unicodedata
from copy import deepcopy

from .errors import CaseError
from .protocol import Case


REQUIRED_TEXT = ("case_features", "diagnostic_discussion", "plan")
GENERIC_REVISION_TEMPLATES = (
    "该项陈述依据不足，需医生结合原始资料确认",
    "该项患者事实存在冲突，需以原始资料复核结果为准",
    "该项事实应以复核更正后的记录为准",
    "该项时间信息需结合原始资料重新确认",
    "该项数值或单位需以原始资料为准",
    "相关事实需医生确认",
)
NUMBER_WITH_CONTEXT = re.compile(r"(?P<prefix>[\u4e00-\u9fffA-Za-z]{0,10})(?P<num>[<>≤≥]?\d+(?:\.\d+)?)(?P<suffix>\s*(?:℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%|岁|天|日|周|月|年)?)")
MEASUREMENT_WITH_UNIT = re.compile(
    r"(?P<num>[<>≤≥]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%)"
)
CANONICAL_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?!\d)")
EVIDENCE_CODES = {
    "EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND", "DATE_SUPPORT",
    "NUMERIC_SUPPORT", "NEGATION_SUPPORT", "PLAN_AS_DONE", "FACT_SUPPORT_LIMITED",
}
HARD_VALIDATION_CODES = {
    "FORMAT", "UNSUPPORTED_BODY_DATE", "UNSUPPORTED_BODY_NUMERIC",
    "UNRESOLVED_BODY_NUMERIC_ROLE",
    "PROSE_BRACKET_MISMATCH", "GENERIC_REVISION_TEMPLATE",
}
REPAIRABLE_VALIDATION_CODES = {
    "EVIDENCE_FORMAT", "SOURCE_ID", "EVIDENCE_KIND", "DATE_SUPPORT",
    "NUMERIC_SUPPORT", "PLAN_AS_DONE", "NON_ACTIONABLE_REVIEW_ITEM",
    "TECHNICAL_REVIEW_ITEM", "UNVERIFIED_PROPOSED_TARGET",
}
TECHNICAL_REVIEW_PATTERNS = (
    r"\b(?:JSON|API|Schema|Verifier|Prompt)\b", r"\bsource_ids?\b",
    r"\b(?:Fact|Claim|Problem)\s*ID\b", r"来源编号(?:错误|无效|不存在)",
    r"模型响应(?:解析)?错误", r"程序(?:运行)?错误", r"调用失败", r"解析失败",
)

PATIENT_FACT = "PATIENT_FACT"
PROPOSED_TARGET = "PROPOSED_TARGET"
CLINICAL_THRESHOLD = "CLINICAL_THRESHOLD"
PLANNED_TEST = "PLANNED_TEST"
UNKNOWN = "UNKNOWN"
VALIDATION_MODES = ("strict", "lite")

_LITE_HARD_CLAIM_CODES = {
    "PATIENT_FACT_CONTRADICTION", "FACT_CONTRADICTION",
    "NUMERIC_VALUE_ERROR", "UNIT_ERROR", "TEMPORAL_CONFLICT", "TIME_MISMATCH",
    "CORRECTION_IGNORED", "NEGATION_SCOPE_ERROR", "UNKNOWN_AS_NEGATIVE",
    "NOT_ADDRESSED", "CONTRADICTS_PROVIDED_DETAIL", "PLAN_AS_COMPLETED",
    "RESULT_AS_AVAILABLE", "DIAGNOSIS_CERTAINTY_UPGRADE", "SOURCE_MISMATCH",
    "INTERNAL_AUDIT_LEAK", "TECHNICAL_REVIEW_ITEM", "UNSAFE_RECOMMENDATION",
}
_LITE_SOFT_CLAIM_CODES = {
    "INFERENCE_UNDER_SUPPORTED", "IMPORTANT_EVIDENCE_OMITTED", "PLAN_ALIGNMENT_WEAK",
    "UNREFERENCED_INFERENCE",
}

_TARGET_METRICS = (
    "低密度脂蛋白胆固醇", "血氧饱和度", "餐后血糖", "空腹血糖",
    "收缩压", "舒张压", "平均动脉压", "血压", "心率", "脉搏",
    "血糖", "体温", "SpO2", "LDL-C",
)
_TARGET_METRIC_PATTERN = "(?:" + "|".join(map(re.escape, _TARGET_METRICS)) + ")"
_TARGET_VALUE_PATTERN = r"[<>≤≥]?\d+(?:\.\d+)?\s*(?:℃|次/分|mmHg|kg/m²|kg|cm|mmol/L|μmol/L|mg/L|g/L|U/L|ng/L|ng/mL|×10[⁹9]/L|%)"
_TARGET_EXPRESSIONS = (
    re.compile(
        rf"(?:(?:拟|计划|建议|应|需|可考虑)\s*)?(?:将|把)?\s*"
        rf"(?P<metric>{_TARGET_METRIC_PATTERN})\s*"
        rf"(?:(?:控制|维持|调整)(?:在|至|为|于)?|(?:降|降低)至)\s*"
        rf"(?P<value>{_TARGET_VALUE_PATTERN})(?:\s*(?:以下|以内|左右))?"
    ),
    re.compile(
        rf"(?P<metric>{_TARGET_METRIC_PATTERN})\s*"
        rf"(?:控制)?目标(?:值)?\s*(?:为|在|至)?\s*"
        rf"(?P<value>{_TARGET_VALUE_PATTERN})(?:\s*(?:以下|以内|左右))?"
    ),
    re.compile(
        rf"(?:目标|控制目标)(?:为|：|:)\s*(?P<metric>{_TARGET_METRIC_PATTERN})?\s*"
        rf"(?P<value>{_TARGET_VALUE_PATTERN})(?:\s*(?:以下|以内|左右))?"
    ),
)


def validate_record(record, case: Case, validation_mode: str = "strict") -> list[dict]:
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(f"unknown validation mode: {validation_mode}")
    issues: list[dict] = []
    if not isinstance(record, dict):
        raise CaseError("INVALID_MODEL_OUTPUT", "模型结果必须是 JSON 对象。")
    expected = {"case_features", "diagnostic_discussion", "plan", "evidence", "review_items"}
    if set(record) != expected:
        issues.append(_issue("error", "FORMAT", "病历顶层字段不完整或含未知字段。"))
    for field in REQUIRED_TEXT:
        if not isinstance(record.get(field), str) or not record[field].strip():
            issues.append(_issue("error", "FORMAT", f"{field} 必须是非空字符串。"))
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        issues.append(_issue("error", "FORMAT", "evidence 必须是数组。"))
        evidence = []
    if not isinstance(record.get("review_items"), list) or any(not isinstance(x, str) for x in record.get("review_items", [])):
        issues.append(_issue("error", "FORMAT", "review_items 必须是字符串数组。"))
    known = {s.source_id: s.content for s in case.sources}
    for index, item in enumerate(evidence):
        if (not isinstance(item, dict) or set(item) != {"statement", "source_ids", "kind"}
                or not isinstance(item.get("statement"), str) or not item["statement"].strip()):
            issues.append(_issue("error", "EVIDENCE_FORMAT", f"evidence[{index}] 结构无效。")); continue
        ids = item.get("source_ids")
        if not isinstance(ids, list) or not ids or any(x not in known for x in ids):
            issues.append(_issue("error", "SOURCE_ID", f"evidence[{index}] 引用了不存在或空的 source_ids。")); continue
        if item.get("kind") not in ("fact", "inference"):
            issues.append(_issue("error", "EVIDENCE_KIND", f"evidence[{index}].kind 非法。"))
        cited = "\n".join(known[x] for x in ids)
        issues.extend(_support_checks(item["statement"], cited, index, item.get("kind")))
    issues.extend(_body_numeric_support_checks(record, case, validation_mode))
    issues.extend(_prose_quality_checks(record))
    return issues


def _prose_quality_checks(record: dict) -> list[dict]:
    issues = []
    for field in REQUIRED_TEXT:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        if not _balanced_brackets(value):
            issues.append(_issue("error", "PROSE_BRACKET_MISMATCH",
                                 f"{field} 存在括号不匹配，可能由不安全的局部修订造成。"))
        matched = [template for template in GENERIC_REVISION_TEMPLATES if template in value]
        if matched:
            issues.append(_issue("error", "GENERIC_REVISION_TEMPLATE",
                                 f"{field} 含无具体对象的修订模板，不能作为临床正文。"))
    reviews = record.get("review_items")
    if isinstance(reviews, list):
        for index, item in enumerate(reviews):
            if (isinstance(item, str)
                    and any(template in item for template in GENERIC_REVISION_TEMPLATES)):
                issues.append(_issue("error", "NON_ACTIONABLE_REVIEW_ITEM",
                                     f"review_items[{index}] 缺少具体、可操作的临床核实对象。"))
            if (isinstance(item, str)
                    and any(re.search(pattern, item, flags=re.IGNORECASE)
                            for pattern in TECHNICAL_REVIEW_PATTERNS)):
                issues.append(_issue("error", "TECHNICAL_REVIEW_ITEM",
                                     f"review_items[{index}] 含技术审计信息，不能进入临床待核事项。"))
    return issues


def _balanced_brackets(text: str) -> bool:
    pairs = {")": "(", "）": "（", "]": "[", "】": "【", "}": "{"}
    openings = set(pairs.values())
    stack = []
    for char in text:
        if char in openings:
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _support_checks(statement: str, cited: str, index: int, kind: str | None) -> list[dict]:
    issues = []
    normalized_statement = normalize_equivalent_dates(statement)
    normalized_cited = normalize_equivalent_dates(cited)
    cited_dates = set(CANONICAL_DATE.findall(normalized_cited))
    for date in CANONICAL_DATE.findall(normalized_statement):
        if date not in cited_dates:
            issues.append(_issue("error", "DATE_SUPPORT",
                                 f"evidence[{index}] 的日期“{date}”未在引用来源中找到。",
                                 evidence_index=index))
    statement_without_dates = CANONICAL_DATE.sub("<DATE>", normalized_statement)
    cited_without_dates = CANONICAL_DATE.sub("<DATE>", normalized_cited)
    for match in NUMBER_WITH_CONTEXT.finditer(statement_without_dates):
        token = match.group("num") + match.group("suffix").replace(" ", "")
        compact = cited_without_dates.replace(" ", "")
        # A value with a unit must match that exact value/unit pair.  Falling
        # back to the bare number can confuse thresholds, dates and other tests.
        supported = token in compact if match.group("suffix").strip() else match.group("num") in compact
        if not supported:
            issues.append(_issue("error", "NUMERIC_SUPPORT",
                                 f"evidence[{index}] 的数值/单位“{token}”未在引用来源中找到。",
                                 evidence_index=index))
    if any(x in statement for x in ("否认", "未见", "无明显", "阴性")) and not any(x in cited for x in ("否认", "未见", "无明显", "阴性", "未闻及")):
        issues.append(_issue("warning", "NEGATION_SUPPORT", f"evidence[{index}] 的否定表述需人工复核。"))
    if any(x in statement for x in ("已完成", "已予", "已行")) and any(x in cited for x in ("待完善", "拟行", "计划", "建议")):
        issues.append(_issue("error", "PLAN_AS_DONE", f"evidence[{index}] 可能把计划写成已完成。"))
    if kind == "fact" and not _has_lexical_support(statement, cited):
        issues.append(_issue("warning", "FACT_SUPPORT_LIMITED",
                             f"evidence[{index}] 与引用来源的字面重合较低，程序无法确认语义支持，需人工或模型核验。"))
    return issues


def recover_evidence(record: dict, case: Case, max_source_chars: int = 320) -> tuple[dict, dict]:
    """Recover only independently verifiable Evidence without changing prose.

    Source reassignment requires one unique source whose normalized text directly
    contains the statement. Similarity, graph scores and clinical knowledge are
    deliberately excluded. Unrecoverable Evidence is dropped; the caller must
    separately enforce the body-level patient number/date safety checks.
    """
    result = deepcopy(record)
    original = result.get("evidence") if isinstance(result, dict) else None
    initial_issues = _evidence_issues(record, case)
    audit = {
        "evidence_validation_initial": _evidence_summary(initial_issues),
        "evidence_repairs": [],
        "evidence_dropped": [],
        "source_reassignment": [],
        "date_normalizations": [],
        "residual_evidence_risks": [],
        "additional_model_calls": 0,
    }
    if not isinstance(original, list):
        audit["evidence_validation_final"] = _evidence_summary(initial_issues)
        audit["recovered_success"] = False
        audit["partial_evidence_recovery"] = False
        return result, audit
    known = {source.source_id: source.content for source in case.sources}
    kept = []
    for index, item in enumerate(original):
        if not _basic_evidence_item(item):
            audit["evidence_dropped"].append(_dropped_evidence(index, item, ["EVIDENCE_FORMAT"]))
            continue
        working = deepcopy(item)
        declared = working.get("source_ids")
        current_sources_valid = (isinstance(declared, list) and bool(declared)
                                 and all(isinstance(value, str) and value in known
                                         for value in declared))
        cited = ("\n".join(known[value] for value in declared)
                 if current_sources_valid else "")
        errors = _support_errors(working, cited, index) if current_sources_valid else [
            _issue("error", "SOURCE_ID", f"evidence[{index}] 来源编号无效。",
                   evidence_index=index)
        ]
        candidates = _unique_direct_support(working, case, index)
        if errors and len(candidates) == 1 and candidates != declared:
            before = list(declared) if isinstance(declared, list) else []
            working["source_ids"] = candidates
            cited = known[candidates[0]]
            errors = _support_errors(working, cited, index)
            if not errors:
                reassignment = {"evidence_index": index, "from_source_ids": before,
                                "to_source_ids": candidates, "strategy": "unique_direct_source_text"}
                audit["source_reassignment"].append(reassignment)
                audit["evidence_repairs"].append(reassignment)
        error_codes = {issue["code"] for issue in errors}
        if (errors and error_codes.issubset({"DATE_SUPPORT", "NUMERIC_SUPPORT"})
                and current_sources_valid and working.get("kind") == "fact"):
            exact = "；".join(known[source_id].strip() for source_id in working["source_ids"])
            if 0 < len(exact) <= max_source_chars and not _support_errors(
                    {**working, "statement": exact}, cited, index):
                before = working["statement"]
                working["statement"] = exact
                errors = []
                audit["evidence_repairs"].append({
                    "evidence_index": index, "strategy": "exact_cited_source",
                    "source_ids": list(working["source_ids"]),
                    "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                    "after_sha256": hashlib.sha256(exact.encode()).hexdigest(),
                })
        if errors:
            audit["evidence_dropped"].append(
                _dropped_evidence(index, working, [issue["code"] for issue in errors]))
            continue
        statement_dates = CANONICAL_DATE.findall(normalize_equivalent_dates(working["statement"]))
        cited_dates = CANONICAL_DATE.findall(normalize_equivalent_dates(cited))
        if statement_dates and set(statement_dates).issubset(set(cited_dates)):
            audit["date_normalizations"].append({
                "evidence_index": index, "canonical_dates": list(dict.fromkeys(statement_dates))
            })
        kept.append(working)
    result["evidence"] = kept
    final_issues = _evidence_issues(result, case)
    audit["evidence_validation_final"] = _evidence_summary(final_issues)
    audit["residual_evidence_risks"] = [
        {"code": issue["code"], "level": issue["level"],
         "evidence_index": issue.get("evidence_index")}
        for issue in final_issues
    ]
    audit["recovered_success"] = bool(initial_issues) and not any(
        item["level"] == "error" for item in final_issues)
    audit["partial_evidence_recovery"] = bool(audit["evidence_dropped"])
    return result, audit


def recover_plan_numeric_targets(record: dict, case: Case) -> tuple[dict, dict]:
    """Remove only safely located, source-unverified treatment target values.

    This recovery never edits patient measurements, Evidence, clinical thresholds,
    or numerics whose semantic role is unclear. A target is changed only when the
    value belongs to one unique target expression in a complete sentence; the
    surrounding plan and clinical objective remain without a fabricated threshold.
    """
    result = deepcopy(record)
    plan = result.get("plan") if isinstance(result, dict) else None
    audit = {
        "applied": False,
        "strategy": "safe_proposed_target_dequantification",
        "repairs": [],
        "unresolved": [],
        "additional_model_calls": 0,
    }
    if not isinstance(plan, str) or not plan.strip():
        return result, audit
    corpus = normalize_equivalent_dates("\n".join(source.content for source in case.sources))
    corpus_without_dates = CANONICAL_DATE.sub("<DATE>", corpus).replace(" ", "")
    replacements = []
    for measurement in MEASUREMENT_WITH_UNIT.finditer(plan):
        token = measurement.group("num") + measurement.group("unit")
        role = classify_numeric_semantic_role(
            plan, measurement.start(), measurement.end(), "plan",
        )
        if role == PROPOSED_TARGET:
            if _proposed_target_supported(plan, measurement.start(), measurement.end(), case):
                continue
        elif token.replace(" ", "") in corpus_without_dates:
            continue
        sentence_start, sentence_end = _sentence_span(plan, measurement.start(), measurement.end())
        sentence = plan[sentence_start:sentence_end]
        context_hash = hashlib.sha256(sentence.encode()).hexdigest()
        if role != PROPOSED_TARGET:
            audit["unresolved"].append({
                "token": token, "semantic_role": role,
                "context_sha256": context_hash,
                "reason": "not_a_safely_rewritable_proposed_target",
            })
            continue
        located = _locate_unique_target_expression(
            sentence, measurement.start() - sentence_start, measurement.end() - sentence_start,
        )
        if located is None:
            audit["unresolved"].append({
                "token": token, "semantic_role": role,
                "context_sha256": context_hash,
                "reason": "target_expression_not_unique_or_incomplete",
            })
            continue
        relative_start, relative_end, metric = located
        replacement = f"{metric}控制目标需由医生结合当前病情和动态监测结果个体化确定"
        replacements.append((sentence_start + relative_start,
                             sentence_start + relative_end, replacement, {
                                 "token": token,
                                 "semantic_role": role,
                                 "metric": metric,
                                 "context_sha256": context_hash,
                                 "strategy": "replace_complete_target_expression",
                             }))
    if not replacements:
        return result, audit
    rewritten = plan
    occupied = []
    applied = []
    for start, end, replacement, detail in sorted(replacements, reverse=True):
        if any(not (end <= left or start >= right) for left, right in occupied):
            audit["unresolved"].append({**detail, "reason": "overlapping_target_expression"})
            continue
        candidate = rewritten[:start] + replacement + rewritten[end:]
        if (not candidate.strip() or not _balanced_brackets(candidate)
                or any(template in candidate for template in GENERIC_REVISION_TEMPLATES)):
            audit["unresolved"].append({**detail, "reason": "post_rewrite_structure_invalid"})
            continue
        rewritten = candidate
        occupied.append((start, end))
        applied.append(detail)
    if applied:
        result["plan"] = rewritten
        audit["applied"] = True
        audit["repairs"] = list(reversed(applied))
        audit["before_sha256"] = hashlib.sha256(plan.encode()).hexdigest()
        audit["after_sha256"] = hashlib.sha256(rewritten.encode()).hexdigest()
    return result, audit


def repair_evidence_numeric_support(record: dict, case: Case,
                                    max_source_chars: int = 320) -> tuple[dict, dict]:
    """Safely replace invalid fact evidence with its short, exact cited text.

    No numeric value is guessed or mapped to a nearby value. Inference evidence
    and long/multi-purpose sources are left unresolved so final validation can
    fail explicitly instead of manufacturing provenance.
    """
    result = deepcopy(record)
    known = {source.source_id: source.content for source in case.sources}
    repaired = []
    unresolved = []
    for index, item in enumerate(result.get("evidence", []) if isinstance(result, dict) else []):
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str):
            continue
        ids = item.get("source_ids")
        if not isinstance(ids, list) or not ids or any(source_id not in known for source_id in ids):
            continue
        cited = "\n".join(known[source_id] for source_id in ids)
        numeric = [issue for issue in _support_checks(item["statement"], cited, index, item.get("kind"))
                   if issue["code"] == "NUMERIC_SUPPORT"]
        if not numeric:
            continue
        exact = "；".join(known[source_id].strip() for source_id in ids)
        can_repair = item.get("kind") == "fact" and 0 < len(exact) <= max_source_chars
        if can_repair:
            remaining = [issue for issue in _support_checks(exact, cited, index, "fact")
                         if issue["code"] == "NUMERIC_SUPPORT"]
            if not remaining:
                before = item["statement"]
                item["statement"] = exact
                repaired.append({"evidence_index": index, "strategy": "exact_cited_source",
                                 "source_ids": list(ids),
                                 "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                                 "after_sha256": hashlib.sha256(exact.encode()).hexdigest()})
                continue
        unresolved.append({"evidence_index": index, "code": "NUMERIC_SUPPORT",
                           "kind": item.get("kind"), "source_ids": list(ids)})
    return result, {"repaired_count": len(repaired), "unresolved_count": len(unresolved),
                    "repairs": repaired, "unresolved": unresolved}


def normalize_equivalent_dates(value: str) -> str:
    """Normalize common equivalent Chinese numeric dates without changing chronology."""
    normalized = _normalize_measurements(value)

    def chinese_full(match):
        canonical = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        follows_with_time = match.end() < len(normalized) and normalized[match.end()].isdigit()
        return canonical + (" " if follows_with_time else "")

    normalized = re.sub(
        r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日",
        chinese_full,
        normalized,
    )
    normalized = re.sub(
        r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)",
        lambda match: f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}",
        normalized,
    )
    normalized = re.sub(
        r"(?<!\d)(\d{4})(?:年|[-/.])(\d{1,2})月?(?![-/.\d])",
        lambda match: f"{int(match.group(1)):04d}-{int(match.group(2)):02d}",
        normalized,
    )
    return normalized


def _normalize_measurements(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return (normalized.replace("µ", "μ").replace("\u00a0", " ")
            .replace("−", "-").replace("‐", "-").replace("‑", "-")
            .replace("–", "-").replace("—", "-"))


def _has_lexical_support(statement: str, cited: str) -> bool:
    clean = lambda value: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ]+", "", value)
    left, right = clean(statement), clean(cited)
    if left and left in right:
        return True
    grams = {left[i:i+2] for i in range(max(0, len(left)-1)) if not left[i:i+2].isdigit()}
    if not grams:
        return True
    return sum(gram in right for gram in grams) / len(grams) >= 0.25


def classify_validation_issue(issue: dict) -> str:
    code = issue.get("code")
    if issue.get("level") == "warning":
        return "soft"
    if code in HARD_VALIDATION_CODES:
        return "hard"
    if code in REPAIRABLE_VALIDATION_CODES:
        return "repairable"
    return "hard"


def classify_claim_issue_for_final(issue: dict, validation_mode: str = "strict") -> dict:
    """Classify an unresolved Claim for the final output boundary.

    Lite mode relaxes only ordinary inference/recommendation uncertainty. It
    never downgrades explicit patient-fact reversals, numeric/unit/time errors,
    state inversions, unsafe recommendations, or structurally unsafe output.
    Missing statement context is deliberately HARD so a privacy-redacted old log
    cannot be made successful merely by changing its status field.
    """
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(f"unknown validation mode: {validation_mode}")
    severity = issue.get("severity")
    code = issue.get("error_type") or issue.get("code")
    statement = issue.get("statement")
    result = {"risk_level": "soft", "reason": "non_high_issue"}
    if severity != "high":
        return result
    if validation_mode == "strict":
        return {"risk_level": "hard", "reason": "strict_high_risk"}
    if not isinstance(statement, str) or not statement.strip():
        return {"risk_level": "hard", "reason": "missing_statement_context"}
    if code in _LITE_HARD_CLAIM_CODES:
        return {"risk_level": "hard", "reason": "lite_safety_floor"}
    if code in _LITE_SOFT_CLAIM_CODES:
        return {"risk_level": "soft", "reason": "lite_ordinary_clinical_inference"}
    if code == "UNSUPPORTED_NEGATIVE":
        if _is_definitive_exclusion(statement):
            return {"risk_level": "hard", "reason": "definitive_exclusion_from_weak_negative"}
        return {"risk_level": "soft", "reason": "qualified_negative_clinical_inference"}
    return {"risk_level": "hard", "reason": "unknown_high_risk_code"}


def classify_numeric_semantic_role(text: str, start: int, end: int, section: str) -> str:
    """Classify one numeric measurement from its complete local sentence.

    Section is supporting context, not the decision rule: a number in ``plan``
    can still be a recalled patient fact, and a target outside ``plan`` remains a
    proposed target when its own wording says so.
    """
    if not isinstance(text, str) or not (0 <= start < end <= len(text)):
        return UNKNOWN
    sentence_start, sentence_end = _sentence_span(text, start, end)
    sentence = text[sentence_start:sentence_end]
    local_start, local_end = start - sentence_start, end - sentence_start
    if _locate_unique_target_expression(sentence, local_start, local_end) is not None:
        return PROPOSED_TARGET
    before = sentence[:local_start]
    numeric_text = sentence[local_start:local_end]
    conditional = bool(
        (numeric_text.startswith(("<", ">", "≤", "≥"))
         and re.search(r"(?:若|如|当|一旦).{0,18}$", before[-24:]))
        or re.search(
            r"(?:达到|超过|高于|低于|小于|大于|阈值|界值|标准|参考范围)\s*$",
            before[-18:],
        )
    )
    if conditional or any(word in sentence for word in ("诊断标准", "判断标准", "参考阈值", "界值")):
        return CLINICAL_THRESHOLD
    planned = any(word in sentence for word in ("拟行", "计划", "待完善", "建议", "复查", "监测", "随访"))
    completed = any(word in sentence for word in (
        "已行", "已完成", "实测", "测得", "检查示", "结果为", "入院时", "既往", "曾服",
    ))
    if planned and not completed and any(word in sentence for word in (
            "检查", "复查", "监测", "检测", "随访", "评估")):
        return PLANNED_TEST
    observed_context = bool(re.search(
        r"(?:当前|目前|入院时|实测|测得).{0,16}(?:为|示|[:：])?", sentence,
    ))
    if section == "case_features" or completed or observed_context:
        return PATIENT_FACT
    if section == "diagnostic_discussion" and any(word in sentence for word in (
            "示", "为", "升高", "降低", "异常", "既往", "目前", "当前")):
        return PATIENT_FACT
    if section == "plan" and any(word in sentence for word in (
            "已予", "已用", "已服", "当前为", "目前为", "既往", "实测", "测得")):
        return PATIENT_FACT
    return UNKNOWN


def _body_numeric_support_checks(record: dict, case: Case,
                                 validation_mode: str = "strict") -> list[dict]:
    corpus = normalize_equivalent_dates("\n".join(source.content for source in case.sources))
    corpus_dates = set(CANONICAL_DATE.findall(corpus))
    corpus_without_dates = CANONICAL_DATE.sub("<DATE>", corpus).replace(" ", "")
    issues = []
    for field in REQUIRED_TEXT:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        normalized = normalize_equivalent_dates(value)
        for date in CANONICAL_DATE.findall(normalized):
            if date not in corpus_dates:
                issues.append(_issue(
                    "error", "UNSUPPORTED_BODY_DATE",
                    f"{field} 含原始 Sources 未支持的患者日期“{date}”。", field=field,
                ))
        without_dates = CANONICAL_DATE.sub("<DATE>", normalized)
        for match in MEASUREMENT_WITH_UNIT.finditer(without_dates):
            token = match.group("num") + match.group("unit")
            role = classify_numeric_semantic_role(
                without_dates, match.start(), match.end(), field,
            )
            supported = token.replace(" ", "") in corpus_without_dates
            if role == PROPOSED_TARGET:
                supported = _proposed_target_supported(
                    without_dates, match.start(), match.end(), case,
                )
            if not supported:
                details = {"field": field, "token": token, "semantic_role": role}
                if role == PROPOSED_TARGET:
                    issues.append(_issue(
                        "warning" if validation_mode == "lite" else "error",
                        "UNVERIFIED_PROPOSED_TARGET",
                        f"{field} 含无法由患者 Sources 判断适用性的拟定治疗目标“{token}”。",
                        **details,
                    ))
                elif role == CLINICAL_THRESHOLD:
                    issues.append(_issue(
                        "warning", "CLINICAL_THRESHOLD_REVIEW",
                        f"{field} 含非患者实测值的临床阈值“{token}”，需审核其医学适用性。",
                        **details,
                    ))
                elif role == PLANNED_TEST:
                    issues.append(_issue(
                        "warning", "PLANNED_NUMERIC_REVIEW",
                        f"{field} 含拟检查或监测相关数值“{token}”，不得表述为既有结果。",
                        **details,
                    ))
                elif role == PATIENT_FACT:
                    issues.append(_issue(
                        "error", "UNSUPPORTED_BODY_NUMERIC",
                        f"{field} 含原始 Sources 未支持的患者数值/单位“{token}”。",
                        **details,
                    ))
                else:
                    issues.append(_issue(
                        "error", "UNRESOLVED_BODY_NUMERIC_ROLE",
                        f"{field} 的数值/单位“{token}”语义角色无法可靠确定。",
                        **details,
                    ))
    return issues


def _is_definitive_exclusion(statement: str) -> bool:
    if any(phrase in statement for phrase in (
            "需排除", "需要排除", "待排除", "进一步排除", "鉴别排除", "是否排除",
            "不能排除", "无法排除", "尚不能排除", "不具排他", "排除与否")):
        return False
    if "明确否定" in statement:
        return True
    if re.search(r"(?:已|可|可以|能够|基本|完全|明确|予以)?排除", statement):
        return True
    return False


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    while left > 0 and text[left - 1] not in "。！？!?；;\r\n":
        left -= 1
    right = end
    while right < len(text) and text[right] not in "。！？!?；;\r\n":
        right += 1
    if right < len(text):
        right += 1
    return left, right


def _locate_unique_target_expression(sentence: str, value_start: int,
                                     value_end: int) -> tuple[int, int, str] | None:
    candidates = []
    for pattern in _TARGET_EXPRESSIONS:
        for match in pattern.finditer(sentence):
            start, end = match.span("value")
            if start <= value_start and value_end <= end:
                metric = match.groupdict().get("metric") or "相关指标"
                candidates.append((match.start(), match.end(), metric))
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _proposed_target_supported(text: str, value_start: int, value_end: int,
                               case: Case) -> bool:
    sentence_start, sentence_end = _sentence_span(text, value_start, value_end)
    sentence = text[sentence_start:sentence_end]
    located = _locate_unique_target_expression(
        sentence, value_start - sentence_start, value_end - sentence_start,
    )
    if located is None:
        return False
    _, _, metric = located
    expected_match = MEASUREMENT_WITH_UNIT.search(text[value_start:value_end])
    if expected_match is None:
        return False
    expected = (expected_match.group("num") + expected_match.group("unit")).replace(" ", "")
    for source in case.sources:
        source_text = _normalize_measurements(source.content)
        for pattern in _TARGET_EXPRESSIONS:
            for candidate in pattern.finditer(source_text):
                candidate_metric = candidate.groupdict().get("metric") or "相关指标"
                candidate_value = candidate.group("value").replace(" ", "")
                if candidate_metric == metric and candidate_value == expected:
                    return True
    return False


def _basic_evidence_item(item) -> bool:
    return (isinstance(item, dict)
            and set(item) == {"statement", "source_ids", "kind"}
            and isinstance(item.get("statement"), str) and bool(item["statement"].strip())
            and item.get("kind") in ("fact", "inference"))


def _support_errors(item: dict, cited: str, index: int) -> list[dict]:
    return [issue for issue in _support_checks(item["statement"], cited, index, item.get("kind"))
            if issue["level"] == "error"]


def _unique_direct_support(item: dict, case: Case, index: int) -> list[str]:
    if item.get("kind") != "fact":
        return []
    statement = normalize_equivalent_dates(item["statement"])
    compact_statement = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ-]+", "", statement)
    matches = []
    for source in case.sources:
        cited = normalize_equivalent_dates(source.content)
        compact_source = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%℃μ-]+", "", cited)
        if (compact_statement and compact_statement in compact_source
                and not _support_errors(item, source.content, index)):
            matches.append(source.source_id)
    return matches if len(matches) == 1 else []


def _evidence_issues(record: dict, case: Case) -> list[dict]:
    return [issue for issue in validate_record(record, case) if issue.get("code") in EVIDENCE_CODES]


def _evidence_summary(issues: list[dict]) -> dict:
    return {
        "error_count": sum(issue.get("level") == "error" for issue in issues),
        "warning_count": sum(issue.get("level") == "warning" for issue in issues),
        "codes": [issue.get("code") for issue in issues],
    }


def _dropped_evidence(index: int, item, codes: list[str]) -> dict:
    statement = item.get("statement", "") if isinstance(item, dict) else ""
    return {
        "evidence_index": index,
        "codes": list(dict.fromkeys(codes)),
        "source_ids": (list(item.get("source_ids", []))
                       if isinstance(item, dict) and isinstance(item.get("source_ids"), list) else []),
        "statement_sha256": hashlib.sha256(str(statement).encode()).hexdigest(),
    }


def _issue(level: str, code: str, message: str, **details) -> dict:
    return {"level": level, "code": code, "message": message, **details}
