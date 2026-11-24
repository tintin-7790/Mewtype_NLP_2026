import unittest

from src.claim_verifier import (deterministic_claim_issues, has_blocking_issues,
                                validate_verification, _guard_source_supported_model_issues)
from src.claim_extractor import extract_claims
from src.errors import CaseError
from src.model_client import VERIFIER_SCHEMA
from src.protocol import Case, Source


def claim(statement, claim_type="patient_fact", section="case_features"):
    return {"claim_id":"C001", "section":section, "statement":statement,
            "claim_type":claim_type, "source_ids":[]}


class ConservativeVerifierTests(unittest.TestCase):
    def test_missing_status_in_plan_is_not_misclassified_as_recommendation(self):
        record = {"case_features":"特点。", "diagnostic_discussion":"讨论。",
                  "plan":"当前用药情况未提供，抗感染治疗是否已启动需确认。",
                  "evidence":[], "review_items":[]}
        plan_claim = next(item for item in extract_claims(record) if item["section"] == "plan")
        self.assertEqual(plan_claim["claim_type"], "missing_information")

    def test_reasonable_clinical_summary_is_not_rejected_for_nonverbatim_wording(self):
        claims = {"C001": claim("发热伴咳嗽原因待查。", "inference", "diagnostic_discussion")}
        response = {"checked_claim_ids":["C001"], "issues":[]}
        validate_verification(response, claims, {"S1"})

    def test_reasonable_differential_and_conditional_plan_are_not_rule_errors(self):
        case = Case("C", "T0", [Source("S1", "患者发热、咳嗽，病原学结果未提供。")])
        values = [claim("结合发热、咳嗽，考虑感染性病变可能。", "inference",
                        "diagnostic_discussion"),
                  {**claim("建议结合临床情况完善病原学检查，具体方案需医生判断。",
                           "clinical_plan", "plan"), "claim_id":"C002"}]
        self.assertEqual(deterministic_claim_issues(case, values), [])

    def test_unreferenced_recommendation_is_downgraded_not_given_fake_source(self):
        item = claim("建议结合病情完善病原学检查，具体由医生判断。", "clinical_plan", "plan")
        response = {"checked_claim_ids":["C001"], "issues":[{
            "claim_id":"C001", "code":"UNREFERENCED_INFERENCE", "severity":"medium",
            "hint":"作为新增建议保留待医生审核。", "source_ids":[]}]}
        issues = validate_verification(response, {"C001":item}, {"S1"})
        self.assertEqual((issues[0]["severity"], issues[0]["error_type"]),
                         ("medium", "UNREFERENCED_INFERENCE"))
        self.assertEqual(issues[0]["source_ids"], [])

    def test_unmentioned_exam_cannot_be_asserted_unperformed(self):
        case = Case("C", "T0", [Source("S1", "患者发热、咳嗽，病原学资料未提供。")])
        issues = deterministic_claim_issues(case, [claim("痰培养未执行。", "missing_information", "review_items")])
        self.assertIn("UNKNOWN_AS_NEGATIVE", {item["error_type"] for item in issues})

    def test_explicit_right_upper_abdominal_pain_negation_is_supported(self):
        case = Case("PRACTICE02", "T0", [Source("P0014", "无发热、恶心、呕吐，无右上腹疼痛。")])
        value = claim("患者无右上腹疼痛。")
        self.assertEqual(deterministic_claim_issues(case, [value]), [])
        model_issue = [{**value, "error_type":"UNSUPPORTED_NEGATIVE", "severity":"high",
                        "revision_hint":"错误地认为未提及。"}]
        kept, dismissed = _guard_source_supported_model_issues(case, model_issue)
        self.assertEqual(kept, [])
        self.assertEqual(dismissed[0]["reason"], "explicit_source_support")
        combined = claim("患者无恶心、呕吐及右上腹疼痛。")
        kept, _ = _guard_source_supported_model_issues(case, [{**combined,
            "error_type":"PATIENT_FACT_CONTRADICTION", "severity":"high",
            "revision_hint":"错误地判定为冲突。"}])
        self.assertEqual(kept, [])

    def test_unmentioned_or_positive_symptom_is_not_protected_as_negative(self):
        negative = claim("患者无胸痛。")
        absent = Case("C", "T0", [Source("S1", "患者咳嗽。")])
        kept, _ = _guard_source_supported_model_issues(absent, [{**negative,
            "error_type":"UNSUPPORTED_NEGATIVE", "severity":"high", "revision_hint":"无来源"}])
        self.assertEqual(len(kept), 1)
        positive = Case("C", "T0", [Source("S1", "患者存在胸痛。")])
        issues = deterministic_claim_issues(positive, [negative])
        self.assertIn("FACT_CONTRADICTION", {item["error_type"] for item in issues})

    def test_unperformed_exam_cannot_become_negative_result(self):
        case = Case("C", "T0", [Source("S1", "痰培养尚未检查。")])
        issues = deterministic_claim_issues(case, [claim("痰培养检查结果阴性。")])
        self.assertIn("RESULT_AS_AVAILABLE", {item["error_type"] for item in issues})

    def test_same_source_can_preserve_time_specific_positive_and_negative(self):
        case = Case("C", "T0", [Source("S1", "昨日出现胸痛，今日无胸痛。")])
        issues = deterministic_claim_issues(case, [claim("今日无胸痛。")])
        self.assertNotIn("FACT_CONTRADICTION", {item["error_type"] for item in issues})

    def test_unprovided_medication_cannot_be_asserted_unused(self):
        case = Case("C", "T0", [Source("S1", "当前用药情况未提供。")])
        issues = deterministic_claim_issues(case, [claim("患者当前未用药。")])
        self.assertIn("UNKNOWN_AS_NEGATIVE", {item["error_type"] for item in issues})

    def test_practice01_current_treatment_and_ct_timing_regression(self):
        case = Case("PRACTICE01", "2026-05-03 20:30", [Source(
            "P0015", "既往予头孢、阿莫西林、左氧氟沙星抗感染治疗。后至我院急诊完善胸部CT提示肺部感染，以肺部感染收住我科。")])
        claims = [claim("当前尚未开始新抗菌、抗结核或抗真菌治疗。", "patient_fact", "plan"),
                  {**claim("患者入院后完成胸部CT检查。"), "claim_id":"C002"}]
        types = {item["error_type"] for item in deterministic_claim_issues(case, claims)}
        self.assertIn("UNKNOWN_AS_NEGATIVE", types)
        self.assertIn("TIME_MISMATCH", types)

    def test_rule_out_diagnosis_cannot_be_upgraded(self):
        case = Case("C", "T0", [Source("S1", "初步诊断：肺炎待排。")])
        issues = deterministic_claim_issues(case, [claim("患者诊断为肺炎。", "inference", "diagnostic_discussion")])
        self.assertIn("DIAGNOSIS_CERTAINTY_UPGRADE", {item["error_type"] for item in issues})

    def test_explicit_patient_fact_contradiction_remains_high_risk(self):
        case = Case("C", "T0", [Source("S1", "患者否认胸痛、咯血。")])
        issues = deterministic_claim_issues(case, [claim("患者存在胸痛。")])
        matched = [item for item in issues if item["error_type"] == "FACT_CONTRADICTION"]
        self.assertEqual(matched[0]["severity"], "high")
        self.assertTrue(has_blocking_issues({"issues":matched}))

    def test_modifier_negation_does_not_reverse_following_positive_fact(self):
        case = Case("C", "T0", [Source(
            "S1", "患者无明显诱因间断出现咳嗽、咳痰，治疗后症状无明显变化。")])
        claims = [claim("患者间断咳嗽、咳痰。"),
                  {**claim("治疗后症状未见明显改善。"), "claim_id":"C002"}]
        issues = deterministic_claim_issues(case, claims)
        self.assertNotIn("FACT_CONTRADICTION", {item["error_type"] for item in issues})

    def test_nonexistent_source_id_is_still_rejected(self):
        claims = {"C001": claim("患者发热。")}
        response = {"checked_claim_ids":["C001"], "issues":[{
            "claim_id":"C001", "code":"SOURCE_MISMATCH", "severity":"high",
            "hint":"引用来源不存在。", "source_ids":["S999"]}]}
        with self.assertRaises(CaseError):
            validate_verification(response, claims, {"S1"})

    def test_verifier_schema_keeps_compact_issue_and_risk_enums(self):
        code_enum = VERIFIER_SCHEMA["schema"]["properties"]["issues"]["items"]["properties"]["code"]["enum"]
        severity_enum = VERIFIER_SCHEMA["schema"]["properties"]["issues"]["items"]["properties"]["severity"]["enum"]
        self.assertIn("UNSUPPORTED_NEGATIVE", code_enum)
        self.assertIn("PLAN_AS_COMPLETED", code_enum)
        self.assertEqual(severity_enum, ["high", "medium", "low"])

    def test_checked_claim_ids_must_cover_all_claims_exactly_once(self):
        claims = {"C001":claim("发热。"), "C002":{**claim("咳嗽。"), "claim_id":"C002"}}
        with self.assertRaises(CaseError) as missing:
            validate_verification({"checked_claim_ids":["C001"], "issues":[]}, claims, {"S1"})
        self.assertEqual(missing.exception.code, "INCOMPLETE_CLAIM_COVERAGE")
        with self.assertRaises(CaseError) as duplicate:
            validate_verification({"checked_claim_ids":["C001","C001","C002"], "issues":[]}, claims, {"S1"})
        self.assertEqual(duplicate.exception.code, "INCOMPLETE_CLAIM_COVERAGE")


if __name__ == "__main__":
    unittest.main()
