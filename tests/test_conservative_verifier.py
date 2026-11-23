import unittest

from src.claim_verifier import (deterministic_claim_issues, has_blocking_issues,
                                normalize_unreferenced_reasoning, validate_verification)
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
        response = {"claims":[{"claim_id":"C001", "verdict":"GROUNDED",
                               "rationale":"症状支持且保留不确定性", "source_ids":["S1"]}], "issues":[]}
        validate_verification(response, claims, {"S1"})

    def test_unreferenced_recommendation_is_downgraded_not_given_fake_source(self):
        item = claim("建议结合病情完善病原学检查，具体由医生判断。", "clinical_plan", "plan")
        response = {"claims":[{"claim_id":"C001", "verdict":"GROUNDED",
                               "rationale":"合理建议", "source_ids":[]}], "issues":[]}
        issues = normalize_unreferenced_reasoning(response, {"C001":item})
        self.assertEqual(response["claims"][0]["verdict"], "REQUIRES_CLINICIAN_REVIEW")
        self.assertEqual(response["claims"][0]["source_ids"], [])
        self.assertEqual((issues[0]["severity"], issues[0]["error_type"]),
                         ("medium", "UNREFERENCED_CLINICAL_RECOMMENDATION"))
        validate_verification(response, {"C001":item}, {"S1"})

    def test_unmentioned_exam_cannot_be_asserted_unperformed(self):
        case = Case("C", "T0", [Source("S1", "患者发热、咳嗽，病原学资料未提供。")])
        issues = deterministic_claim_issues(case, [claim("痰培养未执行。", "missing_information", "review_items")])
        self.assertIn("UNKNOWN_AS_NEGATIVE", {item["error_type"] for item in issues})

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
        response = {"claims":[{"claim_id":"C001", "verdict":"SUPPORTED",
                               "rationale":"原文支持", "source_ids":["S999"]}], "issues":[]}
        with self.assertRaises(CaseError):
            validate_verification(response, claims, {"S1"})

    def test_verifier_schema_keeps_existing_verdict_and_risk_enums(self):
        claim_enum = VERIFIER_SCHEMA["schema"]["properties"]["claims"]["items"]["properties"]["verdict"]["enum"]
        severity_enum = VERIFIER_SCHEMA["schema"]["properties"]["issues"]["items"]["properties"]["severity"]["enum"]
        self.assertEqual(set(claim_enum), {"SUPPORTED", "REFUTED", "NOT_ADDRESSED", "GROUNDED",
                                           "PARTIALLY_GROUNDED", "INSUFFICIENT_EVIDENCE",
                                           "REQUIRES_CLINICIAN_REVIEW"})
        self.assertEqual(severity_enum, ["high", "medium", "low"])


if __name__ == "__main__":
    unittest.main()
