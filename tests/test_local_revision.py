import unittest

from src.claim_extractor import extract_claims
from src.claim_verifier import deterministic_claim_issues, validate_verification
from src.errors import CaseError
from src.local_reviser import apply_conservative_fallback, apply_revision, revision_amplitude
from src.protocol import Case, Source


def record(plan="建议完善相关检查。", reviews=None):
    return {"case_features": "病例特点保持不变。", "diagnostic_discussion": "拟诊讨论保持不变。",
            "plan": plan, "evidence": [{"statement": "既往接受抗感染治疗。",
                "source_ids": ["S1"], "kind": "fact"}], "review_items": reviews or []}


class LocalRevisionTests(unittest.TestCase):
    def test_unknown_current_status_is_located_to_plan(self):
        case = Case("C", "T0", [Source("S1", "患者既往接受抗感染治疗，当前治疗情况未提供。")])
        issues = deterministic_claim_issues(case, extract_claims(record("当前尚未开始抗感染治疗。")))
        self.assertEqual((issues[0]["section"], issues[0]["error_type"]), ("plan", "NOT_ADDRESSED"))

    def test_provided_frequency_cannot_be_called_missing(self):
        case = Case("C", "T0", [Source("S1", "痰中偶带血丝，每周约1–2次。")])
        issues = deterministic_claim_issues(case, extract_claims(record(reviews=["痰中带血丝频次不详。"])))
        self.assertEqual(issues[0]["section"], "review_items")
        self.assertEqual(issues[0]["error_type"], "CONTRADICTS_PROVIDED_DETAIL")

    def test_negation_correction_and_time_rules(self):
        case = Case("C", "T0", [Source("S1", "否认结核患者密切接触史，但与患结核邻居有日常往来。"),
            Source("S2", "口唇无发绀（原记录发绀经复核为无发绀）。"),
            Source("S3", "1天前最高体温38.5℃，入院查体37.1℃。")])
        bad = {"case_features": "无结核接触史。口唇发绀。入院体温38.5℃。",
               "diagnostic_discussion": "病因待查。", "plan": "建议观察。", "evidence": [], "review_items": []}
        types = {issue["error_type"] for issue in deterministic_claim_issues(case, extract_claims(bad))}
        self.assertEqual(types, {"NEGATION_SCOPE_ERROR", "CORRECTION_IGNORED", "TIME_MISMATCH"})

    def test_apply_revision_changes_only_target_section(self):
        before = record("当前尚未开始抗感染治疗。")
        revision = {"section_updates": {"plan": "当前治疗情况未提供，需医生确认。"},
                    "evidence": before["evidence"], "review_items": before["review_items"]}
        after = apply_revision(before, revision, {"plan"})
        self.assertNotEqual(after["plan"], before["plan"])
        self.assertEqual(after["case_features"], before["case_features"])
        self.assertEqual(after["diagnostic_discussion"], before["diagnostic_discussion"])
        with self.assertRaises(CaseError):
            apply_revision(before, {"section_updates": {"case_features": "被改动"}}, {"plan"})
        with self.assertRaises(CaseError):
            apply_revision(before, {"section_updates": {"plan": "已修订。"},
                                    "evidence": before["evidence"], "review_items": ["越界改动"]}, {"plan"})

    def test_verifier_missing_required_fields_is_rejected(self):
        claim_map = {"C001": {"claim_id":"C001", "claim_type":"patient_fact",
                               "section":"case_features", "statement":"发热。"}}
        with self.assertRaises(CaseError):
            validate_verification({"claims":[{"claim_id":"C001", "verdict":"SUPPORTED"}], "issues":[]},
                                  claim_map, {"S1"})

    def test_conservative_fallback_removes_unknown_as_negative_and_records_amplitude(self):
        before = record("当前尚未开始抗感染治疗。")
        issue = {"claim_id":"C003", "section":"plan", "statement":"当前尚未开始抗感染治疗。",
                 "error_type":"UNKNOWN_AS_NEGATIVE", "severity":"high",
                 "revision_hint":"改为当前治疗状态未提供。", "source_ids":[]}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertNotIn("尚未开始", after["plan"])
        self.assertTrue(any("相关事实已作保守处理" in item for item in after["review_items"]))
        self.assertGreater(meta["amplitude"]["plan"]["changed_char_estimate"], 0)
        self.assertNotIn("before_text", meta["amplitude"]["plan"])


if __name__ == "__main__": unittest.main()
