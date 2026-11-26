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
        issue = {**next(item for item in extract_claims(before) if item["section"] == "plan"),
                 "error_type":"UNSUPPORTED_NEGATIVE", "severity":"high",
                 "revision_hint":"改为未知。"}
        revision = {"edits":[{"claim_id":issue["claim_id"], "section":"plan",
                               "action":"replace", "replacement":"当前治疗情况未提供，需医生确认。"}],
                    "review_items_to_add":[]}
        after, meta = apply_revision(before, revision, [issue])
        self.assertNotEqual(after["plan"], before["plan"])
        self.assertEqual(after["case_features"], before["case_features"])
        self.assertEqual(after["diagnostic_discussion"], before["diagnostic_discussion"])
        self.assertEqual(meta["revised_claim_ids"], [issue["claim_id"]])
        with self.assertRaises(CaseError):
            apply_revision(before, {"edits":[{"claim_id":issue["claim_id"],
                "section":"case_features", "action":"replace", "replacement":"被改动"}],
                "review_items_to_add":[]}, [issue])
        with self.assertRaises(CaseError):
            apply_revision(before, {"edits":revision["edits"],
                                    "review_items_to_add":["JSON 解析失败"]}, [issue])
        with self.assertRaises(CaseError):
            apply_revision(before, {"edits":revision["edits"],
                                    "review_items_to_add":["json parse failed"]}, [issue])

    def test_verifier_missing_required_fields_is_rejected(self):
        claim_map = {"C001": {"claim_id":"C001", "claim_type":"patient_fact",
                               "section":"case_features", "statement":"发热。"}}
        with self.assertRaises(CaseError):
            validate_verification({"checked_claim_ids":["C001"]},
                                  claim_map, {"S1"})

    def test_conservative_fallback_removes_unknown_as_negative_and_records_amplitude(self):
        before = record("当前尚未开始抗感染治疗。")
        issue = {**next(item for item in extract_claims(before) if item["section"] == "plan"),
                 "error_type":"UNKNOWN_AS_NEGATIVE", "severity":"high",
                 "revision_hint":"改为当前治疗状态未提供。", "source_ids":[]}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertNotIn("尚未开始", after["plan"])
        self.assertGreater(meta["amplitude"]["plan"]["changed_char_estimate"], 0)
        self.assertNotIn("before_text", meta["amplitude"]["plan"])

    def test_conservative_fallback_qualifies_unsupported_exclusion(self):
        before = record("建议完善相关检查。")
        before["diagnostic_discussion"] = "影像未见明确感染征象，排除肺炎。"
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "UNSUPPORTED_NEGATIVE", "severity": "high",
                 "revision_hint": "不得依据弱阴性排除。", "source_ids": ["S1"]}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["diagnostic_discussion"],
                         "影像未见明确感染征象，尚不能排除肺炎。")
        self.assertEqual(meta["unresolved_claim_ids"], [])
        self.assertEqual(after["case_features"], before["case_features"])

    def test_conservative_fallback_downgrades_diagnosis_certainty(self):
        before = record("建议完善相关检查。")
        before["diagnostic_discussion"] = "患者诊断为肺炎。"
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "DIAGNOSIS_CERTAINTY_UPGRADE", "severity": "high",
                 "revision_hint": "恢复待确认性质。", "source_ids": []}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["diagnostic_discussion"],
                         "患者考虑肺炎，尚待进一步检查确认。")
        self.assertEqual(meta["unresolved_claim_ids"], [])

    def test_conservative_fallback_qualifies_locatable_unsafe_recommendation(self):
        before = record("建议立即开展未经评估的高风险操作。")
        target = next(item for item in extract_claims(before) if item["section"] == "plan")
        issue = {**target, "error_type": "UNSAFE_RECOMMENDATION", "severity": "high",
                 "revision_hint": "改为条件性建议。", "source_ids": []}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["plan"],
                         "是否开展未经评估的高风险操作需由医生结合病情、适应证及风险审慎评估。")
        self.assertEqual(meta["unresolved_claim_ids"], [])

    def test_repeated_text_edits_only_the_claim_span_selected(self):
        before = record("需复核。中间内容保持。需复核。")
        plan_claims = [item for item in extract_claims(before) if item["section"] == "plan"]
        target = {**plan_claims[2], "error_type":"PLAN_ALIGNMENT_WEAK", "severity":"medium"}
        revision = {"edits":[{"claim_id":target["claim_id"], "section":"plan",
                               "action":"replace", "replacement":"第二处已局部修订。"}],
                    "review_items_to_add":[]}
        after, _ = apply_revision(before, revision, [target])
        self.assertEqual(after["plan"], "需复核。中间内容保持。第二处已局部修订。")

    def test_patch_engine_handles_status_number_unit_and_correction_claims(self):
        scenarios = [
            ("痰培养已完成。", "PLAN_AS_COMPLETED", "痰培养处于待完善状态。"),
            ("入院体温38.5℃。", "NUMERIC_VALUE_ERROR", "入院体温37.1℃。"),
            ("白细胞计数12.0mg/L。", "UNIT_ERROR", "白细胞计数单位需按原始报告核对。"),
            ("口唇发绀。", "CORRECTION_IGNORED", "复核记录为口唇无发绀。"),
        ]
        for original, code, replacement in scenarios:
            with self.subTest(code=code):
                before = record(original)
                target = next(item for item in extract_claims(before) if item["section"] == "plan")
                issue = {**target, "error_type":code, "severity":"high",
                         "revision_hint":"按原始资料局部修订。"}
                patch = {"edits":[{"claim_id":target["claim_id"], "section":"plan",
                                    "action":"replace", "replacement":replacement}],
                         "review_items_to_add":[]}
                after, meta = apply_revision(before, patch, [issue])
                self.assertEqual(after["plan"], replacement)
                self.assertEqual(after["case_features"], before["case_features"])
                self.assertEqual(meta["unresolved_claim_ids"], [])


if __name__ == "__main__": unittest.main()
