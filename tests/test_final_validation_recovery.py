import json
import tempfile
import unittest
from pathlib import Path

from src.app import main, parser
from src.claim_extractor import extract_claims
from src.claim_verifier import deterministic_claim_issues
from src.input_adapter import load_input_cases
from src.protocol import Case, Source
from src.validation import (classify_validation_issue, normalize_equivalent_dates,
                            recover_evidence, validate_record)


ROOT = Path(__file__).resolve().parents[1]


def case(*contents):
    return Case("C1", "2026-01-10 14:10", [
        Source(f"S{index:02d}", content) for index, content in enumerate(contents, 1)
    ])


def record(evidence=None, *, features="病例特点。", discussion="现有资料支持主要临床问题，病因尚不能确定。",
           plan="建议结合病情进一步核实，由医生判断。", reviews=None):
    return {"case_features": features, "diagnostic_discussion": discussion, "plan": plan,
            "evidence": evidence or [], "review_items": reviews or []}


def evidence(statement, source_ids, kind="fact"):
    return {"statement": statement, "source_ids": source_ids, "kind": kind}


class FinalValidationRecoveryTests(unittest.TestCase):
    def test_01_iso_and_chinese_full_dates_are_equivalent(self):
        c = case("患者于2026-01-10突发胸痛。")
        value = record([evidence("患者于2026年01月10日突发胸痛。", ["S01"])])
        self.assertNotIn("DATE_SUPPORT", {x["code"] for x in validate_record(value, c)})

    def test_02_slash_and_chinese_dates_are_equivalent(self):
        self.assertEqual(normalize_equivalent_dates("2026/01/10"), "2026-01-10")
        self.assertEqual(normalize_equivalent_dates("2026年1月10日"), "2026-01-10")

    def test_03_unicode_hyphen_is_normalized_before_date_matching(self):
        c = case("患者2026‑01‑10 08:30突发胸痛。")
        value = record([evidence("患者2026年1月10日08:30突发胸痛。", ["S01"])])
        codes = {x["code"] for x in validate_record(value, c)}
        self.assertNotIn("DATE_SUPPORT", codes)
        self.assertNotIn("NUMERIC_SUPPORT", codes)

    def test_04_different_calendar_day_is_rejected(self):
        c = case("检查日期2026-01-10。")
        value = record([evidence("检查日期2026年1月11日。", ["S01"])])
        self.assertIn("DATE_SUPPORT", {x["code"] for x in validate_record(value, c)})

    def test_05_full_date_is_not_split_into_year_month_day_numbers(self):
        c = case("检查日期2026-01-10。")
        value = record([evidence("检查日期2026年1月10日。", ["S01"])])
        issues = validate_record(value, c)
        self.assertFalse([x for x in issues if x["code"] == "NUMERIC_SUPPORT"])

    def test_06_unknown_source_without_direct_support_is_dropped(self):
        c = case("患者胸痛。")
        recovered, audit = recover_evidence(record([evidence("患者发热。", ["BAD"])]), c)
        self.assertEqual(recovered["evidence"], [])
        self.assertTrue(audit["partial_evidence_recovery"])
        self.assertEqual(audit["source_reassignment"], [])

    def test_07_unique_direct_source_can_be_reassigned(self):
        c = case("患者于2026‑01‑10突发胸痛。", "患者否认咯血。")
        recovered, audit = recover_evidence(
            record([evidence("患者于2026年1月10日突发胸痛。", ["BAD"])]), c)
        self.assertEqual(recovered["evidence"][0]["source_ids"], ["S01"])
        self.assertEqual(len(audit["source_reassignment"]), 1)

    def test_08_ambiguous_direct_sources_are_not_guessed(self):
        c = case("患者胸痛。", "患者胸痛。")
        recovered, audit = recover_evidence(record([evidence("患者胸痛。", ["BAD"])]), c)
        self.assertEqual(recovered["evidence"], [])
        self.assertEqual(audit["source_reassignment"], [])

    def test_09_dropping_evidence_preserves_all_prose(self):
        before = record([evidence("诊断阈值为10μmol/L。", ["S01"], "inference")],
                        features="完整病例特点。", discussion="完整拟诊讨论，病因待查。",
                        plan="完整诊疗计划，具体由医生判断。")
        recovered, _ = recover_evidence(before, case("总胆汁酸14.2μmol/L。"))
        for field in ("case_features", "diagnostic_discussion", "plan"):
            self.assertEqual(recovered[field], before[field])
        self.assertEqual(recovered["evidence"], [])

    def test_10_unsupported_body_number_is_hard_even_after_evidence_drop(self):
        c = case("血红蛋白82g/L。")
        value = record([evidence("血红蛋白128g/L。", ["S01"])], features="血红蛋白128g/L。")
        recovered, _ = recover_evidence(value, c)
        issues = validate_record(recovered, c)
        body = next(x for x in issues if x["code"] == "UNSUPPORTED_BODY_NUMERIC")
        self.assertEqual(classify_validation_issue(body), "hard")

    def test_11_supported_body_number_remains_valid(self):
        c = case("血红蛋白82 g/L。")
        value = record(features="血红蛋白82g/L。")
        self.assertNotIn("UNSUPPORTED_BODY_NUMERIC", {x["code"] for x in validate_record(value, c)})

    def test_12_wrong_unit_in_body_remains_hard(self):
        c = case("肌酐185μmol/L。")
        value = record(features="肌酐185mg/L。")
        self.assertIn("UNSUPPORTED_BODY_NUMERIC", {x["code"] for x in validate_record(value, c)})

    def test_13_reasonable_uncertainty_is_not_blocked(self):
        c = case("患者发热伴咳嗽。")
        value = record(discussion="发热伴咳嗽原因待查，现有资料尚不能确定病因。")
        self.assertFalse([x for x in validate_record(value, c) if x["level"] == "error"])

    def test_14_plan_as_completed_evidence_is_dropped(self):
        c = case("拟行胸部CT，结果待回报。")
        recovered, audit = recover_evidence(
            record([evidence("胸部CT已完成。", ["S01"])]), c)
        self.assertEqual(recovered["evidence"], [])
        self.assertIn("PLAN_AS_DONE", audit["evidence_dropped"][0]["codes"])

    def test_15_recovery_never_adds_generic_review_item(self):
        recovered, _ = recover_evidence(
            record([evidence("无支持陈述。", ["BAD"])]), case("患者胸痛。"))
        self.assertEqual(recovered["review_items"], [])

    def test_15b_technical_review_item_is_rejected_by_final_validation(self):
        value = record(reviews=["source_id S999 不存在，模型响应解析错误。"])
        self.assertIn("TECHNICAL_REVIEW_ITEM",
                      {item["code"] for item in validate_record(value, case("患者胸痛。"))})

    def test_16_internal_id_leak_is_still_high_risk(self):
        c = case("患者胸痛。")
        value = record(discussion="F001提示患者胸痛。")
        issues = deterministic_claim_issues(c, extract_claims(value), {"F001"})
        self.assertTrue(any(x["error_type"] == "INTERNAL_AUDIT_LEAK"
                            and x["severity"] == "high" for x in issues))

    def test_17_generic_template_and_broken_bracket_are_still_blocked(self):
        value = record(discussion="原因待查）。该项陈述依据不足，需医生结合原始资料确认。")
        codes = {x["code"] for x in validate_record(value, case("患者胸痛。"))}
        self.assertIn("PROSE_BRACKET_MISMATCH", codes)
        self.assertIn("GENERIC_REVISION_TEMPLATE", codes)

    def test_18_practice05_unicode_date_source_passes_offline(self):
        path = ROOT / "information/临床智录_练习病历数据集JSON/05_心内科CCU_主动脉壁内血肿.json"
        loaded = load_input_cases(path, "practice")
        c = loaded.cases[0]
        source = next(item for item in c.sources if item.field_path == "入院记录.现病史")
        value = record([evidence("患者2026年1月10日08:30突发胸痛。", [source.source_id])])
        codes = {x["code"] for x in validate_record(value, c)}
        self.assertNotIn("DATE_SUPPORT", codes)
        self.assertNotIn("NUMERIC_SUPPORT", codes)

    def test_19_two_argument_cli_contract_is_unchanged(self):
        args = parser().parse_args(["--input", "input.json", "--output", "output.json"])
        self.assertEqual(args.input, "input.json")
        self.assertEqual(args.output, "output.json")

    def test_20_recovery_adds_no_model_call_in_multicase_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_path, log_path = root/"input.json", root/"output.json", root/"log.json"
            responses = root/"responses.json"
            input_path.write_text(json.dumps([
                {"case_id":"C1","record_cutoff":"T0","sources":[{"source_id":"S01","content":"患者胸痛。"}]},
                {"case_id":"C2","record_cutoff":"T0","sources":[{"source_id":"S01","content":"患者发热。"}]},
            ], ensure_ascii=False), encoding="utf-8")
            first = record([evidence("诊断阈值10μmol/L。", ["S01"], "inference")])
            second = record([evidence("患者发热。", ["S01"])])
            responses.write_text(json.dumps([
                {"content": json.dumps(first, ensure_ascii=False)},
                {"content": json.dumps(second, ensure_ascii=False)},
            ], ensure_ascii=False), encoding="utf-8")
            code = main(["--input", str(input_path), "--output", str(output_path),
                         "--log", str(log_path), "--mode", "mock",
                         "--mock-responses", str(responses)])
            values = json.loads(output_path.read_text())
            log = json.loads(log_path.read_text())
            self.assertEqual(code, 0)
            self.assertEqual([item["status"] for item in values], ["success", "success"])
            self.assertEqual(log["total_calls"], 2)
            self.assertEqual(log["cases"][0]["final_validation_recovery"]["additional_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
