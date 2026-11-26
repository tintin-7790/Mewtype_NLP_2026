import json
import tempfile
import unittest
from pathlib import Path

from src.app import main
from src.protocol import Case, Source
from src.validation import (
    CLINICAL_THRESHOLD,
    PATIENT_FACT,
    PLANNED_TEST,
    PROPOSED_TARGET,
    UNKNOWN,
    MEASUREMENT_WITH_UNIT,
    classify_numeric_semantic_role,
    classify_validation_issue,
    recover_plan_numeric_targets,
    validate_record,
)


def case(*contents):
    return Case("C1", "2026-01-10 14:10", [
        Source(f"S{index:02d}", content) for index, content in enumerate(contents, 1)
    ])


def record(*, features="患者因胸痛入院。", discussion="现有资料支持主要临床问题。",
           plan="建议动态评估，由医生判断。", evidence=None, reviews=None):
    return {
        "case_features": features,
        "diagnostic_discussion": discussion,
        "plan": plan,
        "evidence": evidence or [],
        "review_items": reviews or [],
    }


def role(text, section="plan"):
    match = MEASUREMENT_WITH_UNIT.search(text)
    assert match is not None
    return classify_numeric_semantic_role(text, match.start(), match.end(), section)


class PlanNumericSemanticTests(unittest.TestCase):
    def test_patient_blood_pressure_without_source_is_hard(self):
        value = record(features="患者入院血压180/120mmHg。")
        issue = next(item for item in validate_record(value, case("患者胸痛。"))
                     if item["code"] == "UNSUPPORTED_BODY_NUMERIC")
        self.assertEqual(issue["semantic_role"], PATIENT_FACT)
        self.assertEqual(classify_validation_issue(issue), "hard")

    def test_proposed_target_is_not_mislabelled_as_patient_fact(self):
        value = record(plan="继续监测，拟将收缩压控制在<120mmHg，并观察症状。")
        issue = next(item for item in validate_record(value, case("入院血压162/105mmHg。"))
                     if item["code"] == "UNVERIFIED_PROPOSED_TARGET")
        self.assertEqual(issue["semantic_role"], PROPOSED_TARGET)
        self.assertEqual(classify_validation_issue(issue), "repairable")
        self.assertNotIn("UNSUPPORTED_BODY_NUMERIC",
                         {item["code"] for item in validate_record(value, case("患者胸痛。"))})

    def test_same_patient_measurement_does_not_prove_treatment_target(self):
        value = record(plan="拟将收缩压控制在<120mmHg。")
        issues = validate_record(value, case("患者既往测得收缩压<120mmHg。"))
        self.assertIn("UNVERIFIED_PROPOSED_TARGET", {item["code"] for item in issues})

    def test_explicit_source_plan_can_support_same_target(self):
        value = record(plan="拟将收缩压控制在<120mmHg。")
        issues = validate_record(value, case("原治疗计划拟将收缩压控制在<120mmHg。"))
        self.assertNotIn("UNVERIFIED_PROPOSED_TARGET", {item["code"] for item in issues})

    def test_unverified_or_inappropriate_target_is_not_passed_through(self):
        value = record(plan="严密监测，拟将收缩压控制在60mmHg，并评估灌注状态。")
        recovered, audit = recover_plan_numeric_targets(value, case("患者胸痛。"))
        self.assertTrue(audit["applied"])
        self.assertNotIn("60mmHg", recovered["plan"])
        self.assertIn("收缩压控制目标需由医生", recovered["plan"])

    def test_plan_as_completed_remains_blocked_in_evidence(self):
        value = record(evidence=[{
            "statement": "冠脉CTA已完成。", "source_ids": ["S01"], "kind": "fact",
        }])
        codes = {item["code"] for item in validate_record(value, case("拟行冠脉CTA，结果待回报。"))}
        self.assertIn("PLAN_AS_DONE", codes)

    def test_wrong_lab_unit_remains_blocked(self):
        value = record(features="肌酐185mg/L。")
        issues = validate_record(value, case("肌酐185μmol/L。"))
        self.assertTrue(any(item["code"] == "UNSUPPORTED_BODY_NUMERIC"
                            and item["semantic_role"] == PATIENT_FACT for item in issues))

    def test_target_rewrite_preserves_complete_plan_and_structure(self):
        original = "继续镇痛并动态监测；拟将收缩压控制在<120mmHg，同时观察胸痛及灌注变化。"
        recovered, _ = recover_plan_numeric_targets(
            record(plan=original), case("患者胸痛，入院血压162/105mmHg。"),
        )
        self.assertTrue(recovered["plan"].startswith("继续镇痛并动态监测；"))
        self.assertTrue(recovered["plan"].endswith("同时观察胸痛及灌注变化。"))
        self.assertFalse([item for item in validate_record(recovered, case("患者胸痛，入院血压162/105mmHg。"))
                          if item["level"] == "error"])

    def test_evidence_numeric_support_is_still_strict(self):
        value = record(evidence=[{
            "statement": "入院血压120mmHg。", "source_ids": ["S01"], "kind": "fact",
        }])
        self.assertIn("NUMERIC_SUPPORT",
                      {item["code"] for item in validate_record(value, case("入院血压110mmHg。"))})

    def test_recovery_adds_no_broken_text_internal_id_or_generic_template(self):
        recovered, audit = recover_plan_numeric_targets(
            record(plan="建议将血压控制在<120mmHg。"), case("患者胸痛。"),
        )
        self.assertEqual(audit["additional_model_calls"], 0)
        self.assertNotRegex(recovered["plan"], r"\b(?:F|C|PR)\d{3}\b")
        self.assertNotIn("该项陈述依据不足", recovered["plan"])
        self.assertNotIn("PROSE_BRACKET_MISMATCH",
                         {item["code"] for item in validate_record(recovered, case("患者胸痛。"))})

    def test_output_sections_remain_nonempty_and_schema_complete(self):
        recovered, _ = recover_plan_numeric_targets(
            record(plan="建议将血压控制在<120mmHg。"), case("患者胸痛。"),
        )
        self.assertEqual(set(recovered), {
            "case_features", "diagnostic_discussion", "plan", "evidence", "review_items",
        })
        self.assertTrue(all(recovered[field].strip()
                            for field in ("case_features", "diagnostic_discussion", "plan")))

    def test_threshold_planned_test_and_unknown_have_distinct_roles(self):
        self.assertEqual(role("若收缩压<90mmHg，应重新评估。"), CLINICAL_THRESHOLD)
        self.assertEqual(role("计划监测血压<120mmHg并记录趋势。"), PLANNED_TEST)
        self.assertEqual(role("方案参数为120mmHg。"), UNKNOWN)
        self.assertEqual(role("当前实测血压为120mmHg。"), PATIENT_FACT)

    def test_unknown_numeric_role_is_conservatively_blocked(self):
        issues = validate_record(record(plan="方案参数为120mmHg。"), case("患者胸痛。"))
        issue = next(item for item in issues if item["code"] == "UNRESOLVED_BODY_NUMERIC_ROLE")
        self.assertEqual(classify_validation_issue(issue), "hard")

    def test_pipeline_recovers_target_without_an_extra_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            log_path = root / "run_log.json"
            responses = root / "responses.json"
            input_path.write_text(json.dumps([{
                "case_id": "C1", "record_cutoff": "T0",
                "sources": [{"source_id": "S01", "content": "患者胸痛，入院血压162/105mmHg。"}],
            }], ensure_ascii=False), encoding="utf-8")
            responses.write_text(json.dumps([{"content": json.dumps(
                record(plan="拟将收缩压控制在<120mmHg，并动态观察。"), ensure_ascii=False,
            )}], ensure_ascii=False), encoding="utf-8")
            code = main([
                "--input", str(input_path), "--output", str(output_path),
                "--log", str(log_path), "--mode", "mock", "--mock-responses", str(responses),
            ])
            output = json.loads(output_path.read_text(encoding="utf-8"))[0]
            log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(output["status"], "success")
            self.assertNotIn("<120mmHg", output["medical_record"]["plan"])
            self.assertEqual(log["total_calls"], 1)
            recovery = log["cases"][0]["final_validation_recovery"]["plan_numeric_recovery"]
            self.assertTrue(recovery["applied"])


if __name__ == "__main__":
    unittest.main()
