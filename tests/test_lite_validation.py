import json
import tempfile
import unittest
from pathlib import Path

from src.app import _enforce_profile_safety
from src.competition_profile import LOCAL_DEVELOPMENT_PROFILE, PROFILES
from src.errors import CaseError
from src.model_client import MockClient
from src.pipeline import run_case
from src.protocol import Case, Source
from src.validation import (
    classify_claim_issue_for_final,
    classify_validation_issue,
    validate_record,
)


def record(plan="建议结合病情动态评估。"):
    return {
        "case_features": "患者因胸痛入院。",
        "diagnostic_discussion": "现有资料支持主要临床问题。",
        "plan": plan,
        "evidence": [],
        "review_items": [],
    }


class LiteValidationTests(unittest.TestCase):
    def test_treatment_target_is_soft_in_lite_but_strict_in_debug(self):
        case = Case("C", "T", [Source("S1", "患者胸痛。")])
        value = record("拟将收缩压控制在<120mmHg。")
        strict = next(item for item in validate_record(value, case, "strict")
                      if item["code"] == "UNVERIFIED_PROPOSED_TARGET")
        lite = next(item for item in validate_record(value, case, "lite")
                    if item["code"] == "UNVERIFIED_PROPOSED_TARGET")
        self.assertEqual((strict["level"], classify_validation_issue(strict)),
                         ("error", "repairable"))
        self.assertEqual((lite["level"], classify_validation_issue(lite)),
                         ("warning", "soft"))

    def test_patient_measurement_remains_hard_in_lite(self):
        case = Case("C", "T", [Source("S1", "患者胸痛。")])
        value = record(); value["case_features"] = "患者入院血压180/120mmHg。"
        issue = next(item for item in validate_record(value, case, "lite")
                     if item["code"] == "UNSUPPORTED_BODY_NUMERIC")
        self.assertEqual((issue["level"], classify_validation_issue(issue)), ("error", "hard"))

    def test_qualified_negative_inference_is_soft_but_definitive_exclusion_is_hard(self):
        base = {"severity": "high", "error_type": "UNSUPPORTED_NEGATIVE"}
        qualified = classify_claim_issue_for_final(
            {**base, "statement": "现有检查不支持急性心肌梗死，但仍需继续评估。"}, "lite")
        planned = classify_claim_issue_for_final(
            {**base, "statement": "需进一步排除急性心肌梗死。"}, "lite")
        definitive = classify_claim_issue_for_final(
            {**base, "statement": "现已排除急性心肌梗死。"}, "lite")
        self.assertEqual(qualified["risk_level"], "soft")
        self.assertEqual(planned["risk_level"], "soft")
        self.assertEqual(definitive["risk_level"], "hard")

    def test_safety_floor_codes_and_missing_context_remain_hard(self):
        for issue in (
            {"severity": "high", "error_type": "NUMERIC_VALUE_ERROR", "statement": "血压180mmHg。"},
            {"severity": "high", "error_type": "PLAN_AS_COMPLETED", "statement": "CTA已完成。"},
            {"severity": "high", "error_type": "UNSUPPORTED_NEGATIVE", "statement": None},
        ):
            self.assertEqual(classify_claim_issue_for_final(issue, "lite")["risk_level"], "hard")

    def test_formal_profiles_use_lite_while_local_development_stays_strict(self):
        self.assertEqual(PROFILES["baseline_safe"].pipeline, "baseline")
        self.assertTrue(all(profile.final_validation_mode == "lite"
                            for profile in PROFILES.values()))
        self.assertEqual(LOCAL_DEVELOPMENT_PROFILE.final_validation_mode, "strict")

    def test_profile_gate_uses_output_guarantee_result_only(self):
        profile = PROFILES["clinical_d_graph_agent"]
        _enforce_profile_safety(profile, {"clinical": {
            "unresolved_risk_count": 4,
            "unresolved_hard_risk_count": 0,
            "unresolved_soft_risk_count": 4,
            "output_guarantee": {"safe_record_available": True},
        }})
        with self.assertRaises(CaseError):
            _enforce_profile_safety(profile, {"clinical": {
                "unresolved_risk_count": 4,
                "unresolved_hard_risk_count": 1,
                "unresolved_soft_risk_count": 3,
                "output_guarantee": {"safe_record_available": False},
            }})

    def test_lite_weak_negative_is_safely_rewritten_without_review_item_or_extra_call(self):
        case = Case("C", "T", [Source("S1", "心电图未见明确异常，无明显ST-T动态改变。")])
        planner = {"problems": [{
            "problem_id": "PR001", "name": "胸痛", "priority": "high",
            "origin": "documented_problem", "supporting_facts": ["F001"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["评估胸痛原因"], "retrieval_queries": ["胸痛"],
        }]}
        writer = {
            "case_features": "患者因胸痛就诊。",
            "diagnostic_discussion": "现有心电图表现不支持急性心肌梗死，但仍需继续评估。",
            "plan": "建议结合病情动态评估。",
            "evidence": [{"statement": "心电图未见明确异常。",
                          "source_ids": ["S1"], "kind": "fact"}],
            "review_items": [],
        }
        verifier = {"checked_claim_ids": ["C001", "C002", "C003", "C004"], "issues": []}
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "responses.json"
            script.write_text(json.dumps([
                {"content": json.dumps(planner, ensure_ascii=False)},
                {"content": json.dumps(writer, ensure_ascii=False)},
                {"content": json.dumps(verifier, ensure_ascii=False)},
            ], ensure_ascii=False), encoding="utf-8")
            client = MockClient(script)
            result, meta = run_case(
                case, client, "clinical_pipeline", "text", 3,
                clinical_verify=True, clinical_revise=False,
                final_validation_mode="lite",
            )
        self.assertEqual(client.total_calls, 3)
        self.assertEqual(result["review_items"], [])
        self.assertIn("据此尚不能排除急性心肌梗死", result["diagnostic_discussion"])
        self.assertTrue(meta["clinical"]["partial_sanitization_applied"])
        self.assertEqual(meta["clinical"]["unresolved_soft_risk_count"], 0)
        self.assertEqual(meta["clinical"]["unresolved_hard_risk_count"], 0)
        self.assertEqual(meta["clinical"]["verification_status"],
                         "completed_deterministic_fallback")

    def test_practice05_history_cannot_be_downgraded_from_hashes_only(self):
        path = (Path(__file__).resolve().parents[1] / "information/results/05_final_20261125_090337"
                / "run_log.json")
        if not path.exists():
            self.skipTest("PRACTICE05 historical run log is not packaged in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        clinical = payload["cases"][0]["clinical"]
        audit = clinical["final_quality_gate"]["post_revision_high_risk_audit"]
        self.assertEqual([item["claim_id"] for item in audit],
                         ["C014", "C017", "C020", "C031"])
        self.assertTrue(all(item["code"] == "UNSUPPORTED_NEGATIVE" for item in audit))
        self.assertTrue(all(item["source_ids"] == ["P0014"] for item in audit))
        for item in audit:
            classification = classify_claim_issue_for_final(item, "lite")
            self.assertEqual(classification,
                             {"risk_level": "hard", "reason": "missing_statement_context"})


if __name__ == "__main__":
    unittest.main()
