import unittest

from src.evidence_policy import enforce_evidence_policy
from src.protocol import Case, Source
from src.validation import validate_record


def base_record(evidence):
    return {"case_features":"病例特点。", "diagnostic_discussion":"拟诊讨论。",
            "plan":"诊疗计划。", "evidence":evidence, "review_items":[]}


class EvidencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.case = Case("C", "T0", [Source("S1", "2026年5月患者体温38.5℃，否认胸痛。"),
                                      Source("S2", "胸部CT计划完善，结果未提供。")])

    def test_more_than_five_items_are_selected_not_case_failed(self):
        items = [{"statement":f"证据条目{i}。", "source_ids":["S1" if i % 2 else "S2"], "kind":"fact"}
                 for i in range(1, 8)]
        result, report = enforce_evidence_policy(base_record(items), self.case)
        self.assertEqual(len(result["evidence"]), 5)
        self.assertEqual(report["limit_removed"], 2)

    def test_fewer_than_three_valid_items_are_not_padded(self):
        items = [{"statement":"患者体温38.5℃。", "source_ids":["S1"], "kind":"fact"},
                 {"statement":"胸部CT结果未提供。", "source_ids":["S2"], "kind":"fact"}]
        result, report = enforce_evidence_policy(base_record(items), self.case)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertTrue(report["below_recommended_minimum"])

    def test_invalid_source_is_removed_and_disclosed(self):
        items = [{"statement":"患者体温38.5℃。", "source_ids":["S999"], "kind":"fact"}]
        result, report = enforce_evidence_policy(base_record(items), self.case)
        self.assertEqual(result["evidence"], [])
        self.assertEqual(report["invalid_removed"], 1)
        self.assertIn("未能可靠对应原始资料", result["review_items"][0])

    def test_equivalent_year_month_notation_is_accepted(self):
        record = base_record([{"statement":"患者于2026.05就诊。", "source_ids":["S1"], "kind":"fact"}])
        errors = [issue for issue in validate_record(record, self.case) if issue["code"] == "NUMERIC_SUPPORT"]
        self.assertEqual(errors, [])

    def test_true_date_conflict_is_still_reported(self):
        record = base_record([{"statement":"患者于2025.05就诊。", "source_ids":["S1"], "kind":"fact"}])
        errors = [issue for issue in validate_record(record, self.case) if issue["code"] == "NUMERIC_SUPPORT"]
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
