import unittest

from src.evidence_policy import enforce_evidence_policy
from src.protocol import Case, Source
from src.validation import repair_evidence_numeric_support, validate_record


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

    def test_bile_acid_values_are_exact_not_threshold_equivalent(self):
        case = Case("P2", "T0", [Source("P0041", "总胆汁酸 14.2 μmol/L↑"),
                                  Source("P0042", "总胆汁酸 11.8µmol/L↑")])
        correct = base_record([{"statement":"总胆汁酸由14.2μmol/L降至11.8 μmol/L。",
                                "source_ids":["P0041","P0042"], "kind":"fact"}])
        self.assertFalse([x for x in validate_record(correct, case) if x["code"] == "NUMERIC_SUPPORT"])
        wrong = base_record([{"statement":"总胆汁酸高于10μmol/L。",
                              "source_ids":["P0041","P0042"], "kind":"inference"}])
        errors = [x for x in validate_record(wrong, case) if x["code"] == "NUMERIC_SUPPORT"]
        self.assertEqual(len(errors), 1)

    def test_invalid_fact_number_is_repaired_only_with_exact_short_sources(self):
        case = Case("P2", "T0", [Source("P0041", "总胆汁酸 14.2 μmol/L↑"),
                                  Source("P0042", "总胆汁酸 11.8 μmol/L↑")])
        wrong = base_record([{"statement":"总胆汁酸持续≥10μmol/L。",
                              "source_ids":["P0041","P0042"], "kind":"fact"}])
        repaired, audit = repair_evidence_numeric_support(wrong, case)
        self.assertEqual(audit["repaired_count"], 1)
        self.assertIn("14.2", repaired["evidence"][0]["statement"])
        self.assertIn("11.8", repaired["evidence"][0]["statement"])
        self.assertNotIn("10μmol", repaired["evidence"][0]["statement"])
        self.assertFalse([x for x in validate_record(repaired, case) if x["code"] == "NUMERIC_SUPPORT"])

    def test_medical_threshold_inference_is_not_silently_rewritten(self):
        case = Case("P2", "T0", [Source("P0041", "总胆汁酸 14.2 μmol/L↑")])
        wrong = base_record([{"statement":"诊断阈值为10μmol/L。",
                              "source_ids":["P0041"], "kind":"inference"}])
        repaired, audit = repair_evidence_numeric_support(wrong, case)
        self.assertEqual(repaired, wrong)
        self.assertEqual(audit["unresolved_count"], 1)


if __name__ == "__main__":
    unittest.main()
