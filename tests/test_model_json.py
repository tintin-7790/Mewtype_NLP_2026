import unittest

from src.errors import CaseError
from src.model_client import parse_model_json


class ModelJsonParsingTests(unittest.TestCase):
    def test_single_json_is_accepted(self):
        self.assertEqual(parse_model_json('{"ok":true}', stage="planner"), {"ok": True})

    def test_trailing_explanation_is_rejected_with_stage_and_position(self):
        with self.assertRaises(CaseError) as caught:
            parse_model_json('{"ok":true}\n以上为结果。', stage="verifier")
        self.assertEqual(caught.exception.code, "MODEL_INVALID_JSON")
        self.assertIn("verifier", caught.exception.message)
        self.assertIn("JSON_TRAILING_TEXT", caught.exception.message)
        self.assertIn("char", caught.exception.message)

    def test_two_json_values_are_not_silently_truncated(self):
        with self.assertRaises(CaseError) as caught:
            parse_model_json('{"first":1}\n{"second":2}', stage="verifier")
        self.assertIn("MULTIPLE_JSON_VALUES", caught.exception.message)

    def test_one_whole_response_json_fence_is_normalized(self):
        self.assertEqual(parse_model_json('```json\n{"ok":true}\n```', stage="writer"), {"ok": True})
        with self.assertRaises(CaseError) as caught:
            parse_model_json('```json\n{"ok":true}\n```\n额外说明', stage="writer")
        self.assertIn("围栏", caught.exception.message)

    def test_incomplete_json_reports_specific_failure(self):
        with self.assertRaises(CaseError) as caught:
            parse_model_json('{"ok":', stage="planner")
        self.assertIn("INCOMPLETE_JSON", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
