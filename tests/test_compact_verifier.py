import json
import unittest

from src.claim_verifier import run_claim_verifier, validate_verification
from src.errors import CaseError
from src.model_client import Budget, VERIFIER_SCHEMA
from src.pipeline import _stage_failure_detail
from src.protocol import Case, Source


class CaptureClient:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.schema = None
        self.call_events = []

    def call(self, messages, response_format, budget, *, stage="model", schema=None):
        budget.take()
        self.messages = messages
        self.schema = schema
        self.call_events.append({"stage":stage, "json_parse_success":None})
        return json.dumps(self.response, ensure_ascii=False)

    def record_parse_result(self, stage, success, error_type=None, position=None):
        self.call_events[-1]["json_parse_success"] = success


def make_claims(count=28):
    return [{"claim_id":f"C{index:03d}", "section":"case_features",
             "statement":f"第{index}项虚构开发事实。", "claim_type":"patient_fact",
             "source_ids":["S1"], "span_start":None, "span_end":None,
             "item_index":None}
            for index in range(1, count + 1)]


class CompactVerifierTests(unittest.TestCase):
    def test_28_claims_are_compact_and_fully_covered(self):
        claims = make_claims()
        checked = [item["claim_id"] for item in claims]
        client = CaptureClient({"checked_claim_ids":checked, "issues":[]})
        case = Case("PRACTICE-MOCK", "T0", [Source("S1", "虚构开发资料。")])
        result = run_claim_verifier(case, claims, {"unused":"draft"}, {"unused":"plan"},
                                    None, 3, client, "json_schema", Budget(1))
        self.assertEqual(result["checked_claim_ids"], checked)
        self.assertEqual(result["issues"], [])
        self.assertIs(client.schema, VERIFIER_SCHEMA)
        payload = json.loads(client.messages[1]["content"])
        self.assertNotIn("draft_record", payload)
        self.assertNotIn("clinical_plan", payload)
        self.assertEqual(len(payload["claims"]), 28)
        self.assertLess(len(json.dumps(client.response, ensure_ascii=False)), 500)
        self.assertTrue(client.call_events[-1]["json_parse_success"])

    def test_wrong_source_is_rejected_even_with_full_coverage(self):
        claims = make_claims(1)
        response = {"checked_claim_ids":["C001"], "issues":[{
            "claim_id":"C001", "code":"SOURCE_MISMATCH", "severity":"high",
            "hint":"来源编号不支持该陈述。", "source_ids":["S999"]}]}
        with self.assertRaises(CaseError) as raised:
            validate_verification(response, {"C001":claims[0]}, {"S1"})
        self.assertEqual(raised.exception.code, "INVALID_CLAIM_VERIFICATION")

    def test_hard_fact_code_cannot_be_downgraded(self):
        claims = make_claims(1)
        response = {"checked_claim_ids":["C001"], "issues":[{
            "claim_id":"C001", "code":"NUMERIC_VALUE_ERROR", "severity":"medium",
            "hint":"体温数值错误。", "source_ids":["S1"]}]}
        with self.assertRaises(CaseError):
            validate_verification(response, {"C001":claims[0]}, {"S1"})

    def test_length_finish_reason_is_logged_as_suspected_truncation(self):
        client = type("Client", (), {"call_events":[{
            "stage":"verifier", "finish_reason":"length", "max_tokens":2048,
            "token_usage":{"completion_tokens":2048}}]})()
        detail = _stage_failure_detail(client, "verifier")
        self.assertTrue(detail["output_truncation_suspected"])
        self.assertEqual(detail["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
