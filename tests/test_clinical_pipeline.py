import json
import tempfile
import unittest
from pathlib import Path

from src.app import main
from src.errors import CaseError
from src.model_client import MockClient
from src.pipeline import run_case
from src.protocol import Case, Source


def planner():
    return {"problems": [{"problem_id":"PR001","name":"当前治疗状态","priority":"high",
        "origin":"mixed","supporting_facts":["F001"],"contradicting_facts":[],
        "differential_diagnoses":[],"plan_goals":["明确当前治疗状态"],
        "retrieval_queries":["当前 抗感染治疗 状态"]}]}


def draft(plan="当前治疗情况未提供，建议由医生结合病情决定后续治疗。"):
    return {"case_features":"既往接受抗感染治疗，当前治疗情况未提供。",
            "diagnostic_discussion":"当前资料不足以判断现行治疗状态。","plan":plan,
            "evidence":[{"statement":"既往接受抗感染治疗。","source_ids":["S1"],"kind":"fact"}],
            "review_items":["当前治疗情况需确认。"]}


def claims(with_issue=False):
    values = [
        {"claim_id":"C001","verdict":"SUPPORTED","rationale":"原文支持","source_ids":["S1"]},
        {"claim_id":"C002","verdict":"GROUNDED","rationale":"原文支持不确定表述","source_ids":["S1"]},
        {"claim_id":"C003","verdict":"NOT_ADDRESSED",
         "rationale":"当前治疗状态未提供，后续建议需保守","source_ids":["S1"]},
        {"claim_id":"C004","verdict":"SUPPORTED","rationale":"原文支持","source_ids":["S1"]},
        {"claim_id":"C005","verdict":"SUPPORTED","rationale":"原文确实未提供","source_ids":["S1"]},
    ]
    issues = []
    if with_issue:
        issues = [{"claim_id":"C003","section":"plan","statement":"当前尚未开始抗感染治疗。",
                   "error_type":"NOT_ADDRESSED","severity":"high",
                   "revision_hint":"改为当前治疗情况未提供。","source_ids":[]}]
    return {"claims": values, "issues": issues}


class ClinicalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.input = self.root/"input.json"
        self.input.write_text(json.dumps([{"case_id":"C1","record_cutoff":"T0","sources":[
            {"source_id":"S1","content":"患者既往接受抗感染治疗，当前治疗情况未提供。"}]}],
            ensure_ascii=False), encoding="utf-8")
    def tearDown(self): self.temp.cleanup()

    def run_pipeline(self, responses, extra):
        script=self.root/"responses.json"; output=self.root/"output.json"; log=self.root/"log.json"
        script.write_text(json.dumps([{"content":json.dumps(value,ensure_ascii=False)} for value in responses],
                                     ensure_ascii=False),encoding="utf-8")
        code=main(["--input",str(self.input),"--output",str(output),"--log",str(log),"--mode","mock",
                   "--mock-responses",str(script),"--pipeline","clinical_pipeline",*extra])
        return code,json.loads(output.read_text()),json.loads(log.read_text())

    def test_planner_writer_runs_without_embedding_or_verifier(self):
        code,output,log=self.run_pipeline([planner(),draft()],
            ["--clinical-verification","off","--clinical-revision","off"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],2)
        self.assertEqual(log["cases"][0]["clinical"]["evidence_bundle_count"],0)
        self.assertEqual(log["cases"][0]["clinical"]["prompt_profile"],"contrastive")
        self.assertIn("planning_policy",log["cases"][0]["clinical"]["prompt_audit"]["planner"])

    def test_three_call_clinical_pipeline_without_revision(self):
        code,output,log=self.run_pipeline([planner(),draft(),claims()],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertFalse(log["cases"][0]["clinical"]["revision_applied"])
        self.assertEqual([event["stage"] for event in log["cases"][0]["stage_calls"]],
                         ["planner", "writer", "verifier"])
        self.assertTrue(all(event["json_parse_success"] for event in log["cases"][0]["stage_calls"]))

    def test_stage_json_schema_selection_is_logged_in_mock_regression(self):
        code,output,log=self.run_pipeline([planner(),draft(),claims()],
            ["--clinical-verification","on","--clinical-revision","on",
             "--clinical-response-format","json_schema"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        events=log["cases"][0]["stage_calls"]
        self.assertEqual([event["response_format"] for event in events],
                         ["json_schema", "json_schema", "json_schema"])
        for event in events:
            self.assertGreater(event["input_text_chars"], 0)
            self.assertGreater(event["system_prompt_chars"], 0)
            self.assertGreater(event["user_payload_chars"], 0)

    def test_legacy_prompt_profile_remains_available(self):
        code,output,log=self.run_pipeline([planner(),draft()],
            ["--clinical-verification","off","--clinical-revision","off",
             "--clinical-prompt-profile","legacy"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        clinical=log["cases"][0]["clinical"]
        self.assertEqual(clinical["prompt_profile"],"legacy")
        self.assertEqual(clinical["algorithm_mode"],"legacy")
        self.assertEqual([event["stage"] for event in log["cases"][0]["stage_calls"]],
                         ["planner","writer"])

    def test_problem_retrieval_is_injected_and_logged(self):
        config=self.root/"retrieval.json"
        config.write_text(json.dumps({"provider":"mock","dimension":32,"top_k":1,
                                      "min_similarity":-1,"max_chunk_chars":240}),encoding="utf-8")
        code,output,log=self.run_pipeline([planner(),draft()],
            ["--clinical-verification","off","--clinical-revision","off",
             "--retrieval-config",str(config)])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        case_log=log["cases"][0]
        self.assertEqual(case_log["clinical"]["evidence_bundle_count"],1)
        self.assertEqual(case_log["retrieval"]["embedding_calls"],2)

    def test_fourth_call_revises_only_plan(self):
        bad=draft("当前尚未开始抗感染治疗。")
        revised=draft()["evidence"]
        revision={"section_updates":{"plan":"当前治疗情况未提供，后续治疗需由医生结合病情决定。"},
                  "evidence":revised,"review_items":["当前治疗情况需确认。"]}
        code,output,log=self.run_pipeline([planner(),bad,claims(True),revision],
            ["--clinical-verification","on","--clinical-revision","on"])
        result=output[0]["medical_record"]
        self.assertEqual(code,0); self.assertEqual(log["cases"][0]["calls"],4)
        self.assertEqual(result["case_features"],bad["case_features"])
        self.assertEqual(result["diagnostic_discussion"],bad["diagnostic_discussion"])
        self.assertNotEqual(result["plan"],bad["plan"])

    def test_high_risk_without_model_revision_uses_disclosed_fallback(self):
        bad=draft("当前尚未开始抗感染治疗。")
        code,output,log=self.run_pipeline([planner(),bad,claims(True)],
            ["--clinical-verification","on","--clinical-revision","off"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertNotIn("尚未开始抗感染治疗", output[0]["medical_record"]["plan"])
        self.assertEqual(log["cases"][0]["clinical"]["verification_status"],"fallback_disclosed")

    def test_verifier_extra_data_preserves_legal_writer_draft(self):
        script=self.root/"malformed.json"; output=self.root/"malformed-output.json"; log_path=self.root/"malformed-log.json"
        script.write_text(json.dumps([
            {"content":json.dumps(planner(),ensure_ascii=False)},
            {"content":json.dumps(draft(),ensure_ascii=False)},
            {"content":'{"claims":[],"issues":[]}\n额外说明'},
        ], ensure_ascii=False), encoding="utf-8")
        code=main(["--input",str(self.input),"--output",str(output),"--log",str(log_path),
                   "--mode","mock","--mock-responses",str(script),"--pipeline","clinical_pipeline"])
        saved_output=json.loads(output.read_text()); saved_log=json.loads(log_path.read_text())
        self.assertEqual(code,0); self.assertEqual(saved_output[0]["status"],"success")
        self.assertEqual(saved_output[0]["medical_record"]["case_features"], draft()["case_features"])
        self.assertTrue(any("自动核验未完成" in item
                            for item in saved_output[0]["medical_record"]["review_items"]))
        clinical=saved_log["cases"][0]["clinical"]
        self.assertEqual(clinical["verification_status"],"incomplete_writer_preserved")
        self.assertEqual(clinical["verification_error"]["code"],"MODEL_INVALID_JSON")
        event=saved_log["cases"][0]["stage_calls"][2]
        self.assertEqual(event["stage"],"verifier"); self.assertFalse(event["json_parse_success"])
        self.assertEqual(event["parse_error_type"],"JSON_TRAILING_TEXT")
        self.assertIsInstance(event["parse_error_position"],int)

    def test_invalid_verifier_verdict_preserves_writer_draft(self):
        verdict=claims(); verdict["claims"][0]["verdict"]="GROUNDED"
        code,output,log=self.run_pipeline([planner(),draft(),verdict],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(output[0]["medical_record"]["case_features"],draft()["case_features"])
        clinical=log["cases"][0]["clinical"]
        self.assertEqual(clinical["verification_status"],"incomplete_writer_preserved")
        self.assertEqual(clinical["verification_error"]["code"],"INVALID_CLAIM_VERIFICATION")

    def test_invalid_reviser_response_uses_conservative_fallback(self):
        bad=draft("当前尚未开始抗感染治疗。")
        code,output,log=self.run_pipeline([planner(),bad,claims(True),{"unexpected":"value"}],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertNotIn("尚未开始抗感染治疗",output[0]["medical_record"]["plan"])
        clinical=log["cases"][0]["clinical"]
        self.assertEqual(clinical["verification_status"],"fallback_disclosed")
        self.assertEqual(clinical["revision"]["model_revision_error"]["code"],"INVALID_LOCAL_REVISION")

    def test_invalid_contrastive_plan_uses_baseline_fallback_and_logs_mode(self):
        invalid={"problems":[]}
        code,output,log=self.run_pipeline([invalid,draft()],
            ["--clinical-verification","off","--clinical-revision","off",
             "--clinical-prompt-profile","contrastive"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        clinical=log["cases"][0]["clinical"]
        self.assertEqual(clinical["algorithm_mode"],"baseline_fallback")
        self.assertEqual(clinical["planner_fallback"]["primary_error"],"INVALID_CLINICAL_PLAN")
        self.assertEqual([event["stage"] for event in log["cases"][0]["stage_calls"]],
                         ["planner","baseline_fallback"])

    def test_duplicate_direction_fact_is_normalized_without_case_failure(self):
        conflicted=planner()
        conflicted["problems"][0]["differential_diagnoses"]=[{
            "name":"感染性病变", "certainty":"suspected", "origin":"model_inference",
            "supporting_facts":["F001"], "contradicting_facts":["F001"],
            "missing_evidence":[]
        }]
        code,output,log=self.run_pipeline([conflicted,draft()],
            ["--clinical-verification","off","--clinical-revision","off"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        policy=log["cases"][0]["clinical"]["prompt_audit"]["planner"]["planning_policy"]
        self.assertEqual(policy["fact_conflict_normalizations"],1)
        self.assertEqual(policy["ambiguous_fact_ids"],["F001"])
        self.assertEqual(log["cases"][0]["clinical"]["algorithm_mode"],"contrastive")

    def test_invalid_plan_without_fallback_budget_fails_explicitly(self):
        script=self.root/"one-response.json"
        script.write_text(json.dumps([{"content":json.dumps({"problems":[]})}]),encoding="utf-8")
        client=MockClient(script)
        case=Case("C", "T0", [Source("S1", "发热。")])
        with self.assertRaises(CaseError) as raised:
            run_case(case,client,"clinical_pipeline","text",1,
                     clinical_verify=False,clinical_revise=False)
        self.assertEqual(raised.exception.code,"CLINICAL_PLANNER_FALLBACK_BUDGET")

    def test_low_issue_does_not_trigger_revision_call(self):
        verdict=claims(); verdict["issues"]=[{
            "claim_id":"C003","section":"plan","statement":draft()["plan"],
            "error_type":"STYLE_REDUNDANCY","severity":"low",
            "revision_hint":"可精简但不影响临床意义。","source_ids":[]}]
        code,output,log=self.run_pipeline([planner(),draft(),verdict],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertEqual(log["cases"][0]["clinical"]["verification_risk_counts"]["low"],1)

    def test_medium_issue_becomes_review_without_revision_or_failure(self):
        verdict=claims(); verdict["issues"]=[{
            "claim_id":"C003","section":"plan","statement":draft()["plan"],
            "error_type":"PLAN_NEEDS_INDIVIDUALIZATION","severity":"medium",
            "revision_hint":"需结合个体情况由医生确认。","source_ids":[]}]
        code,output,log=self.run_pipeline([planner(),draft(),verdict],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertTrue(any("需医生复核" in item
                            for item in output[0]["medical_record"]["review_items"]))

    def test_unreferenced_grounded_reasoning_is_downgraded_without_case_failure(self):
        verdict=claims(); verdict["claims"][1]["source_ids"]=[]
        code,output,log=self.run_pipeline([planner(),draft(),verdict],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertEqual(log["cases"][0]["clinical"]["verification_risk_counts"]["medium"],1)


if __name__ == "__main__": unittest.main()
