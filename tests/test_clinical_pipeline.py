import json
import tempfile
import unittest
from pathlib import Path

from src.app import main


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

    def test_verifier_extra_data_is_attributed_to_third_stage(self):
        script=self.root/"malformed.json"; output=self.root/"malformed-output.json"; log_path=self.root/"malformed-log.json"
        script.write_text(json.dumps([
            {"content":json.dumps(planner(),ensure_ascii=False)},
            {"content":json.dumps(draft(),ensure_ascii=False)},
            {"content":'{"claims":[],"issues":[]}\n额外说明'},
        ], ensure_ascii=False), encoding="utf-8")
        code=main(["--input",str(self.input),"--output",str(output),"--log",str(log_path),
                   "--mode","mock","--mock-responses",str(script),"--pipeline","clinical_pipeline"])
        saved_output=json.loads(output.read_text()); saved_log=json.loads(log_path.read_text())
        self.assertEqual(code,0); self.assertEqual(saved_output[0]["error"]["code"],"MODEL_INVALID_JSON")
        self.assertIn("verifier", saved_output[0]["error"]["message"])
        event=saved_log["cases"][0]["stage_calls"][2]
        self.assertEqual(event["stage"],"verifier"); self.assertFalse(event["json_parse_success"])
        self.assertEqual(event["parse_error_type"],"JSON_TRAILING_TEXT")
        self.assertIsInstance(event["parse_error_position"],int)

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
        self.assertTrue(any("核验提示需医生复核" in item
                            for item in output[0]["medical_record"]["review_items"]))

    def test_unreferenced_grounded_reasoning_is_downgraded_without_case_failure(self):
        verdict=claims(); verdict["claims"][1]["source_ids"]=[]
        code,output,log=self.run_pipeline([planner(),draft(),verdict],
            ["--clinical-verification","on","--clinical-revision","on"])
        self.assertEqual(code,0); self.assertEqual(output[0]["status"],"success")
        self.assertEqual(log["cases"][0]["calls"],3)
        self.assertEqual(log["cases"][0]["clinical"]["verification_risk_counts"]["medium"],1)


if __name__ == "__main__": unittest.main()
