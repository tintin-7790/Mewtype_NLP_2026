import json
import tempfile
import unittest
from pathlib import Path

from src.claim_extractor import extract_claims
from src.claim_verifier import deterministic_claim_issues
from src.errors import CaseError
from src.model_client import MockClient
from src.output_guarantee import (
    build_sources_fallback,
    guarantee_writer_output,
    recover_from_stage_failure,
    sanitize_writer_record,
)
from src.protocol import Case, Source
from src.pipeline import run_case
from src.validation import validate_record


def case(source_count=3):
    sources = [
        Source("S001", "患者发热、咳嗽3天。", "入院记录.主诉"),
        Source("S002", "初步诊断：肺炎待排。", "入院记录.初步诊断"),
        Source("S003", "计划完善胸部CT检查。", "入院记录.诊疗计划"),
    ]
    for index in range(4, source_count + 1):
        sources.append(Source(f"S{index:03d}", f"第{index}项原始检查资料。",
                              "入院记录.辅助检查"))
    return Case("C1", "T0", sources)


def record(plan="建议结合病情进一步评估。"):
    return {
        "case_features": "患者发热、咳嗽3天。",
        "diagnostic_discussion": "肺炎待排，仍需结合检查确认。",
        "plan": plan,
        "evidence": [{"statement": "患者发热、咳嗽3天。",
                      "source_ids": ["S001"], "kind": "fact"}],
        "review_items": [],
    }


def issue_for(value, section, code="UNSAFE_RECOMMENDATION"):
    claim = next(item for item in extract_claims(value) if item["section"] == section)
    return {**claim, "error_type": code, "severity": "high",
            "revision_hint": "执行安全隔离。", "source_ids": []}


class OutputGuaranteeTests(unittest.TestCase):
    def _run_mock(self, entries, *, max_calls, verify=True):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
            "origin": "mixed", "supporting_facts": ["F001"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["明确病因"], "retrieval_queries": ["发热 咳嗽"],
        }]}
        values = []
        for item in entries:
            if item == "planner":
                values.append({"content": json.dumps(planner, ensure_ascii=False)})
            elif item == "agent":
                values.append({"content": json.dumps({
                    "reviewed_problem_ids": ["PR001"], "issues": [],
                }, ensure_ascii=False)})
            elif item == "writer":
                values.append({"content": json.dumps(record(), ensure_ascii=False)})
            else:
                values.append(item)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "responses.json"
            script.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
            client = MockClient(script)
            result, meta = run_case(
                case(), client, "clinical_pipeline", "text", max_calls,
                clinical_verify=verify, clinical_revise=verify,
            )
        return result, meta, client

    def test_level1_returns_complete_verified_writer(self):
        value, audit = guarantee_writer_output(
            case(), record(), [], verification_completed=True, revision_applied=False,
        )
        self.assertEqual(value, record())
        self.assertEqual(audit["level"], 1)
        self.assertTrue(audit["safe_record_available"])

    def test_incomplete_verifier_preserves_safe_writer_as_level3(self):
        value, audit = guarantee_writer_output(
            case(), record(), [], verification_completed=False, revision_applied=False,
        )
        self.assertEqual(value, record())
        self.assertEqual(audit["level"], 3)
        self.assertFalse(audit["verification_completed"])

    def test_known_unsafe_plan_is_removed_without_erasing_other_plan(self):
        value = record("建议动态观察病情。必须实施高风险治疗。")
        target = [item for item in extract_claims(value) if item["section"] == "plan"][1]
        issue = {**target, "error_type": "UNSAFE_RECOMMENDATION", "severity": "high",
                 "revision_hint": "删除危险建议。", "source_ids": []}
        recovered, audit = guarantee_writer_output(
            case(), value, [issue], verification_completed=True, revision_applied=False,
        )
        self.assertEqual(recovered["plan"], "建议动态观察病情。")
        self.assertNotIn("高风险治疗", recovered["plan"])
        self.assertEqual(audit["level"], 2)
        self.assertIn(target["claim_id"], audit["removed_claim_ids"])

    def test_single_unsafe_plan_is_replaced_from_sources(self):
        value = record("必须实施高风险治疗。")
        issue = issue_for(value, "plan")
        recovered, audit = sanitize_writer_record(case(), value, [issue])
        self.assertNotIn("高风险治疗", recovered["plan"])
        self.assertIn("计划完善胸部CT检查", recovered["plan"])
        self.assertTrue(audit["applied"])

    def test_already_absent_high_statement_does_not_damage_safe_writer(self):
        value = record()
        issue = {"claim_id": "C999", "section": "diagnostic_discussion",
                 "statement": "无法定位的危险断言。", "error_type": "FACT_CONTRADICTION",
                 "severity": "high", "revision_hint": "隔离。", "source_ids": []}
        recovered, audit = sanitize_writer_record(case(), value, [issue])
        # The exact unsafe statement is already absent, so the safe Writer can
        # remain; no partial-text guessing is performed.
        self.assertEqual(recovered, value)
        self.assertFalse(audit["sources_fallback_used"])

    def test_ambiguous_duplicate_claim_locator_uses_sources_fallback(self):
        value = record("必须实施高风险治疗。必须实施高风险治疗。")
        target = next(item for item in extract_claims(value) if item["section"] == "plan")
        issue = {**target, "error_type": "UNSAFE_RECOMMENDATION", "severity": "high",
                 "revision_hint": "隔离。", "source_ids": []}
        recovered, audit = sanitize_writer_record(case(), value, [issue])
        self.assertTrue(audit["sources_fallback_used"])
        self.assertNotIn("高风险治疗", recovered["plan"])

    def test_missing_statement_context_forces_sources_fallback(self):
        issue = {"claim_id": "C999", "section": "diagnostic_discussion",
                 "statement": None, "error_type": "FACT_CONTRADICTION",
                 "severity": "high", "revision_hint": "隔离。", "source_ids": []}
        recovered, audit = sanitize_writer_record(case(), record(), [issue])
        self.assertTrue(audit["sources_fallback_used"])
        self.assertIn("主诉：患者发热、咳嗽3天", recovered["case_features"])

    def test_invalid_evidence_source_rebuilds_only_evidence_field(self):
        value = record()
        value["evidence"][0]["source_ids"] = ["NOT-EXISTS"]
        recovered, audit = sanitize_writer_record(case(), value, [])
        self.assertFalse(audit["sources_fallback_used"])
        self.assertEqual(recovered["case_features"], value["case_features"])
        self.assertEqual(recovered["diagnostic_discussion"], value["diagnostic_discussion"])
        self.assertEqual(recovered["plan"], value["plan"])
        self.assertEqual(audit["field_recovery"]["rebuilt_fields"], ["evidence"])
        self.assertTrue(all(item["source_ids"][0].startswith("S")
                            for item in recovered["evidence"]))

    def test_localized_review_high_risk_preserves_other_fields(self):
        value = record()
        value["review_items"] = ["胸部CT结果待确认。", "目前已排除隐匿性肺炎。"]
        target = extract_claims(value)[-1]
        issue = {**target, "error_type": "UNSUPPORTED_NEGATIVE", "severity": "high",
                 "revision_hint": "删除无依据强否定。", "source_ids": ["S002"]}
        recovered, audit = sanitize_writer_record(case(), value, [issue])
        self.assertEqual(recovered["case_features"], value["case_features"])
        self.assertEqual(recovered["diagnostic_discussion"], value["diagnostic_discussion"])
        self.assertEqual(recovered["plan"], value["plan"])
        self.assertEqual(recovered["review_items"], ["胸部CT结果待确认。"])
        self.assertIn(target["claim_id"], audit["removed_claim_ids"])
        self.assertFalse(audit["sources_fallback_used"])

    def test_unrepairable_diagnostic_claim_rebuilds_only_field_and_evidence(self):
        value = record()
        value["diagnostic_discussion"] = "肺炎已经确诊。"
        value["review_items"] = ["胸部CT结果待确认。"]
        target = issue_for(value, "diagnostic_discussion", "FACT_CONTRADICTION")
        recovered, audit = sanitize_writer_record(case(), value, [target])
        self.assertEqual(recovered["case_features"], value["case_features"])
        self.assertEqual(recovered["plan"], value["plan"])
        self.assertEqual(recovered["review_items"], value["review_items"])
        self.assertNotEqual(recovered["diagnostic_discussion"], value["diagnostic_discussion"])
        self.assertIn("diagnostic_discussion", audit["field_recovery"]["rebuilt_fields"])
        self.assertIn("evidence", audit["field_recovery"]["rebuilt_fields"])
        self.assertEqual(validate_record(recovered, case(), "lite"), [])

    def test_multiple_localized_fields_are_recovered_without_whole_record_fallback(self):
        value = record("必须实施高风险治疗。")
        value["diagnostic_discussion"] = "肺炎已经确诊。"
        issues = [
            issue_for(value, "diagnostic_discussion", "FACT_CONTRADICTION"),
            issue_for(value, "plan", "UNSAFE_RECOMMENDATION"),
        ]
        recovered, audit = sanitize_writer_record(case(), value, issues)
        self.assertFalse(audit["sources_fallback_used"])
        self.assertEqual(recovered["case_features"], value["case_features"])
        self.assertNotIn("已经确诊", recovered["diagnostic_discussion"])
        self.assertNotIn("高风险治疗", recovered["plan"])
        self.assertEqual(
            set(audit["field_recovery"]["rebuilt_fields"]),
            {"diagnostic_discussion", "plan", "evidence"},
        )

    def test_multiple_high_claims_in_one_prose_field_rebuild_once_not_fragment(self):
        value = record()
        value["diagnostic_discussion"] = (
            "可信的病例特异分析。无依据的确定诊断。无依据的确定排除。"
        )
        claims = [item for item in extract_claims(value)
                  if item["section"] == "diagnostic_discussion"]
        issues = [{**claim, "error_type": "FACT_CONTRADICTION", "severity": "high",
                   "revision_hint": "按原始资料恢复字段。", "source_ids": []}
                  for claim in claims[1:]]
        recovered, audit = sanitize_writer_record(case(), value, issues)
        self.assertFalse(audit["sources_fallback_used"])
        self.assertNotIn("无依据", recovered["diagnostic_discussion"])
        self.assertNotEqual(recovered["diagnostic_discussion"], "可信的病例特异分析。")
        self.assertEqual(
            audit["field_recovery"]["fragmentation_prevented_fields"],
            ["diagnostic_discussion"],
        )
        self.assertIn("evidence", audit["field_recovery"]["rebuilt_fields"])

    def test_review_item_removal_does_not_hide_dangerous_body_claim(self):
        risky_case = Case("C2", "T0", [
            Source("S001", "患者胸痛。", "入院记录.主诉"),
            Source("S002", "初步诊断：胸痛原因待查。", "入院记录.初步诊断"),
            Source("S003", "计划完善心电图。", "入院记录.诊疗计划"),
        ])
        value = record()
        value["case_features"] = "患者无胸痛。"
        value["review_items"] = ["目前已排除胸痛相关疾病。"]
        review_claim = extract_claims(value)[-1]
        review_issue = {**review_claim, "error_type": "UNSUPPORTED_NEGATIVE",
                        "severity": "high", "revision_hint": "删除。",
                        "source_ids": ["S001"]}
        recovered, audit = sanitize_writer_record(risky_case, value, [review_issue])
        self.assertNotIn("无胸痛", recovered["case_features"])
        self.assertNotIn("已排除", "".join(recovered["review_items"]))
        self.assertTrue(audit["field_recovery"]["attempted"])
        self.assertEqual(
            [item for item in deterministic_claim_issues(
                risky_case, extract_claims(recovered), set())
             if item.get("severity") == "high"],
            [],
        )

    def test_sources_fallback_is_valid_and_traceable(self):
        recovered, audit = build_sources_fallback(case())
        self.assertEqual(validate_record(recovered, case(), "lite"), [])
        valid_ids = {source.source_id for source in case().sources}
        self.assertTrue(all(set(item["source_ids"]) <= valid_ids
                            for item in recovered["evidence"]))
        self.assertFalse(audit["contains_model_generated_patient_facts"])

    def test_sources_fallback_handles_more_than_fifty_sources(self):
        large = case(55)
        recovered, audit = build_sources_fallback(large)
        self.assertEqual(validate_record(recovered, large, "lite"), [])
        self.assertLessEqual(len(recovered["evidence"]), 5)
        self.assertEqual(audit["source_count"], 55)

    def test_plan_in_auxiliary_remark_is_recovered_as_explicit_plan(self):
        value = Case("PLAN_REMARK", "T0", [
            Source("S001", "反复咳嗽、咳痰2月。", "入院记录.主诉"),
            Source("S002", "初步诊断：肺部感染，肺结核待排。", "入院记录.初步诊断"),
            Source("S003", "入院后待完善痰涂片及培养、痰抗酸染色检查。",
                   "入院记录.辅助检查.备注"),
        ])
        recovered, audit = build_sources_fallback(value)
        self.assertIn("待完善痰涂片及培养", recovered["plan"])
        self.assertEqual(audit["plan_source_ids"], ["S003"])
        self.assertNotIn("未明确记载后续检查", recovered["plan"])

    def test_source_backed_plan_duplicate_is_relocated_out_of_review_items(self):
        value_case = Case("PLAN_REVIEW", "T0", [
            Source("S001", "患者发热、咳嗽3天。", "入院记录.主诉"),
            Source("S002", "初步诊断：肺炎待排。", "入院记录.初步诊断"),
            Source("S003", "入院后待完善胸部CT检查。", "入院记录.辅助检查.备注"),
        ])
        value = record("原始资料记载入院后待完善胸部CT检查。")
        value["review_items"] = [
            "入院后待完善胸部CT检查。",
            "胸部CT检查结果待回报并核实。",
        ]
        recovered, audit = sanitize_writer_record(value_case, value, [])
        self.assertEqual(recovered["review_items"], ["胸部CT检查结果待回报并核实。"])
        self.assertEqual(
            len(audit["field_recovery"]
                ["removed_source_backed_plan_review_item_sha256"]),
            1,
        )
        self.assertTrue(audit["applied"])

    def test_stage_failure_recovers_without_another_model_call(self):
        recovered, audit = recover_from_stage_failure(
            case(), "writer", CaseError("MODEL_TIMEOUT", "timeout"),
        )
        self.assertEqual(audit["level"], 4)
        self.assertFalse(audit["writer_available"])
        self.assertEqual(validate_record(recovered, case(), "lite"), [])

    def test_planner_timeout_recovers_to_sources_with_one_call(self):
        result, meta, client = self._run_mock(
            [{"exception": "timeout"}], max_calls=1, verify=False,
        )
        self.assertEqual(meta["clinical"]["output_guarantee"]["level"], 4)
        self.assertEqual(client.total_calls, 1)
        self.assertTrue(result["case_features"])

    def test_writer_timeout_recovers_to_sources_without_extra_call(self):
        result, meta, client = self._run_mock(
            ["planner", {"exception": "timeout"}], max_calls=2, verify=False,
        )
        self.assertEqual(meta["clinical"]["output_guarantee"]["level"], 4)
        self.assertEqual(client.total_calls, 2)
        self.assertIn("主诉：患者发热、咳嗽3天", result["case_features"])

    def test_writer_invalid_json_recovers_to_sources(self):
        result, meta, client = self._run_mock(
            ["planner", {"content": '{"case_features":'}], max_calls=2, verify=False,
        )
        self.assertEqual(meta["clinical"]["output_guarantee"]["level"], 4)
        self.assertEqual(client.total_calls, 2)
        self.assertEqual(validate_record(result, case(), "lite"), [])

    def test_verifier_timeout_preserves_writer_without_retry(self):
        result, meta, client = self._run_mock(
            ["planner", "writer", {"exception": "timeout"}], max_calls=3, verify=True,
        )
        clinical = meta["clinical"]
        self.assertEqual(client.total_calls, 3)
        self.assertFalse(clinical["verification_completed"])
        self.assertEqual(clinical["verification_error"]["code"], "MODEL_TIMEOUT")
        self.assertEqual(clinical["output_guarantee"]["level"], 3)
        self.assertEqual(result, record())

    def test_verifier_length_truncation_preserves_writer(self):
        result, meta, client = self._run_mock(
            ["planner", "writer", {"content": '{"checked_claim_ids":',
                                     "finish_reason": "length"}],
            max_calls=3, verify=True,
        )
        self.assertEqual(client.total_calls, 3)
        self.assertFalse(meta["clinical"]["verification_completed"])
        self.assertEqual(meta["clinical"]["output_guarantee"]["level"], 3)
        self.assertEqual(result["case_features"], record()["case_features"])

    def test_agent_failure_is_audited_and_writer_still_runs(self):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
            "origin": "mixed", "supporting_facts": ["F001"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["明确病因"], "retrieval_queries": ["发热 咳嗽"],
        }]}
        entries = [
            {"content": json.dumps(planner, ensure_ascii=False)},
            {"exception": "timeout"},
            {"content": json.dumps(record(), ensure_ascii=False)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "responses.json"
            script.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            client = MockClient(script)
            result, meta = run_case(
                case(), client, "clinical_pipeline", "text", 3,
                clinical_verify=False, clinical_revise=False,
                clinical_agent_mode="always",
            )
        self.assertEqual(client.total_calls, 3)
        self.assertEqual(result, record())
        self.assertFalse(meta["clinical"]["adaptive_agent"]["completed"])
        self.assertEqual(meta["clinical"]["adaptive_agent"]["error"]["code"],
                         "MODEL_TIMEOUT")

    def test_exhausted_budget_recovers_from_sources_without_sixth_call(self):
        planner = {"problems": [{
            "problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
            "origin": "mixed", "supporting_facts": ["F001"],
            "contradicting_facts": [], "differential_diagnoses": [],
            "plan_goals": ["明确病因"], "retrieval_queries": ["发热 咳嗽"],
        }]}
        entries = [
            {"content": json.dumps(planner, ensure_ascii=False)},
            {"content": json.dumps({"reviewed_problem_ids": ["PR001"], "issues": []},
                                   ensure_ascii=False)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "responses.json"
            script.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            client = MockClient(script)
            result, meta = run_case(
                case(), client, "clinical_pipeline", "text", 2,
                clinical_verify=False, clinical_revise=False,
                clinical_agent_mode="always",
            )
        self.assertEqual(client.total_calls, 2)
        self.assertEqual(meta["clinical"]["output_guarantee"]["level"], 4)
        self.assertIn("主诉：患者发热、咳嗽3天", result["case_features"])

    def test_multiple_high_risk_types_are_sanitized_in_one_record(self):
        value = record("建议动态观察。必须实施高风险治疗。")
        value["diagnostic_discussion"] = "患者诊断为肺炎。"
        diagnosis_issue = issue_for(
            value, "diagnostic_discussion", "DIAGNOSIS_CERTAINTY_UPGRADE",
        )
        unsafe_claim = [item for item in extract_claims(value)
                        if item["section"] == "plan"][1]
        unsafe_issue = {
            **unsafe_claim, "error_type": "UNSAFE_RECOMMENDATION", "severity": "high",
            "revision_hint": "删除危险建议。", "source_ids": [],
        }
        recovered, audit = sanitize_writer_record(
            case(), value, [diagnosis_issue, unsafe_issue],
        )
        self.assertIn("考虑肺炎", recovered["diagnostic_discussion"])
        self.assertNotIn("诊断为肺炎", recovered["diagnostic_discussion"])
        self.assertNotIn("高风险治疗", recovered["plan"])
        self.assertFalse(audit["sources_fallback_used"])

    def test_empty_sources_is_the_unrecoverable_boundary(self):
        with self.assertRaises(CaseError) as caught:
            build_sources_fallback(Case("EMPTY", "T0", []))
        self.assertEqual(caught.exception.code, "OUTPUT_GUARANTEE_NO_RELIABLE_FACTS")

    def test_diagnosis_certainty_is_conservatively_rewritten(self):
        value = record(); value["diagnostic_discussion"] = "患者诊断为肺炎。"
        issue = issue_for(value, "diagnostic_discussion", "DIAGNOSIS_CERTAINTY_UPGRADE")
        recovered, audit = guarantee_writer_output(
            case(), value, [issue], verification_completed=True, revision_applied=False,
        )
        self.assertIn("考虑肺炎", recovered["diagnostic_discussion"])
        self.assertNotIn("诊断为肺炎", recovered["diagnostic_discussion"])
        self.assertEqual(audit["level"], 2)

    def test_sources_fallback_partitions_clinical_content_and_core_evidence(self):
        rich = Case("RICH", "T0", [
            Source("D01", "张××", "入院记录.一般项目.姓名"),
            Source("D02", "女", "入院记录.一般项目.性别"),
            Source("D03", "78岁", "入院记录.一般项目.年龄"),
            Source("S01", "右下肢水肿1周。", "入院记录.主诉"),
            Source("S02", "右足背凹陷性水肿，久站后加重；无发热、胸闷。",
                   "入院记录.现病史"),
            Source("S03", "右足凹陷性水肿，右踝关节轻度肿胀、压痛。",
                   "入院记录.体格检查.脊柱四肢"),
            Source("S04", "血红蛋白75g/L；白蛋白31.3g/L；D-二聚体3130μg/L；"
                          "下肢静脉超声未见明显血栓。",
                   "入院记录.辅助检查.外院"),
            Source("S05", "右足水肿查因：低蛋白血症，风湿免疫性疾病待排",
                   "入院记录.初步诊断[0]"),
            Source("S06", "中度贫血，营养不良", "入院记录.初步诊断[1]"),
            Source("S07", "计划完善贫血原因评估并动态监测血常规。",
                   "入院记录.诊疗计划"),
        ])
        recovered, audit = build_sources_fallback(rich)
        self.assertIn("主诉：右下肢水肿1周", recovered["case_features"])
        self.assertIn("主要查体：右足凹陷性水肿", recovered["case_features"])
        self.assertIn("血红蛋白75g/L", recovered["case_features"])
        self.assertNotIn("张××", recovered["case_features"])
        self.assertIn("当前主要临床问题", recovered["diagnostic_discussion"])
        self.assertIn("病程与查体依据", recovered["diagnostic_discussion"])
        self.assertIn("检查依据", recovered["diagnostic_discussion"])
        self.assertIn("初步诊断或鉴别方向", recovered["diagnostic_discussion"])
        self.assertIn("计划完善贫血原因评估", recovered["plan"])
        self.assertEqual(set(audit["evidence_source_ids"]),
                         {"S01", "S03", "S04", "S05", "S07"})
        self.assertEqual(validate_record(recovered, rich, "lite"), [])

    def test_hpi_consideration_is_not_misclassified_as_diagnosis_or_plan(self):
        value = Case("NO_PLAN", "T0", [
            Source("S1", "发热伴咳嗽3天。", "入院记录.主诉"),
            Source("S2", "外院曾考虑感染，建议转诊；既往已使用某药5mg。",
                   "入院记录.现病史"),
        ])
        recovered, audit = build_sources_fallback(value)
        self.assertEqual(audit["diagnosis_source_ids"], [])
        self.assertEqual(audit["plan_source_ids"], [])
        self.assertIn("未明确记载后续检查", recovered["plan"])
        self.assertNotIn("某药5mg", recovered["plan"])
        self.assertNotIn("建议转诊", recovered["plan"])
        self.assertEqual(validate_record(recovered, value, "lite"), [])

    def test_resolved_historical_plan_is_not_presented_as_current_fallback_plan(self):
        value = Case("RESOLVED_PLAN", "T0", [
            Source("S1", "发热伴咳嗽3天。", "入院记录.主诉"),
            Source("S2", "原计划复查CT，现已取消。", "入院记录.诊疗计划"),
        ])
        recovered, audit = build_sources_fallback(value)
        self.assertEqual(audit["plan_source_ids"], [])
        self.assertIn("未明确记载后续检查", recovered["plan"])
        self.assertNotIn("已取消", recovered["plan"])

    def test_completed_treatment_is_not_recast_as_future_plan(self):
        value = Case("COMPLETED_TREATMENT", "T0", [
            Source("S1", "PCI术后复查。", "入院记录.主诉"),
            Source("S2", "已行冠脉造影及PCI术，术后出院。", "入院记录.诊疗计划"),
        ])
        recovered, audit = build_sources_fallback(value)
        self.assertEqual(audit["plan_source_ids"], [])
        self.assertIn("未明确记载后续检查", recovered["plan"])
        self.assertNotIn("已行冠脉造影", recovered["plan"])

    def test_mixed_historical_and_pending_plan_keeps_only_pending_clause(self):
        value = Case("MIXED_PLAN", "T0", [
            Source("S1", "胸部不适待查。", "入院记录.主诉"),
            Source("S2", "既往已完成外院检查，本次拟复查胸部CT。",
                   "入院记录.诊疗计划"),
        ])
        recovered, audit = build_sources_fallback(value)
        self.assertEqual(audit["plan_source_ids"], ["S2"])
        self.assertIn("本次拟复查胸部CT", recovered["plan"])
        self.assertNotIn("既往已完成外院检查", recovered["plan"])
        plan_evidence = [item for item in recovered["evidence"]
                         if item["source_ids"] == ["S2"]]
        self.assertEqual(plan_evidence[0]["statement"], "本次拟复查胸部CT。")


if __name__ == "__main__":
    unittest.main()
