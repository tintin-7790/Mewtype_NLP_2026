import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.claim_extractor import extract_claims
from src.claim_verifier import (deterministic_claim_issues, find_internal_ids,
                                is_technical_review_item)
from src.competition_profile import (ACTIVE_COMPETITION_PROFILE,
                                     LOCAL_DEVELOPMENT_PROFILE, PROFILES)
from src.errors import CaseError
from src.local_reviser import (apply_conservative_fallback, apply_revision,
                               run_local_revision)
from src.model_client import Budget, MEDICAL_SCHEMA, VERIFIER_SCHEMA, MockClient
from src.pipeline import run_case
from src.prompt_profiles import load_clinical_prompt, resolve_clinical_prompt_path
from src.protocol import Case, Source
from src.shared_clinical_memory import SharedClinicalMemory
from src.validation import validate_record
from src.writing_style_retriever import render_writer_guidance


ROOT = Path(__file__).resolve().parents[1]


def demo_case():
    return Case("DEMO001", "2026-11-24 10:00", [
        Source("S01", "成人，乏力2周，活动后心悸；现有记录未见明确黑便、肉眼血尿或其他出血表现"),
        Source("S02", "结膜苍白、双下肢轻度水肿；未提供血压、尿量及其他体征"),
        Source("S03", "血红蛋白82 g/L，实验室标低；肌酐185 μmol/L，实验室标高；未提供既往肌酐、红细胞指数、网织红细胞或尿检"),
        Source("S04", "入院初步判断为贫血及肾功能异常，原因待查"),
    ])


def claim(text, section="diagnostic_discussion", claim_id="C001"):
    return {"claim_id": claim_id, "section": section, "statement": text,
            "claim_type": "inference", "source_ids": [], "span_start": 0,
            "span_end": len(text), "item_index": None}


def record(discussion="贫血及肾功能异常原因待查。", reviews=None):
    return {
        "case_features": "乏力2周，活动后心悸，血红蛋白82 g/L，肌酐185 μmol/L。",
        "diagnostic_discussion": discussion,
        "plan": "建议结合病情进一步评估贫血及肾功能异常原因，由医生判断。",
        "evidence": [{"statement": "血红蛋白82 g/L，肌酐185 μmol/L。",
                      "source_ids": ["S03"], "kind": "fact"}],
        "review_items": reviews or [],
    }


class FinalQualityGateTests(unittest.TestCase):
    def test_01_exact_runtime_fact_id_leak_is_detected(self):
        issues = deterministic_claim_issues(
            demo_case(), [claim("F002明确否定黑便与肉眼血尿。")], {"F001", "F002"})
        self.assertIn("INTERNAL_AUDIT_LEAK", {item["error_type"] for item in issues})

    def test_02_problem_and_claim_ids_are_detected_without_broad_regex(self):
        text = "PR001与C014均为内部定位；医学文本F波不应误报。"
        self.assertEqual(find_internal_ids(text, {"PR001", "C014", "F002"}),
                         ["PR001", "C014"])

    def test_03_debug_marker_is_detected(self):
        marker = "CASE_LOCAL_DECISION_CARDS_NOT_PATIENT_EVIDENCE"
        self.assertEqual(find_internal_ids(marker, set()), [marker])

    def test_04_technical_review_item_is_rejected(self):
        text = "需医生复核：F002错误引用来源编号，Verifier 应修正。"
        issues = deterministic_claim_issues(
            demo_case(), [claim(text, "review_items")], {"F002"})
        self.assertTrue(is_technical_review_item(text))
        self.assertIn("TECHNICAL_REVIEW_ITEM", {item["error_type"] for item in issues})

    def test_05_weak_negative_cannot_be_strengthened_to_exclusion(self):
        issues = deterministic_claim_issues(
            demo_case(), [claim("明确否定黑便与肉眼血尿，削弱隐匿性失血可能性。")])
        self.assertIn("UNSUPPORTED_NEGATIVE", {item["error_type"] for item in issues})

    def test_06_conservative_non_exclusion_wording_is_allowed(self):
        issues = deterministic_claim_issues(
            demo_case(), [claim("现有记录未见明确黑便或肉眼血尿，但不能排除隐匿性失血。")])
        self.assertNotIn("UNSUPPORTED_NEGATIVE", {item["error_type"] for item in issues})

    def test_07_planned_exam_cannot_be_written_as_completed(self):
        case = Case("C", "T", [Source("S1", "拟行胸部CT，结果待回报。")])
        issues = deterministic_claim_issues(case, [claim("胸部CT已完成，结果显示异常。", "plan")])
        self.assertIn("PLAN_AS_COMPLETED", {item["error_type"] for item in issues})

    def test_08_rule_out_diagnosis_cannot_be_confirmed(self):
        case = Case("C", "T", [Source("S1", "初步诊断：TIA待排。")])
        issues = deterministic_claim_issues(case, [claim("患者诊断为TIA。")])
        self.assertIn("DIAGNOSIS_CERTAINTY_UPGRADE", {item["error_type"] for item in issues})

    def test_09_no_effect_patch_is_not_accepted(self):
        before = record()
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high"}
        patch = {"edits": [{"claim_id": target["claim_id"], "section": target["section"],
                             "action": "replace", "replacement": target["statement"]}],
                 "review_items_to_add": []}
        with self.assertRaises(CaseError) as caught:
            apply_revision(before, patch, [issue])
        self.assertEqual(caught.exception.code, "LOCAL_REVISION_NO_EFFECT")

    def test_10_targeted_patch_preserves_other_sections(self):
        before = record("F002明确否定黑便。")
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high"}
        patch = {"edits": [{"claim_id": target["claim_id"], "section": target["section"],
                             "action": "replace",
                             "replacement": "现有记录未见明确黑便，但不能据此排除隐匿性失血。"}],
                 "review_items_to_add": []}
        after, meta = apply_revision(before, patch, [issue])
        self.assertEqual(after["case_features"], before["case_features"])
        self.assertNotIn("F002", after["diagnostic_discussion"])
        self.assertEqual(meta["unresolved_claim_ids"], [])

    def test_10b_complete_unsafe_sentence_can_be_removed_without_emptying_section(self):
        before = record("贫血原因待查。必须立即实施高风险治疗。")
        target = [item for item in extract_claims(before)
                  if item["section"] == "diagnostic_discussion"][1]
        issue = {**target, "error_type": "UNSAFE_RECOMMENDATION", "severity": "high"}
        patch = {"edits": [{"claim_id": target["claim_id"],
                              "section": target["section"], "action": "remove",
                              "replacement": ""}], "review_items_to_add": []}
        after, meta = apply_revision(before, patch, [issue])
        self.assertEqual(after["diagnostic_discussion"], "贫血原因待查。")
        self.assertNotIn("高风险治疗", after["diagnostic_discussion"])
        self.assertEqual(meta["revised_claim_ids"], [target["claim_id"]])

    def test_11_post_patch_risk_is_recomputed(self):
        before = record("F002明确否定黑便。")
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high"}
        bad_patch = {"edits": [{"claim_id": target["claim_id"], "section": target["section"],
                                 "action": "replace",
                                 "replacement": "F002明确否定黑便与肉眼血尿。"}],
                     "review_items_to_add": []}
        after, _ = apply_revision(before, bad_patch, [issue])
        remaining = deterministic_claim_issues(demo_case(), extract_claims(after), {"F002"})
        self.assertTrue(any(item["severity"] == "high" for item in remaining))

    def test_12_technical_review_fallback_removes_item(self):
        before = record(reviews=["F002错误引用来源编号，应修正。"])
        target = next(item for item in extract_claims(before) if item["section"] == "review_items")
        issue = {**target, "error_type": "TECHNICAL_REVIEW_ITEM", "severity": "high",
                 "revision_hint": "仅记内部日志。", "source_ids": []}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["review_items"], [])
        self.assertEqual(meta["unresolved_claim_ids"], [])

    def test_13_writing_knowledge_declares_non_evidence_boundary(self):
        guidance = render_writer_guidance(case_text="贫血，肌酐升高", max_chars=1200)
        self.assertIn("NOT_PATIENT_EVIDENCE", guidance)
        self.assertIn("不得迁移", guidance)

    def test_14_shared_memory_maps_only_current_case_fact_ids(self):
        facts = [{"fact_id": "F001", "source_id": "S03", "field_path": "检验"}]
        memory = SharedClinicalMemory(demo_case(), facts)
        memory.ingest_plan({"problems": [{"problem_id": "PR001", "priority": "high",
                            "supporting_facts": ["F001"], "contradicting_facts": [],
                            "differential_diagnoses": [], "plan_goals": ["明确原因"]}]})
        rendered = memory.render("writer")
        self.assertEqual(rendered["case_id"], "DEMO001")
        self.assertEqual(rendered["decision_cards"][0]["supporting_fact_ids"], ["F001"])

    def test_15_shared_memory_is_case_isolated_and_not_persisted(self):
        first = SharedClinicalMemory(demo_case(), [])
        second = SharedClinicalMemory(Case("OTHER", "T", [Source("X1", "资料")]), [])
        self.assertNotEqual(first.render("writer")["case_id"], second.render("writer")["case_id"])
        self.assertFalse(first.audit()["persisted"])
        self.assertFalse(first.audit()["cross_case_scope"])

    def test_16_prompt_manifest_matches_loaded_v2_files(self):
        manifest = json.loads((ROOT / "prompts/contrastive_v2/prompt_sha256.json").read_text())
        for stage, filename in (("planner", "planner_system.txt"),
                                ("counter_evidence", "counter_evidence_system.txt"),
                                ("writer", "writer_system.txt"),
                                ("verifier", "claim_verify_system.txt"),
                                ("reviser", "revise_system.txt")):
            path = resolve_clinical_prompt_path(stage, "contrastive_v2")
            loaded, _ = load_clinical_prompt(stage, "contrastive_v2", "")
            self.assertEqual(loaded, path.read_text())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["files"][filename])

    def test_17_competition_and_development_profiles_remain_isolated(self):
        self.assertEqual(ACTIVE_COMPETITION_PROFILE, "competition_full_v2")
        self.assertEqual(PROFILES[ACTIVE_COMPETITION_PROFILE].pipeline, "clinical_pipeline")
        self.assertFalse(PROFILES[ACTIVE_COMPETITION_PROFILE].requires_mock_embedding)
        self.assertEqual(LOCAL_DEVELOPMENT_PROFILE.environment, "dev_api")
        self.assertTrue(LOCAL_DEVELOPMENT_PROFILE.requires_mock_embedding)

    def test_18_output_schema_is_unchanged_and_demo_record_validates(self):
        self.assertEqual(set(MEDICAL_SCHEMA["schema"]["properties"]),
                         {"case_features", "diagnostic_discussion", "plan",
                          "evidence", "review_items"})
        self.assertEqual(validate_record(record(), demo_case()), [])
        codes = VERIFIER_SCHEMA["schema"]["properties"]["issues"]["items"]["properties"]["code"]["enum"]
        self.assertIn("SOURCE_MISMATCH", codes)
        self.assertNotIn("INTERNAL_AUDIT_LEAK", codes)

    def test_19_output_guarantee_isolates_runtime_id_leak_without_null_record(self):
        plan = {"problems": [{"problem_id": "PR001", "name": "贫血原因待查",
                 "priority": "high", "origin": "mixed", "supporting_facts": ["F001"],
                 "contradicting_facts": [], "differential_diagnoses": [],
                 "plan_goals": ["明确贫血原因"], "retrieval_queries": ["贫血 原因"]}]}
        leaked = record("F001提示贫血，原因待查。")
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "responses.json"
            script.write_text(json.dumps([
                {"content": json.dumps(plan, ensure_ascii=False)},
                {"content": json.dumps(leaked, ensure_ascii=False)},
            ], ensure_ascii=False))
            output, meta = run_case(
                demo_case(), MockClient(script), "clinical_pipeline", "text", 2,
                clinical_verify=False, clinical_revise=False,
            )
        self.assertNotIn("F001", output["diagnostic_discussion"])
        self.assertIn("贫血", output["diagnostic_discussion"])
        clinical = meta["clinical"]
        self.assertEqual(clinical["output_guarantee"]["level"], 3)
        self.assertIn(
            "C002",
            clinical["detected_risks_before_output_guarantee"]
                    ["unresolved_hard_claim_ids"],
        )
        self.assertEqual(clinical["unresolved_hard_risk_count"], 0)

    def test_20_real_corrupted_output_fails_prose_quality_checks(self):
        corrupted = record(
            "本例贫血与肾功能异常共存。该项陈述依据不足，需医生结合原始资料确认。"
            "AKI叠加？原发肾小球病？）。该项陈述依据不足，需医生结合原始资料确认。",
            ["该项陈述依据不足，需医生结合原始资料确认。"],
        )
        codes = {item["code"] for item in validate_record(corrupted, demo_case())}
        self.assertIn("PROSE_BRACKET_MISMATCH", codes)
        self.assertIn("GENERIC_REVISION_TEMPLATE", codes)
        self.assertIn("NON_ACTIONABLE_REVIEW_ITEM", codes)

    def test_21_historical_long_analysis_is_not_replaced_by_fallback_template(self):
        discussion = (
            "贫血原因未明：本例Hb 82 g/L伴结膜苍白、乏力及心悸，支持中度贫血；"
            "隐匿性胃肠道失血虽有贫血提示，但F002明确否定黑便与肉眼血尿，"
            "削弱该诊断可能性。肾功能异常原因未明：肌酐185 μmol/L伴水肿，"
            "需鉴别CKD与AKI。"
        )
        before = record(discussion)
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion" and "F002" in item["statement"])
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high",
                 "revision_hint": "删除内部标识并核对弱阴性。", "source_ids": []}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["diagnostic_discussion"], before["diagnostic_discussion"])
        self.assertEqual(meta["unresolved_claim_ids"], [target["claim_id"]])
        self.assertNotIn("该项陈述依据不足", after["diagnostic_discussion"])

    def test_22_fallback_does_not_add_generic_review_item(self):
        before = record("F002明确否定黑便与肉眼血尿。")
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high",
                 "revision_hint": "需修订。", "source_ids": []}
        after, meta = apply_conservative_fallback(before, [issue])
        self.assertEqual(after["review_items"], [])
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["unresolved_claim_ids"], [target["claim_id"]])

    def test_23_long_sentence_is_one_complete_claim_not_comma_fragments(self):
        value = record("贫血与肾功能异常共存，需鉴别CKD与AKI（包括AKI叠加、原发肾小球病），当前证据有限。")
        claims = [item for item in extract_claims(value)
                  if item["section"] == "diagnostic_discussion"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["statement"], value["diagnostic_discussion"])

    def test_24_generic_model_patch_is_rejected(self):
        before = record("F002明确否定黑便。")
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high"}
        patch = {"edits": [{"claim_id": target["claim_id"], "section": target["section"],
                             "action": "replace",
                             "replacement": "该项陈述依据不足，需医生结合原始资料确认。"}],
                 "review_items_to_add": []}
        with self.assertRaises(CaseError) as caught:
            apply_revision(before, patch, [issue])
        self.assertEqual(caught.exception.code, "LOCAL_REVISION_GENERIC_TEMPLATE")

    def test_25_partially_effective_model_patch_preserves_unresolved_high(self):
        before = record("F002明确否定黑便与肉眼血尿，削弱隐匿性失血可能性。")
        target = next(item for item in extract_claims(before)
                      if item["section"] == "diagnostic_discussion")
        issue = {**target, "error_type": "INTERNAL_AUDIT_LEAK", "severity": "high",
                 "revision_hint": "删除内部 ID 并按原文强度表述。", "source_ids": []}
        revision = {"edits": [{"claim_id": target["claim_id"],
                                "section": "diagnostic_discussion", "action": "replace",
                                "replacement": "明确否定黑便与肉眼血尿，削弱隐匿性失血可能性。"}],
                    "review_items_to_add": []}
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "revision.json"
            script.write_text(json.dumps([{"content": json.dumps(revision, ensure_ascii=False)}]))
            revised, meta = run_local_revision(
                demo_case(), before, {"problems": []}, [issue], None, 3,
                MockClient(script), "text", Budget(1), internal_ids={"F002"},
            )
        self.assertNotIn("F002", revised["diagnostic_discussion"])
        self.assertGreater(meta["patch_acceptance"]["post_patch_high_risk_count"], 0)
        self.assertTrue(meta["unresolved_claim_ids"])


if __name__ == "__main__":
    unittest.main()
