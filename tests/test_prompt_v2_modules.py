import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.app import main, parser
from src.clinical_planner import number_facts
from src.competition_profile import ACTIVE_COMPETITION_PROFILE
from src.facts import organize_facts
from src.input_adapter import load_input_cases
from src.model_client import (COUNTER_EVIDENCE_SCHEMA, MEDICAL_SCHEMA, PLANNER_SCHEMA,
                              REVISER_SCHEMA, VERIFIER_SCHEMA, Budget)
from src.note_writer import run_note_writer
from src.prompt_profiles import load_clinical_prompt, resolve_clinical_prompt_path
from src.protocol import Case, Source
from src.shared_clinical_memory import SharedClinicalMemory
from src.writing_style_retriever import (infer_tags, render_writer_guidance, select_cards,
                                         select_exemplar_lessons)


ROOT = Path(__file__).resolve().parent.parent


def _facts():
    return [{"fact_id": "F001", "source_id": "S1", "field_path": "现病史",
             "fact_text": "PCI术后再发胸痛。", "time": None, "value": None,
             "unit": None, "assertion": "positive", "status": "unknown",
             "correction": None}]


def _plan():
    return {"problems": [{"problem_id": "PR001", "name": "PCI术后再发胸痛",
        "priority": "high", "origin": "mixed", "supporting_facts": ["F001"],
        "contradicting_facts": [], "differential_diagnoses": [],
        "plan_goals": ["明确胸痛原因"], "retrieval_queries": ["PCI 胸痛"]}]}


def _draft():
    return {"case_features": "患者PCI术后再发胸痛。",
            "diagnostic_discussion": "胸痛原因待查。",
            "plan": "建议结合临床情况进一步评估。",
            "evidence": [{"statement": "患者PCI术后再发胸痛。",
                          "source_ids": ["S1"], "kind": "fact"}],
            "review_items": []}


class CaptureClient:
    def call(self, messages, response_format, budget, **kwargs):
        budget.take(); self.messages = messages
        return json.dumps(_draft(), ensure_ascii=False)


class PromptV2ModuleTests(unittest.TestCase):
    def test_cli_defaults_remain_off_and_competition_profile_unchanged(self):
        args = parser().parse_args(["--input", "input.json", "--output", "output.json"])
        self.assertEqual(args.clinical_prompt_profile, "contrastive")
        self.assertEqual(args.clinical_writing_guidance, "off")
        self.assertEqual(args.shared_clinical_memory, "off")
        self.assertEqual(ACTIVE_COMPETITION_PROFILE, "competition_full_v2")

    def test_v2_five_stage_files_are_independent_and_match_integrated_manifest(self):
        manifest = json.loads((ROOT / "prompts" / "contrastive_v2" /
                               "prompt_sha256.json").read_text(encoding="utf-8"))
        for stage, filename in (("planner", "planner_system.txt"),
                                ("counter_evidence", "counter_evidence_system.txt"),
                                ("writer", "writer_system.txt"),
                                ("verifier", "claim_verify_system.txt"),
                                ("reviser", "revise_system.txt")):
            path = resolve_clinical_prompt_path(stage, "contrastive_v2")
            self.assertEqual(path.parent.name, "contrastive_v2")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["files"][filename])
            self.assertTrue(load_clinical_prompt(stage, "contrastive_v2", "")[0].strip())

    def test_v1_loading_is_unchanged_and_separate(self):
        for stage, filename in (("planner", "planner_system.txt"),
                                ("writer", "writer_system.txt"),
                                ("verifier", "claim_verify_system.txt")):
            prompt, example = load_clinical_prompt(stage, "contrastive", "胸痛")
            self.assertEqual(prompt, (ROOT / "prompts" / filename).read_text(encoding="utf-8"))
            self.assertIsNone(example)

    def test_v2_prompts_match_existing_stage_schema_contracts(self):
        schemas = {"planner": PLANNER_SCHEMA, "writer": MEDICAL_SCHEMA,
                   "counter_evidence": COUNTER_EVIDENCE_SCHEMA,
                   "verifier": VERIFIER_SCHEMA, "reviser": REVISER_SCHEMA}
        required_markers = {
            "planner": ("problems", "problem_id", "retrieval_queries"),
            "writer": ("case_features", "diagnostic_discussion", "review_items"),
            "counter_evidence": ("reviewed_problem_ids", "issues", "fact_ids"),
            "verifier": ("checked_claim_ids", "issues", "severity"),
            "reviser": ("edits", "review_items_to_add", "replacement"),
        }
        for stage, schema in schemas.items():
            prompt, _ = load_clinical_prompt(stage, "contrastive_v2", "")
            for marker in required_markers[stage]:
                self.assertIn(marker, prompt)
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["schema"]["additionalProperties"])

    def test_writer_prompt_requires_clinical_analysis_and_source_backed_plan_placement(self):
        prompt, _ = load_clinical_prompt("writer", "contrastive_v2", "")
        for marker in (
            "临床判断 + 具体依据 + 当前限制",
            "不得只复述现病史一两句",
            "约180–350个中文字符",
            "约200–450个中文字符",
            "约150–350个中文字符",
            "一项来源明确但尚待执行的计划应写入 plan",
            "不能仅因“尚未执行”就转移到 review_items",
        ):
            self.assertIn(marker, prompt)
        self.assertIn("软参考，不是合格阈值", prompt)
        self.assertIn("不得为达到长度补造依据", prompt)

    def test_writing_guidance_is_deterministic_bounded_and_non_evidence(self):
        self.assertIn("cardiology", infer_tags("PCI术后胸痛"))
        ids = {card["id"] for card in select_cards(tags={"cardiology"}, max_cards=40)}
        self.assertIn("SP01", ids)
        guidance = render_writer_guidance(case_text="PCI术后胸痛", max_chars=1800)
        self.assertLessEqual(len(guidance), 1800)
        self.assertEqual(guidance, render_writer_guidance(case_text="PCI术后胸痛", max_chars=1800))
        self.assertIn("NOT_PATIENT_EVIDENCE", guidance)
        self.assertNotIn("2025-10-09", guidance)

    def test_synthetic_example_retrieval_returns_lessons_not_patient_record(self):
        lesson = select_exemplar_lessons(tags={"cardiology"}, limit=1)[0]
        self.assertIn("style_takeaways", lesson)
        self.assertNotIn("medical_record", lesson)
        self.assertNotIn("case", lesson)

    def test_writer_guidance_and_memory_are_independent_optional_payloads(self):
        case = Case("C1", "T0", [Source("S1", "PCI术后再发胸痛。", "现病史")])
        off_client = CaptureClient(); off_audit = {}
        run_note_writer(case, _facts(), _plan(), [], off_client, "text", Budget(1), 5,
                        prompt_audit=off_audit)
        off_payload = json.loads(off_client.messages[1]["content"])
        self.assertNotIn("writing_style_context", off_payload)
        self.assertNotIn("shared_decision_cards", off_payload)
        self.assertIn("clinical_plan", off_payload)

        memory = SharedClinicalMemory(case, _facts()); memory.ingest_plan(_plan())
        on_client = CaptureClient(); on_audit = {}
        run_note_writer(case, _facts(), _plan(), [], on_client, "text", Budget(1), 5,
                        prompt_profile="contrastive_v2", prompt_audit=on_audit,
                        writing_guidance="rules_examples",
                        shared_memory_context=memory.render("writer"))
        on_payload = json.loads(on_client.messages[1]["content"])
        self.assertIn("NOT_PATIENT_EVIDENCE", on_payload["writing_style_context"])
        self.assertEqual(on_payload["shared_decision_cards"]["case_id"], "C1")
        self.assertNotIn("clinical_plan", on_payload)
        self.assertNotIn("writing_style_context", on_payload["sources"][0])
        self.assertGreater(on_audit["writer"]["payload_chars"], off_audit["writer"]["payload_chars"])
        self.assertGreaterEqual(on_audit["writer"]["preparation_elapsed_ms"], 0)

    def test_shared_memory_is_case_local_and_tracks_plan_graph_agent(self):
        case = Case("C1", "T0", [Source("S1", "胸痛。")])
        memory = SharedClinicalMemory(case, _facts())
        memory.ingest_plan(_plan())
        memory.annotate_retrieval([{"problem_id": "PR001",
                                    "candidates": [{"source_id": "S1"}]}])
        memory.annotate_counter_review({"issues": [{"problem_id": "PR001"}]})
        counter_view = memory.render("counter_evidence")
        self.assertNotIn("case_clinical_focus", counter_view)
        self.assertNotIn("counter_evidence_review", counter_view["decision_cards"][0])
        rendered = memory.render("writer"); card = rendered["decision_cards"][0]
        self.assertEqual(card["graph_candidate_ids"], ["F001"])
        self.assertEqual(card["counter_issue_ids"], ["CI001"])
        self.assertEqual(card["problem_name"], "PCI术后再发胸痛")
        self.assertEqual(card["supporting_fact_refs"],
                         [{"fact_id": "F001", "source_id": "S1"}])
        self.assertEqual(card["counter_evidence_review"][0]["counter_issue_id"], "CI001")
        self.assertEqual(rendered["case_id"], "C1")
        other = SharedClinicalMemory(Case("C2", "T0", [Source("S2", "头晕。")]), [])
        self.assertEqual(other.render("writer")["decision_cards"], [])

    def test_case02_writer_memory_exposes_unlinked_comorbidities_and_source_plan(self):
        source = next((ROOT / "information" / "临床智录_练习病历数据集JSON").glob("02_*.json"))
        case = load_input_cases(source).cases[0]
        facts = number_facts(organize_facts(case))
        diagnosis_ids = [fact["fact_id"] for fact in facts
                         if "初步诊断" in fact["field_path"]]
        plan = {"problems": [
            {"problem_id": f"PR{index:03d}", "name": name, "priority": priority,
             "origin": "source", "supporting_facts": [diagnosis_ids[index - 1]],
             "contradicting_facts": [], "differential_diagnoses": [],
             "plan_goals": ["核对原始资料并按问题组织"], "retrieval_queries": [name]}
            for index, (name, priority) in enumerate((
                ("妊娠期肝内胆汁淤积症待评估", "high"),
                ("妊娠期糖尿病", "high"),
                ("高龄初产", "medium"),
            ), start=1)
        ]}
        memory = SharedClinicalMemory(case, facts); memory.ingest_plan(plan)
        counter_review = {
            "reviewed_problem_ids": ["PR001", "PR002", "PR003"],
            "issues": [{"problem_id": "PR002", "code": "MISSING_DISCRIMINATING_EVIDENCE",
                        "fact_ids": [diagnosis_ids[1]], "note": "需结合血糖记录核对。"}],
        }
        memory.annotate_counter_review(counter_review)
        writer_memory = memory.render("writer")
        client = CaptureClient(); audit = {}
        run_note_writer(case, facts, plan, [], client, "text", Budget(1), 5,
                        prompt_profile="contrastive_v2", prompt_audit=audit,
                        counter_review=counter_review,
                        shared_memory_context=writer_memory)
        payload = json.loads(client.messages[1]["content"])
        injected = payload["shared_decision_cards"]
        self.assertEqual(len(payload["sources"]), len(case.sources))
        self.assertEqual([card["problem_id"] for card in injected["decision_cards"]],
                         ["PR001", "PR002", "PR003"])
        self.assertEqual(injected["decision_cards"][1]["counter_evidence_review"][0]["code"],
                         "MISSING_DISCRIMINATING_EVIDENCE")
        problem_review = payload["problem_evidence_context"]["problems"][1]
        self.assertEqual(problem_review["counter_evidence_review"]["issues"][0]["code"],
                         "MISSING_DISCRIMINATING_EVIDENCE")
        focus = injected["case_clinical_focus"]
        diagnosis_text = " ".join(item["fact_text"]
                                  for item in focus["documented_diagnosis_fact_refs"])
        for marker in ("妊娠期糖尿病", "高龄", "体外受精", "贫血"):
            self.assertIn(marker, diagnosis_text)
        self.assertTrue(all("assertion" in item
                            for item in focus["documented_diagnosis_fact_refs"]))
        plan_ref = focus["explicit_plan_fact_refs"]
        self.assertEqual(len(plan_ref), 1)
        self.assertEqual(plan_ref[0]["source_id"], "P0049")
        self.assertIn("肝功能、总胆汁酸、凝血功能、血糖谱、胎心监护及产科超声",
                      plan_ref[0]["fact_text"])
        source_ids = {source.source_id for source in case.sources}
        for group in focus.values():
            if not isinstance(group, list):
                continue
            for ref in group:
                self.assertIn(ref["source_id"], source_ids)
        writer_audit = audit["writer"]
        self.assertEqual(writer_audit["problem_context_count"], 3)
        self.assertEqual(writer_audit["shared_memory_card_count"], 3)
        self.assertEqual(writer_audit["shared_memory_problem_ids"],
                         ["PR001", "PR002", "PR003"])
        self.assertEqual(writer_audit["clinical_focus_counts"]["explicit_plan_fact_refs"], 1)
        self.assertEqual(writer_audit["clinical_focus_truncated_reference_counts"], {})

    def test_real_practice_focus_preserves_corrections_treatment_and_plan_timing(self):
        root = ROOT / "information" / "临床智录_练习病历数据集JSON"
        rendered = {}
        for prefix in ("01_", "02_", "03_", "04_", "05_", "06_"):
            case = load_input_cases(next(root.glob(f"{prefix}*.json"))).cases[0]
            facts = number_facts(organize_facts(case))
            focus = SharedClinicalMemory(case, facts).render("writer")["case_clinical_focus"]
            source_text = {source.source_id: source.content for source in case.sources}
            for group in focus.values():
                if not isinstance(group, list):
                    continue
                for ref in group:
                    self.assertIn(ref["source_id"], source_text)
                    if "fact_text" in ref:
                        self.assertIn(ref["fact_text"], source_text[ref["source_id"]])
            rendered[prefix] = focus

        self.assertIn("无发绀", rendered["01_"]["correction_fact_refs"][0]["fact_text"])
        self.assertTrue(any(
            "痰涂片及培养" in ref["fact_text"]
            for ref in rendered["01_"]["explicit_plan_fact_refs"]
        ))
        self.assertTrue(any(ref["fact_id"] == "F014"
                            for ref in rendered["02_"]["treatment_history_fact_refs"]))
        self.assertTrue(any(ref["fact_id"] == "F014"
                            for ref in rendered["05_"]["treatment_history_fact_refs"]))
        self.assertEqual(rendered["04_"]["explicit_plan_fact_refs"], [])
        self.assertIn("为行内镜下黏膜切除术收入我科",
                      rendered["06_"]["explicit_plan_fact_refs"][0]["fact_text"])
        self.assertEqual(rendered["06_"]["explicit_plan_fact_refs"][0]["role"],
                         "documented_plan_or_followup")
        self.assertNotIn("status", rendered["06_"]["explicit_plan_fact_refs"][0])

    def test_plan_focus_includes_pending_actions_but_excludes_resolved_history(self):
        include = {
            "S1": "入院后待复查肝功能。",
            "S2": "拟复查CT。",
            "S3": "需复查血糖。",
            "S4": "嘱定期复查肝功能。",
            "S5": "为行内镜下切除术收入我科。",
        }
        exclude = {
            "S6": "近期无生育计划。",
            "S7": "已按计划免疫接种。",
            "S8": "原计划行手术，现已取消。",
            "S9": "外院曾建议复查CT，现已完成复查。",
            "S10": "既往为行手术收入院，术后出院。",
        }
        sources = [
            Source(source_id, text, "入院记录.现病史")
            for source_id, text in {**include, **exclude}.items()
        ]
        sources.extend([
            Source("S11", "原计划手术，现已取消。", "入院记录.诊疗计划"),
            Source("S12", "3年前拟行手术但患者拒绝。", "入院记录.治疗计划"),
            Source("S13", "拟行手术，现暂缓实施。", "入院记录.处理计划"),
            Source("S14", "一周后门诊复诊。", "入院记录.出院医嘱"),
            Source("S15", "3年前拟行手术。", "入院记录.既往史"),
            Source("S16", "既往已行PCI，本次拟行冠脉造影。", "入院记录.诊疗计划"),
            Source("S17", "3年前拟行手术。", "入院记录.现病史"),
            Source("S18", "出院后3天复查CT提示病灶吸收。", "入院记录.现病史"),
            Source("S19", "拟行手术，尚未实施。", "入院记录.诊疗计划"),
            Source("S20", "原计划手术，未实施。", "入院记录.诊疗计划"),
            Source("S21", "拟行手术，患者拒绝，尚未实施。", "入院记录.诊疗计划"),
            Source("S22", "动态监测血压。", "入院记录.医嘱"),
        ])
        case = Case("PLAN-BOUNDARY", "T0", sources)
        facts = number_facts(organize_facts(case))
        focus = SharedClinicalMemory(case, facts).render("writer")["case_clinical_focus"]
        plan_sources = {ref["source_id"] for ref in focus["explicit_plan_fact_refs"]}
        self.assertEqual(plan_sources, set(include) | {"S14", "S16", "S19", "S22"})
        mixed = next(ref for ref in focus["explicit_plan_fact_refs"]
                     if ref["source_id"] == "S16")
        self.assertEqual(mixed["fact_text"], "本次拟行冠脉造影")

    def test_writer_focus_navigation_is_bounded_without_discarding_case_fact_index(self):
        case = Case("FOCUS-BOUND", "T0", [
            Source(f"S{index:03d}", f"检验指标{index}为{index}.0mmol/L。", "入院记录.辅助检查")
            for index in range(1, 41)
        ])
        facts = number_facts(organize_facts(case))
        memory = SharedClinicalMemory(case, facts)
        focus = memory.render("writer")["case_clinical_focus"]
        self.assertEqual(len(focus["key_measurement_fact_refs"]), 32)
        self.assertEqual(focus["truncated_reference_counts"]["key_measurement_fact_refs"], 8)
        self.assertEqual(memory.audit()["fact_index_count"], 40)

    def test_v2_guidance_memory_pipeline_logs_stage_tokens_and_elapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_path = root / "input.json"
            output = root / "output.json"; log = root / "log.json"; script = root / "responses.json"
            input_path.write_text(json.dumps([{"case_id": "C1", "record_cutoff": "T0",
                "sources": [{"source_id": "S1", "content": "PCI术后再发胸痛。"}]}],
                ensure_ascii=False), encoding="utf-8")
            responses = [_plan(), {"reviewed_problem_ids": ["PR001"], "issues": []}, _draft()]
            script.write_text(json.dumps([{"content": json.dumps(item, ensure_ascii=False),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
                for item in responses], ensure_ascii=False), encoding="utf-8")
            code = main(["--input", str(input_path), "--output", str(output), "--log", str(log),
                         "--mode", "mock", "--mock-responses", str(script),
                         "--pipeline", "clinical_pipeline", "--clinical-verification", "off",
                         "--clinical-revision", "off", "--clinical-agents", "always",
                         "--clinical-prompt-profile", "contrastive_v2",
                         "--clinical-writing-guidance", "rules_examples",
                         "--shared-clinical-memory", "on"])
            self.assertEqual(code, 0)
            saved = json.loads(log.read_text(encoding="utf-8")); case_log = saved["cases"][0]
            clinical = case_log["clinical"]
            self.assertEqual(case_log["token_usage"]["total_tokens"], 45)
            self.assertEqual([event["stage"] for event in case_log["stage_calls"]],
                             ["planner", "counter_evidence", "writer"])
            self.assertTrue(all(event["elapsed_ms"] >= 0 for event in case_log["stage_calls"]))
            self.assertEqual(set(clinical["prompt_audit"]), {"planner", "counter_evidence", "writer"})
            self.assertTrue(clinical["shared_clinical_memory"]["enabled"])
            self.assertEqual(clinical["writing_guidance"]["mode"], "rules_examples")
            self.assertIn("prompts/contrastive_v2/planner_system.txt", saved["prompt_manifest"]["files"])

    def test_v2_full_five_stage_prompt_routing_preserves_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); input_path = root / "input.json"
            output = root / "output.json"; log = root / "log.json"; script = root / "responses.json"
            input_path.write_text(json.dumps([{"case_id": "C1", "record_cutoff": "T0",
                "sources": [{"source_id": "S1", "content": "PCI术后再发胸痛。"}]}],
                ensure_ascii=False), encoding="utf-8")
            verifier = {"checked_claim_ids": ["C001", "C002", "C003", "C004"],
                        "issues": [{"claim_id": "C003", "code": "UNSAFE_RECOMMENDATION",
                                    "severity": "high", "hint": "保持医生审核语气。",
                                    "source_ids": []}]}
            revision = {"edits": [{"claim_id": "C003", "section": "plan", "action": "replace",
                                     "replacement": "后续评估方案需由医生结合病情审核。"}],
                        "review_items_to_add": []}
            responses = [_plan(), {"reviewed_problem_ids": ["PR001"], "issues": []},
                         _draft(), verifier, revision]
            script.write_text(json.dumps([{"content": json.dumps(item, ensure_ascii=False)}
                                          for item in responses], ensure_ascii=False), encoding="utf-8")
            code = main(["--input", str(input_path), "--output", str(output), "--log", str(log),
                         "--mode", "mock", "--mock-responses", str(script),
                         "--pipeline", "clinical_pipeline", "--clinical-agents", "always",
                         "--clinical-prompt-profile", "contrastive_v2",
                         "--clinical-response-format", "json_schema"])
            self.assertEqual(code, 0)
            saved = json.loads(log.read_text(encoding="utf-8")); case_log = saved["cases"][0]
            self.assertEqual([event["stage"] for event in case_log["stage_calls"]],
                             ["planner", "counter_evidence", "writer", "verifier", "reviser"])
            self.assertTrue(all(event["schema_validation_success"]
                                for event in case_log["stage_calls"]))
            self.assertEqual(set(case_log["clinical"]["prompt_audit"]),
                             {"planner", "counter_evidence", "writer", "verifier", "reviser"})


if __name__ == "__main__":
    unittest.main()
