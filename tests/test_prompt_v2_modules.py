import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.app import main, parser
from src.competition_profile import ACTIVE_COMPETITION_PROFILE
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
    def test_research_modules_are_default_off_and_competition_profile_unchanged(self):
        args = parser().parse_args(["--input", "input.json", "--output", "output.json"])
        self.assertEqual(args.clinical_prompt_profile, "contrastive")
        self.assertEqual(args.clinical_writing_guidance, "off")
        self.assertEqual(args.shared_clinical_memory, "off")
        self.assertEqual(ACTIVE_COMPETITION_PROFILE, "baseline_safe")

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

        memory = SharedClinicalMemory(case, _facts()); memory.ingest_plan(_plan())
        on_client = CaptureClient(); on_audit = {}
        run_note_writer(case, _facts(), _plan(), [], on_client, "text", Budget(1), 5,
                        prompt_profile="contrastive_v2", prompt_audit=on_audit,
                        writing_guidance="rules_examples",
                        shared_memory_context=memory.render("writer"))
        on_payload = json.loads(on_client.messages[1]["content"])
        self.assertIn("NOT_PATIENT_EVIDENCE", on_payload["writing_style_context"])
        self.assertEqual(on_payload["shared_decision_cards"]["case_id"], "C1")
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
        rendered = memory.render("writer")
        card = rendered["decision_cards"][0]
        self.assertEqual(card["graph_candidate_ids"], ["F001"])
        self.assertEqual(card["counter_issue_ids"], ["CI001"])
        self.assertEqual(rendered["case_id"], "C1")
        other = SharedClinicalMemory(Case("C2", "T0", [Source("S2", "头晕。")]), [])
        self.assertEqual(other.render("writer")["decision_cards"], [])

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
