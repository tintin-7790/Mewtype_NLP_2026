import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.clinical_graph import build_clinical_graph
from src.clinical_planner import number_facts
from src.errors import TaskError
from src.facts import organize_facts
from src.input_adapter import load_input_cases, normalize_record_cutoff
from src.practice_adapter import adapt_file
from src.protocol import CaseError, load_standard_cases


ROOT = Path(__file__).resolve().parent.parent
PRACTICE_ROOT = ROOT / "information" / "临床智录_练习病历数据集JSON"


def practice_file(number: str) -> Path:
    matches = sorted(PRACTICE_ROOT.glob(f"{number}_*.json"))
    if len(matches) != 1:
        raise AssertionError(f"expected one practice file for {number}, got {len(matches)}")
    return matches[0]


class InputAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="input adapter ")
        self.directory = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, name: str, value) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def formal(case_id="C1", cutoff="T0", source_id="S-ORIGINAL"):
        return {"case_id": case_id, "record_cutoff": cutoff,
                "sources": [{"source_id": source_id, "field_path": "原始.字段",
                             "content": "原始内容", "source_id_origin": "provided"}]}

    def assert_task_error(self, code: str, callback):
        with self.assertRaises(TaskError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_formal_input_is_parsed_unchanged(self):
        path = self.write_json("formal.json", [self.formal()])
        expected = load_standard_cases(path)
        loaded = load_input_cases(path)
        self.assertEqual(loaded.detected_format, "competition")
        self.assertFalse(loaded.audit["adapted"])
        self.assertEqual(loaded.cases, expected)
        case = loaded.cases[0]
        self.assertEqual(case.case_id, "C1")
        self.assertEqual(case.record_cutoff, "T0")
        self.assertEqual(case.sources[0].source_id, "S-ORIGINAL")

    def test_multiple_formal_cases_preserve_order(self):
        raw = [self.formal("C2", "T2", "X9"), self.formal("C1", "T1", "A1")]
        loaded = load_input_cases(self.write_json("formal.json", raw))
        self.assertEqual([case.case_id for case in loaded.cases], ["C2", "C1"])
        self.assertEqual([case.sources[0].source_id for case in loaded.cases], ["X9", "A1"])

    def test_practice_array_is_detected_and_preserves_order(self):
        raw = [
            {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}},
            {"入院记录": {"一般项目": {"记录时间": "2026‑01‑02 10:00"}, "主诉": "胸闷"}},
        ]
        loaded = load_input_cases(self.write_json("input.json", raw))
        self.assertEqual(loaded.detected_format, "practice")
        self.assertEqual([case.case_id for case in loaded.cases], ["PRACTICE001", "PRACTICE002"])
        self.assertEqual([case.record_cutoff for case in loaded.cases],
                         ["2026-01-01 09:00", "2026-01-02 10:00"])
        self.assertEqual([case.sources[-1].content for case in loaded.cases], ["头晕", "胸闷"])
        self.assertEqual([mapping["case_id_origin"] for mapping in loaded.audit["case_mappings"]],
                         ["array_sequence", "array_sequence"])

    def test_practice_array_prefers_provided_case_id(self):
        raw = [
            {"case_id": "ORIGINAL-7", "入院记录": {
                "一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}},
            {"入院记录": {"一般项目": {"记录时间": "2026-01-02 10:00"}, "主诉": "胸闷"}},
        ]
        loaded = load_input_cases(self.write_json("input.json", raw))
        self.assertEqual([case.case_id for case in loaded.cases], ["ORIGINAL-7", "PRACTICE002"])
        self.assertEqual(loaded.audit["case_mappings"][0]["case_id_origin"], "provided")

    def test_mixed_protocol_array_is_rejected(self):
        practice = {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}}
        path = self.write_json("mixed.json", [self.formal(), practice])
        self.assert_task_error("INPUT_FORMAT_MIXED", lambda: load_input_cases(path))

    def test_explicit_modes_are_strict_for_arrays(self):
        practice = {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}}
        practice_path = self.write_json("practice.json", [practice])
        self.assert_task_error("INPUT_PROTOCOL_ERROR",
                               lambda: load_input_cases(practice_path, "competition"))
        formal_path = self.write_json("formal.json", [self.formal()])
        self.assert_task_error("PRACTICE_PROTOCOL_ERROR",
                               lambda: load_input_cases(formal_path, "practice"))

    def test_multi_practice_overrides_are_rejected(self):
        record = {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}}
        path = self.write_json("practice.json", [record, record])
        self.assert_task_error("PRACTICE_OVERRIDE_SCOPE_INVALID",
                               lambda: load_input_cases(path, practice_case_id="ONLY_ONE"))

    def test_practice_01_02_03_are_detected_and_normalized(self):
        expected = {
            "01": ("PRACTICE01", "2026-05-03 20:30", 50),
            "02": ("PRACTICE02", "2026-06-02 10:35", 56),
            "03": ("PRACTICE03", "2026-05-08 14:25", 52),
        }
        for number, values in expected.items():
            with self.subTest(number=number):
                loaded = load_input_cases(practice_file(number))
                case = loaded.cases[0]
                self.assertEqual(loaded.detected_format, "practice")
                self.assertTrue(loaded.audit["adapted"])
                self.assertEqual((case.case_id, case.record_cutoff, len(case.sources)), values)

    def test_auto_sources_match_legacy_adapter(self):
        for number in ("01", "02", "03"):
            with self.subTest(number=number):
                path = practice_file(number)
                automatic = load_input_cases(path).cases[0]
                legacy = self.directory / f"legacy-{number}.json"
                adapt_file(path, legacy, automatic.case_id, automatic.record_cutoff)
                legacy_sources = json.loads(legacy.read_text(encoding="utf-8"))[0]["sources"]
                auto_sources = [source.__dict__ for source in automatic.sources]
                self.assertEqual(auto_sources, legacy_sources)

    def test_derived_source_ids_are_deterministic_and_ordered(self):
        case = load_input_cases(practice_file("03")).cases[0]
        self.assertEqual([source.source_id for source in case.sources],
                         [f"P{index:04d}" for index in range(1, 53)])
        self.assertTrue(all(source.source_id_origin == "program_derived" for source in case.sources))

    def test_unicode_date_hyphens_are_validated(self):
        self.assertEqual(normalize_record_cutoff("2026‑06‑02 10:35"), "2026-06-02 10:35")
        self.assertEqual(normalize_record_cutoff("2026—06—02T10:35:09"), "2026-06-02 10:35:09")
        self.assert_task_error("PRACTICE_RECORD_CUTOFF_INVALID",
                               lambda: normalize_record_cutoff("2026-02-30 10:35"))

    def test_invalid_json_is_task_error(self):
        path = self.directory / "broken.json"
        path.write_text("{", encoding="utf-8")
        self.assert_task_error("INPUT_INVALID_JSON", lambda: load_input_cases(path))

    def test_formal_missing_sources_remains_case_level_error(self):
        path = self.write_json("formal.json", [{"case_id": "C1", "record_cutoff": "T0"}])
        loaded = load_input_cases(path)
        self.assertEqual(loaded.detected_format, "competition")
        self.assertIsInstance(loaded.cases[0], CaseError)
        self.assertEqual(loaded.cases[0].code, "INVALID_SOURCES")

    def test_duplicate_practice_case_ids_are_rejected(self):
        record = {"case_id": "DUP", "入院记录": {
            "一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}}
        path = self.write_json("practice.json", [record, record])
        self.assert_task_error("DUPLICATE_CASE_ID", lambda: load_input_cases(path))

    def test_missing_record_time_requires_explicit_override(self):
        raw = {"入院记录": {"一般项目": {}, "主诉": "头晕"}}
        path = self.write_json("03_practice.json", raw)
        self.assert_task_error("PRACTICE_RECORD_CUTOFF_MISSING", lambda: load_input_cases(path))
        loaded = load_input_cases(path, practice_record_cutoff="2026-05-08 14:25")
        self.assertEqual(loaded.cases[0].record_cutoff, "2026-05-08 14:25")
        self.assertEqual(loaded.audit["case_mappings"][0]["record_cutoff_origin"], "explicit_cli")

    def test_unknown_format_and_weak_practice_shape_are_rejected(self):
        unknown = self.write_json("unknown.json", {"content": "not a protocol"})
        self.assert_task_error("INPUT_FORMAT_UNKNOWN", lambda: load_input_cases(unknown))
        weak = self.write_json("03_weak.json", {"入院记录": {"一般项目": {"记录时间": "2026-01-01 10:00"}}})
        self.assert_task_error("PRACTICE_PROTOCOL_ERROR", lambda: load_input_cases(weak))

    def test_formal_like_object_is_not_silently_adapted(self):
        path = self.write_json("single.json", self.formal())
        self.assert_task_error("INPUT_PROTOCOL_ERROR", lambda: load_input_cases(path))
        self.assert_task_error("INPUT_PROTOCOL_ERROR", lambda: load_input_cases(path, "competition"))

    def test_filename_without_sequence_uses_stable_redacted_fingerprint_or_explicit_id(self):
        raw = {"入院记录": {"一般项目": {"记录时间": "2026-01-01 10:00", "姓名": "甲"},
                              "主诉": "头晕"}}
        path = self.write_json("practice.json", raw)
        first = load_input_cases(path)
        second = load_input_cases(path)
        self.assertRegex(first.cases[0].case_id, r"^PRACTICE_AUTO_[0-9A-F]{12}$")
        self.assertEqual(first.cases[0].case_id, second.cases[0].case_id)
        renamed = json.loads(json.dumps(raw, ensure_ascii=False))
        renamed["入院记录"]["一般项目"]["姓名"] = "乙"
        renamed_path = self.write_json("renamed.json", renamed)
        self.assertEqual(first.cases[0].case_id, load_input_cases(renamed_path).cases[0].case_id)
        self.assertEqual(first.audit["case_mappings"][0]["case_id_origin"],
                         "redacted_content_fingerprint")
        loaded = load_input_cases(path, practice_case_id="PRACTICE_CUSTOM")
        self.assertEqual(loaded.cases[0].case_id, "PRACTICE_CUSTOM")

    def test_root_input_name_does_not_prevent_practice_adaptation(self):
        source = practice_file("03")
        copied = self.directory / "input.json"
        copied.write_bytes(source.read_bytes())
        loaded = load_input_cases(copied)
        self.assertEqual(loaded.detected_format, "practice")
        self.assertRegex(loaded.cases[0].case_id, r"^PRACTICE_AUTO_[0-9A-F]{12}$")
        self.assertEqual(loaded.cases[0].record_cutoff, "2026-05-08 14:25")
        self.assertEqual(len(loaded.cases[0].sources), 52)

    def test_source_traceability_and_practice03_graph_audit(self):
        loaded = load_input_cases(practice_file("03"))
        case = loaded.cases[0]
        by_path = {source.field_path: source for source in case.sources}
        self.assertIn("入院记录.主诉", by_path)
        self.assertTrue(all(source.field_path.startswith("入院记录.") for source in case.sources))
        facts = number_facts(organize_facts(case))
        graph = build_clinical_graph(case, facts)
        self.assertEqual(graph.case_id, "PRACTICE03")
        self.assertEqual(graph.stats()["node_types"]["Source"], len(case.sources))
        valid_sources = {source.source_id for source in case.sources}
        self.assertTrue(all(set(edge.evidence_ids) <= valid_sources for edge in graph.edges))

    def test_empty_fields_do_not_create_or_invent_sources(self):
        raw = {"入院记录": {
            "一般项目": {"记录时间": "2026-01-01 09:00", "备注": ""},
            "主诉": "头晕", "辅助检查": {}, "初步诊断": [],
        }}
        case = load_input_cases(self.write_json("input.json", raw)).cases[0]
        paths = {source.field_path for source in case.sources}
        self.assertNotIn("入院记录.一般项目.备注", paths)
        self.assertNotIn("入院记录.辅助检查", paths)
        self.assertNotIn("入院记录.初步诊断", paths)
        self.assertTrue(any(source.content == "头晕" for source in case.sources))

    def test_adaptation_does_not_modify_input_bytes(self):
        source = practice_file("03")
        copied = self.directory / "input.json"
        copied.write_bytes(source.read_bytes())
        before = hashlib.sha256(copied.read_bytes()).hexdigest()
        load_input_cases(copied)
        after = hashlib.sha256(copied.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_windows_style_characters_in_filename_are_not_protocol_signals(self):
        raw = {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}}
        path = self.write_json(r"C:\病例\input.json", raw)
        loaded = load_input_cases(path)
        self.assertEqual(loaded.detected_format, "practice")
        self.assertRegex(loaded.cases[0].case_id, r"^PRACTICE_AUTO_[0-9A-F]{12}$")

    def test_check_input_tool_reports_only_summary(self):
        raw = {"入院记录": {"一般项目": {
            "记录时间": "2026-01-01 09:00", "姓名": "不应输出的姓名"}, "主诉": "隐私主诉"}}
        path = self.write_json("input.json", raw)
        done = subprocess.run([sys.executable, str(ROOT / "tools" / "check_input.py"),
                               "--input", str(path)], cwd=self.directory,
                              text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        summary = json.loads(done.stdout)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["detected_format"], "practice")
        self.assertNotIn("不应输出的姓名", done.stdout)
        self.assertNotIn("隐私主诉", done.stdout)

    def test_practice07_invalid_json_is_reported_without_repair(self):
        path = practice_file("07")
        self.assert_task_error("INPUT_INVALID_JSON", lambda: load_input_cases(path))

    def test_default_main_accepts_practice_without_intermediate_file(self):
        raw = {"入院记录": {"一般项目": {"记录时间": "2026‑05‑08 14:25"}, "主诉": "发热"}}
        input_path = self.write_json("03_direct.json", raw)
        response = {"case_features": "患者诉发热。", "diagnostic_discussion": "发热原因待查。",
                    "plan": "建议结合临床情况进一步评估。",
                    "evidence": [{"statement": "患者诉发热。", "source_ids": ["P0002"], "kind": "fact"}],
                    "review_items": []}
        responses = self.write_json("responses.json", [{"content": json.dumps(response, ensure_ascii=False)}])
        output = self.directory / "output.json"
        before = {path.name for path in self.directory.iterdir()}
        done = subprocess.run([
            sys.executable, str(ROOT / "main.py"), "--input", str(input_path), "--output", str(output),
            "--mode", "mock", "--mock-responses", str(responses),
        ], cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))[0]
        self.assertEqual(result["case_id"], "PRACTICE03")
        self.assertEqual(result["status"], "success")
        log = json.loads((self.directory / "run_log.json").read_text(encoding="utf-8"))
        self.assertEqual(log["input_adapter"]["detected_format"], "practice")
        after = {path.name for path in self.directory.iterdir()}
        self.assertEqual(after - before, {"output.json", "run_log.json"})

    def test_main_processes_practice_array_in_input_order(self):
        raw = [
            {"入院记录": {"一般项目": {"记录时间": "2026-01-01 09:00"}, "主诉": "头晕"}},
            {"入院记录": {"一般项目": {"记录时间": "2026-01-02 10:00"}, "主诉": "胸闷"}},
        ]
        input_path = self.write_json("input.json", raw)
        responses = []
        for symptom in ("头晕", "胸闷"):
            record = {"case_features": f"患者诉{symptom}。", "diagnostic_discussion": f"{symptom}原因待查。",
                      "plan": "建议结合临床情况进一步评估。",
                      "evidence": [{"statement": f"患者诉{symptom}。",
                                    "source_ids": ["P0002"], "kind": "fact"}],
                      "review_items": []}
            responses.append({"content": json.dumps(record, ensure_ascii=False)})
        response_path = self.write_json("responses.json", responses)
        output = self.directory / "output.json"
        done = subprocess.run([
            sys.executable, str(ROOT / "main.py"), "--input", str(input_path), "--output", str(output),
            "--mode", "mock", "--mock-responses", str(response_path),
        ], cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([item["case_id"] for item in result], ["PRACTICE001", "PRACTICE002"])
        self.assertTrue(all(item["status"] == "success" for item in result))

    def test_explicit_practice_mode_does_not_accept_formal_array(self):
        path = self.write_json("formal.json", [self.formal()])
        self.assert_task_error("PRACTICE_PROTOCOL_ERROR", lambda: load_input_cases(path, "practice"))
        self.assert_task_error("PRACTICE_OVERRIDE_SCOPE_INVALID",
                               lambda: load_input_cases(path, practice_case_id="PRACTICE01"))


if __name__ == "__main__":
    unittest.main()
