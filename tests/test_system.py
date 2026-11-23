import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.errors import TaskError
from src.facts import organize_facts
from src.practice_adapter import adapt_file
from src.protocol import Case, Source, load_standard_cases


ROOT = Path(__file__).resolve().parent.parent


def record(source="S01"):
    return {"case_features": "发热38.5℃。", "diagnostic_discussion": "现有资料支持发热，病因待查。",
            "plan": "建议结合病情进一步检查并观察体温变化。",
            "evidence": [{"statement": "发热38.5℃。", "source_ids": [source], "kind": "fact"}],
            "review_items": []}


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="循证 智录 ")
        self.dir = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def write(self, name, value):
        path = self.dir / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8"); return path
    def standard(self, count=1):
        return [{"case_id": f"C{i+1}", "record_cutoff": "T0",
                 "sources": [{"source_id": "S01", "content": "入院前发热，最高38.5℃。"}]} for i in range(count)]
    def run_main(self, inp, responses, extra=None, cwd=None):
        out = self.dir / "结果 输出.json"; script = self.write("响应.json", responses)
        cmd = [sys.executable, str(ROOT/"main.py"), "--input", str(inp), "--output", str(out),
               "--mode", "mock", "--mock-responses", str(script)] + (extra or [])
        done = subprocess.run(cmd, cwd=cwd or ROOT, text=True, capture_output=True)
        return done, out

    def test_standard_single_and_atomic_repeat(self):
        inp=self.write("输入.json",self.standard()); done,out=self.run_main(inp,[{"content":json.dumps(record(),ensure_ascii=False)}])
        self.assertEqual(done.returncode,0); self.assertEqual(json.loads(out.read_text())[0]["status"],"success")
        done2,_=self.run_main(inp,[{"content":json.dumps(record(),ensure_ascii=False)}]); self.assertEqual(done2.returncode,0)

    def test_multiple_preserves_order(self):
        inp=self.write("i.json",self.standard(2)); payload=json.dumps(record(),ensure_ascii=False)
        done,out=self.run_main(inp,[{"content":payload},{"content":payload}]); self.assertEqual(done.returncode,0)
        self.assertEqual([x["case_id"] for x in json.loads(out.read_text())],["C1","C2"])

    def test_empty_array(self):
        inp=self.write("i.json",[]); done,out=self.run_main(inp,[]); self.assertEqual(done.returncode,0); self.assertEqual(json.loads(out.read_text()),[])

    def test_corrupt_and_duplicate_are_task_errors(self):
        bad=self.dir/"bad.json"; bad.write_text("{",encoding="utf-8"); done,_=self.run_main(bad,[]); self.assertEqual(done.returncode,2)
        inp=self.standard(2); inp[1]["case_id"]="C1"; done,_=self.run_main(self.write("dup.json",inp),[]); self.assertEqual(done.returncode,2)

    def test_missing_real_config_fails_without_fallback(self):
        inp=self.write("i.json",self.standard()); out=self.dir/"o.json"
        done=subprocess.run([sys.executable,str(ROOT/"main.py"),"--input",str(inp),"--output",str(out),"--mode","competition"],capture_output=True)
        self.assertEqual(done.returncode,2); self.assertFalse(out.exists())

    def test_single_case_error_continues(self):
        inp=self.write("i.json",self.standard(2)); payload=json.dumps(record(),ensure_ascii=False)
        done,out=self.run_main(inp,[{"exception":"timeout"},{"content":payload}]); values=json.loads(out.read_text())
        self.assertEqual([x["status"] for x in values],["failed","success"])

    def test_abnormal_model_response_is_case_failure(self):
        inp=self.write("i.json",self.standard()); done,out=self.run_main(inp,[{"exception":"gateway unavailable"}])
        self.assertEqual(done.returncode,0); self.assertEqual(json.loads(out.read_text())[0]["error"]["code"],"MODEL_ERROR")

    def test_invalid_json_missing_body_and_source(self):
        inp=self.write("i.json",self.standard(3))
        missing=record(); missing["plan"]=""
        wrong=record("NOPE")
        done,out=self.run_main(inp,[{"content":"not-json"},{"content":json.dumps(missing,ensure_ascii=False)},{"content":json.dumps(wrong,ensure_ascii=False)}])
        values=json.loads(out.read_text())
        self.assertEqual(done.returncode,0); self.assertEqual([x["status"] for x in values],["failed","failed","success"])
        self.assertEqual(values[2]["medical_record"]["evidence"],[])
        self.assertTrue(any("来源编号无效" in item for item in values[2]["medical_record"]["review_items"]))

    def test_different_working_directory_and_unicode_paths(self):
        inp=self.write("空 格/输入.json",self.standard()); payload=json.dumps(record(),ensure_ascii=False)
        done,out=self.run_main(inp,[{"content":payload}],cwd=self.dir); self.assertEqual(done.returncode,0); self.assertTrue(out.exists())

    def test_practice_adapter_preserves_path_and_text(self):
        raw={"入院记录":{"一般项目":{"体温":"37.1℃"},"现病史":"既往最高38.5℃"}}
        inp=self.write("练习.json",raw); out=self.dir/"adapted.json"; adapt_file(inp,out,"P1","T0")
        sources=json.loads(out.read_text())[0]["sources"]
        self.assertEqual([x["content"] for x in sources],["37.1℃","既往最高38.5℃"])
        self.assertTrue(all(x["source_id_origin"]=="program_derived" and x["field_path"].startswith("入院记录.") for x in sources))
        cli_out=self.dir/"cli adapted.json"
        done=subprocess.run([sys.executable,str(ROOT/"tools/adapt_practice.py"),"--input",str(inp),"--output",str(cli_out),
                             "--case-id","P1","--record-cutoff","T0"],cwd=self.dir,capture_output=True)
        self.assertEqual(done.returncode,0); self.assertTrue(cli_out.exists())

    def test_practice_adapter_requires_explicit_cutoff_and_rejects_bad_json(self):
        inp=self.write("练习.json",{"入院记录":{"主诉":"咳嗽"}})
        with self.assertRaises(TaskError): adapt_file(inp,self.dir/"x.json","P1","")
        bad=self.dir/"badpractice.json"; bad.write_text("[]]",encoding="utf-8")
        with self.assertRaises(TaskError): adapt_file(bad,self.dir/"x.json","P1","T0")

    def test_cough_regression_facts_keep_time_plan_and_correction(self):
        case=Case("C","T0",[
            Source("S1","1天前发热，体温最高38.5℃"), Source("S2","入院时体温37.1℃"),
            Source("S3","否认结核患者密切接触史，但与患结核邻居有日常往来"),
            Source("S4","口唇无发绀（原记录发绀经复核为无发绀）"),
            Source("S5","待完善痰培养"), Source("S6","既往予抗感染治疗")])
        facts=organize_facts(case)
        by_source={key:[x for x in facts if x["source_id"]==key] for key in {x["source_id"] for x in facts}}
        self.assertEqual(by_source["S1"][0]["value"],"38.5"); self.assertEqual(by_source["S2"][0]["value"],"37.1")
        self.assertEqual(by_source["S5"][0]["status"],"planned"); self.assertIsNotNone(by_source["S4"][0]["correction"])
        self.assertEqual([x["assertion"] for x in by_source["S3"]],["negative","positive"])
        self.assertEqual(by_source["S6"][0]["status"],"completed")

    def test_evidence_pipeline_three_calls(self):
        inp=self.write("i.json",self.standard()); facts={"facts":[{"source_id":"S01","fact_text":"最高38.5℃"}]}
        responses=[{"content":json.dumps(facts,ensure_ascii=False)}, {"content":json.dumps(record(),ensure_ascii=False)},
                   {"content":json.dumps({"valid":True,"issues":[],"corrected_record":None},ensure_ascii=False)}]
        done,out=self.run_main(inp,responses,["--pipeline","evidence_pipeline"]); self.assertEqual(done.returncode,0)
        log=json.loads((out.parent/"run_log.json").read_text()); self.assertEqual(log["cases"][0]["calls"],3)

    def test_evidence_pipeline_optional_retrieval_is_logged(self):
        inp=self.write("i.json",self.standard()); facts={"facts":[{"source_id":"S01","fact_text":"最高38.5℃"}]}
        responses=[{"content":json.dumps(facts,ensure_ascii=False)}, {"content":json.dumps(record(),ensure_ascii=False)},
                   {"content":json.dumps({"valid":True,"issues":[],"corrected_record":None},ensure_ascii=False)}]
        retrieval=self.write("retrieval.json",{"provider":"mock","dimension":32,"top_k":2,"min_similarity":-1})
        done,out=self.run_main(inp,responses,["--pipeline","evidence_pipeline","--retrieval-config",str(retrieval)])
        self.assertEqual(done.returncode,0)
        value=json.loads(out.read_text())[0]["medical_record"]
        self.assertEqual(value["evidence"][0]["source_ids"],["S01"])
        log=json.loads((out.parent/"run_log.json").read_text())
        self.assertEqual(log["retrieval"]["provider"],"mock")
        self.assertEqual(log["cases"][0]["retrieval"]["embedding_calls"],2)


if __name__ == "__main__": unittest.main()
