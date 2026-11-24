import json
import tempfile
import unittest
from pathlib import Path

from tools.eval_clinical_pipeline import main


class EvalClinicalPipelineTests(unittest.TestCase):
    def test_aggregates_automatic_and_human_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            input_value=[{"case_id":"C1","record_cutoff":"T0","sources":[
                {"source_id":"S1","content":"发热38.5℃。"}]}]
            record={"case_features":"发热38.5℃。","diagnostic_discussion":"发热原因待查。",
                    "plan":"建议观察体温。","evidence":[{"statement":"发热38.5℃。",
                    "source_ids":["S1"],"kind":"fact"}],"review_items":[]}
            values={"input.json":input_value,"output.json":[{"case_id":"C1","status":"success",
                    "medical_record":record,"error":None}],"log.json":{"total_calls":1,
                    "token_usage":{"total_tokens":10},"elapsed_ms":5},
                    "annotations.json":{"cases":{"C1":{"supported_fact_claims":2,"total_fact_claims":2,
                    "retrieval_ranks":[1,None]}}}}
            for name,value in values.items():
                (root/name).write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")
            manifest={"runs":[{"name":"A","input":"input.json","output":"output.json",
                               "log":"log.json","adjudication":"annotations.json"}]}
            (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            result=root/"summary.json"
            self.assertEqual(main(["--manifest",str(root/"manifest.json"),"--output",str(result)]),0)
            saved=json.loads(result.read_text(encoding="utf-8"))["runs"][0]
            self.assertEqual(saved["structure_compliance"]["value"],1.0)
            self.assertEqual(saved["three_section_completeness"]["value"],1.0)
            self.assertEqual(saved["body_length_chars"]["count"],1)
            self.assertEqual(saved["manual_metrics"]["retrieval_recall_at_k"]["value"],0.5)
            self.assertEqual(saved["manual_metrics"]["retrieval_mrr"],0.5)


if __name__ == "__main__": unittest.main()
