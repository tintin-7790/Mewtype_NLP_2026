import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.smoke_graph_retrieval import DEFAULT_INPUT, build_smoke_report, main


class GraphRetrievalSmokeTests(unittest.TestCase):
    def test_practice03_ppr_smoke_executes_real_graph_path(self):
        report = build_smoke_report(DEFAULT_INPUT, top_k=5, seed_k=5, ppr_iterations=20)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["case_id"], "PRACTICE03")
        self.assertEqual(report["clinical_problem_count"], 3)
        self.assertGreater(report["clinical_relation_edge_count"], 0)
        self.assertEqual(report["ppr_iterations"], 20)
        self.assertTrue(report["ppr_scores"])
        self.assertTrue(report["ranked_fact_ids"])
        self.assertTrue(report["selected_source_ids"])
        self.assertFalse(report["fallback_status"])
        self.assertTrue(report["ranking_changed"])

    def test_smoke_cli_writes_report_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "graph-smoke.json"
            with redirect_stdout(io.StringIO()):
                code = main(["--input", str(DEFAULT_INPUT), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
