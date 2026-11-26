import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app import main
from src.errors import TaskError


def _planner():
    return {"problems": [{
        "problem_id": "PR001", "name": "发热伴咳嗽", "priority": "high",
        "origin": "mixed", "supporting_facts": ["F001"],
        "contradicting_facts": [], "differential_diagnoses": [],
        "plan_goals": ["明确原因"], "retrieval_queries": ["发热 咳嗽"],
    }]}


def _agent():
    return {"reviewed_problem_ids": ["PR001"], "issues": []}


def _writer():
    return {
        "case_features": "患者发热伴咳嗽。",
        "diagnostic_discussion": "发热伴咳嗽原因待查。",
        "plan": "建议由医生结合临床情况决定后续评估。",
        "evidence": [{"statement": "患者发热伴咳嗽。",
                      "source_ids": ["S001"], "kind": "fact"}],
        "review_items": [],
    }


class SuccessfulOfficialEmbedding:
    def __init__(self):
        self.total_calls = 0
        self.total_texts = 0
        self.elapsed_ms = 0.0

    def embed(self, texts):
        self.total_calls += 1
        self.total_texts += len(texts)
        return [[1.0, 0.0] for _ in texts]


class FailingOfficialEmbedding(SuccessfulOfficialEmbedding):
    error_code = "OFFICIAL_EMBEDDING_UNAVAILABLE"

    def embed(self, texts):
        self.total_calls += 1
        self.total_texts += len(texts)
        raise TaskError(self.error_code, "test-only embedding failure")


class TimeoutOfficialEmbedding(FailingOfficialEmbedding):
    error_code = "EMBEDDING_TIMEOUT"


class RetrievalStrategyTests(unittest.TestCase):
    def _run(self, strategy, embedding=None, lexical_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            log_path = root / "run_log.json"
            responses_path = root / "responses.json"
            retrieval_path = root / "retrieval.json"
            input_path.write_text(json.dumps([{
                "case_id": "C1", "record_cutoff": "T0",
                "sources": [
                    {"source_id": "S001", "content": "患者发热伴咳嗽。"},
                    {"source_id": "S002", "content": "患者否认胸痛。"},
                ],
            }], ensure_ascii=False), encoding="utf-8")
            responses_path.write_text(json.dumps([
                {"content": json.dumps(value, ensure_ascii=False)}
                for value in (_planner(), _agent(), _writer())
            ], ensure_ascii=False), encoding="utf-8")
            retrieval_path.write_text('{"provider":"official"}', encoding="utf-8")
            arguments = [
                "--input", str(input_path), "--output", str(output_path),
                "--log", str(log_path), "--mode", "mock",
                "--mock-responses", str(responses_path),
                "--pipeline", "clinical_pipeline",
                "--clinical-verification", "off", "--clinical-revision", "off",
                "--clinical-agents", "always", "--graph-retrieval",
                "off" if strategy == "no_graph" else "ppr",
                "--retrieval-strategy", strategy,
            ]
            load_patch = None
            if embedding is not None:
                arguments.extend(["--retrieval-config", str(retrieval_path)])
                load_patch = patch(
                    "src.app.load_embedding_client",
                    return_value=(embedding, {"provider": "official-test", "model": "bge-m3-test",
                                              "top_k": 2, "min_similarity": -1.0,
                                              "max_chunk_chars": 240}),
                )
            lexical_patch = (patch("src.pipeline.LexicalEvidenceRetriever.retrieve",
                                   side_effect=TaskError("LEXICAL_RETRIEVAL_ERROR", "test-only"))
                             if lexical_failure else None)
            contexts = [context for context in (load_patch, lexical_patch) if context is not None]
            for context in contexts:
                context.start()
            try:
                code = main(arguments)
            finally:
                for context in reversed(contexts):
                    context.stop()
            output = json.loads(output_path.read_text(encoding="utf-8"))
            log = json.loads(log_path.read_text(encoding="utf-8"))
            return code, output, log

    def test_bge_available_uses_dense_graph(self):
        code, output, log = self._run("dense_graph", SuccessfulOfficialEmbedding())
        graph = log["cases"][0]["clinical"]["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertEqual(graph["retrieval_strategy"], "dense_graph")
        self.assertTrue(graph["graph_executed"])
        self.assertEqual(graph["embedding_provider"], "SuccessfulOfficialEmbedding")
        self.assertEqual(graph["retrieval_fallbacks"], [])

    def test_bge_missing_falls_back_to_lexical_graph(self):
        code, output, log = self._run("dense_graph", FailingOfficialEmbedding())
        graph = log["cases"][0]["clinical"]["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertEqual(graph["retrieval_strategy"], "lexical_graph")
        self.assertEqual(graph["seed_provider"], "lexical_bm25_cpu")
        self.assertTrue(graph["graph_executed"])
        self.assertIn("OFFICIAL_EMBEDDING_UNAVAILABLE", graph["graph_fallback_reason"])

    def test_bge_timeout_falls_back_to_lexical_graph(self):
        code, output, log = self._run("dense_graph", TimeoutOfficialEmbedding())
        graph = log["cases"][0]["clinical"]["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertEqual(graph["retrieval_strategy"], "lexical_graph")
        self.assertIn("EMBEDDING_TIMEOUT", graph["graph_fallback_reason"])

    def test_lexical_graph_runs_without_embedding(self):
        code, output, log = self._run("lexical_graph")
        graph = log["cases"][0]["clinical"]["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertIsNone(graph["embedding_provider"])
        self.assertEqual(graph["seed_provider"], "lexical_bm25_cpu")
        self.assertEqual(graph["retrieval_strategy"], "lexical_graph")
        self.assertTrue(graph["selected_source_ids"])

    def test_lexical_failure_falls_back_to_no_graph_and_agent_still_runs(self):
        code, output, log = self._run("lexical_graph", lexical_failure=True)
        clinical = log["cases"][0]["clinical"]
        graph = clinical["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertEqual(graph["retrieval_strategy"], "no_graph")
        self.assertFalse(graph["graph_executed"])
        self.assertIn("LEXICAL_RETRIEVAL_ERROR", graph["graph_fallback_reason"])
        self.assertTrue(clinical["adaptive_agent"]["completed"])
        self.assertEqual(clinical["adaptive_agent"]["retrieval_candidate_count"], 0)

    def test_explicit_no_graph_keeps_agent_and_skips_retrieval(self):
        code, output, log = self._run("no_graph")
        clinical = log["cases"][0]["clinical"]
        graph = clinical["graph_retrieval"]
        self.assertEqual((code, output[0]["status"], log["total_calls"]), (0, "success", 3))
        self.assertEqual(graph["retrieval_strategy"], "no_graph")
        self.assertFalse(graph["graph_requested"])
        self.assertTrue(clinical["adaptive_agent"]["completed"])


if __name__ == "__main__":
    unittest.main()
