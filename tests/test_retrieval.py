import json
import tempfile
import unittest
from pathlib import Path

from src.embedding_client import MockEmbeddingClient, OfficialEmbeddingClient, load_embedding_client
from src.errors import TaskError
from src.protocol import Case, Source
from src.retrieval import DenseEvidenceRetriever, build_retrieval_units


class RetrievalTests(unittest.TestCase):
    def case(self, *contents):
        return Case("CASE-A", "T0", [Source(f"S{i+1}", text, f"入院记录.字段{i+1}")
                                      for i, text in enumerate(contents)])

    def test_semantic_paraphrase_retrieval(self):
        query = "患者体温升高至38.5℃"
        supporting = "1天前出现发热，最高38.5℃。"
        distractor = "否认胸痛、心悸。"
        client = MockEmbeddingClient({query: [1, 0], supporting: [0.99, 0.01], distractor: [0, 1]})
        candidates = DenseEvidenceRetriever(self.case(supporting, distractor), client).retrieve(query, top_k=1)
        self.assertEqual(candidates[0].unit.source_id, "S1")

    def test_negation_mismatch_is_not_treated_as_support(self):
        query = "存在胸痛"
        negative = "患者否认胸痛。"
        positive = "患者诉胸痛。"
        client = MockEmbeddingClient({query: [1, 0], negative: [1, 0], positive: [0.9, 0.1]})
        candidates = DenseEvidenceRetriever(self.case(negative, positive), client).retrieve(query, top_k=2)
        self.assertIn("assertion_mismatch", candidates[0].compatibility_warnings)
        self.assertTrue(candidates[1].compatible)

    def test_same_measurement_different_times_is_flagged(self):
        query = "入院时体温37.1℃"
        earlier = "1天前体温最高38.5℃。"
        current = "入院时体温37.1℃。"
        client = MockEmbeddingClient({query: [1, 0], earlier: [1, 0], current: [0.9, 0.1]})
        candidates = DenseEvidenceRetriever(self.case(earlier, current), client).retrieve(query, top_k=2)
        self.assertIn("time_mismatch", candidates[0].compatibility_warnings)
        self.assertTrue(candidates[1].compatible)

    def test_planned_and_completed_states_are_distinguished(self):
        query = "已完成痰培养"
        planned = "待完善痰培养。"
        completed = "已行痰培养。"
        client = MockEmbeddingClient({query: [1, 0], planned: [1, 0], completed: [0.8, 0.2]})
        candidates = DenseEvidenceRetriever(self.case(planned, completed), client).retrieve(query, top_k=2)
        self.assertIn("status_mismatch", candidates[0].compatibility_warnings)
        self.assertTrue(candidates[1].compatible)

    def test_no_evidence_above_threshold(self):
        query = "患者有咯血"
        source = "患者食欲尚可。"
        client = MockEmbeddingClient({query: [1, 0], source: [0, 1]})
        result = DenseEvidenceRetriever(self.case(source), client, min_similarity=0.5).retrieve(query)
        self.assertEqual(result, [])

    def test_long_case_top_k_finds_key_statement(self):
        key = "复核口唇无发绀。"
        long_source = "。".join([f"一般描述{i}" for i in range(30)] + [key])
        query = "复查未发现口唇发绀"
        units = build_retrieval_units(self.case(long_source), max_chunk_chars=80)
        vectors = {unit.text: [0, 1] for unit in units}
        key_unit = next(unit.text for unit in units if "无发绀" in unit.text)
        vectors[query] = [1, 0]; vectors[key_unit] = [1, 0]
        result = DenseEvidenceRetriever(self.case(long_source), MockEmbeddingClient(vectors),
                                        max_chunk_chars=80).retrieve(query, top_k=1)
        self.assertIn("无发绀", result[0].unit.text)

    def test_source_and_query_vectors_are_cached_per_case(self):
        source = "患者咳嗽3天。"; query = "咳嗽病程3日"
        client = MockEmbeddingClient({source: [1, 0], query: [1, 0]})
        retriever = DenseEvidenceRetriever(self.case(source), client)
        retriever.retrieve(query); retriever.retrieve(query)
        self.assertEqual(client.total_calls, 2)
        self.assertEqual(client.total_texts, 2)
        self.assertTrue(retriever.stats()["source_vector_cache"])

    def test_metadata_and_case_boundary_are_preserved(self):
        case = self.case("入院时体温37.1℃。")
        unit = build_retrieval_units(case)[0]
        self.assertEqual((unit.case_id, unit.source_id, unit.field_path, unit.time),
                         ("CASE-A", "S1", "入院记录.字段1", "入院时"))

    def test_official_adapter_fails_before_guessing_protocol(self):
        with self.assertRaises(TaskError) as caught:
            OfficialEmbeddingClient().embed(["test"])
        self.assertEqual(caught.exception.code, "OFFICIAL_EMBEDDING_PROTOCOL_PENDING")

    def test_official_config_rejects_unpublished_http_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.json"
            path.write_text(json.dumps({"provider": "official", "endpoint": "invented"}), encoding="utf-8")
            with self.assertRaises(TaskError) as caught:
                load_embedding_client(path, "competition")
        self.assertEqual(caught.exception.code, "OFFICIAL_EMBEDDING_CONFIG_UNPUBLISHED")


if __name__ == "__main__":
    unittest.main()
