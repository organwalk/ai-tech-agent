import unittest

from models import RAGEvalRequest, RAGEvalSample
from services.rag_eval_service import RAGEvaluationService


class FakeEmbeddingService:
    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


class FakeRAGService:
    def __init__(self, index_payload=None, hits_by_query=None):
        self.index_payload = index_payload or {"file_ids": set(), "chunk_to_file": {}}
        self.hits_by_query = hits_by_query or {}

    def get_document_index(self, trace_id=""):
        return self.index_payload

    def search_documents_with_details(
        self,
        query_vector,
        file_ids=None,
        n_results=None,
        candidate_k=None,
        query_text="",
        trace_id="",
    ):
        return self.hits_by_query.get(query_text, [])


class RAGEvaluationServiceTests(unittest.TestCase):
    def _build_service(self, index_payload=None, hits_by_query=None):
        service = RAGEvaluationService()
        service.embedding_service = FakeEmbeddingService()
        service.rag_service = FakeRAGService(index_payload=index_payload, hits_by_query=hits_by_query)
        return service

    def test_invalid_when_filter_file_not_found(self):
        service = self._build_service(
            index_payload={
                "file_ids": {"file_a"},
                "chunk_to_file": {"file_a_chunk_1": "file_a"},
            }
        )
        request = RAGEvalRequest(
            samples=[
                RAGEvalSample(
                    query="test",
                    file_ids=["missing_file"],
                    relevant_file_ids=["file_a"],
                )
            ]
        )

        result = service.evaluate(request)
        self.assertEqual(result["summary"]["valid_samples"], 0)
        self.assertEqual(result["summary"]["invalid_samples"], 1)
        reason_counts = result["validation_summary"]["reason_counts"]
        self.assertIn("file_filter_not_found", reason_counts)
        self.assertFalse(result["gate"]["pass"])

    def test_invalid_when_chunk_file_mismatch(self):
        service = self._build_service(
            index_payload={
                "file_ids": {"file_a", "file_b"},
                "chunk_to_file": {"file_b_chunk_1": "file_b"},
            }
        )
        request = RAGEvalRequest(
            samples=[
                RAGEvalSample(
                    query="test",
                    relevant_file_ids=["file_a"],
                    relevant_chunk_ids=["file_b_chunk_1"],
                )
            ]
        )

        result = service.evaluate(request)
        self.assertEqual(result["summary"]["valid_samples"], 0)
        self.assertIn("relevant_chunk_file_mismatch", result["validation_summary"]["reason_counts"])

    def test_primary_file_metric_and_gate_pass(self):
        service = self._build_service(
            index_payload={
                "file_ids": {"file_a", "file_b"},
                "chunk_to_file": {
                    "file_a_chunk_1": "file_a",
                    "file_b_chunk_1": "file_b",
                },
            },
            hits_by_query={
                "query_a": [
                    {
                        "id": "file_b_chunk_1",
                        "metadata": {"file_id": "file_b"},
                        "document": "doc b",
                        "score": 0.9,
                        "semantic_score": 0.8,
                        "lexical_score": 0.6,
                    },
                    {
                        "id": "file_a_chunk_1",
                        "metadata": {"file_id": "file_a"},
                        "document": "doc a",
                        "score": 0.8,
                        "semantic_score": 0.7,
                        "lexical_score": 0.6,
                    },
                ]
            },
        )
        request = RAGEvalRequest(
            samples=[
                RAGEvalSample(
                    query="query_a",
                    relevant_file_ids=["file_a"],
                    relevant_chunk_ids=["file_a_chunk_1"],
                )
            ],
            top_k=8,
        )

        result = service.evaluate(request)
        sample = result["samples"][0]
        self.assertTrue(sample["valid"])
        self.assertEqual(sample["first_hit_rank"], 2)
        self.assertEqual(sample["hit_at_k"], 1)
        self.assertAlmostEqual(sample["mrr_at_k"], 0.5, places=6)
        self.assertGreater(sample["ndcg_at_k"], 0.6)
        self.assertTrue(result["gate"]["pass"])

    def test_gate_fail_when_core_metrics_all_zero(self):
        service = self._build_service(
            index_payload={
                "file_ids": {"file_a", "file_b"},
                "chunk_to_file": {
                    "file_b_chunk_1": "file_b",
                },
            },
            hits_by_query={
                "query_zero": [
                    {
                        "id": "file_b_chunk_1",
                        "metadata": {"file_id": "file_b"},
                        "document": "doc b",
                        "score": 0.9,
                        "semantic_score": 0.8,
                        "lexical_score": 0.6,
                    }
                ]
            },
        )
        request = RAGEvalRequest(
            samples=[
                RAGEvalSample(
                    query="query_zero",
                    relevant_file_ids=["file_a"],
                )
            ]
        )

        result = service.evaluate(request)
        self.assertEqual(result["summary"]["valid_samples"], 1)
        self.assertEqual(result["summary"]["hit_rate_at_k"], 0.0)
        self.assertEqual(result["summary"]["mrr_at_k"], 0.0)
        self.assertEqual(result["summary"]["avg_ndcg_at_k"], 0.0)
        self.assertFalse(result["gate"]["pass"])
        self.assertGreaterEqual(len(result["gate"]["fail_reasons"]), 1)

    def test_runtime_error_is_marked_invalid(self):
        class FailingEmbeddingService:
            def get_embedding(self, text):
                raise RuntimeError("embedding unavailable")

        service = self._build_service(
            index_payload={
                "file_ids": {"file_a"},
                "chunk_to_file": {"file_a_chunk_1": "file_a"},
            },
            hits_by_query={"query_err": []},
        )
        service.embedding_service = FailingEmbeddingService()

        request = RAGEvalRequest(
            samples=[
                RAGEvalSample(
                    query="query_err",
                    relevant_file_ids=["file_a"],
                )
            ]
        )
        result = service.evaluate(request)
        self.assertEqual(result["summary"]["valid_samples"], 0)
        self.assertEqual(result["summary"]["invalid_samples"], 1)
        self.assertIn("runtime_evaluation_error", result["validation_summary"]["reason_counts"])


if __name__ == "__main__":
    unittest.main()
