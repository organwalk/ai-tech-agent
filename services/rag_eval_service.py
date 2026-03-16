import logging
import math
import time
from collections import Counter
from typing import Dict, List, Tuple

from config import Config
from models import RAGEvalRequest, RAGEvalSample
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService
from utils.logging_utils import new_trace_id, truncate_text

logger = logging.getLogger(__name__)


class RAGEvaluationService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.rag_service = RAGService()

    def evaluate(self, request: RAGEvalRequest) -> Dict:
        trace_id = new_trace_id("eval_")
        eval_start = time.perf_counter()
        top_k = request.top_k or Config.RAG_EVAL_TOP_K
        candidate_k = request.candidate_k or Config.RAG_EVAL_CANDIDATE_K

        logger.info(
            "[rag_eval][start] trace=%s samples=%s top_k=%s candidate_k=%s",
            trace_id,
            len(request.samples),
            top_k,
            candidate_k,
        )

        document_index = self.rag_service.get_document_index(trace_id=trace_id)
        validation_ctx = {
            "existing_file_ids": set(document_index.get("file_ids", set())),
            "chunk_to_file": dict(document_index.get("chunk_to_file", {})),
        }

        sample_reports: List[Dict] = []
        reason_counter: Counter = Counter()
        for index, sample in enumerate(request.samples, start=1):
            sample_start = time.perf_counter()
            validation = self._validate_sample(sample, validation_ctx)

            if not validation["valid"]:
                reason_codes = validation["reason_codes"]
                reason_counter.update(reason_codes)
                sample_report = self._build_invalid_sample_report(sample, top_k=sample.top_k or top_k, validation=validation)
            else:
                try:
                    sample_report = self._evaluate_valid_sample(
                        sample=sample,
                        default_top_k=top_k,
                        candidate_k=candidate_k,
                        trace_id=trace_id,
                    )
                except Exception as exc:
                    reason_code = "runtime_evaluation_error"
                    reason_counter.update([reason_code])
                    sample_report = self._build_invalid_sample_report(
                        sample=sample,
                        top_k=sample.top_k or top_k,
                        validation={
                            "reason": f"Evaluation runtime failed: {exc}",
                            "reason_codes": [reason_code],
                        },
                    )
                    logger.error(
                        "[rag_eval][sample_error] trace=%s sample_index=%s query=%s err=%s",
                        trace_id,
                        index,
                        truncate_text(sample.query, Config.LOG_PREVIEW_CHARS),
                        exc,
                    )

            sample_report["sample_index"] = index
            sample_report["duration_ms"] = round((time.perf_counter() - sample_start) * 1000, 2)
            sample_reports.append(sample_report)

        summary = self._aggregate(sample_reports, top_k=top_k)
        validation_summary = {
            "total_samples": len(sample_reports),
            "valid_samples": summary["valid_samples"],
            "invalid_samples": summary["invalid_samples"],
            "reason_counts": dict(sorted(reason_counter.items())),
        }
        gate = self._build_gate(summary)

        if summary["valid_samples"] > 0 and self._is_all_core_zero(summary):
            warning = self._diagnose_zero_metrics(sample_reports)
            logger.warning("[rag_eval][warning] trace=%s %s", trace_id, warning)

        total_duration = (time.perf_counter() - eval_start) * 1000
        logger.info(
            "[rag_eval][end] trace=%s valid=%s invalid=%s hit_rate=%.4f mrr=%.4f ndcg=%.4f gate_pass=%s duration_ms=%.2f",
            trace_id,
            summary["valid_samples"],
            summary["invalid_samples"],
            summary["hit_rate_at_k"],
            summary["mrr_at_k"],
            summary["avg_ndcg_at_k"],
            gate["pass"],
            total_duration,
        )

        return {
            "trace_id": trace_id,
            "config": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "primary_relevance": "file_id",
                "semantic_weight": Config.RAG_SEMANTIC_WEIGHT,
                "lexical_weight": Config.RAG_LEXICAL_WEIGHT,
                "gate_thresholds": self._gate_thresholds(),
            },
            "summary": summary,
            "validation_summary": validation_summary,
            "gate": gate,
            "samples": sample_reports,
            "duration_ms": round(total_duration, 2),
        }

    def _validate_sample(self, sample: RAGEvalSample, validation_ctx: Dict) -> Dict:
        reasons: List[Tuple[str, str]] = []
        existing_file_ids = validation_ctx["existing_file_ids"]
        chunk_to_file = validation_ctx["chunk_to_file"]

        query = (sample.query or "").strip()
        if not query:
            reasons.append(("empty_query", "Query is empty."))

        relevant_file_ids = [str(item).strip() for item in sample.relevant_file_ids if str(item).strip()]
        if not relevant_file_ids:
            reasons.append(("missing_relevant_file_ids", "Primary labels required: relevant_file_ids."))

        missing_relevant_files = [fid for fid in relevant_file_ids if fid not in existing_file_ids]
        if missing_relevant_files:
            reasons.append(
                (
                    "relevant_file_not_found",
                    f"Relevant file_id not found in knowledge base: {missing_relevant_files}",
                )
            )

        filter_file_ids = [str(item).strip() for item in sample.file_ids if str(item).strip()]
        missing_filter_files = [fid for fid in filter_file_ids if fid not in existing_file_ids]
        if missing_filter_files:
            reasons.append(
                (
                    "file_filter_not_found",
                    f"file_ids contains unknown file_id: {missing_filter_files}",
                )
            )

        if filter_file_ids and relevant_file_ids:
            filter_set = set(filter_file_ids)
            relevant_set = set(relevant_file_ids)
            if not (filter_set & relevant_set):
                reasons.append(
                    (
                        "relevant_file_not_in_filter",
                        "No overlap between file_ids filter and relevant_file_ids, retrieval cannot hit labeled files.",
                    )
                )

        chunk_ids = [str(item).strip() for item in sample.relevant_chunk_ids if str(item).strip()]
        if chunk_ids:
            missing_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_to_file]
            if missing_chunks:
                reasons.append(
                    (
                        "relevant_chunk_not_found",
                        f"relevant_chunk_ids contains unknown chunk_id: {missing_chunks}",
                    )
                )

            if relevant_file_ids:
                relevant_file_set = set(relevant_file_ids)
                mismatch_chunks = [
                    chunk_id
                    for chunk_id in chunk_ids
                    if chunk_id in chunk_to_file and chunk_to_file[chunk_id] not in relevant_file_set
                ]
                if mismatch_chunks:
                    reasons.append(
                        (
                            "relevant_chunk_file_mismatch",
                            f"chunk_id does not belong to relevant_file_ids: {mismatch_chunks}",
                        )
                    )

        return {
            "valid": len(reasons) == 0,
            "reason_codes": [code for code, _ in reasons],
            "reason": "; ".join(message for _, message in reasons),
        }

    def _evaluate_valid_sample(
        self,
        sample: RAGEvalSample,
        default_top_k: int,
        candidate_k: int,
        trace_id: str,
    ) -> Dict:
        top_k = sample.top_k or default_top_k

        query_vector = self.embedding_service.get_embedding(sample.query)
        hits = self.rag_service.search_documents_with_details(
            query_vector=query_vector,
            file_ids=sample.file_ids,
            n_results=top_k,
            candidate_k=candidate_k,
            query_text=sample.query,
            trace_id=trace_id,
        )

        primary_relevances, primary_first_hit_rank = self._build_primary_relevance_vector(sample, hits)
        primary_label_count = len(set(str(fid) for fid in sample.relevant_file_ids if str(fid).strip()))
        primary_retrieved_relevant = sum(primary_relevances)

        hit_at_k = 1 if primary_first_hit_rank is not None and primary_first_hit_rank <= top_k else 0
        mrr_at_k = 1.0 / primary_first_hit_rank if primary_first_hit_rank else 0.0
        recall_at_k = primary_retrieved_relevant / max(primary_label_count, 1)
        ndcg_at_k = self._ndcg(primary_relevances, primary_label_count, top_k)

        chunk_relevances, chunk_first_hit_rank = self._build_chunk_relevance_vector(sample, hits)
        keyword_relevances, keyword_first_hit_rank = self._build_keyword_relevance_vector(sample, hits)

        logger.info(
            "[rag_eval][sample] trace=%s query=%s labels(file)=%s hits(file)=%s first_hit=%s",
            trace_id,
            truncate_text(sample.query, Config.LOG_PREVIEW_CHARS),
            primary_label_count,
            primary_retrieved_relevant,
            primary_first_hit_rank,
        )

        relevant_file_set = set(str(fid) for fid in sample.relevant_file_ids if str(fid).strip())
        relevant_chunk_set = set(str(cid) for cid in sample.relevant_chunk_ids if str(cid).strip())
        keyword_set = [item.strip().lower() for item in sample.relevant_keywords if item.strip()]

        preview = []
        for hit in hits[:top_k]:
            metadata = hit.get("metadata") or {}
            hit_file_id = str(metadata.get("file_id", ""))
            hit_chunk_id = str(hit.get("id", ""))
            hit_doc = (hit.get("document") or "").lower()

            preview.append(
                {
                    "chunk_id": hit_chunk_id,
                    "file_id": hit_file_id,
                    "is_primary_relevant": hit_file_id in relevant_file_set,
                    "is_chunk_relevant": bool(relevant_chunk_set) and hit_chunk_id in relevant_chunk_set,
                    "is_keyword_relevant": bool(keyword_set) and any(keyword in hit_doc for keyword in keyword_set),
                    "score": round(hit.get("score", 0.0), 6),
                    "semantic_score": round(hit.get("semantic_score", 0.0), 6),
                    "lexical_score": round(hit.get("lexical_score", 0.0), 6),
                    "text_preview": truncate_text(hit.get("document", ""), 120),
                }
            )

        return {
            "query": sample.query,
            "top_k": top_k,
            "primary_relevance": "file_id",
            "filter_file_ids": sample.file_ids,
            "valid": True,
            "reason": "",
            "reason_codes": [],
            "hit_at_k": hit_at_k,
            "mrr_at_k": round(mrr_at_k, 6),
            "ndcg_at_k": round(ndcg_at_k, 6),
            "recall_at_k": round(recall_at_k, 6),
            "first_hit_rank": primary_first_hit_rank,
            "retrieved_count": len(hits),
            "retrieved_relevant_count": int(primary_retrieved_relevant),
            "label_count": primary_label_count,
            "auxiliary": {
                "chunk_id": self._aux_metrics(chunk_relevances, len(set(sample.relevant_chunk_ids)), top_k, chunk_first_hit_rank),
                "keyword": self._aux_metrics(
                    keyword_relevances,
                    len(set(item.strip().lower() for item in sample.relevant_keywords if item.strip())),
                    top_k,
                    keyword_first_hit_rank,
                ),
            },
            "retrieved": preview,
        }

    def _build_invalid_sample_report(self, sample: RAGEvalSample, top_k: int, validation: Dict) -> Dict:
        return {
            "query": sample.query,
            "top_k": top_k,
            "primary_relevance": "file_id",
            "filter_file_ids": sample.file_ids,
            "valid": False,
            "reason": validation["reason"],
            "reason_codes": validation["reason_codes"],
            "hit_at_k": 0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "recall_at_k": 0.0,
            "first_hit_rank": None,
            "retrieved_count": 0,
            "retrieved_relevant_count": 0,
            "label_count": len(set(sample.relevant_file_ids)),
            "auxiliary": {
                "chunk_id": self._aux_metrics([], len(set(sample.relevant_chunk_ids)), top_k, None),
                "keyword": self._aux_metrics(
                    [],
                    len(set(item.strip().lower() for item in sample.relevant_keywords if item.strip())),
                    top_k,
                    None,
                ),
            },
            "retrieved": [],
        }

    def _build_primary_relevance_vector(self, sample: RAGEvalSample, hits: List[Dict]) -> Tuple[List[int], int]:
        relevant_file_set = set(str(fid) for fid in sample.relevant_file_ids if str(fid).strip())
        relevances = []
        first_hit_rank = None

        for index, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            hit_file_id = str(metadata.get("file_id", ""))
            rel = 1 if hit_file_id in relevant_file_set else 0
            relevances.append(rel)
            if rel == 1 and first_hit_rank is None:
                first_hit_rank = index
        return relevances, first_hit_rank

    def _build_chunk_relevance_vector(self, sample: RAGEvalSample, hits: List[Dict]) -> Tuple[List[int], int]:
        relevant_chunk_set = set(str(cid) for cid in sample.relevant_chunk_ids if str(cid).strip())
        if not relevant_chunk_set:
            return [], None

        relevances = []
        first_hit_rank = None
        for index, hit in enumerate(hits, start=1):
            chunk_id = str(hit.get("id", ""))
            rel = 1 if chunk_id in relevant_chunk_set else 0
            relevances.append(rel)
            if rel == 1 and first_hit_rank is None:
                first_hit_rank = index
        return relevances, first_hit_rank

    def _build_keyword_relevance_vector(self, sample: RAGEvalSample, hits: List[Dict]) -> Tuple[List[int], int]:
        keywords = [item.strip().lower() for item in sample.relevant_keywords if item.strip()]
        if not keywords:
            return [], None

        relevances = []
        first_hit_rank = None
        for index, hit in enumerate(hits, start=1):
            doc_text = (hit.get("document") or "").lower()
            rel = 1 if any(keyword in doc_text for keyword in keywords) else 0
            relevances.append(rel)
            if rel == 1 and first_hit_rank is None:
                first_hit_rank = index
        return relevances, first_hit_rank

    def _aux_metrics(self, relevances: List[int], label_count: int, top_k: int, first_hit_rank: int) -> Dict:
        if label_count <= 0:
            return {
                "label_count": 0,
                "hit_at_k": None,
                "mrr_at_k": None,
                "recall_at_k": None,
                "first_hit_rank": None,
            }

        retrieved_relevant_count = sum(relevances)
        hit_at_k = 1 if first_hit_rank is not None and first_hit_rank <= top_k else 0
        mrr_at_k = 1.0 / first_hit_rank if first_hit_rank else 0.0
        recall_at_k = retrieved_relevant_count / max(label_count, 1)
        return {
            "label_count": label_count,
            "hit_at_k": hit_at_k,
            "mrr_at_k": round(mrr_at_k, 6),
            "recall_at_k": round(recall_at_k, 6),
            "first_hit_rank": first_hit_rank,
        }

    def _ndcg(self, relevances: List[int], label_count: int, top_k: int) -> float:
        if label_count <= 0:
            return 0.0

        dcg = 0.0
        for idx, rel in enumerate(relevances[:top_k], start=1):
            if rel:
                dcg += 1.0 / math.log2(idx + 1)

        ideal_relevant = min(label_count, top_k)
        if ideal_relevant == 0:
            return 0.0

        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))
        if idcg <= 0:
            return 0.0
        return dcg / idcg

    def _aggregate(self, sample_reports: List[Dict], top_k: int) -> Dict:
        valid_reports = [item for item in sample_reports if item.get("valid")]
        invalid_reports = [item for item in sample_reports if not item.get("valid")]
        valid_count = len(valid_reports)

        summary = {
            "top_k": top_k,
            "primary_relevance": "file_id",
            "total_samples": len(sample_reports),
            "valid_samples": valid_count,
            "invalid_samples": len(invalid_reports),
            "hit_rate_at_k": 0.0,
            "mrr_at_k": 0.0,
            "avg_ndcg_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_first_hit_rank": 0.0,
            "auxiliary": {
                "chunk_id": {
                    "evaluated_samples": 0,
                    "hit_rate_at_k": 0.0,
                },
                "keyword": {
                    "evaluated_samples": 0,
                    "hit_rate_at_k": 0.0,
                },
            },
        }

        if valid_count == 0:
            return summary

        summary["hit_rate_at_k"] = round(sum(item["hit_at_k"] for item in valid_reports) / valid_count, 6)
        summary["mrr_at_k"] = round(sum(item["mrr_at_k"] for item in valid_reports) / valid_count, 6)
        summary["avg_ndcg_at_k"] = round(sum(item["ndcg_at_k"] for item in valid_reports) / valid_count, 6)
        summary["avg_recall_at_k"] = round(sum(item["recall_at_k"] for item in valid_reports) / valid_count, 6)

        first_hit_ranks = [item["first_hit_rank"] for item in valid_reports if item["first_hit_rank"]]
        summary["avg_first_hit_rank"] = round(sum(first_hit_ranks) / len(first_hit_ranks), 6) if first_hit_ranks else 0.0

        chunk_eval_items = [
            item for item in valid_reports if (item.get("auxiliary") or {}).get("chunk_id", {}).get("label_count", 0) > 0
        ]
        if chunk_eval_items:
            summary["auxiliary"]["chunk_id"]["evaluated_samples"] = len(chunk_eval_items)
            summary["auxiliary"]["chunk_id"]["hit_rate_at_k"] = round(
                sum(item["auxiliary"]["chunk_id"]["hit_at_k"] for item in chunk_eval_items) / len(chunk_eval_items),
                6,
            )

        keyword_eval_items = [
            item for item in valid_reports if (item.get("auxiliary") or {}).get("keyword", {}).get("label_count", 0) > 0
        ]
        if keyword_eval_items:
            summary["auxiliary"]["keyword"]["evaluated_samples"] = len(keyword_eval_items)
            summary["auxiliary"]["keyword"]["hit_rate_at_k"] = round(
                sum(item["auxiliary"]["keyword"]["hit_at_k"] for item in keyword_eval_items) / len(keyword_eval_items),
                6,
            )

        return summary

    def _gate_thresholds(self) -> Dict:
        return {
            "hit_rate_at_k_min": Config.RAG_EVAL_GATE_HIT_RATE_MIN,
            "mrr_at_k_min": Config.RAG_EVAL_GATE_MRR_MIN,
            "ndcg_at_k_min": Config.RAG_EVAL_GATE_NDCG_MIN,
        }

    def _build_gate(self, summary: Dict) -> Dict:
        thresholds = self._gate_thresholds()
        enabled = Config.RAG_EVAL_GATE_ENABLED

        if not enabled:
            return {
                "enabled": False,
                "pass": True,
                "thresholds": thresholds,
                "fail_reasons": [],
            }

        fail_reasons = []
        valid_samples = summary.get("valid_samples", 0)
        if valid_samples <= 0:
            fail_reasons.append("No valid samples for gate evaluation.")
        else:
            if summary.get("hit_rate_at_k", 0.0) < thresholds["hit_rate_at_k_min"]:
                fail_reasons.append(
                    f"hit_rate_at_k={summary.get('hit_rate_at_k')} < {thresholds['hit_rate_at_k_min']}"
                )
            if summary.get("mrr_at_k", 0.0) < thresholds["mrr_at_k_min"]:
                fail_reasons.append(f"mrr_at_k={summary.get('mrr_at_k')} < {thresholds['mrr_at_k_min']}")
            if summary.get("avg_ndcg_at_k", 0.0) < thresholds["ndcg_at_k_min"]:
                fail_reasons.append(
                    f"avg_ndcg_at_k={summary.get('avg_ndcg_at_k')} < {thresholds['ndcg_at_k_min']}"
                )

        return {
            "enabled": True,
            "pass": len(fail_reasons) == 0,
            "thresholds": thresholds,
            "fail_reasons": fail_reasons,
        }

    def _is_all_core_zero(self, summary: Dict) -> bool:
        return (
            summary.get("hit_rate_at_k", 0.0) == 0.0
            and summary.get("mrr_at_k", 0.0) == 0.0
            and summary.get("avg_ndcg_at_k", 0.0) == 0.0
        )

    def _diagnose_zero_metrics(self, sample_reports: List[Dict]) -> str:
        valid_reports = [item for item in sample_reports if item.get("valid")]
        if not valid_reports:
            return "all metrics are zero because there are no valid samples."

        if all(item.get("retrieved_count", 0) == 0 for item in valid_reports):
            return "all core metrics are zero; likely cause: retrieval returned empty results (index/filter mismatch)."

        with_filter = [item for item in valid_reports if item.get("filter_file_ids")]
        if with_filter and len(with_filter) == len(valid_reports):
            return "all core metrics are zero; likely cause: file_ids filter too strict or mismatched relevant_file_ids."

        return "all core metrics are zero; likely cause: labels mismatched with indexed file_id values."
