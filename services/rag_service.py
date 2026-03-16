import logging
import re
import time
from typing import Any, Dict, List, Optional

import chromadb

from config import Config
from utils.logging_utils import truncate_json

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="learning_materials")
        self.chat_collection = self.chroma_client.get_or_create_collection(name="chat_history")

    def build_where_filter(self, file_ids: List[str]) -> dict:
        if len(file_ids) == 1:
            return {"file_id": file_ids[0]}
        return {"file_id": {"$in": file_ids}}

    def search_documents(
        self,
        query_vector: List[float],
        file_ids: Optional[List[str]] = None,
        n_results: Optional[int] = None,
        query_text: str = "",
        trace_id: str = "",
    ) -> List[str]:
        hits = self.search_documents_with_details(
            query_vector=query_vector,
            file_ids=file_ids,
            n_results=n_results,
            query_text=query_text,
            trace_id=trace_id,
        )
        return [hit["document"] for hit in hits]

    def search_documents_with_details(
        self,
        query_vector: List[float],
        file_ids: Optional[List[str]] = None,
        n_results: Optional[int] = None,
        candidate_k: Optional[int] = None,
        query_text: str = "",
        trace_id: str = "",
    ) -> List[Dict[str, Any]]:
        top_k = n_results or Config.RAG_DOC_TOP_K
        use_candidate_k = max(top_k, candidate_k or Config.RAG_DOC_CANDIDATE_K)
        where_filter = self.build_where_filter(file_ids) if file_ids else None

        return self._query_and_rank(
            collection=self.collection,
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            candidate_k=use_candidate_k,
            where_filter=where_filter,
            trace_id=trace_id,
            source="documents",
        )

    def search_chat_memory(
        self,
        query_vector: List[float],
        window_id: str,
        chapter_id: str,
        n_results: Optional[int] = None,
        query_text: str = "",
        trace_id: str = "",
    ) -> List[str]:
        hits = self.search_chat_memory_with_details(
            query_vector=query_vector,
            window_id=window_id,
            chapter_id=chapter_id,
            n_results=n_results,
            query_text=query_text,
            trace_id=trace_id,
        )
        retrieved_mems = []
        for hit in hits:
            role = (hit["metadata"] or {}).get("role", "unknown")
            retrieved_mems.append(f"[{role}] 曾说过: {hit['document']}")
        return retrieved_mems

    def search_chat_memory_with_details(
        self,
        query_vector: List[float],
        window_id: str,
        chapter_id: str,
        n_results: Optional[int] = None,
        candidate_k: Optional[int] = None,
        query_text: str = "",
        trace_id: str = "",
    ) -> List[Dict[str, Any]]:
        top_k = n_results or Config.RAG_MEMORY_TOP_K
        use_candidate_k = max(top_k, candidate_k or Config.RAG_MEMORY_CANDIDATE_K)
        memory_where_filter = {
            "$and": [
                {"window_id": window_id},
                {"chapter_id": chapter_id},
            ]
        }
        return self._query_and_rank(
            collection=self.chat_collection,
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            candidate_k=use_candidate_k,
            where_filter=memory_where_filter,
            trace_id=trace_id,
            source="memory",
        )

    def _query_and_rank(
        self,
        collection: Any,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        candidate_k: int,
        where_filter: Optional[Dict],
        trace_id: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        start = time.perf_counter()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        ranked_hits = self._hybrid_rank(results=results, query_text=query_text, top_k=top_k)
        duration_ms = (time.perf_counter() - start) * 1000

        if Config.ENABLE_RAG_VERBOSE_LOG:
            top_scores = [round(item["score"], 4) for item in ranked_hits[:3]]
            logger.info(
                "[rag][search] trace=%s source=%s top_k=%s candidate_k=%s where=%s hits=%s top_scores=%s duration_ms=%.2f",
                trace_id or "-",
                source,
                top_k,
                candidate_k,
                truncate_json(where_filter, max_len=Config.LOG_PREVIEW_CHARS),
                len(ranked_hits),
                top_scores,
                duration_ms,
            )
        return ranked_hits

    def _hybrid_rank(self, results: Dict, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []

        query_tokens = self._tokenize(query_text)
        ranked: List[Dict[str, Any]] = []
        seen_docs = set()

        for index, doc in enumerate(documents):
            if not doc or doc in seen_docs:
                continue

            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else None
            chunk_id = ids[index] if index < len(ids) else ""

            semantic_score = self._semantic_score(distance)
            lexical_score = self._lexical_score(query_tokens, doc)
            score = Config.RAG_SEMANTIC_WEIGHT * semantic_score + Config.RAG_LEXICAL_WEIGHT * lexical_score

            ranked.append(
                {
                    "id": chunk_id,
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "score": score,
                }
            )
            seen_docs.add(doc)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return [item for item in ranked[:top_k] if item["score"] > Config.RAG_SCORE_FLOOR]

    def _semantic_score(self, distance: Optional[float]) -> float:
        if distance is None:
            return 0.0
        safe_distance = max(float(distance), 0.0)
        return 1.0 / (1.0 + safe_distance)

    def _lexical_score(self, query_tokens: List[str], doc: str) -> float:
        if not query_tokens:
            return 0.0

        doc_tokens = set(self._tokenize(doc))
        if not doc_tokens:
            return 0.0

        overlap = len(set(query_tokens) & doc_tokens)
        return overlap / max(len(set(query_tokens)), 1)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)

    def get_document_index(self, trace_id: str = "") -> Dict[str, Any]:
        start = time.perf_counter()
        total = self.collection.count()
        if total <= 0:
            return {"file_ids": set(), "chunk_to_file": {}}

        results = self.collection.get(include=["metadatas"], limit=total)
        ids = results.get("ids", []) or []
        metadatas = results.get("metadatas", []) or []

        file_ids = set()
        chunk_to_file = {}
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            file_id = str(metadata.get("file_id", "")).strip()
            if not file_id:
                continue
            file_ids.add(file_id)
            chunk_to_file[str(chunk_id)] = file_id

        if Config.ENABLE_RAG_VERBOSE_LOG:
            logger.info(
                "[rag][index] trace=%s total_chunks=%s file_ids=%s duration_ms=%.2f",
                trace_id or "-",
                len(chunk_to_file),
                len(file_ids),
                (time.perf_counter() - start) * 1000,
            )
        return {"file_ids": file_ids, "chunk_to_file": chunk_to_file}

    def get_document_records(self, limit: Optional[int] = None, trace_id: str = "") -> List[Dict[str, Any]]:
        start = time.perf_counter()
        total = self.collection.count()
        if total <= 0:
            return []

        use_limit = min(limit if limit is not None and limit > 0 else total, total)
        results = self.collection.get(include=["metadatas", "documents"], limit=use_limit)
        ids = results.get("ids", []) or []
        docs = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []

        records: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            file_id = str(metadata.get("file_id", "")).strip()
            records.append(
                {
                    "id": str(chunk_id),
                    "file_id": file_id,
                    "document": docs[index] if index < len(docs) else "",
                    "metadata": metadata,
                }
            )

        if Config.ENABLE_RAG_VERBOSE_LOG:
            logger.info(
                "[rag][records] trace=%s records=%s duration_ms=%.2f",
                trace_id or "-",
                len(records),
                (time.perf_counter() - start) * 1000,
            )
        return records

    def add_document_chunk(self, chunk_id: str, embedding: List[float], document: str, metadata: dict):
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def add_chat_memory(self, msg_id: str, embedding: List[float], content: str, metadata: dict):
        self.chat_collection.add(
            ids=[msg_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
