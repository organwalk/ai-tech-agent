import logging
import time
import traceback

from fastapi import APIRouter

from config import Config
from models import GuideRequest
from prompts.prompt_templates import GUIDE_SYSTEM_PROMPT
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/guide/generate")
async def generate_guide(request: GuideRequest):
    start = time.perf_counter()
    try:
        logger.info("[guide][start] topic=%s file_count=%s", request.topic, len(request.file_ids))

        embedding_service = EmbeddingService()
        rag_service = RAGService()
        llm_service = LLMService()

        reference_context = ""
        retrieved_count = 0
        if request.file_ids:
            topic_vector = embedding_service.get_embedding(request.topic)
            retrieved_docs = rag_service.search_documents(
                topic_vector,
                request.file_ids,
                n_results=Config.RAG_DOC_TOP_K,
                query_text=request.topic,
            )
            retrieved_count = len(retrieved_docs)
            if retrieved_docs:
                reference_context = "\n\n---\n【相关知识库片段】\n".join(retrieved_docs)
            else:
                reference_context = "（在指定文件中未检索到高相关片段）"
        else:
            reference_context = "（未提供参考文件，请基于通识知识生成）"

        messages = [
            {"role": "system", "content": GUIDE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"【学习主题】{request.topic}\n\n【参考内容】\n{reference_context}",
            },
        ]

        guide_content = llm_service.generate_response(
            messages,
            temperature=Config.GUIDE_TEMPERATURE,
            stream=False,
        )
        guide_content = llm_service.clean_markdown_format(guide_content)

        logger.info(
            "[guide][end] topic=%s retrieved=%s result_chars=%s duration_ms=%.2f",
            request.topic,
            retrieved_count,
            len(guide_content),
            (time.perf_counter() - start) * 1000,
        )
        return {"code": 200, "message": "success", "data": guide_content}

    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error("[guide][error] topic=%s err=%s\n%s", request.topic, exc, error_msg)
        return {"code": 500, "message": f"指南生成失败: {str(exc)}", "data": ""}
