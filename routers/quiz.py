import logging
import time
import traceback

from fastapi import APIRouter

from config import Config
from models import QuizAnalysisRequest, QuizRequest
from prompts.prompt_templates import build_quiz_analysis_system_prompt, build_quiz_instruction
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/quiz/generate")
async def generate_quiz(request: QuizRequest):
    start = time.perf_counter()
    try:
        logger.info("[quiz][start] context_chars=%s file_count=%s", len(request.user_data), len(request.file_ids))

        embedding_service = EmbeddingService()
        rag_service = RAGService()
        llm_service = LLMService()

        reference_context = ""
        retrieved_count = 0
        if request.file_ids:
            topic_vector = embedding_service.get_embedding(request.user_data[:500])
            retrieved_docs = rag_service.search_documents(
                topic_vector,
                request.file_ids,
                n_results=Config.RAG_DOC_TOP_K,
                query_text=request.user_data[:500],
            )
            retrieved_count = len(retrieved_docs)
            if retrieved_docs:
                reference_context = "\n\n---\n【相关知识库片段】\n".join(retrieved_docs)
            else:
                reference_context = "（未检索到高相关片段，请基于命题范围生成）"
        else:
            reference_context = "（未选择知识库文件，请基于命题范围生成）"

        final_system_prompt = request.system_prompt + "\n\n" + build_quiz_instruction(reference_context)
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": f"请根据以下范围出题:\n{request.user_data}"},
        ]

        ai_content = llm_service.generate_response(
            messages,
            temperature=Config.QUIZ_TEMPERATURE,
            stream=False,
        )

        if "```json" not in ai_content:
            ai_content = "```json\n" + ai_content
            if not ai_content.endswith("```"):
                ai_content += "\n```"

        logger.info(
            "[quiz][end] retrieved=%s result_chars=%s duration_ms=%.2f",
            retrieved_count,
            len(ai_content),
            (time.perf_counter() - start) * 1000,
        )
        return {"code": 200, "message": "success", "data": ai_content}

    except Exception as exc:
        logger.error("[quiz][error] err=%s", exc)
        return {"code": 200, "data": "```json\n[]\n```"}


@router.post("/api/v1/agent/quiz/analysis")
async def generate_quiz_analysis(request: QuizAnalysisRequest):
    start = time.perf_counter()
    try:
        logger.info("[quiz_analysis][start] prompt_chars=%s", len(request.prompt))

        llm_service = LLMService()
        messages = [
            {"role": "system", "content": build_quiz_analysis_system_prompt(request.requirement)},
            {"role": "user", "content": f"以下是该学员的测试数据:\n{request.prompt}"},
        ]

        analysis_content = llm_service.generate_response(
            messages,
            temperature=Config.QUIZ_ANALYSIS_TEMPERATURE,
            stream=False,
        )
        analysis_content = llm_service.clean_markdown_format(analysis_content)

        logger.info(
            "[quiz_analysis][end] result_chars=%s duration_ms=%.2f",
            len(analysis_content),
            (time.perf_counter() - start) * 1000,
        )
        return {"code": 200, "message": "success", "data": analysis_content}

    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error("[quiz_analysis][error] err=%s\n%s", exc, error_msg)
        return {"code": 500, "message": f"AI 服务处理失败: {str(exc)}", "data": ""}
