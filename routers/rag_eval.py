import logging
import traceback

from fastapi import APIRouter, HTTPException

from models import RAGEvalRequest
from services.rag_eval_service import RAGEvaluationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/rag/evaluate")
async def evaluate_rag(request: RAGEvalRequest):
    try:
        logger.info(
            "[rag_eval][api] received samples=%s top_k=%s candidate_k=%s",
            len(request.samples),
            request.top_k,
            request.candidate_k,
        )
        eval_service = RAGEvaluationService()
        result = eval_service.evaluate(request)
        return {"code": 200, "message": "success", "data": result}
    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error("[rag_eval][api] failed: %s\n%s", exc, error_msg)
        raise HTTPException(status_code=500, detail=f"RAG evaluation failed: {str(exc)}")

