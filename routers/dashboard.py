import logging
import time
import traceback

from fastapi import APIRouter, HTTPException

from config import Config
from models import AnalysisRequest, DiagnosisRequest
from prompts.prompt_templates import DIAGNOSIS_FORMAT_INSTRUCTION
from services.llm_service import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/dashboard/learner/analyze")
async def analyze_learner(request: AnalysisRequest):
    start = time.perf_counter()
    try:
        logger.info(
            "[dashboard][analyze_start] prompt_chars=%s data_chars=%s",
            len(request.system_prompt),
            len(request.context_data),
        )

        llm_service = LLMService()
        messages = [
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": f"以下是学员的历史学习数据JSON，请根据系统指令分析:\n{request.context_data}",
            },
        ]

        analysis_result = llm_service.generate_response(
            messages,
            temperature=Config.DASHBOARD_ANALYSIS_TEMPERATURE,
            stream=False,
        )

        logger.info(
            "[dashboard][analyze_end] result_chars=%s duration_ms=%.2f",
            len(analysis_result),
            (time.perf_counter() - start) * 1000,
        )
        return {"code": 200, "message": "success", "data": analysis_result}

    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error("[dashboard][analyze_error] err=%s\n%s", exc, error_msg)
        raise HTTPException(status_code=500, detail=f"AI Service Connection Error: {str(exc)}")


@router.post("/api/v1/agent/diagnosis/self_study")
async def generate_diagnosis(request: DiagnosisRequest):
    start = time.perf_counter()
    try:
        logger.info("[diagnosis][start] user_chars=%s", len(request.user_content))

        llm_service = LLMService()
        final_system_prompt = request.system_prompt + "\n\n" + DIAGNOSIS_FORMAT_INSTRUCTION

        messages = [
            {"role": "system", "content": final_system_prompt},
            {
                "role": "user",
                "content": f"以下是学员最近的对话/题目记录:\n{request.user_content}",
            },
        ]

        ai_content = llm_service.generate_response(
            messages,
            temperature=Config.DIAGNOSIS_TEMPERATURE,
            stream=False,
        )

        if "---JSON---" not in ai_content:
            ai_content += "\n\n---JSON---\n{}"
        if "---MARKDOWN---" not in ai_content:
            ai_content = "---MARKDOWN---\n" + ai_content

        logger.info(
            "[diagnosis][end] result_chars=%s duration_ms=%.2f",
            len(ai_content),
            (time.perf_counter() - start) * 1000,
        )
        return {"code": 200, "message": "success", "data": ai_content}

    except Exception as exc:
        logger.error("[diagnosis][error] err=%s", exc)
        error_content = (
            "---MARKDOWN---\n"
            "### 诊断生成失败\n"
            f"AI 服务暂时不可用，错误信息: {str(exc)}\n"
            "---JSON---\n"
            "{}"
        )
        return {"code": 200, "data": error_content}
