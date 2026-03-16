import json
import logging

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from models import ChatMemoryRequest, ChatRequest, ToolChatRequest
from services.chat_service import ChatService
from utils.logging_utils import truncate_text

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/chat/memory/add")
async def add_chat_memory(request: ChatMemoryRequest):
    try:
        logger.info(
            "[chat_memory][start] window=%s chapter=%s role=%s msg_id=%s content=%s",
            request.window_id,
            request.chapter_id,
            request.role,
            request.msg_id,
            truncate_text(request.content, 160),
        )

        chat_service = ChatService()
        vector = chat_service.embedding_service.get_embedding(request.content)
        chat_service.rag_service.add_chat_memory(
            msg_id=f"msg_{request.msg_id}",
            embedding=vector,
            content=request.content,
            metadata={"window_id": request.window_id, "chapter_id": request.chapter_id, "role": request.role},
        )

        logger.info(
            "[chat_memory][end] window=%s chapter=%s role=%s dim=%s",
            request.window_id,
            request.chapter_id,
            request.role,
            len(vector) if vector else 0,
        )
        return {"code": 200, "message": "success"}
    except Exception as exc:
        logger.exception("[chat_memory][error] failed")
        return {"code": 500, "message": str(exc)}


@router.post("/api/v1/agent/chat")
async def stream_rag_chat(request: ChatRequest):
    logger.info(
        "[chat_api][start] mode=student window=%s chapter=%s files=%s history=%s query=%s",
        request.windowId,
        request.chapterId,
        len(request.fileIds),
        len(request.history),
        truncate_text(request.query, 180),
    )

    chat_service = ChatService()

    async def event_generator():
        try:
            async for chunk in chat_service.stream_chat(
                request.query,
                request.windowId,
                request.chapterId,
                request.fileIds,
                request.history,
            ):
                yield chunk
        except Exception as exc:
            logger.exception("[chat_api][error] mode=student")
            error_data = {"msg": f"AI 服务异常: {str(exc)}"}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/agent/chat/share")
async def tool_stream_rag_chat(request: ToolChatRequest):
    logger.info(
        "[chat_api][start] mode=teacher window=%s chapter=%s user=%s files=%s history=%s query=%s",
        request.windowId,
        request.chapterId,
        request.userId,
        len(request.fileIds),
        len(request.history),
        truncate_text(request.query, 180),
    )

    chat_service = ChatService()

    async def event_generator():
        try:
            async for chunk in chat_service.stream_chat_with_tools(request):
                yield chunk
        except Exception as exc:
            logger.exception("[chat_api][error] mode=teacher")
            error_data = {"msg": f"AI 服务异常: {str(exc)}"}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
