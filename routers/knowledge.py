import logging
import time
import traceback

from fastapi import APIRouter, HTTPException

from config import Config
from models import ParseRequest
from services.embedding_service import EmbeddingService
from services.file_parser import FileParser
from services.rag_service import RAGService
from utils.text_chunker import TextChunker

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/v1/agent/knowledge/parse")
async def parse_document(request: ParseRequest):
    start = time.perf_counter()
    try:
        logger.info("[knowledge][start] file_id=%s window=%s url=%s", request.file_id, request.window_id, request.file_url)

        file_parser = FileParser()
        text_chunker = TextChunker(
            chunk_size=Config.TEXT_CHUNK_SIZE,
            overlap=Config.TEXT_CHUNK_OVERLAP,
        )
        embedding_service = EmbeddingService()
        rag_service = RAGService()

        content, content_type = file_parser.download_file(request.file_url)
        text_content = file_parser.extract_text(content, content_type, request.file_url)
        chunks = text_chunker.chunk_text(text_content)

        logger.info(
            "[knowledge][parsed] file_id=%s chars=%s chunks=%s content_type=%s",
            request.file_id,
            len(text_content),
            len(chunks),
            content_type,
        )

        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            vector = embedding_service.get_embedding(chunk)
            rag_service.add_document_chunk(
                chunk_id=f"{request.file_id}_chunk_{index}",
                embedding=vector,
                document=chunk,
                metadata={
                    "file_id": request.file_id,
                    "window_id": request.window_id,
                    "chunk_index": index,
                    "chunk_length": len(chunk),
                    "source_url": request.file_url,
                },
            )

        logger.info(
            "[knowledge][end] file_id=%s chunks=%s duration_ms=%.2f",
            request.file_id,
            len(chunks),
            (time.perf_counter() - start) * 1000,
        )
        return {
            "code": 200,
            "message": "success",
            "data": f"成功解析并入库 {len(chunks)} 个文本块",
        }

    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error("[knowledge][error] file_id=%s err=%s\n%s", request.file_id, exc, error_msg)
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(exc)}")
