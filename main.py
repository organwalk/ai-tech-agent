import logging
import sys
import time

from fastapi import FastAPI, Request

from config import Config
from routers import chat, dashboard, guide, knowledge, quiz, rag_eval
from utils.logging_utils import new_trace_id


app = FastAPI(title="DeepSeek Analysis & Quiz Service (Ark Version)")


logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


app.include_router(knowledge.router)
app.include_router(guide.router)
app.include_router(dashboard.router)
app.include_router(quiz.router)
app.include_router(chat.router)
app.include_router(rag_eval.router)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    trace_id = request.headers.get("x-trace-id", new_trace_id("req_"))
    request.state.trace_id = trace_id

    logger.info(
        "[access][start] trace=%s method=%s path=%s",
        trace_id,
        request.method,
        request.url.path,
    )
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["x-trace-id"] = trace_id

    logger.info(
        "[access][end] trace=%s method=%s path=%s status=%s duration_ms=%.2f",
        trace_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
async def root():
    return {"message": "AI Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
