import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_int(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_json(relative_path: str) -> Dict[str, Any]:
    with open(BASE_DIR / relative_path, "r", encoding="utf-8") as f:
        return json.load(f)


class Config:
    API_KEY = os.getenv("ARK_API_KEY", "")
    BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    MODEL_ID = os.getenv("ARK_MODEL_ID", "deepseek-v3-2-251201")
    EMBED_MODEL_ID = os.getenv("ARK_EMBED_MODEL_ID", "ep-20260225005055-qq86c")
    CLIENT_TIMEOUT = _get_int("ARK_CLIENT_TIMEOUT", 1800)

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "knowledge_db"))

    # Logging and observability
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_PREVIEW_CHARS = _get_int("LOG_PREVIEW_CHARS", 220)
    ENABLE_RAG_VERBOSE_LOG = _get_bool("ENABLE_RAG_VERBOSE_LOG", True)
    ENABLE_TOOL_VERBOSE_LOG = _get_bool("ENABLE_TOOL_VERBOSE_LOG", True)

    # LLM
    TOOL_CALL_TEMPERATURE = _get_float("TOOL_CALL_TEMPERATURE", 0.2)
    FINAL_ANSWER_TEMPERATURE = _get_float("FINAL_ANSWER_TEMPERATURE", 0.7)
    GUIDE_TEMPERATURE = _get_float("GUIDE_TEMPERATURE", 0.7)
    QUIZ_TEMPERATURE = _get_float("QUIZ_TEMPERATURE", 0.8)
    QUIZ_ANALYSIS_TEMPERATURE = _get_float("QUIZ_ANALYSIS_TEMPERATURE", 0.8)
    DASHBOARD_ANALYSIS_TEMPERATURE = _get_float("DASHBOARD_ANALYSIS_TEMPERATURE", 1.3)
    DIAGNOSIS_TEMPERATURE = _get_float("DIAGNOSIS_TEMPERATURE", 1.1)
    STREAM_DELAY_SECONDS = _get_float("STREAM_DELAY_SECONDS", 0.01)

    # Tool loop and calls
    TOOL_MAX_ROUNDS = _get_int("TOOL_MAX_ROUNDS", 5)
    TOOL_DUPLICATE_CALL_LIMIT = _get_int("TOOL_DUPLICATE_CALL_LIMIT", 2)
    TOOL_CALL_TIMEOUT_SECONDS = _get_int("TOOL_CALL_TIMEOUT_SECONDS", 10)
    TOOL_DEFAULT_PAGE_NUM = _get_int("TOOL_DEFAULT_PAGE_NUM", 1)
    TOOL_DEFAULT_PAGE_SIZE = _get_int("TOOL_DEFAULT_PAGE_SIZE", 10)

    # RAG retrieval
    RAG_DOC_TOP_K = _get_int("RAG_DOC_TOP_K", 8)
    RAG_DOC_CANDIDATE_K = _get_int("RAG_DOC_CANDIDATE_K", 24)
    RAG_MEMORY_TOP_K = _get_int("RAG_MEMORY_TOP_K", 8)
    RAG_MEMORY_CANDIDATE_K = _get_int("RAG_MEMORY_CANDIDATE_K", 24)
    RAG_SEMANTIC_WEIGHT = _get_float("RAG_SEMANTIC_WEIGHT", 0.75)
    RAG_LEXICAL_WEIGHT = _get_float("RAG_LEXICAL_WEIGHT", 0.25)
    RAG_SCORE_FLOOR = _get_float("RAG_SCORE_FLOOR", 0.0)

    # Document parsing
    TEXT_CHUNK_SIZE = _get_int("TEXT_CHUNK_SIZE", 500)
    TEXT_CHUNK_OVERLAP = _get_int("TEXT_CHUNK_OVERLAP", 50)

    # RAG evaluation
    RAG_EVAL_TOP_K = _get_int("RAG_EVAL_TOP_K", 8)
    RAG_EVAL_CANDIDATE_K = _get_int("RAG_EVAL_CANDIDATE_K", 24)
    RAG_EVAL_GATE_ENABLED = _get_bool("RAG_EVAL_GATE_ENABLED", True)
    RAG_EVAL_GATE_HIT_RATE_MIN = _get_float("RAG_EVAL_GATE_HIT_RATE_MIN", 0.50)
    RAG_EVAL_GATE_MRR_MIN = _get_float("RAG_EVAL_GATE_MRR_MIN", 0.30)
    RAG_EVAL_GATE_NDCG_MIN = _get_float("RAG_EVAL_GATE_NDCG_MIN", 0.40)

    tool_config = _load_json("tool_api_registry.json")
    TOOL_JAVA_BASE_URL = tool_config["TOOL_JAVA_BASE_URL"]
    TOOL_API_REGISTRY = tool_config["TOOL_API_REGISTRY"]

    student_tool_config = _load_json("student_tool_api_registry.json")
    STUDENT_TOOL_JAVA_BASE_URL = student_tool_config["TOOL_JAVA_BASE_URL"]
    STUDENT_TOOL_API_REGISTRY = student_tool_config["TOOL_API_REGISTRY"]
