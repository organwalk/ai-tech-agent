import json
import uuid
from typing import Any


def new_trace_id(prefix: str = "") -> str:
    trace = uuid.uuid4().hex[:12]
    return f"{prefix}{trace}" if prefix else trace


def truncate_text(value: Any, max_len: int = 220) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(+{len(text) - max_len} chars)"


def truncate_json(value: Any, max_len: int = 220) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return truncate_text(text, max_len=max_len)

