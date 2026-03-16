import json
import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import requests

from config import Config
from utils.logging_utils import truncate_json, truncate_text

logger = logging.getLogger(__name__)


class ToolService:
    _TEACHER_INTENT_TOOL_KEYWORDS = {
        "knowledge_explanation": ["knowledge", "guide", "chapter"],
        "practice_and_quiz": ["quiz", "record"],
        "progress_tracking": ["student", "record", "feedback", "chat"],
        "resource_lookup": ["knowledge", "guide", "chapter"],
    }

    _STUDENT_INTENT_TOOL_KEYWORDS = {
        "knowledge_explanation": ["knowledge", "guide"],
        "practice_and_quiz": ["quiz", "record"],
        "progress_tracking": ["record", "chat"],
        "resource_lookup": ["knowledge", "guide"],
    }

    _JSON_SCHEMA_TYPE_MAP = {
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "string": "string",
    }

    def __init__(self):
        self.tool_registry = Config.TOOL_API_REGISTRY
        self.student_tool_registry = Config.STUDENT_TOOL_API_REGISTRY

    def build_tools_from_registry(self, intent: Optional[str] = None) -> List[Dict]:
        return self._build_tools(
            registry=self.tool_registry,
            intent=intent,
            intent_mapping=self._TEACHER_INTENT_TOOL_KEYWORDS,
        )

    def build_student_tools_from_registry(self, intent: Optional[str] = None) -> List[Dict]:
        return self._build_tools(
            registry=self.student_tool_registry,
            intent=intent,
            intent_mapping=self._STUDENT_INTENT_TOOL_KEYWORDS,
        )

    def _build_tools(self, registry: Dict, intent: Optional[str], intent_mapping: Dict[str, List[str]]) -> List[Dict]:
        selected_tool_names = self._select_tool_names(registry, intent, intent_mapping)
        tools = []

        for tool_func_name in selected_tool_names:
            tool_api_cfg = registry[tool_func_name]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_func_name,
                        "description": tool_api_cfg.get("description", tool_func_name),
                        "parameters": self._build_tool_parameters(tool_api_cfg),
                    },
                }
            )

        return tools

    def _select_tool_names(self, registry: Dict, intent: Optional[str], intent_mapping: Dict[str, List[str]]) -> List[str]:
        all_names = list(registry.keys())
        if not intent or intent not in intent_mapping:
            return all_names

        keywords = intent_mapping[intent]
        selected = [name for name in all_names if self._contains_any(name, keywords)]
        return selected or all_names

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        lowered = text.lower()
        return any(keyword.lower() in lowered for keyword in keywords)

    def _build_tool_parameters(self, tool_api_cfg: Dict) -> Dict:
        parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

        body_fields = self._extract_body_fields(tool_api_cfg)
        for field_name, field_info in body_fields.items():
            parameters["properties"][field_name] = {
                "type": self._normalize_field_type(field_info.get("type", "string")),
                "description": field_info.get("description", ""),
            }

        for path_param in self._extract_url_params(tool_api_cfg.get("url", "")):
            if path_param == "TOOL_JAVA_BASE_URL":
                continue
            parameters["properties"].setdefault(
                path_param,
                {
                    "type": "string",
                    "description": f"URL path parameter: {path_param}",
                },
            )

        if not parameters["properties"]:
            parameters["properties"]["intent_reason"] = {
                "type": "string",
                "description": "Brief reason for this tool call.",
            }

        return parameters

    def _extract_body_fields(self, tool_api_cfg: Dict) -> Dict:
        request_body = tool_api_cfg.get("request_body", {})
        if not isinstance(request_body, dict):
            return {}

        if "fields" in request_body and isinstance(request_body["fields"], dict):
            return request_body["fields"]

        return {
            key: value
            for key, value in request_body.items()
            if isinstance(value, dict) and key not in {"description", "type", "examples"}
        }

    def _normalize_field_type(self, raw_type: str) -> str:
        return self._JSON_SCHEMA_TYPE_MAP.get(str(raw_type).lower(), "string")

    def parse_tool_arguments(self, raw_arguments: Optional[str]) -> Dict:
        if not raw_arguments:
            return {}

        if isinstance(raw_arguments, dict):
            return raw_arguments

        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("[tool][args] parse failed raw=%s", truncate_text(raw_arguments, Config.LOG_PREVIEW_CHARS))
            return {}

    def call_tool(
        self,
        tool_func_name: str,
        context_params: Dict,
        tool_args: Optional[Dict] = None,
        token: str = "",
        trace_id: str = "",
    ) -> str:
        return self._call_registry_tool(
            registry=self.tool_registry,
            base_url=Config.TOOL_JAVA_BASE_URL,
            tool_func_name=tool_func_name,
            context_params=context_params,
            tool_args=tool_args,
            token=token,
            trace_id=trace_id,
            source="teacher",
        )

    def call_student_tool(
        self,
        tool_func_name: str,
        context_params: Dict,
        tool_args: Optional[Dict] = None,
        token: str = "",
        trace_id: str = "",
    ) -> str:
        return self._call_registry_tool(
            registry=self.student_tool_registry,
            base_url=Config.STUDENT_TOOL_JAVA_BASE_URL,
            tool_func_name=tool_func_name,
            context_params=context_params,
            tool_args=tool_args,
            token=token,
            trace_id=trace_id,
            source="student",
        )

    def _call_registry_tool(
        self,
        registry: Dict,
        base_url: str,
        tool_func_name: str,
        context_params: Dict,
        tool_args: Optional[Dict],
        token: str,
        trace_id: str,
        source: str,
    ) -> str:
        if tool_func_name not in registry:
            logger.error("[tool][not_found] trace=%s source=%s tool=%s", trace_id or "-", source, tool_func_name)
            return f"Tool {tool_func_name} not found."

        call_start = time.perf_counter()
        tool_api_cfg = registry[tool_func_name]
        merged_params = self._merge_params(context_params, tool_args)

        request_url = tool_api_cfg["url"].replace("{TOOL_JAVA_BASE_URL}", base_url)
        request_url, consumed_path_keys = self._resolve_url_params(request_url, merged_params)

        method = tool_api_cfg.get("method", "GET").upper()
        headers = {"Authorization": token} if token else {}

        if Config.ENABLE_TOOL_VERBOSE_LOG:
            logger.info(
                "[tool][start] trace=%s source=%s tool=%s method=%s url=%s params=%s",
                trace_id or "-",
                source,
                tool_func_name,
                method,
                request_url,
                truncate_json(merged_params, Config.LOG_PREVIEW_CHARS),
            )

        try:
            if method == "GET":
                query_params = self._strip_consumed_keys(merged_params, consumed_path_keys)
                response = requests.get(
                    request_url,
                    params=query_params,
                    headers=headers,
                    timeout=Config.TOOL_CALL_TIMEOUT_SECONDS,
                )
            else:
                request_body = self._build_request_body(tool_api_cfg, merged_params, consumed_path_keys)
                response = requests.post(
                    request_url,
                    json=request_body,
                    headers=headers,
                    timeout=Config.TOOL_CALL_TIMEOUT_SECONDS,
                )

            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            result = json.dumps(data, ensure_ascii=False)

            logger.info(
                "[tool][end] trace=%s source=%s tool=%s status=%s duration_ms=%.2f result=%s",
                trace_id or "-",
                source,
                tool_func_name,
                response.status_code,
                (time.perf_counter() - call_start) * 1000,
                truncate_text(result, Config.LOG_PREVIEW_CHARS),
            )
            return result

        except Exception as exc:
            logger.error(
                "[tool][error] trace=%s source=%s tool=%s duration_ms=%.2f err=%s",
                trace_id or "-",
                source,
                tool_func_name,
                (time.perf_counter() - call_start) * 1000,
                exc,
            )
            return f"Internal tool call failed: {exc}"

    def _merge_params(self, context_params: Dict, tool_args: Optional[Dict]) -> Dict:
        merged = {}
        for key, value in (context_params or {}).items():
            if value is not None and value != "":
                merged[key] = value
        for key, value in (tool_args or {}).items():
            if value is not None and value != "":
                merged[key] = value
        return merged

    def _extract_url_params(self, url: str) -> List[str]:
        return re.findall(r"\{([^}]+)\}", url)

    def _resolve_url_params(self, request_url: str, merged_params: Dict) -> Tuple[str, Set[str]]:
        consumed_path_keys: Set[str] = set()
        for key in self._extract_url_params(request_url):
            if key in merged_params:
                request_url = request_url.replace(f"{{{key}}}", str(merged_params[key]))
                consumed_path_keys.add(key)
        return request_url, consumed_path_keys

    def _strip_consumed_keys(self, merged_params: Dict, consumed_path_keys: Set[str]) -> Dict:
        return {
            key: value
            for key, value in merged_params.items()
            if key not in consumed_path_keys and key != "intent_reason"
        }

    def _build_request_body(self, tool_api_cfg: Dict, merged_params: Dict, consumed_path_keys: Set[str]) -> Dict:
        body_fields = self._extract_body_fields(tool_api_cfg)
        request_body = {}

        if body_fields:
            for field_name in body_fields:
                if field_name in merged_params and field_name not in consumed_path_keys:
                    request_body[field_name] = merged_params[field_name]
        else:
            request_body = self._strip_consumed_keys(merged_params, consumed_path_keys)

        request_body = self._append_condition_fields_if_needed(body_fields, request_body, merged_params)

        if "pageNum" not in request_body and (not body_fields or "pageNum" in body_fields):
            request_body["pageNum"] = Config.TOOL_DEFAULT_PAGE_NUM
        if "pageSize" not in request_body and (not body_fields or "pageSize" in body_fields):
            request_body["pageSize"] = Config.TOOL_DEFAULT_PAGE_SIZE

        return request_body

    def _append_condition_fields_if_needed(self, body_fields: Dict, request_body: Dict, merged_params: Dict) -> Dict:
        has_condition_schema = "exactConditions" in body_fields or "fuzzyConditions" in body_fields
        if not has_condition_schema:
            return request_body

        if "exactConditions" in merged_params:
            request_body["exactConditions"] = merged_params["exactConditions"]
        if "fuzzyConditions" in merged_params:
            request_body["fuzzyConditions"] = merged_params["fuzzyConditions"]

        if "exactConditions" in request_body or "fuzzyConditions" in request_body:
            return request_body

        exact_conditions = {}
        fuzzy_conditions = {}

        for key, value in merged_params.items():
            if key in {"windowId", "chapterId", "userId", "pageNum", "pageSize", "intent_reason"}:
                continue
            if self._should_use_exact_match(key, value):
                exact_conditions[key] = value
            else:
                fuzzy_conditions[key] = value

        if exact_conditions:
            request_body["exactConditions"] = exact_conditions
        if fuzzy_conditions:
            request_body["fuzzyConditions"] = fuzzy_conditions

        return request_body

    def _should_use_exact_match(self, param_name: str, param_value) -> bool:
        id_keywords = ["id", "status", "type", "is"]
        lower_param = param_name.lower()
        if any(keyword in lower_param for keyword in id_keywords):
            return True
        return isinstance(param_value, (int, float, bool))
