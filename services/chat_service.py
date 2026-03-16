import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from config import Config
from models import ChatMessageItem, ToolChatRequest
from prompts.prompt_templates import (
    PromptProfile,
    build_chat_system_prompt,
    build_chat_user_prompt,
    build_tool_round_reflection_prompt,
)
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.personalization_service import PersonalizationService
from services.rag_service import RAGService
from services.tool_service import ToolService
from services.tool_tracker import ToolTracker
from utils.logging_utils import new_trace_id, truncate_json, truncate_text

logger = logging.getLogger(__name__)


class ChatService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.embedding_service = EmbeddingService()
            cls._instance.rag_service = RAGService()
            cls._instance.llm_service = LLMService()
            cls._instance.tool_service = ToolService()
        return cls._instance

    def __init__(self):
        pass

    async def stream_chat(
        self,
        query: str,
        window_id: str,
        chapter_id: str,
        file_ids: Optional[List[str]],
        history: List[ChatMessageItem],
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._stream_chat_core(
            role="student",
            query=query,
            window_id=window_id,
            chapter_id=chapter_id,
            file_ids=file_ids or [],
            history=history,
            user_id="",
            token="",
        ):
            yield chunk

    async def stream_chat_with_tools(self, request: ToolChatRequest) -> AsyncGenerator[str, None]:
        async for chunk in self._stream_chat_core(
            role="teacher",
            query=request.query,
            window_id=request.windowId,
            chapter_id=request.chapterId,
            file_ids=request.fileIds or [],
            history=request.history,
            user_id=request.userId,
            token=request.token,
        ):
            yield chunk

    async def _stream_chat_core(
        self,
        role: str,
        query: str,
        window_id: str,
        chapter_id: str,
        file_ids: List[str],
        history: List[ChatMessageItem],
        user_id: str,
        token: str,
    ) -> AsyncGenerator[str, None]:
        trace_id = new_trace_id("chat_")
        req_start = time.perf_counter()
        logger.info(
            "[chat][start] trace=%s role=%s query=%s window=%s chapter=%s files=%s history=%s",
            trace_id,
            role,
            truncate_text(query, Config.LOG_PREVIEW_CHARS),
            window_id,
            chapter_id,
            len(file_ids),
            len(history),
        )

        try:
            embed_start = time.perf_counter()
            topic_vector = self.embedding_service.get_embedding(query)
            logger.info(
                "[chat][embedding] trace=%s vector_dim=%s duration_ms=%.2f",
                trace_id,
                len(topic_vector) if topic_vector else 0,
                (time.perf_counter() - embed_start) * 1000,
            )

            reference_context = ""
            if file_ids:
                retrieved_docs = self.rag_service.search_documents(
                    topic_vector,
                    file_ids,
                    n_results=Config.RAG_DOC_TOP_K,
                    query_text=query,
                    trace_id=trace_id,
                )
                if retrieved_docs:
                    reference_context = "\n---\n".join(retrieved_docs)
                logger.info(
                    "[chat][rag_docs] trace=%s retrieved=%s context_chars=%s",
                    trace_id,
                    len(retrieved_docs),
                    len(reference_context),
                )

            memory_context = ""
            if window_id and chapter_id:
                retrieved_mems = self.rag_service.search_chat_memory(
                    topic_vector,
                    window_id,
                    chapter_id,
                    n_results=Config.RAG_MEMORY_TOP_K,
                    query_text=query,
                    trace_id=trace_id,
                )
                memory_context = "\n".join(retrieved_mems)
                logger.info(
                    "[chat][rag_memory] trace=%s retrieved=%s context_chars=%s",
                    trace_id,
                    len(retrieved_mems),
                    len(memory_context),
                )

            profile = PersonalizationService.build_profile(query=query, history=history, role=role)
            prompt_profile = PromptProfile(
                role=profile.role,
                intent=profile.intent,
                learner_level=profile.learner_level,
                response_style=profile.response_style,
            )
            logger.info(
                "[chat][profile] trace=%s intent=%s level=%s style=%s",
                trace_id,
                profile.intent,
                profile.learner_level,
                profile.response_style,
            )

            messages_payload = self._build_message_payload(
                system_prompt=build_chat_system_prompt(prompt_profile),
                user_prompt=build_chat_user_prompt(query, reference_context, memory_context),
                history=history,
            )

            if file_ids:
                yield f"event: citation\ndata: {json.dumps({'docs': file_ids}, ensure_ascii=False)}\n\n"

            if role == "teacher":
                tool_list_data = self.tool_service.build_tools_from_registry(intent=profile.intent)
            else:
                tool_list_data = self.tool_service.build_student_tools_from_registry(intent=profile.intent)
            logger.info("[chat][tools] trace=%s available_tools=%s", trace_id, len(tool_list_data))

            messages_payload = self._run_tool_loop(
                role=role,
                messages_payload=messages_payload,
                tool_list_data=tool_list_data,
                window_id=window_id,
                chapter_id=chapter_id,
                user_id=user_id,
                token=token,
                trace_id=trace_id,
            )

            stream_response = self.llm_service.generate_stream_response(
                messages_payload,
                temperature=Config.FINAL_ANSWER_TEMPERATURE,
            )

            chunk_count = 0
            for chunk in stream_response:
                if not chunk.choices:
                    continue
                delta_content = chunk.choices[0].delta.content
                if not delta_content:
                    continue

                data_dict = {"content": delta_content, "is_finish": False}
                yield f"event: message\ndata: {json.dumps(data_dict, ensure_ascii=False)}\n\n"
                await asyncio.sleep(Config.STREAM_DELAY_SECONDS)
                chunk_count += 1

            end_dict = {"content": "", "is_finish": True}
            yield f"event: message\ndata: {json.dumps(end_dict, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

            logger.info(
                "[chat][end] trace=%s chunks=%s total_duration_ms=%.2f",
                trace_id,
                chunk_count,
                (time.perf_counter() - req_start) * 1000,
            )

        except Exception as exc:
            logger.exception("[chat][error] trace=%s failed", trace_id)
            error_data = {"msg": f"AI 服务异常: {str(exc)}"}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    def _build_message_payload(self, system_prompt: str, user_prompt: str, history: List[ChatMessageItem]) -> List[Dict]:
        messages_payload: List[Dict] = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages_payload.append({"role": msg.role, "content": msg.content})
        messages_payload.append({"role": "user", "content": user_prompt})
        return messages_payload

    def _run_tool_loop(
        self,
        role: str,
        messages_payload: List[Dict],
        tool_list_data: List[Dict],
        window_id: str,
        chapter_id: str,
        user_id: str,
        token: str,
        trace_id: str,
    ) -> List[Dict]:
        if not tool_list_data:
            return messages_payload

        tracker = ToolTracker(duplicate_limit=Config.TOOL_DUPLICATE_CALL_LIMIT)
        context_params = {
            "windowId": window_id,
            "chapterId": chapter_id,
        }
        if user_id:
            context_params["userId"] = user_id

        logger.info(
            "[chat][tool_loop_start] trace=%s max_rounds=%s duplicate_limit=%s",
            trace_id,
            Config.TOOL_MAX_ROUNDS,
            Config.TOOL_DUPLICATE_CALL_LIMIT,
        )

        for round_index in range(1, Config.TOOL_MAX_ROUNDS + 1):
            tracker.start_round()
            round_start = time.perf_counter()

            tool_first_response = self.llm_service.generate_with_tools(
                messages_payload,
                tool_list_data,
                temperature=Config.TOOL_CALL_TEMPERATURE,
            )
            tool_resp_msg = tool_first_response.choices[0].message

            if not tool_resp_msg.tool_calls:
                logger.info("[chat][tool_round] trace=%s round=%s no_tool_calls", trace_id, round_index)
                break

            logger.info(
                "[chat][tool_round] trace=%s round=%s tool_calls=%s",
                trace_id,
                round_index,
                len(tool_resp_msg.tool_calls),
            )

            assistant_message = {
                "role": "assistant",
                "content": tool_resp_msg.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_resp_msg.tool_calls
                ],
            }
            messages_payload.append(assistant_message)

            for tool_call in tool_resp_msg.tool_calls:
                tool_func_name = tool_call.function.name
                tool_args = self.tool_service.parse_tool_arguments(tool_call.function.arguments)
                is_allowed, _signature = tracker.register(tool_func_name, tool_args)

                logger.info(
                    "[chat][tool_call] trace=%s round=%s tool=%s args=%s",
                    trace_id,
                    round_index,
                    tool_func_name,
                    truncate_json(tool_args, Config.LOG_PREVIEW_CHARS),
                )

                if not is_allowed:
                    messages_payload.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Skip duplicated tool call: same tool and same arguments were used repeatedly.",
                        }
                    )
                    logger.warning(
                        "[chat][tool_call_skip] trace=%s round=%s tool=%s reason=duplicated",
                        trace_id,
                        round_index,
                        tool_func_name,
                    )
                    continue

                if role == "teacher":
                    tool_api_result = self.tool_service.call_tool(
                        tool_func_name=tool_func_name,
                        context_params=context_params,
                        tool_args=tool_args,
                        token=token,
                        trace_id=trace_id,
                    )
                else:
                    tool_api_result = self.tool_service.call_student_tool(
                        tool_func_name=tool_func_name,
                        context_params=context_params,
                        tool_args=tool_args,
                        token=token,
                        trace_id=trace_id,
                    )

                messages_payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_api_result,
                    }
                )
                logger.info(
                    "[chat][tool_result] trace=%s round=%s tool=%s result_preview=%s",
                    trace_id,
                    round_index,
                    tool_func_name,
                    truncate_text(tool_api_result, Config.LOG_PREVIEW_CHARS),
                )

            remaining_rounds = Config.TOOL_MAX_ROUNDS - tracker.round_count
            if remaining_rounds > 0:
                messages_payload.append(
                    {
                        "role": "system",
                        "content": build_tool_round_reflection_prompt(
                            round_index=tracker.round_count,
                            remaining_rounds=remaining_rounds,
                            executed_tools=tracker.recent_tools(),
                        ),
                    }
                )

            logger.info(
                "[chat][tool_round_end] trace=%s round=%s duration_ms=%.2f recent_tools=%s",
                trace_id,
                round_index,
                (time.perf_counter() - round_start) * 1000,
                tracker.recent_tools(),
            )

        return messages_payload
