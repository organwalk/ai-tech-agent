import re
from dataclasses import dataclass
from typing import List

from models import ChatMessageItem


@dataclass(frozen=True)
class PersonalizationProfile:
    role: str
    intent: str
    learner_level: str
    response_style: str
    preferred_tool_groups: List[str]


class PersonalizationService:
    _INTENT_RULES = {
        "knowledge_explanation": [
            "是什么",
            "原理",
            "解释",
            "概念",
            "例子",
            "讲解",
            "理解",
        ],
        "practice_and_quiz": [
            "练习",
            "测验",
            "题目",
            "quiz",
            "测试",
            "错题",
        ],
        "progress_tracking": [
            "进度",
            "掌握",
            "薄弱",
            "成绩",
            "报告",
            "分析",
            "反馈",
        ],
        "resource_lookup": [
            "资料",
            "文档",
            "知识库",
            "参考",
            "指南",
            "文件",
        ],
    }

    _BEGINNER_HINTS = ["不会", "看不懂", "入门", "基础", "新手", "从头"]
    _ADVANCED_HINTS = ["优化", "架构", "性能", "原理", "深入", "工程化", "最佳实践"]

    @classmethod
    def build_profile(
        cls,
        query: str,
        history: List[ChatMessageItem],
        role: str,
    ) -> PersonalizationProfile:
        intent = cls._detect_intent(query)
        learner_level = cls._detect_level(query, history, role)
        response_style = cls._build_response_style(intent, learner_level, role)
        preferred_tool_groups = cls._map_tool_groups(intent, role)
        return PersonalizationProfile(
            role=role,
            intent=intent,
            learner_level=learner_level,
            response_style=response_style,
            preferred_tool_groups=preferred_tool_groups,
        )

    @classmethod
    def _detect_intent(cls, query: str) -> str:
        lowered = query.lower()
        for intent, keywords in cls._INTENT_RULES.items():
            if any(keyword in lowered for keyword in keywords):
                return intent
        return "general_consultation"

    @classmethod
    def _detect_level(cls, query: str, history: List[ChatMessageItem], role: str) -> str:
        if role == "teacher":
            return "instructor"

        context = f"{query} " + " ".join(msg.content for msg in history[-6:])
        if any(keyword in context for keyword in cls._BEGINNER_HINTS):
            return "beginner"
        if any(keyword in context for keyword in cls._ADVANCED_HINTS):
            return "advanced"

        sentence_count = len(re.findall(r"[。！？!?]", context))
        return "intermediate" if sentence_count > 6 else "beginner"

    @classmethod
    def _build_response_style(cls, intent: str, learner_level: str, role: str) -> str:
        if role == "teacher":
            if intent == "progress_tracking":
                return "数据先行、问题定位、给出教学干预建议"
            return "结论清晰、结构化、强调可执行决策"

        if learner_level == "beginner":
            return "分步骤、少术语、带类比示例"
        if intent == "practice_and_quiz":
            return "先给思路，再给标准答案和错因"
        return "概念-例子-练习三段式输出"

    @classmethod
    def _map_tool_groups(cls, intent: str, role: str) -> List[str]:
        if role == "teacher":
            mapping = {
                "progress_tracking": ["records", "feedback", "chat"],
                "practice_and_quiz": ["quiz", "record"],
                "resource_lookup": ["knowledge", "guide"],
            }
        else:
            mapping = {
                "practice_and_quiz": ["quiz", "record"],
                "resource_lookup": ["knowledge", "guide"],
                "knowledge_explanation": ["knowledge", "guide", "chat"],
            }
        return mapping.get(intent, ["knowledge", "guide", "chat"])

