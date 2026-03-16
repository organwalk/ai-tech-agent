from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptProfile:
    role: str
    intent: str
    learner_level: str
    response_style: str


CHAT_TOOL_RULES = (
    "【工具调用规则】\n"
    "1. 只要问题涉及系统动态数据，请优先调用工具，不要主观猜测。\n"
    "2. 工具调用后要立即判断信息是否充分，充分则停止调用并直接回答。\n"
    "3. 严禁重复调用相同工具+相同参数，避免无效循环。\n"
    "4. 回答时使用自然语言，不直接暴露底层 JSON 字段。\n"
    "5. 若工具返回空数据或失败，明确说明当前限制并给出下一步建议。"
)


def build_chat_system_prompt(profile: PromptProfile) -> str:
    role_label = "教师用户" if profile.role == "teacher" else "学员用户"
    role_instruction = (
        "请提供决策支持、班级诊断和教学建议，突出可执行项。"
        if profile.role == "teacher"
        else "请提供分层讲解、步骤化引导和可练习建议，帮助学习者逐步掌握。"
    )
    return (
        f"你是一名专业的智能学习助手，当前服务对象是【{role_label}】。\n"
        f"{role_instruction}\n\n"
        "【个性化画像】\n"
        f"- 当前意图: {profile.intent}\n"
        f"- 学习阶段: {profile.learner_level}\n"
        f"- 回答风格: {profile.response_style}\n\n"
        f"{CHAT_TOOL_RULES}"
    )


def build_chat_user_prompt(query: str, reference_context: str, memory_context: str) -> str:
    return (
        f"【参考资料】\n{reference_context or '无'}\n\n"
        f"【历史对话记忆】\n{memory_context or '无'}\n\n"
        f"【用户最新问题】\n{query}"
    )


def build_tool_round_reflection_prompt(round_index: int, remaining_rounds: int, executed_tools: List[str]) -> str:
    tool_summary = "、".join(executed_tools) if executed_tools else "无"
    return (
        f"【第 {round_index} 轮工具调用后反思】\n"
        f"- 已调用工具: {tool_summary}\n"
        f"- 剩余可调用轮次: {remaining_rounds}\n"
        "- 请判断当前信息是否足够回答用户。\n"
        "- 若足够，立即停止工具调用并直接回答。\n"
        "- 若不足，仅调用最必要的下一步工具。"
    )


GUIDE_SYSTEM_PROMPT = (
    "你是一位专业、耐心的学习导师。"
    "请基于用户给出的主题和参考内容，生成结构清晰、可执行的学习指南。\n"
    "【输出要求】\n"
    "1. 必须包含三个部分: 核心概念、示例讲解、简答题(附答案)。\n"
    "2. 使用 Markdown 排版。\n"
    "3. 参考内容不足时，可补充常识，但应优先使用给定资料。\n"
    "4. 直接输出内容，不要使用 ```markdown 代码块。"
)


QUIZ_JSON_TEMPLATE = (
    "[\n"
    '  {\n'
    '    "id": "q1",\n'
    '    "text": "题目描述",\n'
    '    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],\n'
    '    "correct": "B",\n'
    '    "analysis": "答案解析"\n'
    "  }\n"
    "]"
)


def build_quiz_instruction(reference_context: str) -> str:
    return (
        "【命题背景参考资料】\n"
        f"{reference_context}\n\n"
        "【输出格式要求】\n"
        "1. 仅输出 JSON 数组。\n"
        f"2. 严格遵循以下结构:\n{QUIZ_JSON_TEMPLATE}\n"
        "3. 使用 Markdown 的 ```json 代码块包裹。\n"
        "4. 不要输出额外开场白或结尾语。"
    )


def build_quiz_analysis_system_prompt(requirement: str) -> str:
    return (
        "你是一位资深教育专家与 AI 导师。"
        "请基于学员答卷数据，输出深度试卷分析与改进建议。\n"
        "【输出要求】\n"
        f"{requirement}\n"
        "1. 直接输出 Markdown 正文。\n"
        "2. 不要使用 ```markdown 代码块。\n"
        "3. 不要输出开场白或结束语。"
    )


DIAGNOSIS_JSON_TEMPLATE = (
    "{\n"
    '  "skills": [\n'
    "    {\n"
    '      "tag": "技能/知识点",\n'
    '      "desc": "一句话评价",\n'
    '      "color": "#67C23A",\n'
    '      "level": "S",\n'
    '      "levelDesc": "炉火纯青"\n'
    "    }\n"
    "  ],\n"
    '  "focus_score": 80\n'
    "}"
)


DIAGNOSIS_FORMAT_INSTRUCTION = (
    "【输出格式控制】\n"
    "请严格分为两部分输出:\n"
    "第一部分:\n"
    "---MARKDOWN---\n"
    "(学习状态分析、知识盲区、改进建议)\n\n"
    "第二部分:\n"
    "---JSON---\n"
    f"{DIAGNOSIS_JSON_TEMPLATE}\n\n"
    "规则:\n"
    "1. level 仅可为 S/A/B/C。\n"
    "2. color 对应 S:#67C23A, A:#409EFF, B:#E6A23C, C:#F56C6C。\n"
    "3. focus_score 为 0-100 整数。"
)

