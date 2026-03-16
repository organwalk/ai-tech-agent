from pydantic import BaseModel
from typing import List, Optional


class ParseRequest(BaseModel):
    file_id: str
    file_url: str
    window_id: str


class GuideRequest(BaseModel):
    topic: str
    file_ids: List[str]


class AnalysisRequest(BaseModel):
    system_prompt: str
    context_data: str


class DiagnosisRequest(BaseModel):
    system_prompt: str
    user_content: str


class QuizRequest(BaseModel):
    system_prompt: str
    user_data: str
    file_ids: List[str] = []


class QuizAnalysisRequest(BaseModel):
    prompt: str
    requirement: str


class ChatMessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    windowId: str
    chapterId: str
    fileIds: List[str] = []
    history: List[ChatMessageItem] = []


class ChatMemoryRequest(BaseModel):
    window_id: str
    chapter_id: str
    msg_id: str
    role: str
    content: str


class ToolChatRequest(BaseModel):
    query: str
    windowId: str
    fileIds: List[str] = []
    history: List[ChatMessageItem] = []
    userId: str = ""
    chapterId: str = ""
    token: str = ""


class RAGEvalSample(BaseModel):
    query: str
    file_ids: List[str] = []
    relevant_chunk_ids: List[str] = []
    relevant_file_ids: List[str] = []
    relevant_keywords: List[str] = []
    needs_review: bool = False
    top_k: Optional[int] = None


class RAGEvalRequest(BaseModel):
    samples: List[RAGEvalSample]
    top_k: int = 8
    candidate_k: int = 24
