from typing import List, Optional
from volcenginesdkarkruntime import Ark
from config import Config


class LLMService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = Ark(
                api_key=Config.API_KEY,
                base_url=Config.BASE_URL,
                timeout=Config.CLIENT_TIMEOUT
            )
        return cls._instance

    def __init__(self):
        self.client = self._client
        self.model_id = Config.MODEL_ID

    def generate_response(self, messages: List[dict], temperature: float = 0.7, stream: bool = False, thinking_type: str = "disabled") -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            stream=stream,
            thinking={"type": thinking_type}
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content

    def generate_stream_response(self, messages: List[dict], temperature: float = 0.7, thinking_type: str = "disabled"):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            stream=True,
            thinking={"type": thinking_type}
        )
        return response

    def generate_with_tools(self, messages: List[dict], tools: List[dict], temperature: float = 0.2, tool_choice: str = "auto"):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            stream=False
        )
        return response

    def clean_markdown_format(self, content: str) -> str:
        if content.startswith("```markdown"):
            content = content[11:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
