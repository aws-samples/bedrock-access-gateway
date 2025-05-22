from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # e.g., "gemini-1.5-pro-preview-0409"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatCompletionChoice]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # e.g., gemini-1.5-pro-preview-0409
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
