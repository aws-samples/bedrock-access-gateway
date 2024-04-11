import time
import uuid
from typing import Literal, Iterable

from pydantic import BaseModel, Field


class Model(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str | None = "model"
    owned_by: str | None = "bedrock"


class Models(BaseModel):
    object: str | None = "list"
    data: list[Model] = []


class ResponseFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal["function"] = "function"
    function: ResponseFunction


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: str | None = "auto"


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image"
    image_url: ImageUrl


class SystemMessage(BaseModel):
    name: str | None = None
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    name: str | None = None
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]


class AssistantMessage(BaseModel):
    name: str | None = None
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: object


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class ChatRequest(BaseModel):
    messages: list[SystemMessage | UserMessage | AssistantMessage | ToolMessage]
    model: str
    frequency_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    presence_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    stream: bool | None = False
    temperature: float | None = Field(default=1.0, le=2.0, ge=0.0)
    top_p: float | None = Field(default=1.0, le=1.0, ge=0.0)
    user: str | None = None  # Not used
    max_tokens: int | None = 2048
    n: int | None = 1  # Not used
    tools: list[Tool] | None = None
    tool_choice: str | object = "auto"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseMessage(BaseModel):
    # tool_calls
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class BaseChoice(BaseModel):
    index: int
    finish_reason: str | None
    logprobs: dict | None = None


class Choice(BaseChoice):
    message: ChatResponseMessage


class ChoiceDelta(BaseChoice):
    delta: ChatResponseMessage


class BaseChatResponse(BaseModel):
    # id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid.uuid4())[:8])
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = "fp"


class ChatResponse(BaseChatResponse):
    choices: list[Choice]
    object: Literal["chat.completion"] = "chat.completion"
    usage: Usage


class ChatStreamResponse(BaseChatResponse):
    choices: list[ChoiceDelta]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"


class EmbeddingsRequest(BaseModel):
    input: str | list[str] | Iterable[int | Iterable[int]]
    model: str
    encoding_format: Literal["float", "base64"] = "float"  # not used.
    dimensions: int | None = None  # not used.
    user: str | None = None  # not used.


class Embedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingsUsage
