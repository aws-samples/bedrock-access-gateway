import time
from typing import Literal

from pydantic import BaseModel, Field


class Model(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str | None = "model"
    owned_by: str | None = "bedrock"


class Models(BaseModel):
    object: str | None = "list"
    data: list[Model] = []


class ChatRequestMessage(BaseModel):
    name: str | None = None
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatRequestMessage]
    model: str
    frequency_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    presence_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    stream: bool | None = False
    temperature: float | None = Field(default=1.0, le=2.0, ge=0.0)
    top_p: float | None = Field(default=1.0, le=1.0, ge=0.0)
    user: str | None = None  # Not used
    max_tokens: int | None = 2048
    n: int | None = 1  # Not used


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseMessage(BaseModel):
    # tool_calls
    role: Literal["assistant"] | None = None
    content: str | None = None


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
    system_fingerprint: str = "fp_e97c09dd4e26"


class ChatResponse(BaseChatResponse):
    choices: list[Choice]
    object: Literal["chat.completion"] = "chat.completion"
    usage: Usage


class ChatStreamResponse(BaseChatResponse):
    choices: list[ChoiceDelta]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str
    # Cohere Embed
    input_type: Literal["search_document", "search_query", "classification", "clustering"] | None = None
    truncate: Literal["NONE", "LEFT", "RIGHT"] | None = None
    # Titan Embeddings
    embedding_config: dict | None = None


class BaseEmbeddingsResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str


class Embedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseEmbeddingsResponse):
    data: list[Embedding]
    object: Literal["list"] = "list"
    usage: EmbeddingsUsage