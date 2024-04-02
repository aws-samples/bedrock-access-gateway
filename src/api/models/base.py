import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterable

from api.schema import (
    # Chat
    ChatResponse,
    ChatRequest,
    ChatStreamResponse,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
)


class BaseChatModel(ABC):
    """Represent a basic chat model

    Currently, only Bedrock model is supported, but may be used for SageMaker models if needed.
    """

    @abstractmethod
    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Handle a basic chat completion requests."""
        pass

    @abstractmethod
    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Handle a basic chat completion requests with stream response."""
        pass

    def _generate_message_id(self) -> str:
        return "chatcmpl-" + str(uuid.uuid4())[:8]

    def _stream_response_to_bytes(self, response: ChatStreamResponse) -> bytes:
        return "data: {}\n\n".format(response.model_dump_json()).encode("utf-8")


class BaseEmbeddingsModel(ABC):
    """Represents a basic embeddings model.

    Currently, only Bedrock-provided models are supported, but it may be used for SageMaker models if needed.
    """

    @abstractmethod
    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Handle a basic embeddings request."""
        pass

    def _generate_message_id(self) -> str:
        return "embeddings-" + str(uuid.uuid4())[:8]
