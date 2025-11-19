from typing import Annotated

from fastapi import APIRouter, Body, Depends, Header, Request
from fastapi.responses import StreamingResponse
from langfuse.decorators import langfuse_context, observe

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


def extract_langfuse_metadata(chat_request: ChatRequest, headers: dict) -> dict:
    """Extract Langfuse tracing metadata from request body and headers.
    
    Metadata can be provided via:
    1. extra_body.langfuse_metadata dict in the request
    2. HTTP headers: X-Chat-Id, X-User-Id, X-Session-Id, X-Message-Id
    3. user field in the request (for user_id)
    
    Returns a dict with: user_id, session_id, chat_id, message_id, and any custom metadata
    """
    metadata = {}
    
    # Extract from extra_body if present
    if chat_request.extra_body and isinstance(chat_request.extra_body, dict):
        langfuse_meta = chat_request.extra_body.get("langfuse_metadata", {})
        if isinstance(langfuse_meta, dict):
            metadata.update(langfuse_meta)
    
    # Extract from headers
    headers_lower = {k.lower(): v for k, v in headers.items()}
    
    # Map headers to metadata fields - support both standard and OpenWebUI-prefixed headers
    header_mapping = {
        "x-chat-id": "chat_id",
        "x-openwebui-chat-id": "chat_id",  # OpenWebUI sends this format
        "x-user-id": "user_id",
        "x-openwebui-user-id": "user_id",  # OpenWebUI sends this format
        "x-session-id": "session_id",
        "x-openwebui-session-id": "session_id",  # OpenWebUI sends this format
        "x-message-id": "message_id",
        "x-openwebui-message-id": "message_id",  # OpenWebUI sends this format
    }
    
    for header_key, meta_key in header_mapping.items():
        if header_key in headers_lower and headers_lower[header_key]:
            # Don't override if already set (standard headers take precedence)
            if meta_key not in metadata:
                metadata[meta_key] = headers_lower[header_key]
    
    # Use the 'user' field from request as user_id if not already set
    if "user_id" not in metadata and chat_request.user:
        metadata["user_id"] = chat_request.user
    
    return metadata


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
@observe(as_type="generation", name="chat_completion")
async def chat_completions(
    request: Request,
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            ],
        ),
    ],
):
    # Extract metadata for Langfuse tracing
    metadata = extract_langfuse_metadata(chat_request, dict(request.headers))
    
    # Create trace name using chat_id if available
    trace_name = f"chat:{metadata.get('chat_id', 'unknown')}"
    
    # Update trace with metadata, user_id, and session_id
    langfuse_context.update_current_trace(
        name=trace_name,
        user_id=metadata.get("user_id"),
        session_id=metadata.get("session_id"),
        metadata=metadata,
        input={
            "model": chat_request.model,
            "messages": [msg.model_dump() for msg in chat_request.messages],
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "tools": [tool.model_dump() for tool in chat_request.tools] if chat_request.tools else None,
        }
    )

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    
    if chat_request.stream:
        return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
    
    response = await model.chat(chat_request)
    
    # Update trace with output for non-streaming
    langfuse_context.update_current_trace(
        output={
            "message": response.choices[0].message.model_dump() if response.choices else None,
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
        }
    )
    
    return response
