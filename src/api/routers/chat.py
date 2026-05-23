from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL, ENABLE_REQUEST_METADATA

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
)

# Headers forwarded by OpenWebUI when ENABLE_FORWARD_USER_INFO_HEADERS=true
_USER_INFO_HEADERS = [
    ("x-openwebui-user-name", "user"),
    ("x-openwebui-user-id", "user_id"),
    ("x-openwebui-user-email", "user_email"),
    ("x-openwebui-user-role", "user_role"),
]


def _extract_request_metadata(request: Request) -> dict[str, str]:
    """Extract user info headers into a Bedrock requestMetadata dict."""
    metadata = {}
    for header, key in _USER_INFO_HEADERS:
        if val := request.headers.get(header):
            metadata[key] = val[:256]  # Bedrock enforces 256 char limit per value
    return metadata


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
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
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Extract request metadata from forwarded user info headers
    request_metadata = _extract_request_metadata(request) if ENABLE_REQUEST_METADATA else {}

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request, request_metadata=request_metadata),
            media_type="text/event-stream",
        )
    return await model.chat(chat_request, request_metadata=request_metadata)
