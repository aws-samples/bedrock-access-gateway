import json
import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL, USAGE_USER_HEADER, USAGE_CHAT_ID_HEADER

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


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
    TRACE_LEVEL = 5
    if logger.isEnabledFor(TRACE_LEVEL):
        logger.log(
            TRACE_LEVEL,
            "Request headers: %s",
            json.dumps(dict(request.headers), indent=2),
        )
        logger.log(
            TRACE_LEVEL,
            "Incoming chat completion request (raw parsed body): %s",
            json.dumps(chat_request.model_dump(), indent=2, default=str),
        )
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Exception will be raised if model not supported.
    # Compute effective max_tokens (same logic as bedrock.py _parse_request)
    effective_max_tokens = (
        chat_request.max_completion_tokens
        if chat_request.max_completion_tokens is not None
        else chat_request.max_tokens
    )
    model = BedrockModel()
    model.request_meta = {
        "user_email": request.headers.get(USAGE_USER_HEADER, "-") if USAGE_USER_HEADER else "-",
        "chat_id": request.headers.get(USAGE_CHAT_ID_HEADER, "-") if USAGE_CHAT_ID_HEADER else "-",
        "max_tokens": effective_max_tokens,
        "user_agent": request.headers.get("user-agent", "-"),
    }
    model.validate(chat_request)
    if chat_request.stream:
        return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
    return await model.chat(chat_request)
