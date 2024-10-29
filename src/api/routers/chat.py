import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from api.setting import DEFAULT_MODEL

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/completions", response_model=ChatResponse | ChatStreamResponse, response_model_exclude_unset=True)
async def chat_completions(
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
        ]
):
    log.debug("Received chat request: %s", chat_request)
    try:
        if chat_request.model.lower().startswith("gpt-"):
            log.info("Using default model instead of GPT-* variant.")
            chat_request.model = DEFAULT_MODEL

        # Exception will be raised if model not supported.
        model = BedrockModel()
        model.validate(chat_request)
        if chat_request.stream:
            log.info("Streaming response requested.")
            return StreamingResponse(
                content=model.chat_stream(chat_request), media_type="text/event-stream"
            )
        log.info("Processing chat request.")
        response = model.chat(chat_request)
        log.debug("Chat response: %s", response)
        return response

    except ValueError as e:
        # Handle validation errors
        log.warning("Validation error: %s", e)
        raise e  # Optionally, re-raise or handle with HTTPException

    except Exception as e:
        # Log unexpected exceptions with stack trace
        log.exception("Unexpected error while processing chat request: %s", e)
        raise e
