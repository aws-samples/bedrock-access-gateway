from typing import Annotated

from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models import get_model, SUPPORTED_BEDROCK_MODELS
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from api.setting import DEFAULT_MODEL

router = APIRouter()

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/completions", response_model=ChatResponse | ChatStreamResponse)
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
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL
    if chat_request.model not in SUPPORTED_BEDROCK_MODELS.keys():
        raise HTTPException(status_code=400, detail="Unsupported Model Id " + chat_request.model)
    try:
        model = get_model(chat_request.model)

        if chat_request.stream:
            return StreamingResponse(
                content=model.chat_stream(chat_request), media_type="text/event-stream"
            )
        return model.chat(chat_request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
