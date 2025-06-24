from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.auth_utils import validate_model_access, get_user_inference_profile
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
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
    ],
    request: Request,
):
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Validate user access to the requested model
    if not validate_model_access(request, chat_request.model):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to model {chat_request.model}"
        )

    # Get user's inference profile for cost tracking
    user_inference_profile = get_user_inference_profile(request, chat_request.model)
    
    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    
    # Pass inference profile to model for cost tracking
    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request, user_inference_profile=user_inference_profile), 
            media_type="text/event-stream"
        )
    return await model.chat(chat_request, user_inference_profile=user_inference_profile)
