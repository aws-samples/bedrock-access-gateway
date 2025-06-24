from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from api.auth import api_key_auth
from api.auth_utils import get_user_context, validate_model_access
from api.models.bedrock import BedrockModel
from api.schema import Model, Models

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

chat_model = BedrockModel()


async def validate_model_id(model_id: str):
    if model_id not in chat_model.list_models():
        raise HTTPException(status_code=500, detail="Unsupported Model Id")


@router.get("", response_model=Models)
async def list_models(request: Request):
    """List models available to the authenticated user."""
    user_context = get_user_context(request)
    
    # Get all available models
    all_models = chat_model.list_models()
    
    # Filter models based on user permissions
    if user_context:
        # Multi-tenant mode: filter by allowed models
        allowed_models = []
        for model_id in all_models:
            if validate_model_access(request, model_id):
                allowed_models.append(model_id)
        model_list = [Model(id=model_id) for model_id in allowed_models]
    else:
        # Single-key mode: return all models
        model_list = [Model(id=model_id) for model_id in all_models]
    
    return Models(data=model_list)


@router.get(
    "/{model_id}",
    response_model=Model,
)
async def get_model(
    model_id: Annotated[
        str,
        Path(description="Model ID", example="anthropic.claude-3-sonnet-20240229-v1:0"),
    ],
    request: Request,
):
    """Get model details if user has access."""
    await validate_model_id(model_id)
    
    # Check user access to this specific model
    if not validate_model_access(request, model_id):
        raise HTTPException(
            status_code=403, 
            detail=f"Access denied to model {model_id}"
        )
    
    return Model(id=model_id)
