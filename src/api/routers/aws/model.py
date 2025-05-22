from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
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
async def list_models():
    model_list = [Model(id=model_id) for model_id in chat_model.list_models()]
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
):
    await validate_model_id(model_id)
    return Model(id=model_id)
