import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import Models, Model

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

chat_model = BedrockModel()


async def validate_model_id(model_id: str):
    try:
        log.debug("Validating model ID: %s", model_id)
        if model_id not in chat_model.list_models():
            log.warning("Unsupported Model ID: %s", model_id)
            raise HTTPException(status_code=500, detail="Unsupported Model Id")
        log.info("Model ID validated successfully: %s", model_id)
    except Exception as e:
        log.exception("Error during model validation: %s", e)
        raise

@router.get("", response_model=Models)
async def list_models():
    try:
        log.info("Listing available models.")
        model_list = [
            Model(id=model_id) for model_id in chat_model.list_models()
        ]
        log.debug("Available models: %s", model_list)
        return Models(data=model_list)
    except Exception as e:
        log.exception("Error while listing models: %s", e)
        raise HTTPException(status_code=500, detail="Unable to list models")



@router.get(
    "/{model_id}",
    response_model=Model,
)
async def get_model(
        model_id: Annotated[
            str,
            Path(description="Model ID", example="anthropic.claude-3-sonnet-20240229-v1:0"),
        ]
):
    try:
        log.info("Fetching details for model ID: %s", model_id)
        await validate_model_id(model_id)
        return Model(id=model_id)
    except HTTPException as e:
        log.warning("HTTPException: %s", e.detail)
        raise
    except Exception as e:
        log.exception("Unexpected error while fetching model: %s", e)
        raise HTTPException(status_code=500, detail="Unable to fetch model details")
