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
    import logging

    from api.models.bedrock import get_bedrock_model_list
    from api.services.model_availability import get_model_availability_service
    from api.setting import AWS_REGION

    logger = logging.getLogger(__name__)

    # Get the full model list
    all_models = chat_model.list_models()

    # Filter models based on availability checking
    try:
        availability_service = get_model_availability_service()
        available_models = await availability_service.get_available_models(AWS_REGION)

        if available_models:
            # Filter to only include available models or custom models
            bedrock_models = get_bedrock_model_list()
            filtered_models = [
                model_id
                for model_id in all_models
                if model_id in available_models or bedrock_models.get(model_id, {}).get("type") == "custom"
            ]
            original_count = len(all_models)
            filtered_count = len(filtered_models)
            logger.info(f"Filtered models based on availability: {original_count} -> {filtered_count}")
        else:
            logger.info("Model availability checking returned no available models or is disabled")
            filtered_models = all_models
    except Exception as e:
        logger.warning(f"Error checking model availability, including all models: {e}")
        filtered_models = all_models

    # Build the response
    model_list = []
    for model_id in filtered_models:
        # Model ID already includes the name for custom models
        model = Model(id=model_id)
        model_list.append(model)

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
    from api.models.bedrock import get_bedrock_model_list

    await validate_model_id(model_id)
    model = Model(id=model_id)

    # Add model name if available
    model_info = get_bedrock_model_list().get(model_id, {})
    if "name" in model_info:
        model.name = model_info["name"]

    return model
