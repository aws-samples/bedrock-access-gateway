from typing import Annotated

from fastapi import APIRouter, Depends, Body, HTTPException

from api.auth import api_key_auth
from api.models import get_embeddings_model, SUPPORTED_BEDROCK_EMBEDDING_MODELS
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL

router = APIRouter()

router = APIRouter(
    prefix="/embeddings",
    tags=["items"],
    dependencies=[Depends(api_key_auth)],
)


@router.post("/", response_model=EmbeddingsResponse)
async def embeddings(
        embeddings_request: Annotated[
            EmbeddingsRequest,
            Body(
                examples=[
                    {
                        "model": "cohere.embed-multilingual-v3",
                        "input": [
                            "Your text string goes here"
                        ],
                    }
                ],
            ),
        ]
):
    if embeddings_request.model.lower().startswith("text-embedding-"):
        embeddings_request.model = DEFAULT_EMBEDDING_MODEL
    if embeddings_request.model not in SUPPORTED_BEDROCK_EMBEDDING_MODELS.keys():
        raise HTTPException(status_code=400, detail="Unsupported Model Id " + embeddings_request.model)
    try:
        model = get_embeddings_model(embeddings_request.model)
        return model.embed(embeddings_request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
