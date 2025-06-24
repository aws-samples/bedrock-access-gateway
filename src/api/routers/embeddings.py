from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request, HTTPException

from api.auth import api_key_auth
from api.auth_utils import validate_model_access
from api.models.bedrock import get_embeddings_model
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL

router = APIRouter(
    prefix="/embeddings",
    dependencies=[Depends(api_key_auth)],
)


@router.post("", response_model=EmbeddingsResponse)
async def embeddings(
    embeddings_request: Annotated[
        EmbeddingsRequest,
        Body(
            examples=[
                {
                    "model": "cohere.embed-multilingual-v3",
                    "input": ["Your text string goes here"],
                }
            ],
        ),
    ],
    request: Request,
):
    if embeddings_request.model.lower().startswith("text-embedding-"):
        embeddings_request.model = DEFAULT_EMBEDDING_MODEL
    
    # Validate user access to the requested model
    if not validate_model_access(request, embeddings_request.model):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to model {embeddings_request.model}"
        )
    
    # Exception will be raised if model not supported.
    model = get_embeddings_model(embeddings_request.model)
    return model.embed(embeddings_request)
