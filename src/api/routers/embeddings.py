import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Body

from api.auth import api_key_auth
from api.models.bedrock import get_embeddings_model
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

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
                        "input": [
                            "Your text string goes here"
                        ],
                    }
                ],
            ),
        ]
):
    log.debug("Received embeddings request: %s", embeddings_request)
    try:
        if embeddings_request.model.lower().startswith("text-embedding-"):
            embeddings_request.model = DEFAULT_EMBEDDING_MODEL
        # Exception will be raised if model not supported.
        model = get_embeddings_model(embeddings_request.model)
        # Generate embeddings
        log.info("Generating embeddings for input.")
        response = model.embed(embeddings_request)
        log.debug("Generated embeddings response: %s", response)

        return response

    except ValueError as e:
        # Handle validation-related issues
        log.warning("Validation error: %s", e)
        raise e  # Optionally re-raise or return an HTTPException

    except Exception as e:
        # Log unexpected exceptions with stack trace
        log.exception("Unexpected error during embeddings generation: %s", e)
        raise e
