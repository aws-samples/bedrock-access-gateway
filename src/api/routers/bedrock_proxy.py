import os
import logging
from typing import Dict, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from aws_bedrock_token_generator import provide_token

from api.auth import api_key_auth
from api.setting import AWS_REGION, DEBUG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bedrock")

# Get static token if provided (convenience feature)
AWS_BEARER_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

def get_aws_bearer_token() -> str:
    """Get AWS bearer token - static if provided, otherwise auto-generate"""
    if AWS_BEARER_TOKEN:
        logger.debug("Using static AWS bearer token")
        return AWS_BEARER_TOKEN
    
    # Default: auto-generate token using AWS SDK credentials
    try:
        token = provide_token(region=AWS_REGION)
        logger.debug("Generated fresh AWS Bedrock token")
        return token
    except Exception as e:
        logger.error(f"Failed to generate AWS token: {e}")
        raise HTTPException(status_code=503, detail="Failed to generate AWS authentication token. Ensure AWS credentials are configured or set AWS_BEARER_TOKEN_BEDROCK")


def get_aws_url(model_id: str, endpoint_path: str) -> str:
    """Convert proxy path to AWS Bedrock URL"""
    encoded_model_id = quote(model_id, safe='')
    base_url = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"
    return f"{base_url}/model/{encoded_model_id}/{endpoint_path}"


def get_proxy_headers(request: Request) -> Dict[str, str]:
    """Get headers to forward to AWS, replacing Authorization"""
    headers = dict(request.headers)

    # Remove proxy authorization and add AWS bearer token
    headers.pop("authorization", None)
    headers.pop("host", None)  # Let httpx set the correct host

    # Get fresh AWS token (static or auto-generated)
    aws_token = get_aws_bearer_token()
    headers["Authorization"] = f"Bearer {aws_token}"

    return headers


@router.api_route("/model/{model_id}/{endpoint_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def transparent_proxy(
    request: Request,
    background_tasks: BackgroundTasks,
    model_id: str,
    endpoint_path: str,
    _: None = Depends(api_key_auth)
):
    """
    Transparent HTTP proxy to AWS Bedrock.
    Forwards all requests as-is, only changing auth and URL.
    Supports both static tokens and auto-refresh tokens.
    """

    # Build AWS URL
    aws_url = get_aws_url(model_id, endpoint_path)

    # Get headers to forward
    proxy_headers = get_proxy_headers(request)

    # Get request body
    body = await request.body()

    if DEBUG:
        logger.info(f"Proxying {request.method} to: {aws_url}")
        logger.info(f"Headers: {dict(proxy_headers)}")
        if body:
            logger.info(f"Body length: {len(body)} bytes")

    try:
        # Always use streaming for transparent pass-through
        client = httpx.AsyncClient()

        # Add cleanup task
        async def cleanup_client():
            await client.aclose()

        background_tasks.add_task(cleanup_client)

        # Use a custom response class that captures headers from the stream
        stream_request = client.stream(
            method=request.method,
            url=aws_url,
            headers=proxy_headers,
            content=body,
            params=request.query_params,
            timeout=120.0
        )
        
        # Start the stream to get response object
        response = await stream_request.__aenter__()
        
        # Schedule cleanup
        async def cleanup_stream():
            await stream_request.__aexit__(None, None, None)
        background_tasks.add_task(cleanup_stream)

        async def stream_generator():
            async for chunk in response.aiter_bytes():
                if chunk:  # Only yield non-empty chunks
                    yield chunk

        # Create StreamingResponse with AWS response headers and status
        return StreamingResponse(
            content=stream_generator(),
            status_code=response.status_code,
            headers=dict(response.headers)
        )

    except httpx.RequestError as e:
        logger.error(f"Proxy request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"AWS returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail="Proxy error")