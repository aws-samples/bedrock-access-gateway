import os
import logging
from typing import Dict, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, Response

from api.auth import api_key_auth
from api.setting import AWS_REGION, DEBUG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bedrock")

# Get AWS bearer token from environment
AWS_BEARER_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

if not AWS_BEARER_TOKEN:
    logger.warning("AWS_BEARER_TOKEN_BEDROCK not set - bedrock proxy endpoints will not work")


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

    if AWS_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {AWS_BEARER_TOKEN}"

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
    """
    if not AWS_BEARER_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="AWS_BEARER_TOKEN_BEDROCK not configured"
        )

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

        async def stream_generator():
            async with client.stream(
                method=request.method,
                url=aws_url,
                headers=proxy_headers,
                content=body,
                params=request.query_params,
                timeout=120.0
            ) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:  # Only yield non-empty chunks
                        yield chunk

        return StreamingResponse(content=stream_generator())

    except httpx.RequestError as e:
        logger.error(f"Proxy request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"AWS returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail="Proxy error")