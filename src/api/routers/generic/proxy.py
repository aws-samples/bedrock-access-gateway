import os
import httpx

from fastapi import APIRouter, Request, Response

proxy_target = os.getenv("PROXY_TARGET")

router = APIRouter()

@router.api_route("/{path:path}", methods=["POST"])
async def proxy(request: Request, path: str):
    body = await request.body()

    # Forward most headers (excluding host, content-length)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ["host", "content-length"]
    }

    # Build the full proxy URL
    proxy_url = f"https://{proxy_target}/{path}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            proxy_url,
            content=body,
            headers=headers,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )
