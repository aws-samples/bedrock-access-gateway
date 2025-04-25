import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum
import httpx
import os

from api.routers import chat, embeddings, model
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION

METADATA_URL = "http:///metadata.google.internal/computeMetadata/v1"
HEADERS = {"Metadata-Flavor": "Google"}

async def get_project_id_and_region():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Get project ID
            project_id_resp = await client.get(f"{METADATA_URL}/project/project-id", headers=HEADERS)
            project_id_resp.raise_for_status()
            project_id = project_id_resp.text

            # Get full region path
            region_resp = await client.get(f"{METADATA_URL}/instance/region", headers=HEADERS)
            region_resp.raise_for_status()
            full_region = region_resp.text

        region = full_region.split("/")[-1]
        return project_id, region

    except httpx.HTTPError as e:
        return None, None
    
def get_gcp_target():
    """
    Check if the environment variable is set to use GCP.
    """

    project_id, location = get_project_id_and_region()
    project_id = project_id if project_id else os.getenv("GCP_PROJECT_ID")
    location = location if location else os.getenv("GCP_LOCATION")

    if project_id and location:
        return f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/"

    return None

def get_proxy_target():
    """
    Check if the environment variable is set to use a proxy.
    """
    proxy_target = os.getenv("PROXY_TARGET")
    if proxy_target:
        return proxy_target
    gcp_target = get_gcp_target()
    if gcp_target:
        return gcp_target

    return None

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
app = FastAPI(**config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

proxy_target = get_proxy_target()

if proxy_target:
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def proxy(request: Request, path: str):
        # Build safe target URL
        target_url = f"{proxy_target.rstrip('/')}/{path.lstrip('/')}".rstrip("/")

        # Sanitize headers
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "accept-encoding"}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    # TODO: can we avoid deserializing the body and re-serializing it?
                    content=await request.body(),
                    params=request.query_params,
                    timeout=30.0,
                )
        except httpx.RequestError as e:
            logging.error(f"Proxy request failed: {e}")
            return Response(status_code=502, content=f"Upstream request failed: {e}")

        # filter out headers that could cause issues to client (because we act as a proxy)
        response_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        }
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type", "application/octet-stream"),
        )
    
else:
    app.include_router(model.router, prefix=API_ROUTE_PREFIX)
    app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
    app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)


@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)