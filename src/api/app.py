import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum
import httpx
import os
from contextlib import asynccontextmanager

from api.routers import chat, embeddings, model
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION

METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
HEADERS = {"Metadata-Flavor": "Google"}

proxy_target: str | None = None  # Global, will be populated on startup

async def get_project_id_and_region():
    try:
        project_id: str | None = None
        region: str | None = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Get project ID
            target_url = f"{METADATA_URL}/project/project-id"
            project_id_resp = await client.get(target_url, headers=HEADERS)
            project_id_resp.raise_for_status()
            project_id = project_id_resp.text

            # Get full region path
            target_url = f"{METADATA_URL}/instance/region"
            region_resp = await client.get(target_url, headers=HEADERS)
            region_resp.raise_for_status()
            region_full_path = region_resp.text

        region = region_full_path.split("/")[-1]
        return project_id, region

    except httpx.HTTPError as e:
        logging.error(f"Google metadata request failed: {e}")
        return project_id, region
    
async def get_gcp_target():
    """
    Check if the environment variable is set to use GCP.
    """

    project_id, region = await get_project_id_and_region()
    project_id = project_id if project_id else os.getenv("GCP_PROJECT_ID")
    region = region if region else os.getenv("GCP_REGION")

    if project_id and region:
        return f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/"

    return None

async def get_proxy_target():
    """
    Check if the environment variable is set to use a proxy.
    """
    proxy_target = os.getenv("PROXY_TARGET")
    if proxy_target:
        return proxy_target
    gcp_target = await get_gcp_target()
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


async def setup_routing(app: FastAPI):
    """
    Setup routing for the application.
    This function is called during the lifespan of the application.
    """
    if proxy_target:
        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        async def proxy(request: Request, path: str):
            # Build safe target URL
            path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
            target_url = f"{proxy_target.rstrip('/')}/{path_no_prefix.lstrip('/')}".rstrip("/")

            # Sanitize headers
            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in {"host", "content-length", "accept-encoding", "connection"}
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_target
    proxy_target = await get_proxy_target()
    if proxy_target:
        logging.info(f"Proxy target set to: {proxy_target}")
    else:
        logging.info("No proxy target set. Using internal routers.")

    await setup_routing(app)

    yield  # After this point app will shutdown
app = FastAPI(**config, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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