import logging

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum
import httpx
import json
import os
from contextlib import asynccontextmanager

from api.routers import chat, embeddings, model
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION

from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.modelmapper import USE_MODEL_MAPPING, get_model, load_model_map

if USE_MODEL_MAPPING:
    load_model_map()

# Utility: get service account access token
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = AuthRequest()
    credentials.refresh(auth_request)
    return credentials.token

def get_gcp_target():
    """
    Check if the environment variable is set to use GCP.
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION")
    endpoint = os.getenv("GCP_ENDPOINT", "openai")

    if project_id and region:
        return f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/{endpoint}/"

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

proxy_target = get_proxy_target()
if proxy_target:
    logging.info(f"Proxy target set to: {proxy_target}")
else:
    logging.info("No proxy target set. Using internal routers.")

app = FastAPI(**config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if proxy_target:
    logging.info(f"Proxy target set to: {proxy_target}")
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def proxy(request: Request, path: str):
        # Build safe target URL
        path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
        target_url = f"{proxy_target.rstrip('/')}/{path_no_prefix.lstrip('/')}".rstrip("/")

        # remove hop-by-hop headers
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "accept-encoding", "connection", "authorization"}
        }

        # Fetch service account token
        access_token = get_access_token()
        headers["Authorization"] = f"Bearer {access_token}"

        try:
            content = await request.body()

            if USE_MODEL_MAPPING:
                request_model = content.get("model", None)
                content["model"] = get_model(request_model)
                content = json.dumps(content)

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=content,
                    params=request.query_params,
                    timeout=30.0,
                )
        except httpx.RequestError as e:
            logging.error(f"Proxy request failed: {e}")
            return Response(status_code=502, content=f"Upstream request failed: {e}")

        # remove hop-by-hop headers
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
    logging.info("No proxy target set. Using internal routers.")
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