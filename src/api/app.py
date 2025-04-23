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

def get_gcp_target():
    """
    Check if the environment variable is set to use GCP.
    """

    # TODO: are these the best env vars to check for?
    project_id = os.getenv("GCP_PROJECT_ID")
    project_id = project_id if project_id else os.getenv("GCLOUD_PROJECT")
    location = os.getenv("GCP_REGION")

    if project_id and location:
        return f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"

    return None

def get_docker_target():
    """
    Check if the environment variable is set to use Docker.
    """

    # test if http://model-runner.docker.internal/ is reachable
    try:
        response = httpx.get("http://model-runner.docker.internal/")
        if response.status_code == 200:
            return "http://model-runner.docker.internal/"
    except httpx.RequestError as e:
        logging.error(f"Error connecting to Docker target: {e}")

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
    docker_target = get_docker_target()
    if docker_target:
        return docker_target

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
        async with httpx.AsyncClient() as client:
            # Read body only once
            body = await request.body()

        # Build safe target URL
        target_url = proxy_target.rstrip('/')
        if path:
            target_url += '/' + path.lstrip('/')

        logging.info(f"Forwarding request to: {target_url}")
        logging.info(f"Request method: {request.method}")
        logging.info(f"Incoming path: {request.url.path}")
        logging.info(f"Request body:\n{body.decode('utf-8', errors='replace')}")

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
                    content=body,
                    params=dict(request.query_params),
                    timeout=30.0,
                )
        except httpx.RequestError as e:
            logging.error(f"Proxy request failed: {e}")
            return Response(status_code=502, content=f"Upstream request failed: {e}")

        logging.info(f"Upstream responded with {response.status_code}")

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type", "application/octet-stream"),
        )            
            # # Build the target URL
            # target_url = f"{proxy_target}/{path}".rstrip("/")

            # body = await request.body()

            # # Print or log the request body
            # logging.info(f"Forwarding request to: {target_url}")
            # logging.info(f"Request method: {request.method}")
            # logging.info(f"Request body:\n{body.decode('utf-8', errors='replace')}")


            # logging.info(f"Incoming path: {request.url.path}")
            # # Forward the request
            # resp = await client.request(
            #     method=request.method,
            #     url=target_url,
            #     headers=request.headers.raw,
            #     # TODO: can we avoid deserializing the body and re-serializing it?
            #     content=body,
            #     params=request.query_params,
            # )

            # # async with httpx.AsyncClient() as client:
            # #     try:
            # #         resp = await client.request(
            # #             method=request.method,
            # #             url=target_url,
            # #             headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            # #             content=body,
            # #             params=request.query_params,
            # #         )
            # #     except httpx.RequestError as e:
            # #         logging.error(f"Proxy request failed: {e}")
            # #         return Response(status_code=502, content=f"Upstream request failed: {e}")


            # # TODO: can we avoid deserializing the response body and re-serializing it?
            # # Return the response with the same status code and headers
            # # return response.json(), response.status_code, dict(response.headers)
    
            # return Response(
            #     content=resp.content,
            #     status_code=resp.status_code,
            #     headers=dict(resp.headers),
            #     media_type=resp.headers.get("content-type", "application/octet-stream"),
            # )

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
