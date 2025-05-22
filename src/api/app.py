import os
import logging
import uvicorn

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.routers.aws import chat, embeddings, model
from api.routers.gcp import predict
from api.routers.generic import proxy
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, GCP_ENDPOINT, GCP_PROJECT_ID, REGION, SUMMARY, PROVIDER, TITLE, USE_MODEL_MAPPING, VERSION

from api.modelmapper import load_model_map

if USE_MODEL_MAPPING:
    load_model_map()


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

if os.getenv("PROXY_TARGET"):
    logging.info("Proxy target set to: Generic")
    app.include_router(proxy.router, prefix=API_ROUTE_PREFIX)
elif PROVIDER.lower() == "gcp":
    logging.info("Proxy target set to: GCP")
    app.include_router(predict.router, prefix=API_ROUTE_PREFIX)
else:
    logging.info("Proxy target set to: AWS")
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
