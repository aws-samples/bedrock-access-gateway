import logging

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.routers import chat, embeddings, model, bedrock_proxy
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION

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


app.include_router(model.router, prefix=API_ROUTE_PREFIX)
app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)
app.include_router(bedrock_proxy.router, prefix=API_ROUTE_PREFIX)


@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger = logging.getLogger(__name__)
    
    # Log essential info only - avoid sensitive data and performance overhead
    logger.warning(
        "Request validation failed: %s %s - %s", 
        request.method, 
        request.url.path,
        str(exc).split('\n')[0]  # First line only
    )
    
    return PlainTextResponse(str(exc), status_code=400)


handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
