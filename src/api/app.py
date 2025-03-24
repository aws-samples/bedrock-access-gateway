import logging

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.routers import chat, embeddings, model
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


@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


handler = Mangum(app)

if __name__ == "__main__":
    import os
    import multiprocessing
    
    # Check environment setting via environment variable (default is development)
    env = os.environ.get("ENV", "development")
    
    if env == "production":
        # Production environment: Set workers count (recommended: CPU cores * 2 + 1)
        workers_count = (multiprocessing.cpu_count() * 2) + 1
        uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=workers_count)
        logging.info(f"Server started in production mode. Workers: {workers_count}")
    else:
        # Development environment: Single process with auto-reload enabled
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
        logging.info("Server started in development mode.")
