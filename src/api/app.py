import json
import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.routers import chat, embeddings, model
from api.setting import API_ROUTE_PREFIX, DEBUG, DESCRIPTION, SUMMARY, TITLE, VERSION, TRACE

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

# Register custom TRACE level (5) below DEBUG (10) for per-chunk streaming logs
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

logging.basicConfig(level=logging.INFO)
# Only set DEBUG on our own 'api' loggers, not boto3/botocore/urllib3
if TRACE:
    logging.getLogger("api").setLevel(TRACE_LEVEL)
elif DEBUG:
    logging.getLogger("api").setLevel(logging.DEBUG)
app = FastAPI(**config)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
origins_list = [origin.strip() for origin in allowed_origins.split(",")] if allowed_origins != "*" else ["*"]

# Warn if CORS allows all origins
if origins_list == ["*"]:
    logging.warning("CORS is configured to allow all origins (*). Set ALLOWED_ORIGINS environment variable to restrict access.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,  # nosec - configurable via ALLOWED_ORIGINS env var
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
    logger = logging.getLogger(__name__)

    error_count = len(exc.errors()) if hasattr(exc, 'errors') else 'unknown'
    logger.warning(
        "Request validation failed: %s %s - %s validation errors:\n%s",
        request.method,
        request.url.path,
        error_count,
        "\n".join(f"  - {e.get('loc', '?')}: {e.get('msg', '?')}" for e in exc.errors()) if hasattr(exc, 'errors') else str(exc),
    )

    # Log the raw body at TRACE level so we can see what the client actually sent
    if logger.isEnabledFor(TRACE_LEVEL):
        try:
            body = await request.body()
            raw = json.loads(body)
            # Truncate message content for readability
            if "messages" in raw:
                for msg in raw["messages"]:
                    if "content" in msg:
                        c = msg["content"]
                        if isinstance(c, str) and len(c) > 200:
                            msg["content"] = c[:200] + f"... ({len(c)} chars)"
                        elif isinstance(c, list):
                            for item in c:
                                if isinstance(item, dict) and "text" in item and len(item["text"]) > 200:
                                    item["text"] = item["text"][:200] + f"... ({len(item['text'])} chars)"
            logger.log(TRACE_LEVEL, "Rejected request body (truncated): %s", json.dumps(raw, indent=2, default=str))
        except Exception:
            logger.log(TRACE_LEVEL, "Rejected request body (raw): %s", (await request.body()).decode("utf-8", errors="replace")[:2000])

    return PlainTextResponse(str(exc), status_code=400)


handler = Mangum(app)

if __name__ == "__main__":
    # Bind to 0.0.0.0 for container environments, network is handled by network policies and load balancers
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)  # nosec B104
