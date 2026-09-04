import os

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"
ENABLE_APPLICATION_INFERENCE_PROFILES = os.environ.get("ENABLE_APPLICATION_INFERENCE_PROFILES", "true").lower() != "false"
ENABLE_PROMPT_CACHING = os.environ.get("ENABLE_PROMPT_CACHING", "false").lower() != "false"

# Multimodal image inputs may reference a remote URL that the gateway fetches on
# behalf of the caller. See api/image_url.py for the checks applied to that URL.
ENABLE_IMAGE_URL_FETCH = os.environ.get("ENABLE_IMAGE_URL_FETCH", "true").lower() != "false"
# Optional comma-separated allowlist of hosts an image url may point at.
# Empty means any host that resolves to a globally routable address is allowed.
IMAGE_URL_ALLOWED_HOSTS = frozenset(
    host.strip().lower() for host in os.environ.get("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if host.strip()
)
IMAGE_URL_MAX_SIZE_MB = int(os.environ.get("IMAGE_URL_MAX_SIZE_MB", "10"))
