import os
from urllib.parse import urlparse


API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""


def _env_or_none(key: str) -> str | None:
    return os.environ.get(key) or None


def _validate_https_url(name: str, value: str | None) -> str | None:
    if value is None:
        return None

    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(f"{name} must be a valid https URL when set. Got: {value}")
    return value


DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
BEDROCK_URL = _validate_https_url("BEDROCK_URL", _env_or_none("BEDROCK_URL"))
BEDROCK_RUNTIME_URL = _validate_https_url("BEDROCK_RUNTIME_URL", _env_or_none("BEDROCK_RUNTIME_URL"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"
ENABLE_APPLICATION_INFERENCE_PROFILES = os.environ.get("ENABLE_APPLICATION_INFERENCE_PROFILES", "true").lower() != "false"
ENABLE_PROMPT_CACHING = os.environ.get("ENABLE_PROMPT_CACHING", "false").lower() != "false"
MODEL_WHITELIST_FILE = _env_or_none("MODEL_WHITELIST_FILE")
MODEL_WHITELIST_JSON = _env_or_none("MODEL_WHITELIST_JSON")
