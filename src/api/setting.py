import os

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL")
USE_MODEL_MAPPING = os.getenv("USE_MODEL_MAPPING", "true").lower() != "false"

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "default")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_ENDPOINT = os.getenv("GCP_ENDPOINT", "openapi")

PROVIDER = os.getenv("PROVIDER", "GCP" if GCP_PROJECT_ID and GCP_REGION else "AWS")
REGION = os.getenv("REGION", GCP_REGION if PROVIDER == "GCP" else AWS_REGION)
