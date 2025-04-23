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
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"
# Custom models are always enabled by default
ENABLE_CUSTOM_MODELS = True

# Guardrail configuration
ENABLE_GUARDRAILS = os.environ.get("ENABLE_GUARDRAILS", "false").lower() == "true"
DEFAULT_GUARDRAIL_ID = os.environ.get("DEFAULT_GUARDRAIL_ID", "")
DEFAULT_GUARDRAIL_VERSION = os.environ.get("DEFAULT_GUARDRAIL_VERSION", "DRAFT")
DEFAULT_GUARDRAIL_TRACE = os.environ.get("DEFAULT_GUARDRAIL_TRACE", "ENABLED")

# Log warning if guardrails are enabled but no ID is provided
if ENABLE_GUARDRAILS and not DEFAULT_GUARDRAIL_ID:
    import logging
    logging.getLogger(__name__).warning("ENABLE_GUARDRAILS is set to true but DEFAULT_GUARDRAIL_ID is not provided. Guardrails will not be applied.")
