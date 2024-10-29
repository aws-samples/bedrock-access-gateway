import os
import logging 

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = "/api/v1"

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.

List of Amazon Bedrock models currently supported:
- Anthropic Claude 2 / 3 /3.5 (Haiku/Sonnet/Opus)
- Meta Llama 2 / 3
- Mistral / Mixtral
- Cohere Command R / R+
- Cohere Embedding
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_REGION_OPUS = os.environ.get("AWS_REGION_OPUS", "us-west-2")
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3"
)

log.info("Settings loaded with region: %s", AWS_REGION)