import os

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = "/api/v1"

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.

List of Amazon Bedrock models currently supported:

# Chat
- anthropic.claude-instant-v1
- anthropic.claude-v2:1
- anthropic.claude-v2
- anthropic.claude-3-sonnet-20240229-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- meta.llama2-13b-chat-v1
- meta.llama2-70b-chat-v1
- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1
- mistral.mistral-large-2402-v1:0

# Embeddings
- cohere.embed-multilingual-v3
- cohere.embed-english-v3
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3"
)
