# Bedrock Access Gateway Project Overview

## Project Purpose
Bedrock Access Gateway provides OpenAI-compatible RESTful APIs for Amazon Bedrock models, making it easy to use AWS foundation models without changing existing code that uses OpenAI APIs.

## Key Features
- Support for OpenAI-compatible endpoints including chat completions, embeddings, and model APIs
- Streaming responses via server-sent events (SSE)
- Support for tool calling
- Support for multimodal APIs
- Cross-region inference
- Support for reasoning with models like Claude 3.7 Sonnet and DeepSeek R1
- Support for custom imported models

## Architecture
- FastAPI-based RESTful API
- AWS deployment options:
  - ALB + Lambda 
  - ALB + Fargate
- Proxies requests to AWS Bedrock service
- Handles authentication, request transformation, and response formatting
- Implements model availability filtering with Parameter Store caching

## Current Development
Current branch focuses on model availability filtering to ensure that only models the user has access to are returned from the models API, with optimization for Lambda cold starts.