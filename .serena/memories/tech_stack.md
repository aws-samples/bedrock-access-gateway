# Tech Stack

## Language and Runtime
- Python 3.12
- UV Python package manager (preferred over pip/python)

## Web Framework
- FastAPI for API development
- Uvicorn as ASGI server
- Mangum for AWS Lambda integration

## AWS Services
- Amazon Bedrock (core service being proxied)
- AWS Lambda or ECS/Fargate for deployment
- Application Load Balancer for request routing
- Amazon ECR for container images
- AWS Secrets Manager for API key storage
- AWS Parameter Store for model availability caching

## Package Dependencies
- fastapi: Web framework
- pydantic: Data validation
- uvicorn: ASGI server
- mangum: Lambda adapter
- tiktoken: Tokenization for OpenAI compatibility
- boto3/botocore: AWS SDK
- requests: HTTP client
- numpy: Numerical operations

## Testing
- pytest
- pytest-asyncio for async testing

## Development Tools
- Ruff for linting and formatting
- pre-commit for git hooks