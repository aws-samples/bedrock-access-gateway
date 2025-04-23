# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Install dependencies: `cd src && uv pip install -r requirements.txt`
- Run Python scripts: `cd src && uv run <script_name>.py`
- Run locally: `cd src && uv run -m uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Build Docker: `cd scripts && bash ./push-to-ecr.sh`
- Lint: `pipx run ruff check`
- Format: `pipx run ruff format`

## Environment Configuration
- We use `uv` instead of regular `python` or `pip` commands
- Enable debug mode: `export DEBUG=true`
- Set AWS region: `export AWS_REGION=us-east-1`
- Custom models are enabled by default

## Code Style
- Python version: 3.12
- Line length: 120 characters max
- Indentation: 4 spaces
- Quote style: Double quotes for strings
- Imports: Group (standard lib, third-party, internal) and alphabetically sorted
- Use FastAPI patterns for API development
- Type annotations required for all functions and classes
- Use abstract base classes (ABC) for interfaces
- Snake case for variables/functions, PascalCase for classes
- Explicit error handling with specific exception types
- Use HTTPException for API errors
- Document public functions with docstrings

## Architecture
This project provides OpenAI-compatible RESTful APIs for Amazon Bedrock models, making it easy to use AWS foundation models without changing existing code that uses OpenAI APIs.

## Testing
- Test API functionality: `cd src && uv run test_api.py`
- Test custom models: `cd src && uv run test_custom_models.py`