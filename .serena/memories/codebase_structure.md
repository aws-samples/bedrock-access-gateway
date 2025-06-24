# Codebase Structure

## Main Directories
- `/src`: Source code for the application
  - `/src/api`: Main API application code
    - `/src/api/app.py`: FastAPI application entry point
    - `/src/api/routers`: API route handlers
    - `/src/api/models`: Data models and Bedrock service integration
    - `/src/api/services`: Service layer (includes model availability service)
    - `/src/api/schema.py`: Data schemas for API requests/responses
    - `/src/api/auth.py`: Authentication logic
    - `/src/api/setting.py`: Configuration settings
  - `/src/Dockerfile`: Docker configuration for Lambda deployment
  - `/src/Dockerfile_ecs`: Docker configuration for ECS/Fargate deployment
  - `/src/requirements.txt`: Python dependencies

- `/scripts`: Helper scripts for deployment and building
  - `/scripts/push-to-ecr.sh`: Script to build and push Docker images to ECR

- `/deployment`: Infrastructure deployment templates
  - `/deployment/BedrockProxy.template`: CloudFormation template for Lambda deployment
  - `/deployment/BedrockProxyFargate.template`: CloudFormation template for Fargate deployment
  - `/deployment/terraform`: Terraform configuration files

- `/docs`: Documentation files
  - `/docs/Usage.md`: Usage guide for the API endpoints
  - `/docs/Troubleshooting.md`: Troubleshooting guide

- `/assets`: Static assets like images

## Key Files
- `.claude/settings.local.json`: Claude AI configuration file
- `CLAUDE.md`: Instructions for Claude AI when working with this codebase
- `ruff.toml`: Ruff linter and formatter configuration
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `CUSTOM_MODELS_IMPLEMENTATION.md`: Documentation for custom model implementation

## Current Feature Branch
- `feature/add-model-access-filtering`: Implementing model availability filtering with Lambda cold start optimization