# Suggested Commands

## Environment Setup
- Install dependencies: `cd src && uv pip install -r requirements.txt`
- Enable debug mode: `export DEBUG=true`
- Set AWS region: `export AWS_REGION=us-east-1`

## Running the Application
- Run locally: `cd src && uv run -m uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Run Python scripts: `cd src && uv run <script_name>.py`

## Testing
- Test API functionality: `cd src && uv run test_api.py`
- Test custom models: `cd src && uv run test_custom_models.py`
- Test with custom models enabled: `ENABLE_CUSTOM_MODELS=true uv run test_api.py`

## Code Quality
- Lint: `pipx run ruff check`
- Format: `pipx run ruff format`

## Deployment
- Build and push Docker images: `cd scripts && bash ./push-to-ecr.sh`

## Git Commands
- View current branch: `git branch`
- View changes: `git diff`
- Stage changes: `git add <file>`
- Commit changes: `git commit -m "message"`
- Push changes: `git push origin <branch>`

## File Operations
- List files: `ls -la`
- Search for patterns: `grep -r "pattern" .`
- Find files: `find . -name "filename"`