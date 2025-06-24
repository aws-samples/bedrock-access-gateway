# Task Completion Checklist

When completing tasks in this project, please follow these steps:

1. **Linting**:
   - Run `pipx run ruff check` to lint your code
   - Fix any linting errors

2. **Formatting**:
   - Run `pipx run ruff format` to format your code according to project standards

3. **Testing**:
   - For API functionality: `cd src && uv run test_api.py`
   - For custom model functionality: `cd src && uv run test_custom_models.py`
   - To enable debug mode: `export DEBUG=true` before running tests
   - To set AWS region: `export AWS_REGION=us-east-1` before running tests

4. **Local Development**:
   - To run locally: `cd src && uv run -m uvicorn api.app:app --host 0.0.0.0 --port 8000`

5. **Docker Build**:
   - To build and push Docker images: `cd scripts && bash ./push-to-ecr.sh`

6. **Review**:
   - Ensure all code follows Python 3.12 standards
   - Verify proper type annotations are used
   - Confirm proper exception handling is implemented
   - Check that docstrings are included for public functions