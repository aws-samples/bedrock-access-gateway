# Code Style Guidelines

## Python Style
- Python version: 3.12
- Line length: 120 characters max
- Indentation: 4 spaces
- Quote style: Double quotes for strings
- Imports: Group by (standard lib, third-party, internal) and alphabetically sort within groups
- Type annotations required for all functions and classes
- Use abstract base classes (ABC) for interfaces
- Snake case for variables/functions, PascalCase for classes

## Error Handling
- Explicit error handling with specific exception types
- Use HTTPException for API errors

## Documentation
- Document public functions with docstrings
- Follow FastAPI patterns for API development

## Linting and Formatting
- Use Ruff for linting and formatting
  - `pipx run ruff check` for linting
  - `pipx run ruff format` for code formatting

## Configuration
- Default `ruff.toml` settings:
  - line-length = 120
  - indent-width = 4
  - target-version = "py312"
  - Select rules: E (errors), F (flake8), I (isort)
  - Ignore rules: E501 (line length), C901 (complexity), F401 (unused imports)
  - Double quote style for strings