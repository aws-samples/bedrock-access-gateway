import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# api.auth refuses to import without an API key, and api.setting reads its
# defaults at import time, so both are set before any api module is imported.
os.environ.setdefault("API_KEY", "test-api-key")
