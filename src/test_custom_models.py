import json
import logging
import os
import sys
import unittest

import boto3
from botocore.config import Config
from fastapi.testclient import TestClient

# Add the src directory to the path so we can import from api
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.app import app
from api.models.bedrock import list_bedrock_models
from api.setting import DEFAULT_API_KEYS, ENABLE_CUSTOM_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create a test client with authentication
client = TestClient(app)
# Set up default authentication headers
client.headers.update({"Authorization": f"Bearer {DEFAULT_API_KEYS}"})

# Configure AWS client
config = Config(connect_timeout=60, read_timeout=120, retries={"max_attempts": 1})
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_custom_model_id() -> str:
    """Find a custom model to test with.

    Returns:
        The ID of a custom model to test with

    Raises:
        AssertionError: If custom models are not enabled or no custom models are found
    """
    # Make sure custom models are enabled
    assert ENABLE_CUSTOM_MODELS is True, "Custom models are not enabled"

    # Get the list of models
    model_list = list_bedrock_models()
    logger.info(f"Found {len(model_list)} total models in region {AWS_REGION}")

    # Find custom models
    custom_models = {k: v for k, v in model_list.items() if v.get("type") == "custom"}
    logger.info(f"Found {len(custom_models)} custom models in region {AWS_REGION}")

    # Log all model types for debugging
    model_types = {}
    for _, v in model_list.items():
        model_type = v.get("type", "unknown")
        model_types[model_type] = model_types.get(model_type, 0) + 1
    logger.info(f"Model types distribution: {model_types}")

    # Fail tests if no custom models are available
    assert custom_models, (
        f"No custom models found in region {AWS_REGION}. Tests require at least one custom imported model."
    )

    # Return the first custom model ID
    model_id = list(custom_models.keys())[0]
    logger.info(f"Using custom model: {model_id}")
    return model_id


class TestCustomModels(unittest.TestCase):
    """Integration tests for custom imported models."""

    def test_list_models_includes_custom_models(self):
        """Test that the model list includes custom models."""
        # Verify custom models are available - will fail if none found
        get_custom_model_id()

        # Get the list of models
        response = client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)

        # Parse the response
        data = response.json()
        self.assertIn("data", data)

        # Get model IDs
        model_ids = [model["id"] for model in data["data"]]

        # Check if any custom models are included
        custom_model_found = False
        for model_id in model_ids:
            if "-id:custom." in model_id:
                custom_model_found = True
                logger.info(f"Found custom model in response: {model_id}")
                break

        self.assertTrue(custom_model_found, "No custom models found in model list")

    def test_custom_model_invocation(self):
        """Test that a custom model can be invoked successfully."""
        # Get a custom model ID - will fail if none available
        model_id = get_custom_model_id()

        # Create a request payload
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100,
        }

        # Send the request
        response = client.post("/api/v1/chat/completions", json=payload)

        # Check the response status
        self.assertEqual(response.status_code, 200, f"Failed to invoke custom model: {response.text}")

        # Parse the response
        data = response.json()

        # Verify the response structure
        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("message", data["choices"][0])
        self.assertIn("content", data["choices"][0]["message"])

        # Verify that the content is not empty
        content = data["choices"][0]["message"]["content"]
        self.assertTrue(content and len(content) > 0, "Response content is empty")

        # Log the response for debugging
        logger.info(f"Received response from custom model: {content[:100]}...")

    def test_custom_model_streaming(self):
        """Test that a custom model can be used with streaming."""
        # Get a custom model ID - will fail if none available
        model_id = get_custom_model_id()

        # Create a request payload with streaming enabled
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello, what's your name?"}],
            "max_tokens": 100,
            "stream": True,
        }

        # Send the request
        response = client.post("/api/v1/chat/completions", json=payload)

        # Check the response status
        self.assertEqual(response.status_code, 200, "Failed to stream from custom model")

        # For the TestClient, we'll verify the headers indicate streaming
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream"), "Response is not a streaming response"
        )

        # Read some content to verify it's working
        chunk_count = 0
        content = response.content.decode("utf-8")

        # Simple check - streaming responses contain "data:" lines
        self.assertIn("data:", content, "No streaming data found in response")

        # Count data chunks
        for line in content.split("\n"):
            if line.startswith("data:"):
                chunk_count += 1

        # Verify we got some chunks
        self.assertTrue(chunk_count > 0, "No chunks received from streaming")
        logger.info(f"Received {chunk_count} data chunks from streaming response")


if __name__ == "__main__":
    unittest.main()
