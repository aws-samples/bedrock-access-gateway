"""
Shared test fixtures and pre-import patches.

boto3 and tiktoken are patched before any test module imports the bedrock module,
preventing real AWS calls and encoding downloads during test collection.
"""

from unittest.mock import patch, MagicMock

import pytest


def _mock_client_factory(service_name, **kwargs):
    """Factory that returns mock clients for bedrock services."""
    mock = MagicMock()
    if service_name == "bedrock":
        mock.list_foundation_models.return_value = {"modelSummaries": []}
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = []
        mock.get_paginator.return_value = paginator_mock
    return mock


# Patch boto3 and tiktoken at module level so they're active before bedrock imports
_boto3_patch = patch("boto3.client", side_effect=_mock_client_factory)
_tiktoken_patch = patch(
    "tiktoken.get_encoding", return_value=MagicMock(encode=lambda x: x.split())
)
_boto3_patch.start()
_tiktoken_patch.start()


@pytest.fixture(autouse=True)
def patch_profile_metadata(monkeypatch):
    """Populate profile_metadata for cross-region model IDs."""
    import api.models.bedrock as bedrock_module

    monkeypatch.setattr(bedrock_module, "profile_metadata", {
        "us.anthropic.claude-opus-4-7": {
            "underlying_model_id": "anthropic.claude-opus-4-7",
            "profile_type": "SYSTEM_DEFINED",
        },
        "us.anthropic.claude-sonnet-4-5-20250514-v1:0": {
            "underlying_model_id": "anthropic.claude-sonnet-4-5-20250514-v1:0",
            "profile_type": "SYSTEM_DEFINED",
        },
    })
    monkeypatch.setattr(bedrock_module, "ENABLE_PROMPT_CACHING", False)
