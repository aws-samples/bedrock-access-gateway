"""
Tests for Opus 4.7 Adaptive Thinking Support.

Verifies adaptive thinking request format, sampling parameter stripping,
legacy model compatibility, and assistant prefill handling.

Run with: pytest -v
"""

import pytest

from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, UserMessage, AssistantMessage


# --- Helpers ---


def _make_chat_request(
    model: str,
    reasoning_effort=None,
    temperature=None,
    top_p=None,
    max_tokens=2048,
    messages=None,
) -> ChatRequest:
    """Helper to construct a valid ChatRequest for testing."""
    if messages is None:
        messages = [UserMessage(content="Hello, world!")]
    return ChatRequest(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _parse_request(chat_request: ChatRequest) -> dict:
    """Call _parse_request on a BedrockModel instance."""
    model = BedrockModel()
    return model._parse_request(chat_request)


# --- Test constants ---

OPUS_47_MODELS = [
    "us.anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-7",
]

REASONING_EFFORTS = ["low", "medium", "high"]

LEGACY_CLAUDE_MODELS = [
    "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
]


# =============================================================================
# Adaptive shape correctness
# For Opus 4.7 + non-null reasoning_effort, additionalModelRequestFields
# contains exactly `thinking` + `output_config`, never `reasoning_config`/`budget_tokens`.
# =============================================================================


@pytest.mark.parametrize("model_id", OPUS_47_MODELS)
@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_opus47_has_thinking_and_output_config(model_id, effort):
    """Opus 4.7 + reasoning_effort produces thinking + output_config, not reasoning_config."""
    request = _make_chat_request(model=model_id, reasoning_effort=effort)
    args = _parse_request(request)

    assert "additionalModelRequestFields" in args, (
        f"Expected additionalModelRequestFields for model={model_id}, effort={effort}"
    )

    additional = args["additionalModelRequestFields"]

    assert additional == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }, f"Unexpected additionalModelRequestFields shape: {additional}"

    assert "reasoning_config" not in additional
    assert "budget_tokens" not in str(additional)


# =============================================================================
# Sampling parameter stripping
# For Opus 4.7, inferenceConfig never contains temperature/topP regardless
# of what the client sends. maxTokens IS preserved.
# =============================================================================


@pytest.mark.parametrize("model_id", OPUS_47_MODELS)
@pytest.mark.parametrize("temperature,top_p", [
    (0.7, 0.9),
    (0.0, 1.0),
    (2.0, 0.5),
    (1.0, None),
    (None, 0.8),
])
def test_opus47_strips_temperature_and_top_p(model_id, temperature, top_p):
    """Opus 4.7 strips temperature/topP but preserves maxTokens."""
    max_tokens = 4096
    request = _make_chat_request(
        model=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    args = _parse_request(request)

    inference_config = args["inferenceConfig"]

    assert "temperature" not in inference_config, (
        f"temperature should be stripped for Opus 4.7, got: {inference_config}"
    )
    assert "topP" not in inference_config, (
        f"topP should be stripped for Opus 4.7, got: {inference_config}"
    )
    assert "maxTokens" in inference_config
    assert inference_config["maxTokens"] == max_tokens


# =============================================================================
# Legacy model compatibility
# Non-Opus-4.7 Claude + reasoning_effort produces reasoning_config with budget_tokens.
# =============================================================================


@pytest.mark.parametrize("model_id", LEGACY_CLAUDE_MODELS)
@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_legacy_claude_uses_reasoning_config(model_id, effort):
    """Legacy Claude + reasoning_effort produces reasoning_config with budget_tokens."""
    request = _make_chat_request(model=model_id, reasoning_effort=effort)
    args = _parse_request(request)

    assert "additionalModelRequestFields" in args, (
        f"Expected additionalModelRequestFields for legacy model={model_id}"
    )

    additional = args["additionalModelRequestFields"]

    assert "reasoning_config" in additional, (
        f"Expected 'reasoning_config' for legacy Claude, got: {additional}"
    )
    reasoning_config = additional["reasoning_config"]
    assert reasoning_config["type"] == "enabled"
    assert "budget_tokens" in reasoning_config
    assert isinstance(reasoning_config["budget_tokens"], int)
    assert reasoning_config["budget_tokens"] > 0

    assert "thinking" not in additional
    assert "output_config" not in additional


# =============================================================================
# Assistant prefill handling
# Opus 4.7 conversations ending with assistant role get a user message appended.
# When conversation ends with user message, no extra message is appended.
# =============================================================================


@pytest.mark.parametrize("model_id", OPUS_47_MODELS)
@pytest.mark.parametrize("assistant_text", [
    "Sure, let me help.",
    "Here is the code:",
    "I'll continue from where I left off.",
])
def test_opus47_assistant_ending_gets_user_appended(model_id, assistant_text):
    """Conversations ending with assistant message get a continuation user message appended."""
    messages = [
        UserMessage(content="Hello"),
        AssistantMessage(content=assistant_text),
    ]
    request = _make_chat_request(model=model_id, messages=messages)
    args = _parse_request(request)

    result_messages = args["messages"]

    assert result_messages[-1]["role"] == "user", (
        f"Expected last message role='user', got: {result_messages[-1]['role']}"
    )
    last_content = result_messages[-1]["content"]
    assert any("text" in item for item in last_content), (
        f"Expected text content in continuation message, got: {last_content}"
    )


@pytest.mark.parametrize("model_id", OPUS_47_MODELS)
@pytest.mark.parametrize("user_text", [
    "Hello, world!",
    "What is the meaning of life?",
    "Please write some code.",
])
def test_opus47_user_ending_no_extra_message(model_id, user_text):
    """Conversations ending with user message are not modified."""
    messages = [UserMessage(content=user_text)]
    request = _make_chat_request(model=model_id, messages=messages)
    args = _parse_request(request)

    result_messages = args["messages"]

    user_messages = [m for m in result_messages if m["role"] == "user"]
    assert len(user_messages) == 1, (
        f"Expected exactly 1 user message, got {len(user_messages)}: {result_messages}"
    )
    assert result_messages[-1]["role"] == "user"
