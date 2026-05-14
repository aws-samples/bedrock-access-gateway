"""Unit tests for BedrockModel._sanitize_tool_pairs / _parse_messages.

Drop-in for upstream PR (aws-samples/bedrock-access-gateway). Place at
``test/test_parse_messages_sanitize.py`` in the upstream repo and run with::

    pytest test/test_parse_messages_sanitize.py

The tests exercise the strict Bedrock Converse pairing rule: every ``toolUse``
must be followed by a matching ``toolResult`` block. They assert that the
sanitiser drops orphan tool_calls / tool_results from the OpenAI message list
before translation, and that well-formed sequences are unchanged.
"""

import json
from typing import Iterable

import pytest
from api.models.bedrock import BedrockModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ResponseFunction,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def _tc(call_id: str, query: str = "x") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=ResponseFunction(
            name="kb_lookup",
            arguments=json.dumps({"query": query}),
        ),
    )


def _ids(parsed: list, kind: str) -> list[str]:
    """Collect ``toolUse`` or ``toolResult`` ids from a parsed message list."""
    out: list[str] = []
    for msg in parsed:
        for block in msg.get("content", []):
            if kind in block:
                out.append(block[kind]["toolUseId"])
    return out


def _assert_balanced(parsed: list) -> None:
    use_ids = _ids(parsed, "toolUse")
    result_ids = _ids(parsed, "toolResult")
    assert set(use_ids) == set(result_ids), f"Unbalanced tool pairs: toolUse={use_ids} toolResult={result_ids}"


@pytest.fixture
def model() -> BedrockModel:
    return BedrockModel()


@pytest.fixture
def model_id() -> str:
    return "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_orphan_tool_calls_are_dropped(model: BedrockModel, model_id: str) -> None:
    """Assistant ``tool_calls`` without matching ``ToolMessage`` are removed."""
    request = ChatRequest(
        model=model_id,
        messages=[
            UserMessage(role="user", content="ask"),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[_tc("orphan-1"), _tc("orphan-2")],
            ),
            UserMessage(role="user", content="continue"),
        ],
    )

    parsed = model._parse_messages(request)

    assert _ids(parsed, "toolUse") == []
    _assert_balanced(parsed)


def test_well_formed_sequence_is_unchanged(model: BedrockModel, model_id: str) -> None:
    """Sanitiser is a no-op when every ``toolUse`` has a matching ``toolResult``."""
    request = ChatRequest(
        model=model_id,
        messages=[
            UserMessage(role="user", content="ask"),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[_tc("ok-1")],
            ),
            ToolMessage(role="tool", tool_call_id="ok-1", content="result"),
            UserMessage(role="user", content="thanks"),
        ],
    )

    parsed = model._parse_messages(request)

    assert _ids(parsed, "toolUse") == ["ok-1"]
    assert _ids(parsed, "toolResult") == ["ok-1"]
    _assert_balanced(parsed)


def test_orphan_tool_result_is_dropped(model: BedrockModel, model_id: str) -> None:
    """``ToolMessage`` with no matching ``tool_call`` is removed."""
    request = ChatRequest(
        model=model_id,
        messages=[
            UserMessage(role="user", content="hi"),
            ToolMessage(
                role="tool",
                tool_call_id="orphan-result",
                content="leftover",
            ),
            UserMessage(role="user", content="continue"),
        ],
    )

    parsed = model._parse_messages(request)

    assert _ids(parsed, "toolResult") == []
    _assert_balanced(parsed)


def test_partial_pairs_keep_only_paired_ids(model: BedrockModel, model_id: str) -> None:
    """Orphan tool_calls are dropped; paired ones and the assistant text remain."""
    request = ChatRequest(
        model=model_id,
        messages=[
            UserMessage(role="user", content="ask"),
            AssistantMessage(
                role="assistant",
                content="Let me check.",
                tool_calls=[_tc("paired"), _tc("orphan")],
            ),
            ToolMessage(role="tool", tool_call_id="paired", content="result"),
            UserMessage(role="user", content="thanks"),
        ],
    )

    parsed = model._parse_messages(request)

    assert _ids(parsed, "toolUse") == ["paired"]
    assert _ids(parsed, "toolResult") == ["paired"]
    # Assistant text content survives even though one of its tool_calls was orphaned.
    assistant_texts: Iterable[str] = (
        block["text"]
        for msg in parsed
        if msg.get("role") == "assistant"
        for block in msg.get("content", [])
        if "text" in block
    )
    assert "Let me check." in list(assistant_texts)
    _assert_balanced(parsed)


def test_assistant_with_only_orphan_tool_calls_is_dropped(model: BedrockModel, model_id: str) -> None:
    """An assistant turn that carried only orphaned ``tool_calls`` disappears."""
    request = ChatRequest(
        model=model_id,
        messages=[
            UserMessage(role="user", content="ask"),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[_tc("orphan-only")],
            ),
            UserMessage(role="user", content="continue"),
        ],
    )

    parsed = model._parse_messages(request)

    # The assistant turn collapses entirely; the two user messages get merged
    # into a single user turn by ``_reframe_multi_payloard``.
    assert all(msg["role"] == "user" for msg in parsed)
    _assert_balanced(parsed)
