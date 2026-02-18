import os

os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
os.environ.setdefault("ENABLE_CROSS_REGION_INFERENCE", "false")
os.environ.setdefault("ENABLE_APPLICATION_INFERENCE_PROFILES", "false")

from api.models.bedrock import BedrockModel
from api.schema import AssistantMessage, ChatRequest, ResponseFunction, ToolCall, ToolMessage, UserMessage


def test_parse_assistant_text_converts_think_tags_to_reasoning_blocks():
    model = BedrockModel()

    content = model._parse_assistant_text("<think>Plan first</think>Now answer")

    assert content == [
        {"reasoningContent": {"text": "Plan first"}},
        {"text": "Now answer"},
    ]


def test_parse_assistant_text_preserves_unbalanced_think_tags_as_plain_text():
    model = BedrockModel()

    content = model._parse_assistant_text("<think>Plan first")

    assert content == [{"text": "<think>Plan first"}]


def test_parse_messages_keeps_reasoning_before_tool_use():
    model = BedrockModel()

    request = ChatRequest(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            UserMessage(content="How is the weather in New York?"),
            AssistantMessage(
                content="<think>Need weather tool lookup.</think>I will check now.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ResponseFunction(
                            name="get_current_weather",
                            arguments='{"city":"New York"}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="call_1",
                content="Sunny and 24C.",
            ),
        ],
    )

    messages = model._parse_messages(request)

    assert len(messages) == 3
    assert messages[1]["role"] == "assistant"
    assistant_content = messages[1]["content"]
    assert assistant_content[0] == {"reasoningContent": {"text": "Need weather tool lookup."}}
    assert any("toolUse" in block for block in assistant_content)

