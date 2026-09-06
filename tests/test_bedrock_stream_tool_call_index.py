import unittest

from api.models.bedrock import BedrockModel


class BedrockStreamToolCallIndexTest(unittest.TestCase):
    def setUp(self):
        self.model = BedrockModel()
        self.model.think_emitted = False
        self.model._stream_tool_index_by_content_block = {}
        self.model._next_stream_tool_index = 0

    def test_first_streamed_tool_call_uses_non_negative_index(self):
        start_chunk = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": "tooluse_123",
                        "name": "get_weather",
                    }
                },
            }
        }
        delta_chunk = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": "{\"city\":\"Boston\"}",
                    }
                },
            }
        }

        start_response = self.model._create_response_stream(
            model_id="us.anthropic.claude-sonnet-4-6",
            message_id="chatcmpl-test",
            chunk=start_chunk,
        )
        delta_response = self.model._create_response_stream(
            model_id="us.anthropic.claude-sonnet-4-6",
            message_id="chatcmpl-test",
            chunk=delta_chunk,
        )

        self.assertEqual(start_response.choices[0].delta.tool_calls[0].index, 0)
        self.assertEqual(delta_response.choices[0].delta.tool_calls[0].index, 0)

    def test_multiple_streamed_tool_calls_are_indexed_by_tool_ordinal(self):
        tool_one_start = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": "tooluse_1",
                        "name": "get_weather",
                    }
                },
            }
        }
        tool_two_start = {
            "contentBlockStart": {
                "contentBlockIndex": 2,
                "start": {
                    "toolUse": {
                        "toolUseId": "tooluse_2",
                        "name": "lookup_news",
                    }
                },
            }
        }

        tool_one_response = self.model._create_response_stream(
            model_id="us.anthropic.claude-sonnet-4-6",
            message_id="chatcmpl-test",
            chunk=tool_one_start,
        )
        tool_two_response = self.model._create_response_stream(
            model_id="us.anthropic.claude-sonnet-4-6",
            message_id="chatcmpl-test",
            chunk=tool_two_start,
        )

        self.assertEqual(tool_one_response.choices[0].delta.tool_calls[0].index, 0)
        self.assertEqual(tool_two_response.choices[0].delta.tool_calls[0].index, 1)


if __name__ == "__main__":
    unittest.main()
