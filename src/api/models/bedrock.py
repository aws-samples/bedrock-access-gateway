import base64
import json
import logging
import re
import time
from abc import ABC
from typing import AsyncIterable, Iterable, Literal

import boto3
import numpy as np
import requests
import tiktoken
from fastapi import HTTPException

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    # Chat
    ChatResponse,
    ChatRequest,
    Choice,
    ChatResponseMessage,
    Usage,
    ChatStreamResponse,
    ImageContent,
    TextContent,
    ToolCall,
    ChoiceDelta,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Function,
    ResponseFunction,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Embedding,


)
from api.setting import DEBUG, AWS_REGION

logger = logging.getLogger(__name__)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    # Disable Titan embedding.
    # "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")


class BedrockModel(BaseChatModel):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
    _supported_models = {
        "amazon.titan-text-premier-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "anthropic.claude-instant-v1": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "anthropic.claude-v2:1": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "anthropic.claude-v2": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "anthropic.claude-3-opus-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "meta.llama2-13b-chat-v1": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "meta.llama2-70b-chat-v1": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "meta.llama3-8b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "meta.llama3-70b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        # Llama 3.1 8b cross-region inference profile
        "us.meta.llama3-1-8b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "meta.llama3-1-8b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        # Llama 3.1 70b cross-region inference profile
        "us.meta.llama3-1-70b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "meta.llama3-1-70b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "meta.llama3-1-405b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        # Llama 3.2 1B cross-region inference profile
        "us.meta.llama3-2-1b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        # Llama 3.2 3B cross-region inference profile
        "us.meta.llama3-2-3b-instruct-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        # Llama 3.2 11B cross-region inference profile
        "us.meta.llama3-2-11b-instruct-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": False,
        },
        # Llama 3.2 90B cross-region inference profile
        "us.meta.llama3-2-90b-instruct-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "mistral.mistral-7b-instruct-v0:2": {
            "system": False,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "mistral.mixtral-8x7b-instruct-v0:1": {
            "system": False,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "mistral.mistral-small-2402-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        },
        "mistral.mistral-large-2402-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "mistral.mistral-large-2407-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "cohere.command-r-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "cohere.command-r-plus-v1:0": {
            "system": True,
            "multimodal": False,
            "tool_call": True,
            "stream_tool_call": False,
        },
        "apac.anthropic.claude-3-sonnet-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "apac.anthropic.claude-3-haiku-20240307-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # claude 3 Haiku cross-region inference profile
        "us.anthropic.claude-3-haiku-20240307-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "eu.anthropic.claude-3-haiku-20240307-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # claude 3 Opus cross-region inference profile
        "us.anthropic.claude-3-opus-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # claude 3 Sonnet cross-region inference profile
        "us.anthropic.claude-3-sonnet-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "eu.anthropic.claude-3-sonnet-20240229-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # claude 3.5 Sonnet cross-region inference profile
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # claude 3.5 Sonnet v2 cross-region inference profile(Now only us-west-2)
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "system": True,
            "multimodal": True,
            "tool_call": True,
            "stream_tool_call": True,
        },
        # Amazon Nova models - AWS's proprietary large language models
        "us.amazon.nova-lite-v1:0": {
            "system": True,      # Supports system prompts for context setting
            "multimodal": True,  # Capable of processing both text and images
            "tool_call": True,
            "stream_tool_call": True,
        },
        "us.amazon.nova-micro-v1:0": {
            "system": True,      # Supports system prompts for context setting
            "multimodal": False, # Text-only model, no image processing capabilities
            "tool_call": True,
            "stream_tool_call": True,
        },
        "us.amazon.nova-pro-v1:0": {
            "system": True,      # Supports system prompts for context setting
            "multimodal": True,  # Capable of processing both text and images
            "tool_call": True,
            "stream_tool_call": True,
        },
    }

    def list_models(self) -> list[str]:
        return list(self._supported_models.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in self._supported_models.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"

        # check if tool call is supported
        elif chat_request.tools and not self._is_tool_call_supported(chat_request.model, stream=chat_request.stream):
            tool_call_info = "Tool call with streaming" if chat_request.stream else "Tool call"
            error = f"{tool_call_info} is currently not supported by {chat_request.model}"

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""
        if DEBUG:
            logger.info("Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(args))

        try:
            if stream:
                response = bedrock_runtime.converse_stream(**args)
            else:
                response = bedrock_runtime.converse(**args)
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        message_id = self.generate_message_id()
        response = self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]

        chat_response = self._create_response(
            model=chat_request.model,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        response = self._invoke_bedrock(chat_request, stream=True)
        message_id = self.generate_message_id()

        stream = response.get("stream")
        for chunk in stream:
            stream_response = self._create_response_stream(
                model_id=chat_request.model, message_id=message_id, chunk=chunk
            )
            if not stream_response:
                continue
            if DEBUG:
                logger.info("Proxy response :" + stream_response.model_dump_json())
            if stream_response.choices:
                yield self.stream_response_to_bytes(stream_response)
            elif (
                    chat_request.stream_options
                    and chat_request.stream_options.include_usage
            ):
                # An empty choices for Usage as per OpenAI doc below:
                # if you set stream_options: {"include_usage": true}.
                # an additional chunk will be streamed before the data: [DONE] message.
                # The usage field on this chunk shows the token usage statistics for the entire request,
                # and the choices field will always be an empty array.
                # All other chunks will also include a usage field, but with a null value.
                yield self.stream_response_to_bytes(stream_response)

        # return an [DONE] message at the end.
        yield self.stream_response_to_bytes()

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts.
        Note that not all models support system prompts.

        example output: [{"text" : system_prompt}]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """

        system_prompts = []
        for message in chat_request.messages:
            if message.role != "system":
                # ignore system messages here
                continue
            assert isinstance(message.content, str)
            system_prompts.append({"text": message.content})

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                if message.content:
                    # Text message
                    messages.append(
                        {
                            "role": message.role,
                            "content": self._parse_content_parts(
                                message, chat_request.model
                            ),
                        }
                    )
                else:
                    # Tool use message
                    tool_input = json.loads(message.tool_calls[0].function.arguments)
                    messages.append(
                        {
                            "role": message.role,
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": message.tool_calls[0].id,
                                        "name": message.tool_calls[0].function.name,
                                        "input": tool_input
                                    }
                                }
                            ],
                        }
                    )
            elif isinstance(message, ToolMessage):
                # Bedrock does not support tool role,
                # Add toolResult to content
                # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.tool_call_id,
                                    "content": [{"text": message.content}],
                                }
                            }
                        ],
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return self._reframe_multi_payloard(messages)


    def _reframe_multi_payloard(self, messages: list) -> list:
        """ Receive messages and reformat them to comply with the Claude format

With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
with Claude format requests, you cannot repeatedly receive messages from the same role.

This method searches through the OpenAI format messages in order and reformats them to the Claude format.

```
openai_format_messages=[
{"role": "user", "content": "hogehoge"},
{"role": "user", "content": "fugafuga"},
]

bedrock_format_messages=[
{
    "role": "user",
    "content": [
        {"text": "hogehoge"},
        {"text": "fugafuga"}
    ]
},
]
```
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message['role']
            next_content = message['content']

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append({
                        "role": current_role,
                        "content": current_content
                    })
                # Switch to the new role
                current_role = next_role
                current_content = []

            # Add the message content to current_content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)

        # Add the last role's messages to the list
        if current_content:
            reformatted_messages.append({
                "role": current_role,
                "content": current_content
            })

        return reformatted_messages


    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """

        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        # Base inference parameters.
        inference_config = {
            "temperature": chat_request.temperature,
            "maxTokens": chat_request.max_tokens,
            "topP": chat_request.top_p,
        }

        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        # add tool config
        if chat_request.tools:
            args["toolConfig"] = {
                "tools": [
                    self._convert_tool_spec(t.function) for t in chat_request.tools
                ]
            }

            if chat_request.tool_choice and not chat_request.model.startswith("meta.llama3-1-"):
                if isinstance(chat_request.tool_choice, str):
                    # auto (default) is mapped to {"auto" : {}}
                    # required is mapped to {"any" : {}}
                    if chat_request.tool_choice == "required":
                        args["toolConfig"]["toolChoice"] = {"any": {}}
                    else:
                        args["toolConfig"]["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    assert "function" in chat_request.tool_choice
                    args["toolConfig"]["toolChoice"] = {
                        "tool": {"name": chat_request.tool_choice["function"].get("name", "")}}
        return args

    def _create_response(
            self,
            model: str,
            message_id: str,
            content: list[dict] = None,
            finish_reason: str | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> ChatResponse:

        message = ChatResponseMessage(
            role="assistant",
        )
        if finish_reason == "tool_use":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html#tool-use-examples
            tool_calls = []
            for part in content:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            id=tool["toolUseId"],
                            type="function",
                            function=ResponseFunction(
                                name=tool["name"],
                                arguments=json.dumps(tool["input"]),
                            ),
                        )
                    )
            message.tool_calls = tool_calls
            message.content = None
        else:
            message.content = ""
            if content:
                message.content = content[0]["text"]

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
            self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        if DEBUG:
            logger.info("Bedrock response chunk: " + str(chunk))

        finish_reason = None
        message = None
        usage = None
        if "messageStart" in chunk:
            message = ChatResponseMessage(
                role=chunk["messageStart"]["role"],
                content="",
            )
        if "contentBlockStart" in chunk:
            # tool call start
            delta = chunk["contentBlockStart"]["start"]
            if "toolUse" in delta:
                # first index is content
                index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            type="function",
                            id=delta["toolUse"]["toolUseId"],
                            function=ResponseFunction(
                                name=delta["toolUse"]["name"],
                                arguments="",
                            ),
                        )
                    ]
                )
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                # stream content
                message = ChatResponseMessage(
                    content=delta["text"],
                )
            else:
                # tool use
                index = chunk["contentBlockDelta"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            function=ResponseFunction(
                                arguments=delta["toolUse"]["input"],
                            )
                        )
                    ]
                )
        if "messageStop" in chunk:
            message = ChatResponseMessage()
            finish_reason = chunk["messageStop"]["stopReason"]

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=metadata["usage"]["inputTokens"],
                        completion_tokens=metadata["usage"]["outputTokens"],
                        total_tokens=metadata["usage"]["totalTokens"],
                    ),
                )
        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                usage=usage,
            )

        return None

    def _parse_image(self, image_url: str) -> tuple[bytes, str]:
        """Try to get the raw data from an image url.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
        returns a tuple of (Image Data, Content Type)
        """
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)
        # if already base64 encoded.
        # Only supports 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)

        # Send a request to the image URL
        response = requests.get(image_url)
        # Check if the request was successful
        if response.status_code == 200:

            content_type = response.headers.get("Content-Type")
            if not content_type.startswith("image"):
                content_type = "image/jpeg"
            # Get the image content
            image_content = response.content
            return image_content, content_type
        else:
            raise HTTPException(
                status_code=500, detail="Unable to access the image url"
            )

    def _parse_content_parts(
            self,
            message: UserMessage,
            model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self._is_multimodal_supported(model_id):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    def _is_tool_call_supported(self, model_id: str, stream: bool = False) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["stream_tool_call"] if stream else feature["tool_call"]

    def _is_multimodal_supported(self, model_id: str) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["multimodal"]

    def _is_system_prompt_supported(self, model_id: str) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["system"]

    def _convert_tool_spec(self, func: Function) -> dict:
        return {
            "toolSpec": {
                "name": func.name,
                "description": func.description,
                "inputSchema": {
                    "json": func.parameters,
                },
            }
        }

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "finished": "stop",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "complete": "stop",
                "content_filtered": "content_filter"
            }
            return finish_reason_mapping.get(finish_reason.lower(), finish_reason.lower())
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        try:
            return bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def _create_response(
            self,
            embeddings: list[float],
            model: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
            encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response


class CohereEmbeddingsModel(BedrockEmbeddingsModel):

    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        texts = []
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            # For encoded input
            # The workaround is to use tiktoken to decode to get the original text.
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    # Iterable[int]
                    encodings.append(inner)
                else:
                    # Iterable[Iterable[int]]
                    text = ENCODER.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(ENCODER.decode(encodings))

        # Maximum of 2048 characters
        args = {
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END",  # "NONE|START|END"
        }
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=response_body["embeddings"],
            model=embeddings_request.model,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):

    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        if isinstance(embeddings_request.input, str):
            input_text = embeddings_request.input
        elif (
                isinstance(embeddings_request.input, list)
                and len(embeddings_request.input) == 1
        ):
            input_text = embeddings_request.input[0]
        else:
            raise ValueError(
                "Amazon Titan Embeddings models support only single strings as input."
            )
        args = {
            "inputText": input_text,
            # Note: inputImage is not supported!
        }
        if embeddings_request.model == "amazon.titan-embed-image-v1":
            args["embeddingConfig"] = (
                embeddings_request.embedding_config
                if embeddings_request.embedding_config
                else {"outputEmbeddingLength": 1024}
            )
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=[response_body["embedding"]],
            model=embeddings_request.model,
            input_tokens=response_body["inputTextTokenCount"],
        )


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    match model_name:
        case "Cohere Embed Multilingual" | "Cohere Embed English":
            return CohereEmbeddingsModel()
        case _:
            logger.error("Unsupported model id " + model_id)
            raise HTTPException(
                status_code=400,
                detail="Unsupported embedding model id " + model_id,
            )
