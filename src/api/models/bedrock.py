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
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    Embedding,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Error,
    ErrorMessage,
    Function,
    ImageContent,
    ResponseFunction,
    TextContent,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from api.setting import (
    AWS_REGION,
    DEBUG,
    DEFAULT_MODEL,
    ENABLE_CROSS_REGION_INFERENCE,
    ENABLE_CUSTOM_MODELS,
)

logger = logging.getLogger(__name__)

config = Config(connect_timeout=60, read_timeout=120, retries={"max_attempts": 1})
custom_model_config = Config(connect_timeout=60, read_timeout=120, retries={"max_attempts": 10, "mode": "standard"})

# Initialize the Bedrock clients
if DEBUG:
    logger.info(f"Initializing Bedrock clients with region: {AWS_REGION}")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=AWS_REGION,
    config=config,
)


def get_inference_region_prefix():
    if AWS_REGION.startswith("ap-"):
        return "apac"
    return AWS_REGION[:2]


# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
cr_inference_prefix = get_inference_region_prefix()

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    # Disable Titan embedding.
    # "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")


def list_bedrock_models() -> dict:
    """Automatically getting a list of supported models.

    Returns a model list combines:
        - ON_DEMAND models.
        - Cross-Region Inference Profiles (if enabled via Env)
        - Custom Imported Models (if enabled via Env)
    """
    model_list = {}
    if DEBUG:
        logger.info(f"Getting bedrock models with region: {AWS_REGION}, ENABLE_CUSTOM_MODELS: {ENABLE_CUSTOM_MODELS}")

    try:
        profile_list = []
        if ENABLE_CROSS_REGION_INFERENCE:
            # List system defined inference profile IDs
            response = bedrock_client.list_inference_profiles(maxResults=1000, typeEquals="SYSTEM_DEFINED")
            profile_list = [p["inferenceProfileId"] for p in response["inferenceProfileSummaries"]]

        # List foundation models, only cares about text outputs here.
        response = bedrock_client.list_foundation_models(byOutputModality="TEXT")

        for model in response["modelSummaries"]:
            model_id = model.get("modelId", "N/A")
            stream_supported = model.get("responseStreamingSupported", True)
            status = model["modelLifecycle"].get("status", "ACTIVE")

            # currently, use this to filter out rerank models and legacy models
            if not stream_supported or status not in ["ACTIVE", "LEGACY"]:
                continue

            inference_types = model.get("inferenceTypesSupported", [])
            input_modalities = model["inputModalities"]
            # Add on-demand model list
            if "ON_DEMAND" in inference_types:
                model_list[model_id] = {"modalities": input_modalities, "type": "foundation"}

            # Add cross-region inference model list.
            profile_id = cr_inference_prefix + "." + model_id
            if profile_id in profile_list:
                model_list[profile_id] = {"modalities": input_modalities, "type": "foundation"}

        # Get custom imported models if enabled
        if ENABLE_CUSTOM_MODELS:
            if DEBUG:
                logger.info("Custom models enabled, checking for custom and imported models")

            # Define the regions to check - always include the default region
            regions_to_check = [AWS_REGION]

            # Only use local region for custom models
            # No additional regions are checked

            if DEBUG:
                logger.info(f"Checking the following regions for custom and imported models: {regions_to_check}")

            # Check each region for custom and imported models
            for region in regions_to_check:
                if DEBUG:
                    logger.info(f"Checking region {region} for custom and imported models")

                # Create a client specifically for this region
                try:
                    regional_bedrock_client = boto3.client(
                        service_name="bedrock",
                        region_name=region,
                        config=config,
                    )

                    # List all custom models (fine-tuned models)
                    try:
                        if DEBUG:
                            logger.info(f"Listing custom models in {region}")
                        custom_models_response = regional_bedrock_client.list_custom_models()

                        if DEBUG:
                            logger.info(
                                f"Custom models response from {region}: {json.dumps(custom_models_response, default=str)}"
                            )

                        for model in custom_models_response.get("modelSummaries", []):
                            model_arn = model.get("modelArn")
                            if not model_arn:
                                continue

                            # Extract model name/ID from ARN
                            model_id_from_arn = model_arn.split("/")[-1]

                            # For regular custom models, we might not have a friendly name
                            # Try to get a friendly name if available, otherwise use ID
                            model_name = model.get("modelName", model_id_from_arn)

                            # Create a more descriptive ID that includes model name
                            # Format: [model_name]-id:custom.[aws_id]
                            sanitized_name = model_name.replace(" ", "-").replace("/", "-")
                            model_id = f"{sanitized_name}-id:custom.{model_id_from_arn}"

                            # Store the original ID for reference when invoking models
                            original_id = f"custom.{model_id_from_arn}"

                            # Custom models are currently only supporting TEXT modality
                            # We could potentially get more details via GetCustomModel API
                            model_list[model_id] = {
                                "modalities": ["TEXT"],
                                "type": "custom",
                                "arn": model_arn,
                                "name": model_name,
                                "region": region,
                                "original_id": original_id,
                            }

                            if DEBUG:
                                logger.info(f"Added custom model from {region}: {model_id}")
                    except Exception as e:
                        logger.warning(f"Unable to list custom models in {region}: {str(e)}")

                    # List all imported models (models imported from external sources)
                    try:
                        if DEBUG:
                            logger.info(f"Listing imported models in {region}")
                        imported_models_response = regional_bedrock_client.list_imported_models()

                        if DEBUG:
                            logger.info(
                                f"Imported models response from {region}: {json.dumps(imported_models_response, default=str)}"
                            )

                        for model in imported_models_response.get("modelSummaries", []):
                            model_arn = model.get("modelArn")
                            model_name = model.get("modelName", "")

                            if not model_arn:
                                continue

                            # Extract ID from ARN
                            model_id_from_arn = model_arn.split("/")[-1]

                            # Create a more descriptive ID that includes model name
                            # Format: [model_name]-id:custom.[aws_id]
                            # This way we have a descriptive ID for display
                            sanitized_name = model_name.replace(" ", "-").replace("/", "-")
                            model_id = f"{sanitized_name}-id:custom.{model_id_from_arn}"

                            # Also store the original AWS ID for reference when invoking models
                            original_id = f"custom.{model_id_from_arn}"

                            # Add model to the list
                            modalities = ["TEXT"]  # Default to TEXT modality
                            model_list[model_id] = {
                                "modalities": modalities,
                                "type": "custom",
                                "arn": model_arn,
                                "name": model_name,
                                "region": region,
                                "original_id": original_id,
                            }

                            if DEBUG:
                                logger.info(f"Added imported model from {region}: {model_id} (Name: {model_name})")
                    except Exception as e:
                        logger.warning(f"Unable to list imported models in {region}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error connecting to region {region}: {str(e)}")

    except Exception as e:
        logger.error(f"Unable to list models: {str(e)}")

    if not model_list:
        # In case stack not updated.
        model_list[DEFAULT_MODEL] = {"modalities": ["TEXT", "IMAGE"], "type": "foundation"}

    if DEBUG:
        logger.info(f"Final model list contains {len(model_list)} models")
        custom_models = {k: v for k, v in model_list.items() if v.get("type") == "custom"}
        logger.info(f"Custom models in final list: {len(custom_models)}")
        if custom_models:
            logger.info(f"Custom model IDs: {list(custom_models.keys())}")

    return model_list


# Initialize the model list.
bedrock_model_list = list_bedrock_models()


class BedrockModel(BaseChatModel):
    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        global bedrock_model_list
        bedrock_model_list = list_bedrock_models()
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        model_id = chat_request.model

        # Check if model is directly supported
        if model_id in bedrock_model_list.keys():
            # For models with custom format, extract the AWS ID
            if "-id:custom." in model_id:
                # Extract information from the ID
                parts = model_id.split("-id:")
                if len(parts) == 2:
                    model_name = parts[0]
                    original_id = parts[1]

                    # Store the display name for logging
                    chat_request.custom_model_display_name = model_name

                    # Store the original model ID for response consistency
                    chat_request._model_original = model_id

                    # Update the model ID to the AWS format for invocation
                    chat_request.model = original_id
            return

        # Special case handling for direct custom.* format (backward compatibility)
        if model_id.startswith("custom."):
            # Check if this is a valid custom model ID
            for listed_id, info in bedrock_model_list.items():
                if "-id:" in listed_id and listed_id.split("-id:")[1] == model_id:
                    # Found a matching custom model
                    # Store the display name if available
                    if "name" in info:
                        chat_request.custom_model_display_name = info["name"]
                    return

        # If we get here, the model is not supported
        error = f"Unsupported model {model_id}, please use models API to get a list of supported models"

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""
        if DEBUG:
            logger.info("Raw request: " + chat_request.model_dump_json())

        # Check if this is a custom model or foundation model
        model_id = chat_request.model
        model_info = bedrock_model_list.get(model_id, {})

        # If model_info is empty, this might be using the original AWS ID format
        # after being processed by the validate method
        if not model_info and "-id:custom." not in model_id:
            # Look for a model with this original_id
            for _, info in bedrock_model_list.items():
                if info.get("original_id") == model_id:
                    model_info = info
                    break

        model_type = model_info.get("type", "foundation")

        if model_type == "custom":
            # Custom imported model invocation path
            return await self._invoke_custom_model(chat_request, model_info, stream)
        else:
            # Foundation model invocation path - standard converse API
            # convert OpenAI chat request to Bedrock SDK request
            args = self._parse_request(chat_request)
            if DEBUG:
                logger.info("Bedrock request: " + json.dumps(str(args)))

            try:
                if stream:
                    # Run the blocking boto3 call in a thread pool
                    response = await run_in_threadpool(bedrock_runtime.converse_stream, **args)
                else:
                    # Run the blocking boto3 call in a thread pool
                    response = await run_in_threadpool(bedrock_runtime.converse, **args)
            except bedrock_runtime.exceptions.ValidationException as e:
                logger.error("Validation Error: " + str(e))
                raise HTTPException(status_code=400, detail=str(e))
            except bedrock_runtime.exceptions.ThrottlingException as e:
                logger.error("Throttling Error: " + str(e))
                raise HTTPException(status_code=429, detail=str(e))
            except Exception as e:
                logger.error(e)
                raise HTTPException(status_code=500, detail=str(e))
            return response

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        message_id = self.generate_message_id()
        response = await self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]

        # Use the original model ID from the request for the response
        # If we transformed a descriptive ID to the AWS format in validation, use the original
        # This ensures we maintain the user-friendly format in responses
        original_request_model = getattr(chat_request, "_model_original", chat_request.model)
        response_model_id = original_request_model

        if DEBUG and getattr(chat_request, "custom_model_display_name", None):
            logger.info(f"Using model: {chat_request.model} (Name: {chat_request.custom_model_display_name})")

        chat_response = self._create_response(
            model=response_model_id,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

    async def _async_iterate(self, stream):
        """Helper method to convert sync iterator to async iterator"""
        for chunk in stream:
            await run_in_threadpool(lambda: chunk)
            yield chunk

    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        try:
            response = await self._invoke_bedrock(chat_request, stream=True)
            message_id = self.generate_message_id()

            # Use the original model ID from the request for the response
            original_request_model = getattr(chat_request, "_model_original", chat_request.model)
            response_model_id = original_request_model

            if DEBUG and getattr(chat_request, "custom_model_display_name", None):
                logger.info(
                    f"Streaming using model: {chat_request.model} (Name: {chat_request.custom_model_display_name})"
                )

            # Check if this is a custom model response
            if isinstance(response, dict) and "response_stream" in response:
                # Custom model streaming implementation
                response["model_id"] = response_model_id  # Use the display ID
                async for chunk in self._handle_custom_model_stream(response, message_id, chat_request):
                    yield chunk
                return

            # Standard foundation model streaming
            stream = response.get("stream")
            async for chunk in self._async_iterate(stream):
                args = {"model_id": response_model_id, "message_id": message_id, "chunk": chunk}
                stream_response = self._create_response_stream(**args)
                if not stream_response:
                    continue
                if DEBUG:
                    logger.info("Proxy response :" + stream_response.model_dump_json())
                if stream_response.choices:
                    yield self.stream_response_to_bytes(stream_response)
                elif chat_request.stream_options and chat_request.stream_options.include_usage:
                    # An empty choices for Usage as per OpenAI doc below:
                    # if you set stream_options: {"include_usage": true}.
                    # an additional chunk will be streamed before the data: [DONE] message.
                    # The usage field on this chunk shows the token usage statistics for the entire request,
                    # and the choices field will always be an empty array.
                    # All other chunks will also include a usage field, but with a null value.
                    yield self.stream_response_to_bytes(stream_response)

            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()
        except Exception as e:
            error_event = Error(error=ErrorMessage(message=str(e)))
            yield self.stream_response_to_bytes(error_event)

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
                        "content": self._parse_content_parts(message, chat_request.model),
                    }
                )
            elif isinstance(message, AssistantMessage):
                if message.content:
                    # Text message
                    messages.append(
                        {
                            "role": message.role,
                            "content": self._parse_content_parts(message, chat_request.model),
                        }
                    )
                if message.tool_calls:
                    # Tool use message
                    for tool_call in message.tool_calls:
                        tool_input = json.loads(tool_call.function.arguments)
                        messages.append(
                            {
                                "role": message.role,
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": tool_call.id,
                                            "name": tool_call.function.name,
                                            "input": tool_input,
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
        """Receive messages and reformat them to comply with the Claude format

        With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
        with Claude format requests, you cannot repeatedly receive messages from the same role.

        This method searches through the OpenAI format messages in order and reformats them to the Claude format.

        ```
        openai_format_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Who are you?"},
        ]

        bedrock_format_messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Hello"},
                    {"text": "Who are you?"}
                ]
            },
        ]
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message["role"]
            next_content = message["content"]

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append({"role": current_role, "content": current_content})
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
            reformatted_messages.append({"role": current_role, "content": current_content})

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

        if chat_request.stop is not None:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            inference_config["stopSequences"] = stop

        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        if chat_request.reasoning_effort:
            # From OpenAI api, the max_token is not supported in reasoning mode
            # Use max_completion_tokens if provided.

            max_tokens = (
                chat_request.max_completion_tokens if chat_request.max_completion_tokens else chat_request.max_tokens
            )
            budget_tokens = self._calc_budget_tokens(max_tokens, chat_request.reasoning_effort)
            inference_config["maxTokens"] = max_tokens
            # unset topP - Not supported
            inference_config.pop("topP")

            args["additionalModelRequestFields"] = {
                "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
            }
        # add tool config
        if chat_request.tools:
            tool_config = {"tools": [self._convert_tool_spec(t.function) for t in chat_request.tools]}

            if chat_request.tool_choice and not chat_request.model.startswith("meta.llama3-1-"):
                if isinstance(chat_request.tool_choice, str):
                    # auto (default) is mapped to {"auto" : {}}
                    # required is mapped to {"any" : {}}
                    if chat_request.tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    assert "function" in chat_request.tool_choice
                    tool_config["toolChoice"] = {"tool": {"name": chat_request.tool_choice["function"].get("name", "")}}
            args["toolConfig"] = tool_config
        return args

    def _create_response(
        self,
        model: str,
        message_id: str,
        content: list[dict] | None = None,
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
            for c in content:
                if "reasoningContent" in c:
                    message.reasoning_content = c["reasoningContent"]["reasoningText"].get("text", "")
                elif "text" in c:
                    message.content = c["text"]
                else:
                    logger.warning("Unknown tag in message content " + ",".join(c.keys()))

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

    def _create_response_stream(self, model_id: str, message_id: str, chunk: dict) -> ChatStreamResponse | None:
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
            elif "reasoningContent" in delta:
                # ignore "signature" in the delta.
                if "text" in delta["reasoningContent"]:
                    message = ChatResponseMessage(
                        reasoning_content=delta["reasoningContent"]["text"],
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
                            ),
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
            raise HTTPException(status_code=500, detail="Unable to access the image url")

    def _parse_content_parts(
        self,
        message: UserMessage | AssistantMessage,
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
                if not self.is_supported_modality(model_id, modality="IMAGE"):
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

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        model = bedrock_model_list.get(model_id, {})
        modalities = model.get("modalities", [])
        if modality in modalities:
            return True
        return False

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

    def _calc_budget_tokens(self, max_tokens: int, reasoning_effort: Literal["low", "medium", "high"]) -> int:
        # Helper function to calculate budget_tokens based on the max_tokens.
        # Ratio for efforts:  Low - 30%, medium - 60%, High: Max token - 1
        # Note that The minimum budget_tokens is 1,024 tokens so far.
        # But it may be changed for different models in the future.
        if reasoning_effort == "low":
            return int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            return int(max_tokens * 0.6)
        else:
            return max_tokens - 1

    async def _invoke_custom_model(self, chat_request: ChatRequest, model_info: dict, stream: bool = False) -> dict:
        """Invoke a custom imported model through the InvokeModel API.

        Custom models use a different API (InvokeModel) than foundation models, and
        require using the model ARN instead of ID.
        """
        if DEBUG:
            logger.info(f"Invoking custom model: {chat_request.model}")

        # Get the model ARN from the model_info
        model_arn = model_info.get("arn")
        if not model_arn:
            raise HTTPException(status_code=400, detail=f"ARN not found for custom model {chat_request.model}")

        # Get the region for this model - default to AWS_REGION if not specified
        model_region = model_info.get("region", AWS_REGION)
        if DEBUG:
            logger.info(f"Using region {model_region} for custom model {chat_request.model}")

        # Create a custom runtime client with exponential backoff retries
        # This helps handle the ModelNotReadyException which can occur with imported models
        custom_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=model_region,
            config=custom_model_config,
        )

        # Create a simplified prompt from the messages
        prompt = self._create_prompt_from_messages(chat_request.messages)

        # Prepare the body for the custom model
        # Format depends on the underlying model - we're using a standard text generation format
        body = {
            "prompt": prompt,
            "max_tokens": chat_request.max_tokens,
            "temperature": chat_request.temperature,
            "top_p": chat_request.top_p,
        }

        # Additional parameters may be added depending on the specific custom model requirements
        if chat_request.stop:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            body["stop_sequences"] = stop

        # Serialize the body for the API call
        serialized_body = json.dumps(body)

        try:
            if stream:
                # Streaming response for custom models
                response = await run_in_threadpool(
                    lambda: custom_runtime.invoke_model_with_response_stream(
                        modelId=model_arn,
                        contentType="application/json",
                        accept="application/json",
                        body=serialized_body,
                    )
                )

                # Return a special response structure for the streaming handler
                return {
                    "response_stream": response.get("body"),
                    "content_type": response.get("contentType"),
                    "model_id": chat_request.model,
                }
            else:
                # Non-streaming response
                response = await run_in_threadpool(
                    lambda: custom_runtime.invoke_model(
                        modelId=model_arn,
                        contentType="application/json",
                        accept="application/json",
                        body=serialized_body,
                    )
                )

                # Parse the response body
                response_body = json.loads(response.get("body").read())

                if DEBUG:
                    logger.info(f"Custom model response: {response_body}")

                # Extract the completion text - format depends on the model type
                # We'll handle common formats used by popular models
                completion_text = ""
                input_tokens = 0
                output_tokens = 0

                # Handle different response formats based on model type
                if "completion" in response_body:
                    completion_text = response_body["completion"]
                elif "generations" in response_body and isinstance(response_body["generations"], list):
                    # Handle formats like Anthropic Claude style
                    completion_text = response_body["generations"][0].get("text", "")
                elif "generated_text" in response_body:
                    # Handle formats like Hugging Face style
                    completion_text = response_body["generated_text"]
                elif "choices" in response_body and isinstance(response_body["choices"], list):
                    # Handle formats like OpenAI style
                    completion_text = response_body["choices"][0].get("text", "")

                # Try to extract token usage info if available
                if "usage" in response_body:
                    input_tokens = response_body["usage"].get("prompt_tokens", 0)
                    output_tokens = response_body["usage"].get("completion_tokens", 0)

                # Create a response structure that matches the converse API response
                return {
                    "output": {"message": {"content": [{"text": completion_text}]}},
                    "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
                    "stopReason": "finished",
                }

        except custom_runtime.exceptions.ModelNotReadyException as e:
            # This exception is specific to imported models
            logger.error(f"Model not ready: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Custom model {chat_request.model} is not ready yet: {str(e)}")
        except custom_runtime.exceptions.ValidationException as e:
            logger.error(f"Validation Error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except custom_runtime.exceptions.ThrottlingException as e:
            logger.error(f"Throttling Error: {str(e)}")
            raise HTTPException(status_code=429, detail=str(e))
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"AWS Client Error: {error_code} - {error_message}")
            raise HTTPException(status_code=500, detail=f"Custom model error ({error_code}): {error_message}")
        except Exception as e:
            logger.error(f"Error invoking custom model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _create_prompt_from_messages(self, messages: list) -> str:
        """Create a simple text prompt from the messages list.

        This is a simplified version for custom models that may not support
        the structured conversation format used by foundation models.
        """
        prompt_parts = []

        for message in messages:
            role = message.role

            # Format content based on message type
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                # Combine text parts
                content_parts = []
                for part in message.content:
                    if hasattr(part, "text"):
                        content_parts.append(part.text)
                content = " ".join(content_parts)
            else:
                continue

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool ({message.tool_call_id}): {content}")

        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

    async def _handle_custom_model_stream(
        self, response: dict, message_id: str, chat_request: ChatRequest
    ) -> AsyncIterable[bytes]:
        """Process streaming responses from custom models."""
        try:
            stream = response.get("response_stream")
            model_id = response.get("model_id", chat_request.model)

            # Accumulated text for token counting
            accumulated_text = ""

            # Start with the message start chunk
            start_response = ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[ChoiceDelta(index=0, delta=ChatResponseMessage(role="assistant", content=""))],
            )
            yield self.stream_response_to_bytes(start_response)

            # Process each chunk from the custom model stream
            async for chunk in self._async_iterate(stream):
                if chunk.get("contentType") != "application/json":
                    continue

                # Parse the chunk
                try:
                    chunk_body = json.loads(chunk.get("chunk").get("bytes").decode())
                    if DEBUG:
                        logger.info(f"Custom model stream chunk: {chunk_body}")

                    # Extract the delta text
                    delta_text = ""
                    if "completion" in chunk_body:
                        delta_text = chunk_body["completion"]
                    elif "delta" in chunk_body:
                        delta_text = chunk_body["delta"]
                    elif "content" in chunk_body:
                        delta_text = chunk_body["content"]
                    elif "text" in chunk_body:
                        delta_text = chunk_body["text"]

                    if delta_text:
                        accumulated_text += delta_text

                        # Create a stream response for this chunk
                        stream_response = ChatStreamResponse(
                            id=message_id,
                            model=model_id,
                            choices=[ChoiceDelta(index=0, delta=ChatResponseMessage(content=delta_text))],
                        )
                        yield self.stream_response_to_bytes(stream_response)
                except Exception as e:
                    logger.warning(f"Error parsing custom model stream chunk: {str(e)}")
                    continue

            # Estimate token count for usage info
            input_tokens = len(self._create_prompt_from_messages(chat_request.messages).split())
            output_tokens = len(accumulated_text.split())

            # Send usage information if requested
            if chat_request.stream_options and chat_request.stream_options.include_usage:
                usage_response = ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )
                yield self.stream_response_to_bytes(usage_response)

            # Send completion message
            end_response = ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[ChoiceDelta(index=0, delta=ChatResponseMessage(), finish_reason="stop")],
            )
            yield self.stream_response_to_bytes(end_response)

            # Send [DONE] marker
            yield self.stream_response_to_bytes()

        except Exception as e:
            logger.error(f"Error in custom model streaming: {str(e)}")
            error_event = Error(error=ErrorMessage(message=str(e)))
            yield self.stream_response_to_bytes(error_event)

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
                "content_filtered": "content_filter",
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
        except bedrock_runtime.exceptions.ThrottlingException as e:
            logger.error("Throttling Error: " + str(e))
            raise HTTPException(status_code=429, detail=str(e))
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
        response = self._invoke_model(args=self._parse_args(embeddings_request), model_id=embeddings_request.model)
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
        elif isinstance(embeddings_request.input, list) and len(embeddings_request.input) == 1:
            input_text = embeddings_request.input[0]
        else:
            raise ValueError("Amazon Titan Embeddings models support only single strings as input.")
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
        response = self._invoke_model(args=self._parse_args(embeddings_request), model_id=embeddings_request.model)
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
