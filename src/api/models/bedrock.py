import base64
import json
import logging
from typing import AsyncIterable, Iterable

import boto3
import requests
import tiktoken
from fastapi import HTTPException

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    # Chat
    ChatResponse,
    ChatRequest,
    ChatRequestMessage,
    Choice,
    ChatResponseMessage,
    Usage,
    ChatStreamResponse,
    ChoiceDelta,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Embedding, TextContent,
)
from api.setting import DEBUG, AWS_REGION

logger = logging.getLogger(__name__)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

SUPPORTED_BEDROCK_MODELS = {
    "anthropic.claude-instant-v1": "Claude Instant",
    "anthropic.claude-v2:1": "Claude",
    "anthropic.claude-v2": "Claude",
    "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet",
    "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku",
    "meta.llama2-13b-chat-v1": "Llama 2 Chat 13B",
    "meta.llama2-70b-chat-v1": "Llama 2 Chat 70B",
    "mistral.mistral-7b-instruct-v0:2": "Mistral 7B Instruct",
    "mistral.mixtral-8x7b-instruct-v0:1": "Mixtral 8x7B Instruct",
}

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    # Disable Titan embedding.
    # "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
class BedrockModel(BaseChatModel):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str, with_stream: bool = False):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        if with_stream:
            return bedrock_runtime.invoke_model_with_response_stream(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        return bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept=self.accept,
            contentType=self.content_type,
        )

    def _create_response(
            self,
            model: str,
            message: str,
            message_id: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> ChatResponse:
        choice = Choice(
            index=0,
            message=ChatResponseMessage(
                role="assistant",
                content=message,
            ),
            finish_reason="stop",
        )
        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response

    def _create_response_stream(
            self, model: str, message_id: str, chunk_message: str, finish_reason: str | None
    ) -> ChatStreamResponse:
        choice = ChoiceDelta(
            index=0,
            delta=ChatResponseMessage(
                role="assistant",
                content=chunk_message,
            ),
            finish_reason=finish_reason,
        )
        response = ChatStreamResponse(
            id=message_id,
            model=model,
            choices=[choice],
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response


def get_model(model_id: str) -> BedrockModel:
    model_name = SUPPORTED_BEDROCK_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    if model_name in ["Claude Instant", "Claude", "Claude 3 Sonnet", "Claude 3 Haiku"]:
        return ClaudeModel()
    elif model_name in ["Llama 2 Chat 13B", "Llama 2 Chat 70B"]:
        return Llama2Model()
    elif model_name in ["Mistral 7B Instruct", "Mixtral 8x7B Instruct"]:
        return MistralModel()
    else:
        logger.error("Unsupported model id " + model_id)
        raise ValueError("Invalid model ID")


class ClaudeModel(BedrockModel):
    anthropic_version = "bedrock-2023-05-31"

    def _get_base64_image(self, image_url: str):
        # Send a request to the image URL
        response = requests.get(image_url)
        # Check if the request was successful
        if response.status_code == 200:
            # Get the image content
            image_content = response.content
            # Encode the image content as base64
            base64_image = base64.b64encode(image_content)
            return base64_image.decode('utf-8')
        else:
            raise HTTPException(status_code=500, detail="Unable to access the image url")

    def _parse_messages(self, messages: list[ChatRequestMessage]) -> list[dict]:
        # Refer to: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        converted_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                converted_messages.append({"role": msg.role, "content": msg.content})
                continue

            content_parts = []
            for part in msg.content:
                if isinstance(part, TextContent):
                    content_parts.append(part.model_dump())
                else:
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._get_base64_image(part.image_url.url)
                        }
                    })

            converted_messages.append({"role": msg.role, "content": content_parts})
        return converted_messages

    def _parse_args(self, chat_request: ChatRequest) -> dict:
        args = {
            "anthropic_version": self.anthropic_version,
            "max_tokens": chat_request.max_tokens,
            "top_p": chat_request.top_p,
            "temperature": chat_request.temperature,
        }
        start = 0
        if chat_request.messages[0].role == "system":
            args["system"] = chat_request.messages[0].content
            start = 1
        args["messages"] = self._parse_messages(chat_request.messages[start:])
        return args

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        response = self._invoke_model(
            args=self._parse_args(chat_request), model_id=chat_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            model=chat_request.model,
            message=response_body["content"][0]["text"],
            message_id=response_body["id"],
            input_tokens=response_body["usage"]["input_tokens"],
            output_tokens=response_body["usage"]["output_tokens"],
        )

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        response = self._invoke_model(
            args=self._parse_args(chat_request),
            model_id=chat_request.model,
            with_stream=True,
        )
        msg_id = ""
        chunk_id = 0
        for event in response.get("body"):
            if DEBUG:
                logger.info("Bedrock response chunk: " + str(event))
            chunk = json.loads(event["chunk"]["bytes"])
            chunk_id += 1
            if chunk["type"] == "message_start":
                msg_id = chunk["message"]["id"]
                continue

            if chunk["type"] == "message_delta":
                chunk_message = ""
                finish_reason = "stop"

            elif chunk["type"] == "content_block_delta":
                chunk_message = chunk["delta"]["text"]
                finish_reason = None
            else:
                continue
            response = self._create_response_stream(
                model=chat_request.model,
                message_id=msg_id,
                chunk_message=chunk_message,
                finish_reason=finish_reason,
            )

            yield self._stream_response_to_bytes(response)


class Llama2Model(BedrockModel):

    def _convert_prompt(self, messages: list[ChatRequestMessage]) -> str:
        """Create a prompt message follow below example:

        <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s>
        <s>[INST] {user_message_2} [/INST]
        """
        if DEBUG:
            logger.info("Convert below messages to prompt for Llama 2: ")
            for msg in messages:
                logger.info(msg.model_dump_json())
        bos_token = "<s>"
        eos_token = "</s>"
        prompt = bos_token + "[INST] "
        start = 0
        end_turn = False
        if messages[0].role == "system":
            prompt += "<<SYS>>\n" + messages[0].content + "\n<<SYS>>\n\n"
            start = 1
        # TODO: Add validation
        for i in range(start, len(messages)):
            msg = messages[i]
            if not isinstance(msg.content, str):
                raise HTTPException(status_code=400, detail="Content must be a string for Llama 2 model")
            if msg.role == "user":
                if end_turn:
                    prompt += bos_token + "[INST] "
                prompt += msg.content + " [/INST] "
                end_turn = False
            else:
                prompt += msg.content + eos_token
                end_turn = True
        if DEBUG:
            logger.info("Converted prompt: " + prompt.replace("\n", "\\n"))
        return prompt

    def _parse_args(self, chat_request: ChatRequest) -> dict:
        prompt = self._convert_prompt(chat_request.messages)
        return {
            "prompt": prompt,
            "max_gen_len": chat_request.max_tokens,
            "temperature": chat_request.temperature,
            "top_p": chat_request.top_p,
        }

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        response = self._invoke_model(
            args=self._parse_args(chat_request), model_id=chat_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))
        message_id = self._generate_message_id()

        return self._create_response(
            model=chat_request.model,
            message=response_body["generation"],
            message_id=message_id,
            input_tokens=response_body["prompt_token_count"],
            output_tokens=response_body["generation_token_count"],
        )

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        response = self._invoke_model(
            args=self._parse_args(chat_request),
            model_id=chat_request.model,
            with_stream=True,
        )
        msg_id = ""
        chunk_id = 0
        for event in response.get("body"):
            if DEBUG:
                logger.info("Bedrock response chunk: " + str(event))
            chunk = json.loads(event["chunk"]["bytes"])
            chunk_id += 1
            response = self._create_response_stream(
                model=chat_request.model,
                message_id=msg_id,
                chunk_message=chunk["generation"],
                finish_reason=chunk["stop_reason"],
            )
            yield self._stream_response_to_bytes(response)


class MistralModel(BedrockModel):
    def _convert_prompt(self, messages: list[ChatRequestMessage]) -> str:
        """Create a prompt message follow below example:

        <s>[INST] {your_system_message}\n{user_message_1} [/INST] {model_reply_1}</s>
        <s>[INST] {user_message_2} [/INST]
        """
        if DEBUG:
            logger.info("Convert below messages to prompt for Llama 2: ")
            for msg in messages:
                logger.info(msg.model_dump_json())
        bos_token = "<s>"
        eos_token = "</s>"
        prompt = bos_token + "[INST] "
        start = 0
        end_turn = False
        if messages[0].role == "system":
            prompt += messages[0].content + "\n"
            start = 1
        # TODO: Add validation
        for i in range(start, len(messages)):
            msg = messages[i]
            if not isinstance(msg.content, str):
                raise HTTPException(status_code=400, detail="Content must be a string for Mistral/Mixtral model")
            if msg.role == "user":
                if end_turn:
                    prompt += bos_token + "[INST] "
                prompt += msg.content + " [/INST] "
                end_turn = False
            else:
                prompt += msg.content + eos_token
                end_turn = True
        if DEBUG:
            logger.info("Converted prompt: " + prompt.replace("\n", "\\n"))
        return prompt

    def _parse_args(self, chat_request: ChatRequest) -> dict:
        prompt = self._convert_prompt(chat_request.messages)
        return {
            "prompt": prompt,
            "max_tokens": chat_request.max_tokens,
            "temperature": chat_request.temperature,
            "top_p": chat_request.top_p,
        }

    def chat(self, chat_request: ChatRequest) -> ChatResponse:

        response = self._invoke_model(
            args=self._parse_args(chat_request), model_id=chat_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))
        message_id = self._generate_message_id()

        return self._create_response(
            model=chat_request.model,
            message=response_body["outputs"][0]["text"],
            message_id=message_id,
        )

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        response = self._invoke_model(
            args=self._parse_args(chat_request),
            model_id=chat_request.model,
            with_stream=True,
        )
        msg_id = ""
        chunk_id = 0
        for event in response.get("body"):
            if DEBUG:
                logger.info("Bedrock response chunk: " + str(event))
            chunk = json.loads(event["chunk"]["bytes"])
            chunk_id += 1
            response = self._create_response_stream(
                model=chat_request.model,
                message_id=msg_id,
                chunk_message=chunk["outputs"][0]["text"],
                finish_reason=chunk["outputs"][0]["stop_reason"],
            )
            yield self._stream_response_to_bytes(response)


class BedrockEmbeddingsModel(BaseEmbeddingsModel):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        return bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept=self.accept,
            contentType=self.content_type,
        )

    def _create_response(
            self,
            embeddings: list[float],
            model: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> EmbeddingsResponse:
        data = [
            Embedding(
                index=i,
                embedding=embedding
            ) for i, embedding in enumerate(embeddings)
        ]
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


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    if model_name in ["Cohere Embed Multilingual", "Cohere Embed English"]:
        return CohereEmbeddingsModel()
    elif model_name in ["Titan Embeddings G1 - Text", "Titan Multimodal Embeddings G1"]:
        return TitanEmbeddingsModel()
    else:
        logger.error("Unsupported model id " + model_id)
        raise ValueError("Invalid model ID")


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
            args["embeddingConfig"] = embeddings_request.embedding_config if embeddings_request.embedding_config else {
                "outputEmbeddingLength": 1024}
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
            input_tokens=response_body["inputTextTokenCount"]
        )
