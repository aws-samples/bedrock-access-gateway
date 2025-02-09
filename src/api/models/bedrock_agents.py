import base64
import json
import logging
import re
import time
from abc import ABC
from typing import AsyncIterable, Iterable, Literal

import boto3
from botocore.config import Config
import numpy as np
import requests
import tiktoken
from fastapi import HTTPException

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.models.bedrock import BedrockModel
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

KB_PREFIX = 'kb-'
AGENT_PREFIX = 'ag-'

DEFAULT_KB_MODEL = 'anthropic.claude-3-haiku-20240307-v1:0'
DEFAULT_KB_MODEL_ARN = f'arn:aws:bedrock:{AWS_REGION}::foundation-model/{DEFAULT_KB_MODEL}'

logger = logging.getLogger(__name__)

config = Config(connect_timeout=1, read_timeout=120, retries={"max_attempts": 1})

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)

bedrock_agent = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=AWS_REGION,
    config=config,
)

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    # Disable Titan embedding.
    # "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")


class BedrockAgents(BedrockModel):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features


    _supported_models = {
        
    }
    
    # get list of active knowledgebases
    def get_kbs(self):

        bedrock_ag = boto3.client(
            service_name="bedrock-agent",
            region_name=AWS_REGION,
            config=config,
        )
        
        # List knowledge bases
        response = bedrock_ag.list_knowledge_bases(maxResults=100)

        # Print knowledge base information
        for kb in response['knowledgeBaseSummaries']:
            name = f"{KB_PREFIX}{kb['name']}"
            val = {
                "system": True,      # Supports system prompts for context setting
                "multimodal": True,  # Capable of processing both text and images
                "tool_call": True,
                "stream_tool_call": True,
                "kb_id": kb['knowledgeBaseId'],
                "model_id": DEFAULT_KB_MODEL
            }
            self._supported_models[name] = val
    
    def get_latest_agent_alias(self, client, agent_id):

        # List all aliases for the agent
        response = client.list_agent_aliases(
            agentId=agent_id,
            maxResults=100  # Adjust based on your needs
        )
        
        if not response.get('agentAliasSummaries'):
            return None
            
        # Sort aliases by creation time to get the latest one
        aliases = response['agentAliasSummaries']
        latest_alias = None
        latest_creation_time = None
        
        for alias in aliases:
            # Only consider aliases that are in PREPARED state
            if alias['agentAliasStatus'] == 'PREPARED':
                creation_time = alias.get('creationDateTime')
                if latest_creation_time is None or creation_time > latest_creation_time:
                    latest_creation_time = creation_time
                    latest_alias = alias
        
        if latest_alias:
            return latest_alias['agentAliasId']
            
        return None

        

    def get_agents(self):
        bedrock_ag = boto3.client(
            service_name="bedrock-agent",
            region_name=AWS_REGION,
            config=config,
        )
        # List Agents
        response = bedrock_ag.list_agents(maxResults=100)

        # Prepare agent for display
        for agent in response['agentSummaries']:
           
            if (agent['agentStatus'] != 'PREPARED'):
                continue

            name = f"{AGENT_PREFIX}{agent['agentName']}"
            agentId = agent['agentId']

            aliasId = self.get_latest_agent_alias(bedrock_ag, agentId)
            if (aliasId is None):
                continue

            val = {
                "system": False,      # Supports system prompts for context setting
                "multimodal": True,  # Capable of processing both text and images
                "tool_call": True,
                "stream_tool_call": False,
                "agent_id": agentId,
                "alias_id": aliasId
            }
            self._supported_models[name] = val
    
    def get_models(self):
        
        client = boto3.client(
            service_name="bedrock",
            region_name=AWS_REGION,
            config=config,
        )
        response = client.list_foundation_models(byInferenceType='ON_DEMAND')

        # Prepare agent for display
        for model in response['modelSummaries']:
            if ((model['modelLifecycle']['status'] == 'ACTIVE') and 
                #('ON_DEMAND' in model['inferenceTypesSupported']) and 
                ('EMBEDDING' not in model['outputModalities'])) :
                name = f"{model['modelId']}"
                stream_support = False

                if (('responseStreamingSupported' in model.keys()) and 
                    (model['responseStreamingSupported'] is True)):
                    stream_support = True

                val = {
                    "system": True,      # Supports system prompts for context setting
                    "multimodal": len(model['inputModalities'])>1,  # Capable of processing both text and images
                    "tool_call": True,
                    "stream_tool_call": stream_support,
                }
                self._supported_models[name] = val

    def supported_models(self):
        logger.info("BedrockAgents.supported_models")
        return list(self._supported_models.keys())

    def list_models(self) -> list[str]:
        logger.info("BedrockAgents.list_models")
        self.get_models()
        self.get_kbs()
        self.get_agents()
        return list(self._supported_models.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        #logger.info(f"BedrockAgents.validate: {chat_request}")

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
            logger.info("BedrockAgents._invoke_bedrock: Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        try:
            
            if stream:
                response = bedrock_client.converse_stream(**args)
            else:
                response = bedrock_client.converse(**args)


        except bedrock_client.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""
        #chat: {chat_request}")
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
        logger.info(f"BedrockAgents.chat_stream: {chat_request}")
        
        response = ''
        message_id = self.generate_message_id()

        if (chat_request.model.startswith(KB_PREFIX)):
            response = self._invoke_kb(chat_request, stream=True)    
        elif (chat_request.model.startswith(AGENT_PREFIX)):
            response = self._invoke_agent(chat_request, stream=True)

            _event_stream = response["completion"]
            
            chunk_count = 1
            message = ChatResponseMessage(
                role="assistant",
                content="",
            )
            stream_response = ChatStreamResponse(
                id=message_id,
                model=chat_request.model,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield self.stream_response_to_bytes(stream_response)

            for event in _event_stream:
                #print(f'\n\nChunk {chunk_count}: {event}')
                chunk_count += 1
                if "chunk" in event:
                    _data = event["chunk"]["bytes"].decode("utf8")
                    message = ChatResponseMessage(content=_data)

                    stream_response = ChatStreamResponse(
                        id=message_id,
                        model=chat_request.model,
                        choices=[
                            ChoiceDelta(
                                index=0,
                                delta=message,
                                logprobs=None,
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield self.stream_response_to_bytes(stream_response)
                    
                    #message = self._make_fully_cited_answer(_data, event, False, 0)
            
            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()
            return None
        else:
            response = self._invoke_bedrock(chat_request, stream=True)

        print(response)
        stream = response.get("stream")
        # chunk_count = 1
        for chunk in stream:
            #print(f'\n\nChunk {chunk_count}: {chunk}')
            # chunk_count += 1

            stream_response = self._create_response_stream(
                model_id=chat_request.model, message_id=message_id, chunk=chunk
            )
            #print(f'stream_response: {stream_response}')
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
        #logger.info(f"BedrockAgents._parse_system_prompts: {chat_request}")

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

        #logger.info(f"BedrockAgents._parse_messages: {chat_request}")

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

    # This function invokes knowledgebase
    def _invoke_kb(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke kb with default model"""
        if DEBUG:
            logger.info("BedrockAgents._invoke_kb: Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        

        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        model = self._supported_models[chat_request.model]
        logger.info(f"model: {model}")

        args['modelId'] = model['model_id']
        logger.info(f"args: {args}")
        
        ################

        try:
            query = args['messages'][0]['content'][0]['text']
            messages = args['messages']
            query = messages[len(messages)-1]['content'][0]['text']
            logger.info(f"Query: {query}")

            # Step 1 - Retrieve Context
            retrieval_request_body = {
                "retrievalQuery": {
                    "text": query
                },
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 2
                    }
                }
            }
                
            # Make the retrieve request
            response = bedrock_agent.retrieve(knowledgeBaseId=model['kb_id'], **retrieval_request_body)
            logger.info(f"retrieve response: {response}")
            
            # Extract and return the results
            context = ''
            if "retrievalResults" in response:
                for result in response["retrievalResults"]:
                    result = result["content"]["text"]
                    #logger.info(f"Result: {result}")
                    context = f"{context}\n{result}"
                    
                    
            # Step 2 - Append context in the prompt
            args['messages'][0]['content'][0]['text'] = f"Context: {context} \n\n {query}"

            #print(args)

            # Step 3 - Make the converse request
            if stream:
                response = bedrock_client.converse_stream(**args)
            else:
                response = bedrock_client.converse(**args)

            logger.info(f'kb response: {response}')

        except Exception as e:
            print(f"Error retrieving from knowledge base: {str(e)}")
            raise

        ###############
        return response
    
    # This function invokes knowledgebase
    def _invoke_agent(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke agent """
        if DEBUG:
            logger.info("BedrockAgents._invoke_agent: Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        

        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        model = self._supported_models[chat_request.model]
        #logger.info(f"model: {model}")
        logger.info(f"args: {args}")
        
        ################

        try:
            query = args['messages'][0]['content'][0]['text']
            messages = args['messages']
            query = messages[len(messages)-1]['content'][0]['text']
            query = f"My customer id is 1. {query}"
            logger.info(f"Query: {query}")

            # Step 1 - Retrieve Context
            request_params = {
                'agentId': model['agent_id'],
                'agentAliasId': model['alias_id'],
                'sessionId': 'unique-session-id',  # Generate a unique session ID
                'inputText': query
            }
                
            # Make the retrieve request
            # Invoke the agent
            response = bedrock_agent.invoke_agent(**request_params)
            return response
            #logger.info(f'agent response: {response} ----\n\n')

            _event_stream = response["completion"]
            
            chunk_count = 1
            for event in _event_stream:
                #print(f'\n\nChunk {chunk_count}: {event}')
                chunk_count += 1
                if "chunk" in event:
                    _data = event["chunk"]["bytes"].decode("utf8")
                    _agent_answer = self._make_fully_cited_answer(
                        _data, event, False, 0)
                    
            
            #print(f'_agent_answer: {_agent_answer}')
            
            # Process the response
            #completion = response.get('completion', '')
            return response

        except Exception as e:
            print(f"Error retrieving from knowledge base: {str(e)}")
            raise

        ###############
        return response

    def _make_fully_cited_answer(
        self, orig_agent_answer, event, enable_trace=False, trace_level="none"):
        _citations = event.get("chunk", {}).get("attribution", {}).get("citations", [])
        if _citations:
            if enable_trace:
                print(
                    f"got {len(event['chunk']['attribution']['citations'])} citations \n"
                )
        else:
            return orig_agent_answer

        # remove <sources> tags to work around a bug
        _pattern = r"\n\n<sources>\n\d+\n</sources>\n\n"
        _cleaned_text = re.sub(_pattern, "", orig_agent_answer)
        _pattern = "<sources><REDACTED></sources>"
        _cleaned_text = re.sub(_pattern, "", _cleaned_text)
        _pattern = "<sources></sources>"
        _cleaned_text = re.sub(_pattern, "", _cleaned_text)

        _fully_cited_answer = ""
        _curr_citation_idx = 0

        for _citation in _citations:
            if enable_trace and trace_level == "all":
                print(f"full citation: {_citation}")

            _start = _citation["generatedResponsePart"]["textResponsePart"]["span"][
                "start"
            ] - (
                _curr_citation_idx + 1
            )  # +1
            _end = (
                _citation["generatedResponsePart"]["textResponsePart"]["span"]["end"]
                - (_curr_citation_idx + 2)
                + 4
            )  # +2
            _refs = _citation.get("retrievedReferences", [])
            if len(_refs) > 0:
                _ref_url = (
                    _refs[0].get("location", {}).get("s3Location", {}).get("uri", "")
                )
            else:
                _ref_url = ""
                _fully_cited_answer = _cleaned_text
                break

            _fully_cited_answer += _cleaned_text[_start:_end] + " [" + _ref_url + "] "

            if _curr_citation_idx == 0:
                _answer_prefix = _cleaned_text[:_start]
                _fully_cited_answer = _answer_prefix + _fully_cited_answer

            _curr_citation_idx += 1

            if enable_trace and trace_level == "all":
                print(f"\n\ncitation {_curr_citation_idx}:")
                print(
                    f"got {len(_citation['retrievedReferences'])} retrieved references for this citation\n"
                )
                print(f"citation span... start: {_start}, end: {_end}")
                print(
                    f"citation based on span:====\n{_cleaned_text[_start:_end]}\n===="
                )
                print(f"citation url: {_ref_url}\n============")

        if enable_trace and trace_level == "all":
            print(
                f"\nfullly cited answer:*************\n{_fully_cited_answer}\n*************"
            )

        return _fully_cited_answer

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
        #logger.info(f'chunk: {chunk}')
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
            return bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except bedrock_client.exceptions.ValidationException as e:
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
  

