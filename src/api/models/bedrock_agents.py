# Original Credit: GitHub user dhapola 
import base64
import json
import logging
import re
import time
from abc import ABC
from typing import AsyncIterable

import boto3
from botocore.config import Config
import numpy as np
import requests
import tiktoken
from fastapi import HTTPException
from api.models.model_manager import ModelManager

from api.models.bedrock import (
    BedrockModel, 
    bedrock_client, 
    bedrock_runtime)

from api.schema import (
    ChatResponse,
    ChatRequest,
    ChatResponseMessage,
    ChatStreamResponse,
    ChoiceDelta
)
                                
from api.setting import (DEBUG, AWS_REGION, AGENT_PREFIX)

from md import MetaData

logger = logging.getLogger(__name__)
config = Config(
            connect_timeout=60,      # Connection timeout: 60 seconds
            read_timeout=900,        # Read timeout: 15 minutes (suitable for long streaming responses)
            retries={
                'max_attempts': 8,   # Maximum retry attempts
                'mode': 'adaptive'   # Adaptive retry mode
            },
            max_pool_connections=50  # Maximum connection pool size
        )

bedrock_agent = boto3.client(
            service_name="bedrock-agent",
            region_name=AWS_REGION,
            config=config,
        )

bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=AWS_REGION,
    config=config,
)


class BedrockAgents(BedrockModel):

    def __init__(self):
        """Append agents to model list."""
        super().__init__()
        self.get_agents()
    
    def get_latest_agent_aliases(self, client, agent_id):#, limit=2):

        # List all aliases for the agent
        response = client.list_agent_aliases(
            agentId=agent_id,
            maxResults=100  # Adjust based on your needs
        )
        
        if not response.get('agentAliasSummaries'):
            return None

        # Sort aliases by createdAt descending
        aliases = response.get('agentAliasSummaries', [])

        sorted_aliases = sorted(
            [a for a in aliases if a.get('agentAliasName')],
            key=lambda a: a['createdAt'],
            reverse=True
        )

        # Init
        result = {}
        seen_statuses = set()

        for alias in sorted_aliases:
            if "PREPARED" in alias.get('agentAliasStatus'):
                name = alias.get('agentAliasName').replace('AgentTestAlias', 'DRAFT')
                result[name]=alias

            #if len(result) >= limit:
            #    break

        return result

    def get_agents(self):
        bedrock_ag = boto3.client(
            service_name="bedrock-agent",
            region_name=AWS_REGION,
            config=config,
        )
        # List Agents
        response = bedrock_agent.list_agents(maxResults=100)

        # Prepare agent for display
        for agent in response['agentSummaries']:
           
            if (agent['agentStatus'] != 'PREPARED'):
                continue

            agentId = agent['agentId']

            all_latest_aliases = self.get_latest_agent_aliases(bedrock_ag, agentId)
            if not all_latest_aliases:
                continue

            for alias_name, latest_alias in all_latest_aliases.items():         
                key_alias_id = 'agentAliasId'

                name = f"{AGENT_PREFIX}{agent['agentName']}-{alias_name}"

                val = {
                    "system": False,      # Supports system prompts for context setting. These are already set in Bedrock Agent configuration
                    "multimodal": True,  # Capable of processing both text and images
                    "tool_call": False,  # Tool Use not required for Agents
                    "stream_tool_call": True,
                    "agent_id": agentId,
                    "alias_id": latest_alias[key_alias_id]
                }
                
                model = {}
                model[name]=val
                self._model_manager.add_model(model)
    

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        try:
            
            if stream:
                response = bedrock_runtime.converse_stream(**args)
            else:
                response = bedrock_runtime.converse(**args)


        except bedrock_client.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""
        #chat: {chat_request}")

        message_id = self.generate_message_id()

        if (chat_request.model.startswith(AGENT_PREFIX)):
            response = self._invoke_agent(chat_request)
            output = ""
            
            for event in response["completion"]:
                output += event["chunk"]["bytes"].decode("utf-8")

            # Minimal response (stop reason, token I/O counts not returned)
            chat_response = self._create_response(
                model=chat_request.model,
                message_id=message_id,
                content=[{"text": output}],
                finish_reason="",
                input_tokens=0,
                output_tokens=0
            )
        else:
            # Just use what we know works
            chat_response = await super().chat(chat_request)

        return chat_response
    
    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:

        """Default implementation for Chat Stream API"""
        
        response = ''
        message_id = self.generate_message_id()

        if (chat_request.model.startswith(AGENT_PREFIX)):
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
            return 
        else:
            response = await self._invoke_bedrock(chat_request, stream=True)

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
    
    def _invoke_agent(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke agent """
        if DEBUG:
            logger.info("BedrockAgents._invoke_agent: Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        

        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        model = self._model_manager.get_all_models()[chat_request.model]
        
        ################

        try:
            query = args['messages'][0]['content'][0]['text']
            messages = args['messages']
            query = messages[len(messages)-1]['content'][0]['text']

            md = MetaData(query)
            md_args = {}
            session_state = {}
            
            if md.has_metadata:
                md_args = md.get_metadata_args()
                query = md.get_clean_query()
                kb_id = "D3Q2K57HXU"

                session_state['knowledgeBaseConfigurations'] = [{
                    'knowledgeBaseId': kb_id, # TODO: Don't hard-wire!
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'filter': md_args
                        }
                    }
                }]

            # Step 1 - Retrieve Context
            # TODO: Session state
            request_params = {
                'agentId': model['agent_id'],
                'agentAliasId': model['alias_id'],
                'sessionId': 'unique-session-id',  # Generate a unique session ID
                'inputText': query,
            }

            # Append KB config if present
            if session_state:
                request_params['sessionState'] = session_state
                
            # Make the retrieve request
            # Invoke the agent
            response = bedrock_agent_runtime.invoke_agent(**request_params)
            return response
            
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    