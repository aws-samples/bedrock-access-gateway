[中文](./Usage_CN.md)

# Usage Guide

Assuming you have set up below environment variables after deployed:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

## Models API

You can use this API to get a list of supported model IDs.

Also, you can use this API to refresh the model list if new models are added to Amazon Bedrock.


**Example Request**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Example Response**

```bash
[
  ...
  {
    "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  {
    "id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  ...
]
```

## Embedding API

**Important Notice**: Please carefully review the following points before using this proxy API for embedding.

1. If you have previously used OpenAI embedding models to create vectors, be aware that switching to a new model may not be straightforward. Different models have varying dimensions (e.g., embed-multilingual-v3.0 has 1024 dimensions), and even for the same text, they may produce different results.
2. If you are using OpenAI embedding models for encoded integers (such as with LangChain), this solution will attempt to decode the integers using `tiktoken` to retrieve the original text. However, there is no guarantee that the decoded text will be accurate.
3. If you are using OpenAI embedding models for long texts, you should verify the maximum number of tokens supported for Bedrock models, e.g. for optimal performance, Bedrock recommends limiting the text length to less than 512 tokens.


**Example Request**

```bash
curl $OPENAI_BASE_URL/embeddings \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "input": "The food was delicious and the waiter...",
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }'
```

**Example Response**

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.02279663,
                -0.024612427,
                0.012863159,
                ...
                0.01612854,
                0.0038928986
            ],
            "index": 0
        }
    ],
    "model": "cohere.embed-multilingual-v3",
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
}
```

Alternatively, you can use the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

text = "hello"
# will output like [0.003578186, 0.028717041, 0.031021118, -0.0014066696,...]
print(get_embedding(text))
```

Or LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:5])
doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])
```

## Multimodal API

**Important Notice**: Please carefully review the following points before using this proxy API for Multimodal.

1. This API is only supported by Claude 3 model.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in these images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/aws-samples/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

If you need to use this API with non-public images, you can do base64 the image first and pass the encoded string. 
Replace `image/jpeg` with the actual content type. Currently, only 'image/jpeg', 'image/png', 'image/gif' or 'image/webp' is supported.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in this images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,<your image data>"
                    }
                }
            ]
        }
    ]
}'
```

**Example Response**

```json
{
    "id": "msg_01BY3wcz41x7XrKhxY3VzWke",
    "created": 1712543069,
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The image contains the following objects:\n\n1. A peach-colored short-sleeve button-up shirt\n2. An olive green plaid long coat/jacket\n3. A pair of white sneakers or canvas shoes\n4. A brown shoulder bag or purse\n5. A makeup brush or cosmetic applicator\n6. A tube or container (possibly lipstick or lip balm)\n7. A pair of sunglasses\n8. A thought bubble icon\n9. A footprint icon\n10. A leaf or plant icon\n11. A flower icon\n12. A cloud icon\n\nIn total, there are 12 distinct objects depicted in the illustrated scene."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 197,
        "completion_tokens": 147,
        "total_tokens": 344
    }
}
```


## Tool Call

**Important Notice**: Please carefully review the following points before using this Tool Call for Chat completion API.

1. Function Call is now deprecated in favor of Tool Call by OpenAI, hence it's not supported here, you should use Tool Call instead.
2. This API is only supported by Claude 3 model.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Shanghai today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or state which is required."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "Use this tool to get the current location if user does not provide a location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ],
    "tool_choice": "auto"
}'
```

**Example Response**

```json
{
    "id": "msg_01PjrKDWhYGsrTNdeqzWd6D9",
    "created": 1712543689,
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "0",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"Shanghai\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "total_tokens": 320
    }
}
```

You can try it with different questions, such as:
1. Hello, who are you?  (No tools are needed)
2. What is the weather like today?  (Should use get_current_location tool first)


## Reasoning

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.
- The only model supports Reasoning is Claude 3.7 Sonnet (extended thinking) so far. Please make sure the model supports reasoning.
- The reasoning mode (or thinking mode) is not enabled by default, you must pass additional `reasoning_effort` parameter in your request.
- The reasoning response (CoT, thoughts) is added in an additional tag 'reasoning_content' which is not officially supported by OpenAI. This is to follow [Deepseek Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model#api-example). This may be changed in the future.
- Please provide the right max_tokens (or max_completion_tokens) in your request. The budget_tokens is based on reasoning_effort (low: 30%, medium: 60%, high: 100% of max tokens), ensuring minimum budget_tokens of 1,024 with Anthropic recommending at least 4,000 tokens for comprehensive reasoning. Check [Bedrock Document](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-37.html) for more details.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "stream": false
}'
```

**Example Response**

```json
{
    "id": "chatcmpl-83fb7a88",
    "created": 1740545278,
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "3.9 is bigger than 3.11.\n\nWhen comparing decimal numbers, we need to understand what these numbers actually represent:...",
                "reasoning_content": "I need to compare the decimal numbers 3.9 and 3.11.\n\nFor decimal numbers, we first compare the whole number parts, and if they're equal, we compare the decimal parts. \n\nBoth numbers ..."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 51,
        "completion_tokens": 565,
        "total_tokens": 616
    }
}
```

You can also use OpenAI SDK (run `pip3 install -U openai` first )

- Non-Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "which one is bigger, 3.9 or 3.11?"}]
response = client.chat.completions.create(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

- Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
    stream=True,
)

reasoning_content = ""
content = ""

for chunk in response:
    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
```