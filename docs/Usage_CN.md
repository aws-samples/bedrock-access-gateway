[English](./Usage.md)

# Usage Guide

假设您在部署后已设置以下环境变量:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

**API 示例:**
- [Models API](#models-api)
- [Embedding API](#embedding-api)
- [Multimodal API](#multimodal-api)
- [Tool Call](#tool-call)
- [Reasoning](#reasoning)

## Models API

你可以通过这个API 获取支持的models 列表。 另外，如果Amazon Bedrock有新模型加入后，你也可以用它来更新刷新模型列表。

**Request 示例**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Response 示例**

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

**重要**: 在使用此代理 API 之前,请仔细阅读以下几点:

1. 如果您之前使用 OpenAI Embedding模型来创建向量,请注意切换到新模型可能没有那么直接。不同模型具有不同的维度(例如,embed-multilingual-v3.0 有 1024 个维度),即使对于相同的文本,它们也可能产生不同的结果。
2. 如果您使用 OpenAI Embedding模型传入的是整数编码(例如与 LangChain 一起使用),此方案将尝试使用 `tiktoken` 进行解码以检索原始文本。但是,无法保证解码后的文本准确无误。
3. 如果您对长文本使用 OpenAI Embedding,您应该验证 Bedrock 模型支持的最大Token数,例如为获得最佳性能,Bedrock 建议将文本长度限制在少于 512 个Token。

**Request 示例**

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

**Response 示例**

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

或者你可以使用OpenAI 的SDK

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

或者 LangChain

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

**Request 示例**

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
                        "url": "https://github.com/aws-samples/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

如果您需要使用此API处理非公开图像,您可以先对图像进行base64编码,然后传递编码后的字符串。
将"image/jpeg"替换为实际的内容类型(content type)。目前仅支持"image/jpeg"、"image/png"、"image/gif"或"image/webp"。

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

**Response 示例**

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

**重要**:在使用此代理API进行Tool Call之前,请仔细阅读以下几点:

1. OpenAI 已经废弃使用Function Call,而推荐使用Tool Call,因此Function Call在此处不受支持,您应该改为Tool Call。

**Request 示例**

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

**Response 示例**

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


**重要**: 使用此 reasoning 推理模式前，请仔细阅读以下要点。

- 目前仅 Claude 3.7 Sonnet / Deepseek R1 模型支持推理功能。使用前请确保所用模型支持推理。
- Claude 3.7 Sonnet 推理模式（或思考模式）默认未启用，您必须在请求中传递额外的 reasoning_effort 参数，参数值可选:low，medium, high。另外，请在请求中提供正确的 max_tokens（或 max_completion_tokens）参数。budget_tokens 基于 reasoning_effort 设置（低：30%，中：60%，高：100% 的max tokens），确保最小 budget_tokens 为 1,024，Anthropic 建议至少使用 4,000 个令牌以获得全面的推理。详情请参阅 [Bedrock Document](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-37.html)。
- Deepseek R1 会自动使用推理模式，不需要在中传递额外的 reasoning_effort 参数（否则会报错）
- 推理结果（思维链结果、思考过程）被添加到名为 'reasoning_content' 的额外标签中，这不是 OpenAI 官方支持的格式。此设计遵循 [Deepseek Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model#api-example)  的规范。未来可能会有所变动。

**Request 示例**

- Claude 3.7 Sonnet

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

- DeepSeek R1

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.deepseek.r1-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "stream": false
}'
```


**Response 示例**

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

或者使用 OpenAI SDK (请先运行`pip3 install -U openai` 升级到最新版本)

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
    elif chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
```