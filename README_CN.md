[English](./README.md)

# Bedrock Access Gateway

使用兼容OpenAI的API访问Amazon Bedrock

## 概述

Amazon Bedrock提供了广泛的基础模型(如Claude 3 Sonnet/Haiku、Llama 2、Mistral/Mixtral等),以及构建生成式AI应用程序的多种功能。更多详细信息,请查看[Amazon Bedrock](https://aws.amazon.com/bedrock)。

有时,您可能已经使用OpenAI的API或SDK构建了应用程序,并希望在不修改代码的情况下试用Amazon
Bedrock的模型。或者,您可能只是希望在AutoGen等工具中评估这些基础模型的功能。 好消息是, 这里提供了一种方便的途径,让您可以使用
OpenAI 的 API 或 SDK 无缝集成并试用 Amazon Bedrock 的模型,而无需对现有代码进行修改。

如果您觉得这个项目有用,请考虑给它点个一个免费的小星星 ⭐。

功能列表：

- [x] 支持 server-sent events (SSE)的流式响应
- [x] 支持 Model APIs
- [x] 支持 Chat Completion APIs
- [x] 支持 Function Call/Tool Call (**beta**)
- [x] 支持 Embedding APIs (**beta**)
- [x] 支持 Multimodal APIs (**beta**)

> 注意： 不支持旧的 [text completion](https://platform.openai.com/docs/api-reference/completions) API，请更改为使用Chat Completion API。

支持的Amazon Bedrock模型列表（Model IDs）：

- anthropic.claude-instant-v1
- anthropic.claude-v2:1
- anthropic.claude-v2
- anthropic.claude-3-sonnet-20240229-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- meta.llama2-13b-chat-v1
- meta.llama2-70b-chat-v1
- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1
- mistral.mistral-large-2402-v1:0
- cohere.embed-multilingual-v3 (embedding)
- cohere.embed-english-v3 (embedding)

> 注意: 默认模型为 `anthropic.claude-3-sonnet-20240229-v1:0`， 可以通过更改Lambda环境变量进行更改。

## 使用指南

### 前提条件

请确保您已满足以下先决条件:

- 可以访问Amazon Bedrock基础模型。

如果您还没有获得模型访问权限,请参考[配置](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)指南。

### 架构图

下图展示了本方案的参考架构。请注意,它还包括一个新的**VPC**,其中只有两个公共子网用于应用程序负载均衡器(ALB)。

![Architecture](assets/arch.svg)

> 注意: 你可以使用 Lambda Web Adapter + Function URL (
> 参见[示例](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming))来代替
> ALB,或者使用 AWS Fargate 来代替 Lambda,以获得更好的流响应性能。

### 部署

请按以下步骤将Bedrock代理API部署到您的AWS账户中。仅支持Amazon Bedrock可用的区域(如us-west-2)。 部署预计用时**3-5分钟** 🕒。

**第一步: 自定义您的API Key (可选)**

> 注意:这一步是使用任意字符串（不带空格）创建一个自定义的API Key(凭证),将用于后续访问代理API。此API Key不必与您实际的OpenAI
> Key一致,您甚至无需拥有OpenAI API Key。建议您执行此步操作并且请确保保管好此API Key。

1. 打开AWS管理控制台,导航到Systems Manager服务。
2. 在左侧导航窗格中,单击"参数存储"。
3. 单击"创建参数"按钮。
4. 在"创建参数"窗口中,选择以下选项:
    - 名称:输入参数的描述性名称(例如"BedrockProxyAPIKey")。
    - 描述:可选,为参数提供描述。
    - 层级:选择**标准**。
    - 类型:选择**SecureString**。
    - 值: 随意字符串（不带空格）。
5. 单击"创建参数"。
6. 记录您使用的参数名称(例如"BedrockProxyAPIKey")。您将在下一步中需要它。

**第二步: 部署CloudFormation堆栈**

1. 登录AWS管理控制台,切换到要部署CloudFormation堆栈的区域。
2. 单击以下按钮在该区域启动CloudFormation堆栈。

   [![Launch Stack](assets/launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/create/template?stackName=BedrockProxyAPI&templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-proxy-api/latest/BedrockProxy.template)
3. 单击"下一步"。
4. 在"指定堆栈详细信息"页面,提供以下信息:
    - 堆栈名称: 可以根据需要更改名称。
    - ApiKeyParam(如果在步骤1中设置了API Key):输入您用于存储API密钥的参数名称(例如"BedrockProxyAPIKey")，否则,请将此字段留空。
      单击"下一步"。
5. 在"配置堆栈选项"页面,您可以保留默认设置或根据需要进行自定义。
6. 单击"下一步"。
7. 在"审核"页面,查看您即将创建的堆栈详细信息。勾选底部的"我确认，AWS CloudFormation 可能创建 IAM 资源。"复选框。
8. 单击"创建堆栈"。

仅此而已 🎉 。部署完成后,点击CloudFormation堆栈,进入"输出"选项卡,你可以从"APIBaseUrl"
中找到API Base URL,它应该类似于`http://xxxx.xxx.elb.amazonaws.com/api/v1` 这样的格式。

### SDK/API使用

你只需要API Key和API Base URL。如果你没有设置自己的密钥,那么默认将使用API Key `bedrock`。

现在,你可以尝试使用代理API了。假设你想测试Claude 3 Sonnet模型,那么使用"anthropic.claude-3-sonnet-20240229-v1:0"作为模型ID。

- **API 使用示例**

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
# 旧版本请使用OPENAI_API_BASE
# https://github.com/openai/openai-python/issues/624
export OPENAI_API_BASE=<API base url>
```

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

- **SDK 使用示例**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

## 其他例子

### AutoGen

例如在AutoGen studio配置和使用模型

![AutoGen Model](assets/autogen-model.png)

### LangChain

请确保使用的示`ChatOpenAI(...)` ，而不是`OpenAI(...)`

```python
# pip install langchain-openai
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_api_base=os.environ['OPENAI_BASE_URL'],
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=chat)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = llm_chain.invoke(question)
print(response)

```

## FAQs

### 关于隐私

这个方案不会收集您的任何数据。而且,它默认情况下也不会记录任何请求或响应。

### 为什么没有使用API Gateway 而是使用了Application Load Balancer?

简单的答案是API Gateway不支持 server-sent events (SSE) 用于流式响应。

### 支持哪些区域?

只支持Amazon Bedrock可用的区域,即:

- 美国东部(弗吉尼亚北部)
- 美国西部(俄勒冈州)
- 亚太地区(新加坡)
- 亚太地区(东京)
- 欧洲(法兰克福)

注意，并非所有模型都在上面区可用。

### 我可以构建并使用自己的ECR镜像吗?

是的,你可以克隆repo并自行构建容器镜像(src/Dockerfile),然后推送到你自己的ECR仓库。 脚本可以参考`scripts/push-to-ecr.sh`。

在部署之前,请在CloudFormation模板中替换镜像仓库URL。

### 我可以在本地运行吗?

是的,你可以在本地运行,那么API Base URL应该类似于`http://localhost:8000/api/v1`

### 使用代理API会有任何性能牺牲或延迟增加吗?

与 AWS SDK 调用相比,本方案参考架构会在响应上会有额外的延迟,你可以自己部署并测试。

另外,你也可以使用 Lambda Web Adapter + Function URL (
参见 [示例](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming))来代替 ALB
或使用 AWS Fargate 来代替 Lambda,以获得更好的流响应性能。

### 有计划支持SageMaker模型吗?

目前没有支持SageMaker模型的计划。这取决于是否有客户需求。

### 有计划支持Bedrock自定义模型吗?

不支持微调模型和设置了已预配吞吐量的模型。如有需要,你可以克隆repo并进行自定义。

### 如何升级?

如果架构没有变化,你可以简单地将最新镜像部署到Lambda中,以使用新功能(手动),而无需重新部署整个CloudFormation堆栈。

## 安全

更多信息,请参阅[CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

## 许可证

本项目根据MIT-0许可证获得许可。请参阅LICENSE文件。
