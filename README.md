[中文](./README_CN.md)

# Bedrock Access Gateway

OpenAI-Compatible RESTful APIs for Amazon Bedrock

## Overview

Amazon Bedrock offers a wide range of foundation models (such as Claude 3 Sonnet/Haiku, Llama 2, Mistral/Mixtral etc.)
and a broad set of capabilities for you to build generative AI applications.
Check [Amazon Bedrock](https://aws.amazon.com/bedrock) for more details.

Sometimes, you might have applications developed using OpenAI APIs or SDKs, and you want to experiment with Amazon
Bedrock without modifying your codebase. Or you may simply wish to evaluate the capabilities of these foundation models
in tools like AutoGen etc. Well, this repository allows you to access Amazon Bedrock models seamlessly through OpenAI
APIs and SDKs, enabling you to test these models without code changes.

If you find this GitHub repository useful, please consider giving it a free star to show your appreciation and support
for the project.

Features:

- [x] Support streaming response via server-sent events (SSE)
- [x] Support Model APIs
- [x] Support Chat Completion APIs
- [ ] Support Function Call/Tool Call
- [ ] Support Embedding APIs
- [ ] Support Image APIs

> NOTE: 1. The legacy [text completion](https://platform.openai.com/docs/api-reference/completions) API is not
> supported, you should move to chat completion API. 2. May support other APIs such as fine-tuning, Assistants API etc.
> in the future.

Supported Amazon Bedrock models (Model IDs):

- anthropic.claude-instant-v1
- anthropic.claude-v2:1
- anthropic.claude-v2
- anthropic.claude-3-sonnet-20240229-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- meta.llama2-13b-chat-v1
- meta.llama2-70b-chat-v1
- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1

> Note: The default model is set to `anthropic.claude-3-sonnet-20240229-v1:0`. You can change it via Lambda environment
> variables.

## Get Started

### Prerequisites

Please make sure you have met below prerequisites:

- Access to Amazon Bedrock foundation models.

If you haven't got model access, please follow
the [Set Up](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) guide

### Architecture

The following diagram illustrates the reference architecture. Note that it also includes a new **VPC** with two public
subnets only for the Application Load Balancer (ALB).

![Architecture](assets/arch.svg)

> Note: You can use Lambda Web Adapter + Function URL (
> See [Example](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming))
> to replace ALB or AWS Fargate to replace Lambda to get better performance on streaming response.

### Deployment

Please follow below steps to deploy the Bedrock Proxy APIs into your AWS account. Only support regions where Amazon
Bedrock is available (such as us-west-2). The deployment will take approximately 3-5 minutes.

**Step 1: Create you own custom API key (Optional)**

> NOTE: This step is to use any string (without spaces) you like to create a custom API Key (credential) that will be
> used to access the proxy API later. This key does not have to match your actual OpenAI key, and you don't even need to
> have an OpenAI API key. It is recommended that you take this step and ensure that you keep the key safe and private.

1. Open the AWS Management Console and navigate to the Systems Manager service.
2. In the left-hand navigation pane, click on "Parameter Store".
3. Click on the "Create parameter" button.
4. In the "Create parameter" window, select the following options:
    - Name: Enter a descriptive name for your parameter (e.g., "BedrockProxyAPIKey").
    - Description: Optionally, provide a description for the parameter.
    - Tier: Select **Standard**.
    - Type: Select **SecureString**.
    - Value: Any string (without spaces).
5. Click "Create parameter".
6. Make a note of the parameter name you used (e.g., "BedrockProxyAPIKey"). You'll need this in the next step.

**Step 2: Deploy the CloudFormation stack**

1. Sign in to AWS Management Console, switch to the region to deploy the CloudFormation Stack to.
2. Click the following button to launch the CloudFormation Stack in that region.

   [![Launch Stack](assets/launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/create/template?stackName=BedrockProxyAPI&templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-proxy-api/latest/BedrockProxy.template)

3. Click "Next".
4. On the "Specify stack details" page, provide the following information:
    - Stack name: Change the stack name if needed.
    - ApiKeyParam (if you set up an API key in Step 1): Enter the parameter name you used for storing the API key (
      e.g., "BedrockProxyAPIKey"). If you did not set up an API key, leave this field blank.
      Click "Next".
5. On the "Configure stack options" page, you can leave the default settings or customize them according to your needs.
6. Click "Next".
7. On the "Review" page, review the details of the stack you're about to create. Check the "I acknowledge that AWS
   CloudFormation might create IAM resources" checkbox at the bottom.
8. Click "Create stack".

That is it! Once deployed, click the CloudFormation stack and go to **Outputs** tab, you can find the API Base URL
from `APIBaseUrl`, the value should look like `http://xxxx.xxx.elb.amazonaws.com/api/v1`.

### SDK/API Usage

All you need is the API Key and the API Base URL. And if you didn't
set up your own key, then the default API Key `bedrock` will be used.

Now, you can try out the proxy APIs. Let's say you want to test Claude 3 Sonnet model, then
use `anthropic.claude-3-sonnet-20240229-v1:0` as the Model ID.

- **Example API Usage**

```bash
curl https://<API base url>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API Key>" \
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

- **Example SDK Usage**

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_API_BASE=<API base url>
```

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

## Other Examples

### AutoGen

Below is an image of setting up the model in AutoGen studio.

![AutoGen Model](assets/autogen-model.png)

### LangChain

Make sure you use `ChatOpenAI(...)` instead of `OpenAI(...)`

```python
# pip install langchain-openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    openai_api_key="xxxx",
    openai_api_base="http://xxx.elb.amazonaws.com/api/v1",
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

### About Privacy

This application does not collect any of your data. Furthermore, it does not log any requests or responses by default.

### Why not used API Gateway instead of Application Load Balancer?

Short answer is that API Gateway does not support server-sent events (SSE) for streaming response.

### Which regions are supported?

This solution only supports the regions where Amazon Bedrock is available, so:

- US East (N. Virginia)
- US West (Oregon)
- Asia Pacific (Singapore)
- Asia Pacific (Tokyo)
- Europe (Frankfurt)

Note that not all models are available in those regions.

### Can I build and use my own ECR image

Yes, you can clone the repo and build the container image by yourself (src/Dockerfile) and then push to your ECR repo.

Replace the repo url in the CloudFormation template before you deploy.

### Can I run this locally

Yes, you can run this locally, then the API base url should be like `http://localhost:8000/api/v1`

### Any performance sacrifice or latency increase by using the proxy APIs

Comparing with the AWS SDK call, the referenced architecture will bring additional latency on response, you can try and
test that own you own.

Also, you can use Lambda Web Adapter + Function URL (
See [Example](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming))
> to replace ALB or AWS Fargate to replace Lambda to get better performance on streaming response.

### Any plan to support SageMaker models?

Currently, there is no plan of supporting SageMaker models. This depends on if there are customer asks.

### Any plan to support Bedrock custom models?

Fine-tuned models and models with Provisioned Throughput are not supported. You can clone the repo and make the
customization if needed.

### How to upgrade?

If there is no changes on architecture, you can simply deploy the latest image to your Lambda to use the new
features (manually) without redeploying the whole CloudFormation stack.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

