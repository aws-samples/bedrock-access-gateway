[中文](./README_CN.md)

# Bedrock Access Gateway

OpenAI-compatible RESTful APIs for Amazon Bedrock

## Breaking Changes

This solution can now **automatically detect** new models supported in Amazon Bedrock. 
So whenever new models are added to Amazon Bedrock, you can immediately try them without the need to wait for code changes to this repo. 

This is to use the `ListFoundationModels` api and the `ListInferenceProfiles` api by Amazon Bedrock, due to this change, additional IAM permissions are required to your Lambda/Fargate role.

If you are facing error: 'Unsupported model xxx, please use models API to get a list of supported models' even the model ID is correct, 
please either update your existing stack (**Recommended**) with the new template in the deployment folder or manually add below permissions to the related Lambda/Fargate role.

```json
{
   "Action": [
       "bedrock:ListFoundationModels",
       "bedrock:ListInferenceProfiles"
   ],
   "Resource": "*",
   "Effect": "Allow"
}
```

Please raise an GitHub issue if you still have problems.

## Overview

Amazon Bedrock offers a wide range of foundation models (such as Claude 3 Opus/Sonnet/Haiku, Llama 2/3, Mistral/Mixtral,
etc.) and a broad set of capabilities for you to build generative AI applications. Check the [Amazon Bedrock](https://aws.amazon.com/bedrock) landing page for additional information.

Sometimes, you might have applications developed using OpenAI APIs or SDKs, and you want to experiment with Amazon Bedrock without modifying your codebase. Or you may simply wish to evaluate the capabilities of these foundation models in tools like AutoGen etc. Well, this repository allows you to access Amazon Bedrock models seamlessly through OpenAI APIs and SDKs, enabling you to test these models without code changes.

If you find this GitHub repository useful, please consider giving it a free star ⭐ to show your appreciation and support for the project.

**Features:**

- [x] Support streaming response via server-sent events (SSE)
- [x] Support Model APIs
- [x] Support Chat Completion APIs
- [x] Support Tool Call (**new**)
- [x] Support Embedding API (**new**)
- [x] Support Multimodal API (**new**)
- [x] Support Cross-Region Inference (**new**)

Please check [Usage Guide](./docs/Usage.md) for more details about how to use the new APIs.

> **Note:** The default model is set to `anthropic.claude-3-sonnet-20240229-v1:0` which can be changed via Lambda environment variables (`DEFAULT_MODEL`). You can call the [Models API](./docs/Usage.md#models-api) to get the full list of model IDs supported.

## Get Started

### Prerequisites

Please make sure you have met below prerequisites:

- Access to Amazon Bedrock foundation models.

> For more information on how to request model access, please refer to the [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) (Set Up > Model access)

### Architecture

The following diagram illustrates the reference architecture. Note that it also includes a new **VPC** with two public subnets only for the Application Load Balancer (ALB).

![Architecture](assets/arch.svg)

You can also choose to use [AWS Fargate](https://aws.amazon.com/fargate/) behind the ALB instead of [AWS Lambda](https://aws.amazon.com/lambda/), the main difference is the latency of the first byte for streaming response (Fargate is lower).

Alternatively, you can use Lambda Function URL to replace ALB, see [example](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming)

### Deployment

Please follow the steps below to deploy the Bedrock Proxy APIs into your AWS account. Only supports regions where Amazon Bedrock is available (such as `us-west-2`). The deployment will take approximately **3-5 minutes** 🕒.

**Step 1: Create your own custom API key (Optional)**

#### Store API Key in ParameterStore

> **Note:** This step is to use any string (without spaces) you like to create a custom API Key (credential) that will be used to access the proxy API later. This key does not have to match your actual OpenAI key, and you don't need to have an OpenAI API key. It is recommended that you take this step and ensure that you keep the key safe and private.

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

#### Store API Key in ENV variable

1. Provide an ENV variable to the container named: `API_KEY` with the API key value.

**Step 2: Deploy the CloudFormation stack**

1. Sign in to AWS Management Console, switch to the region to deploy the CloudFormation Stack to.
2. Click the following button to launch the CloudFormation Stack in that region. Choose one of the following:

      [<kbd> <br> ALB + Lambda 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-access-gateway/latest/BedrockProxy.template&stackName=BedrockProxyAPI)

      [<kbd> <br> ALB + Fargate 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-access-gateway/latest/BedrockProxyFargate.template&stackName=BedrockProxyAPI)
3. Click "Next".
4. On the "Specify stack details" page, provide the following information:
    - Stack name: Change the stack name if needed.
    - ApiKeyParam (if you set up an API key in Step 1): Enter the parameter name you used for storing the API key (e.g., `BedrockProxyAPIKey`). If you did not set up an API key, leave this field blank. Click "Next".
5. On the "Configure stack options" page, you can leave the default settings or customize them according to your needs.
6. Click "Next".
7. On the "Review" page, review the details of the stack you're about to create. Check the "I acknowledge that AWS CloudFormation might create IAM resources" checkbox at the bottom.
8. Click "Create stack".

That is it! 🎉 Once deployed, click the CloudFormation stack and go to **Outputs** tab, you can find the API Base URL from `APIBaseUrl`, the value should look like `http://xxxx.xxx.elb.amazonaws.com/api/v1`.

### Troubleshooting

If you encounter any issues, please check the [Troubleshooting Guide](./docs/Troubleshooting.md) for more details.

### SDK/API Usage

All you need is the API Key and the API Base URL. If you didn't set up your own key, then the default API Key (`bedrock`) will be used.

Now, you can try out the proxy APIs. Let's say you want to test Claude 3 Sonnet model (model ID: `anthropic.claude-3-sonnet-20240229-v1:0`)...

**Example API Usage**

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
# For older versions
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

**Example SDK Usage**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

Please check [Usage Guide](./docs/Usage.md) for more details about how to use embedding API, multimodal API and tool call.



## Other Examples

### LangChain

Make sure you use `ChatOpenAI(...)` instead of `OpenAI(...)`

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

### About Privacy

This application does not collect any of your data. Furthermore, it does not log any requests or responses by default.

### Why not used API Gateway instead of Application Load Balancer?

Short answer is that API Gateway does not support server-sent events (SSE) for streaming response.

### Which regions are supported?

Generally speaking, all regions that Amazon Bedrock supports will also be supported, if not, please raise an issue in Github.

Note that not all models are available in those regions.

### Which models are supported?

You can use the [Models API](./docs/Usage.md#models-api) to get/refresh a list of supported models in the current region.

### Can I build and use my own ECR image

Yes, you can clone the repo and build the container image by yourself (`src/Dockerfile`) and then push to your ECR repo. You can use `scripts/push-to-ecr.sh`

Replace the repo url in the CloudFormation template before you deploy.

### Can I run this locally

Yes, you can run this locally, e.g. run below command under `src` folder:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API base url should look like `http://localhost:8000/api/v1`.

### Any performance sacrifice or latency increase by using the proxy APIs

Comparing with the AWS SDK call, the referenced architecture will bring additional latency on response, you can try and test that on you own.

Also, you can use Lambda Web Adapter + Function URL (see [example](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming)) to replace ALB or AWS Fargate to replace Lambda to get better performance on streaming response.

### Any plan to support SageMaker models?

Currently, there is no plan to support SageMaker models. This may change provided there's a demand from customers.

### Any plan to support Bedrock custom models?

Fine-tuned models and models with Provisioned Throughput are currently not supported. You can clone the repo and make the customization if needed.

### How to upgrade?

To use the latest features, you don't need to redeploy the CloudFormation stack. You simply need to pull the latest image.

To do so, depends on which version you deployed:

- **Lambda version**: Go to AWS Lambda console, find the Lambda function, then find and click the `Deploy new image` button and click save.
- **Fargate version**: Go to ECS console, click the ECS cluster, go the `Tasks` tab, select the only task that is running and simply click `Stop selected` menu. A new task with latest image will start automatically.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
