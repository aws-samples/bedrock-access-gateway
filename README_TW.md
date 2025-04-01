# Bedrock Access Gateway

OpenAI 兼容的 Amazon Bedrock RESTful API

[English](./README.md) | [簡體中文](./README_CN.md)

## 最新消息 🔥

此專案支援 **Claude 3.7 Sonnet** 和 **DeepSeek R1** 的推理功能，詳情請參閱 [使用方法](./docs/Usage.md#reasoning)。您需要先執行 Models API 以刷新模型列表。

## 概述

Amazon Bedrock 提供了多種基礎模型（如 Claude 3 Opus/Sonnet/Haiku、Llama 2/3、Mistral/Mixtral 等）以及廣泛的功能，讓您可以建立生成式 AI 應用程式。更多資訊請參閱 [Amazon Bedrock](https://aws.amazon.com/bedrock) 登陸頁面。

有時候，您可能已經使用 OpenAI API 或 SDK 開發了應用程式，並希望在不修改代碼庫的情況下試驗 Amazon Bedrock。或者您可能只是希望在 AutoGen 等工具中評估這些基礎模型的功能。這個倉庫允許您通過 OpenAI API 和 SDK 無縫訪問 Amazon Bedrock 模型，讓您可以在不更改代碼的情況下測試這些模型。

如果您覺得這個 GitHub 倉庫有用，請考慮給它一個免費的星星 ⭐ 以表示對專案的支持和感謝。

**功能：**

- [x] 支援通過伺服器發送事件（SSE）進行流式回應
- [x] 支援模型 API
- [x] 支援聊天完成 API
- [x] 支援工具調用
- [x] 支援嵌入 API
- [x] 支援多模態 API
- [x] 支援跨區域推理
- [x] 支援推理（**新功能**）

更多詳情請參閱 [使用指南](./docs/Usage.md)。

## 快速入門

### 先決條件

請確保您已滿足以下先決條件：

- 訪問 Amazon Bedrock 基礎模型。

> 有關如何請求模型訪問的更多資訊，請參閱 [Amazon Bedrock 使用者指南](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)（設置 > 模型訪問）

### 架構

下圖展示了參考架構。請注意，它還包括一個新的 **VPC**，其中只有兩個公共子網用於應用程式負載均衡器（ALB）。

![架構](assets/arch.png)

您也可以選擇在 ALB 後面使用 [AWS Fargate](https://aws.amazon.com/fargate/) 而不是 [AWS Lambda](https://aws.amazon.com/lambda/)，主要區別在於流式回應的首字節延遲（Fargate 更低）。

或者，您可以使用 Lambda Function URL 替換 ALB，請參閱 [示例](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming)。

### 部署

請按照以下步驟將 Bedrock Proxy API 部署到您的 AWS 帳戶。僅支援 Amazon Bedrock 可用的區域（例如 `us-west-2`）。部署大約需要 **3-5 分鐘** 🕒。

**步驟 1：在 Secrets Manager 中創建自己的 API 密鑰（必須）**

> **注意：** 此步驟是使用您喜歡的任何字符串（無空格）創建自定義 API 密鑰（憑證），稍後將用於訪問代理 API。此密鑰不必與您的實際 OpenAI 密鑰匹配，您也不需要擁有 OpenAI API 密鑰。請保持密鑰的安全和私密。

1. 打開 AWS 管理控制台並導航到 AWS Secrets Manager 服務。
2. 點擊 "存儲新秘密" 按鈕。
3. 在 "選擇秘密類型" 頁面，選擇：

   秘密類型：其他類型的秘密
   鍵/值對：

   - 鍵：api_key
   - 值：輸入您的 API 密鑰值

   點擊 "下一步"

4. 在 "配置秘密" 頁面：
   秘密名稱：輸入名稱（例如 "BedrockProxyAPIKey"）
   描述：（可選）添加您的秘密描述
5. 點擊 "下一步" 並檢查所有設置，然後點擊 "存儲"

創建後，您將在 Secrets Manager 控制台中看到您的秘密。記下秘密 ARN。

**步驟 2：部署 CloudFormation 堆棧**

1. 登錄 AWS 管理控制台，切換到要部署 CloudFormation 堆棧的區域。
2. 點擊以下按鈕在該區域啟動 CloudFormation 堆棧。選擇以下之一：

   [<kbd> <br> ALB + Lambda 一鍵部署 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-access-gateway/latest/BedrockProxy.template&stackName=BedrockProxyAPI)

   [<kbd> <br> ALB + Fargate 一鍵部署 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://aws-gcr-solutions.s3.amazonaws.com/bedrock-access-gateway/latest/BedrockProxyFargate.template&stackName=BedrockProxyAPI)

3. 點擊 "下一步"。
4. 在 "指定堆棧詳細信息" 頁面，提供以下信息：

   - 堆棧名稱：如有需要，修改堆棧名稱。
   - ApiKeySecretArn：輸入您用於存儲 API 密鑰的秘密 ARN。

   點擊 "下一步"。

5. 在 "配置堆棧選項" 頁面，您可以保留默認設置或根據需要自定義設置。點擊 "下一步"。
6. 在 "審查" 頁面，審查您即將創建的堆棧詳細信息。勾選底部的 "我承認 AWS CloudFormation 可能會創建 IAM 資源" 复选框。點擊 "創建堆棧"。

就是這樣！🎉 部署完成後，點擊 CloudFormation 堆棧並轉到 **輸出** 標籤，您可以從 `APIBaseUrl` 中找到 API 基本 URL，值應該類似於 `http://xxxx.xxx.elb.amazonaws.com/api/v1`。

### 故障排除

如果遇到任何問題，請參閱 [故障排除指南](./docs/Troubleshooting.md) 以獲取更多詳細信息。

### SDK/API 使用

您只需要 API 密鑰和 API 基本 URL。如果您沒有設置自己的密鑰，則將使用默認 API 密鑰（`bedrock`）。

現在，您可以嘗試代理 API。假設您想測試 Claude 3 Sonnet 模型（模型 ID：`anthropic.claude-3-sonnet-20240229-v1:0`）...

**API 使用示例**

```bash
export OPENAI_API_KEY=<API 密鑰>
export OPENAI_BASE_URL=<API 基本 URL>
# 對於舊版本
# https://github.com/openai/openai-python/issues/624
export OPENAI_API_BASE=<API 基本 URL>
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

**SDK 使用示例**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

更多詳情請參閱 [使用指南](./docs/Usage.md) 以了解如何使用嵌入 API、多模態 API 和工具調用。

## 其他示例

### LangChain

請確保使用 `ChatOpenAI(...)` 而不是 `OpenAI(...)`

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

template = """問題：{question}

答案：讓我們一步一步來思考。"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=chat)

question = "Justin Beiber 出生那年哪支 NFL 球隊贏得了超級碗？"
response = llm_chain.invoke(question)
print(response)

```

## 常見問題

### 關於隱私

此應用程式不會收集您的任何數據。此外，默認情況下，它不會記錄任何請求或回應。

### 為什麼不使用 API Gateway 而是使用應用程式負載均衡器？

簡短的回答是 API Gateway 不支援流式回應的伺服器發送事件（SSE）。

### 支援哪些區域？

一般來說，Amazon Bedrock 支援的所有區域也將被支援，如果沒有，請在 Github 中提出問題。

請注意，並非所有模型都在這些區域中可用。

### 支援哪些模型？

您可以使用 [Models API](./docs/Usage.md#models-api) 來獲取/刷新當前區域支援的模型列表。

### 我可以構建並使用自己的 ECR 映像嗎？

是的，您可以克隆倉庫並自行構建容器映像（`src/Dockerfile`），然後推送到您的 ECR 倉庫。您可以使用 `scripts/push-to-ecr.sh`

在部署之前替換 CloudFormation 模板中的倉庫 URL。

### 我可以在本地運行嗎？

是的，您可以在本地運行，例如在 `src` 文件夾下運行以下命令：

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

API 基本 URL 應類似於 `http://localhost:8000/api/v1`。

### 使用代理 API 是否會犧牲性能或增加延遲

與 AWS SDK 調用相比，參考架構將在回應上帶來額外的延遲，您可以自行嘗試和測試。

此外，您可以使用 Lambda Web Adapter + Function URL（請參閱 [示例](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming)）替換 ALB 或使用 AWS Fargate 替換 Lambda 以獲得更好的流式回應性能。

### 有計劃支援 SageMaker 模型嗎？

目前沒有計劃支援 SageMaker 模型。如果有客戶需求，這可能會改變。

### 有計劃支援 Bedrock 自定義模型嗎？

目前不支援微調模型和具有預配置吞吐量的模型。如果需要，您可以克隆倉庫並進行自定義。

### 如何升級？

要使用最新功能，您不需要重新部署 CloudFormation 堆棧。您只需拉取最新映像即可。

具體操作取決於您部署的版本：

- **Lambda 版本**：進入 AWS Lambda 控制台，找到 Lambda 函數，然後找到並點擊 `部署新映像` 按鈕並點擊保存。
- **Fargate 版本**：進入 ECS 控制台，點擊 ECS 集群，進入 `任務` 標籤，選擇唯一正在運行的任務並簡單點擊 `停止選定` 菜單。新的任務將自動啟動最新映像。

## 安全

更多信息請參閱 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

## 授權

此庫根據 MIT-0 授權許可。請參閱 LICENSE 文件。
