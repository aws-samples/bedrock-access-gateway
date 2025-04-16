# Custom Imported Models Implementation

## Overview

This document describes the implementation of custom imported model support in the Bedrock Access Gateway. Custom models are models that you have imported into Bedrock, and this feature allows you to use them through the OpenAI-compatible API interface just like foundation models.

## User-Friendly Model IDs

One of the key features of this implementation is the creation of user-friendly model IDs that include the model name. Instead of cryptic AWS IDs like `custom.a1b2c3d4`, models are presented with descriptive IDs in the format:

```
{model-name}-id:custom.{aws_id}
```

For example: `mistral-7b-instruct-id:custom.a1b2c3d4`

This makes it easier to identify models when using the Models API, while maintaining compatibility with the original AWS ID format.

## Changes Made

1. **Model Discovery**
   - Extended `list_bedrock_models()` to include:
     - Custom models via `bedrock_client.list_custom_models()`
     - Imported models via `bedrock_client.list_imported_models()`
   - Created user-friendly model IDs that include the model name
   - Added type field to model metadata to distinguish between "foundation" and "custom" models
   - Added region information to each model to support cross-region invocation
   - Stored model ARN for custom models for invocation purposes

2. **Configuration**
   - Custom models are always enabled by default
   - Added retry configuration for custom model invocation
   - The implementation uses the local AWS region by default

3. **Model Validation**
   - Added ID transformation logic in the `validate()` method to handle both:
     - Descriptive model IDs (e.g., `mistral-7b-instruct-id:custom.a1b2c3d4`)
     - Original AWS IDs (e.g., `custom.a1b2c3d4`)
   - Stores the original display ID to preserve it in responses

4. **Model Invocation**
   - Added branching logic in `_invoke_bedrock()` to handle custom models differently
   - Implemented `_invoke_custom_model()` method to handle custom model invocation via `InvokeModel`/`InvokeModelWithResponseStream`
   - Added custom model response parsing to handle various model output formats
   - Implemented special handling for `ModelNotReadyException`
   - Added region-specific client creation for cross-region models

5. **Streaming Support**
   - Added `_handle_custom_model_stream()` method to handle streaming responses
   - Added support for parsing different streaming formats from custom models

6. **Message Formatting**
   - Implemented `_create_prompt_from_messages()` to convert OpenAI-style chat messages to text format for custom models

7. **Documentation**
   - Updated README.md to include new feature
   - Updated Usage.md with custom model usage examples
   - Updated FAQs to indicate custom model support

## Usage

### Listing Custom Models

To list available custom models in your AWS account, use the Models API:

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[] | select(.id | startswith("custom.") or contains("-id:custom."))'
```

### Using Custom Models

Custom models can be used with either their descriptive ID or the original AWS ID:

```bash
# Using descriptive ID
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "mistral-7b-instruct-id:custom.a1b2c3d4",
    "messages": [
      { "role": "user", "content": "Hello, world!" }
    ]
  }'

# Using original AWS ID (also supported)
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "custom.a1b2c3d4",
    "messages": [
      { "role": "user", "content": "Hello, world!" }
    ]
  }'
```

## Troubleshooting

### Model Not Found

If your custom model isn't appearing:

1. Check if the model exists in your AWS account:
   ```bash
   aws bedrock list-imported-models --region us-east-1
   ```

2. Restart the gateway to refresh the model list:
   ```bash
   # For Lambda deployments
   # Go to Lambda console > Find your function > Click "Deploy new image"
   
   # For Fargate deployments
   # Go to ECS console > Find your cluster > Tasks tab > Stop running task
   ```

### Invocation Errors

Common errors when invoking custom models:

- `ModelNotReadyException`: The model is still being prepared - wait a few minutes and try again
- `ValidationException`: Check that your input format is compatible with the model