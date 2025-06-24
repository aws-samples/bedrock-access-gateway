# Multi-Tenant API Key Setup Guide

This guide explains how to configure and use the multi-tenant API key feature for individual cost tracking and usage monitoring through Application Inference Profiles.

## Overview

The multi-tenant feature allows you to:
- Assign individual API keys to different users/teams
- Track costs and usage per user through Application Inference Profiles
- Control model access permissions per user
- Maintain compatibility with existing single-key setups

## Prerequisites

1. **Application Inference Profiles**: Create Application Inference Profiles in AWS Bedrock for each user/team you want to track separately.
2. **AWS Secrets Manager**: Update your existing secret with the new multi-tenant configuration structure.

## Step 1: Create Application Inference Profiles

For each user or team, create an Application Inference Profile in AWS Bedrock:

```bash
# Example AWS CLI command to create an Application Inference Profile
aws bedrock create-inference-profile \
  --inference-profile-name "engineering-team" \
  --description "Cost tracking for engineering team" \
  --model-source modelArn="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0" \
  --tags "Department=Engineering,CostCenter=eng-001"
```

## Step 2: Update Secrets Manager Configuration

Update your existing AWS Secrets Manager secret with the new structure:

### Current Single-Key Format
```json
{
  "api_key": "your-current-api-key"
}
```

### New Multi-Tenant Format (with backward compatibility)
```json
{
  "api_key": "your-current-api-key",
  "multi_tenant_config": {
    "enabled": true,
    "users": {
      "engineering-team-api-key-uuid": {
        "inference_profiles": [
          "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/engineering-team"
        ],
        "allowed_models": [
          "anthropic.claude-3-sonnet-*",
          "anthropic.claude-3-haiku-*"
        ],
        "metadata": {
          "user_id": "engineering_team",
          "department": "engineering",
          "email": "engineering@company.com"
        }
      }
    }
  }
}
```

## Step 3: Generate API Keys for Users

Generate secure, unique API keys for each user/team:

```bash
# Example: Generate a UUID-based API key
python3 -c "import uuid; print(f'team-eng-{uuid.uuid4()}')"
```

## Step 4: Configure Model Access Patterns

The `allowed_models` field supports pattern matching:

- `"*"`: Allow all models
- `"anthropic.claude-3-sonnet-*"`: Allow all Claude 3 Sonnet variants
- `"anthropic.claude-3-haiku-20240307-v1:0"`: Allow only specific model version

### Example Access Patterns

```json
{
  "engineering_team": [
    "anthropic.claude-3-sonnet-*",
    "anthropic.claude-3-haiku-*"
  ],
  "marketing_team": [
    "anthropic.claude-3-haiku-*"
  ],
  "data_science_team": [
    "*"
  ],
  "external_contractors": [
    "anthropic.claude-3-haiku-20240307-v1:0"
  ]
}
```

## Step 5: Test the Configuration

### Using curl
```bash
# Test with engineering team API key
curl $OPENAI_BASE_URL/models \
  -H "Authorization: Bearer engineering-team-api-key-uuid"

# Test chat completion
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer engineering-team-api-key-uuid" \
  -d '{
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using OpenAI SDK
```python
from openai import OpenAI

# Engineering team client
eng_client = OpenAI(
    api_key="engineering-team-api-key-uuid",
    base_url="http://your-bedrock-gateway.com/api/v1"
)

# Marketing team client  
mkt_client = OpenAI(
    api_key="marketing-team-api-key-uuid",
    base_url="http://your-bedrock-gateway.com/api/v1"
)

# Engineering team can use Sonnet
eng_response = eng_client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello from engineering!"}]
)

# Marketing team limited to Haiku
mkt_response = mkt_client.chat.completions.create(
    model="anthropic.claude-3-haiku-20240307-v1:0", 
    messages=[{"role": "user", "content": "Hello from marketing!"}]
)
```

## Step 6: Monitor Costs and Usage

### CloudWatch Metrics
With Application Inference Profiles, you'll automatically get CloudWatch metrics for:
- Request counts per profile
- Token usage per profile
- Error rates per profile
- Latency metrics per profile

### Cost Allocation
- Each Application Inference Profile appears as a separate line item in AWS Cost Explorer
- Use tags on profiles to enable detailed cost allocation reports
- Set up budget alerts per profile for cost control

### Example CloudWatch Dashboard Query
```json
{
  "metrics": [
    ["AWS/Bedrock", "InvocationClientErrors", "InferenceProfileName", "engineering-team"],
    ["AWS/Bedrock", "InvocationServerErrors", "InferenceProfileName", "engineering-team"],
    ["AWS/Bedrock", "Invocations", "InferenceProfileName", "engineering-team"]
  ],
  "period": 300,
  "stat": "Sum",
  "region": "us-west-2",
  "title": "Engineering Team Bedrock Usage"
}
```

## Migration Strategy

### Option 1: Gradual Migration
1. Deploy the updated code with `multi_tenant_config.enabled = false`
2. Test with existing single API key
3. Enable multi-tenant mode: `multi_tenant_config.enabled = true`
4. Gradually distribute individual API keys to users

### Option 2: Immediate Migration
1. Create all Application Inference Profiles
2. Update Secrets Manager with full multi-tenant configuration
3. Deploy updated code with `multi_tenant_config.enabled = true`
4. Distribute API keys to all users immediately

## Troubleshooting

### Common Issues

1. **403 Access Denied to Model**
   - Check that the model ID matches the allowed_models patterns
   - Verify the user's API key is correctly configured

2. **Invalid API Key**
   - Ensure the API key exists in the multi_tenant_config.users object
   - Check that multi_tenant_config.enabled is true

3. **Application Inference Profile Not Found**
   - Verify the Application Inference Profile ARN is correct
   - Ensure the profile exists and is active in your AWS account

### Debug Mode
Enable debug logging by setting `DEBUG=true` environment variable to see detailed authentication flow.

### Fallback Behavior
- If multi-tenant is disabled: Falls back to single API key validation
- If multi-tenant config is invalid: Falls back to single API key validation
- If Application Inference Profile is invalid: Uses original model ID

## Security Considerations

1. **API Key Management**
   - Use strong, unique API keys for each user/team
   - Rotate API keys regularly
   - Store keys securely and share them through secure channels

2. **Principle of Least Privilege**
   - Grant minimum necessary model access to each user
   - Use specific model versions rather than wildcards when possible
   - Regularly review and update access permissions

3. **Monitoring**
   - Set up CloudWatch alerts for unusual usage patterns
   - Monitor failed authentication attempts
   - Track model access denials

## Cost Optimization Tips

1. **Model Selection**
   - Use Haiku for simple tasks (lower cost)
   - Reserve Sonnet/Opus for complex tasks requiring higher quality
   - Guide users on appropriate model selection

2. **Usage Budgets**
   - Set up AWS Budgets for each Application Inference Profile
   - Configure alerts at 80% and 95% of budget thresholds
   - Implement cost controls through allowed_models restrictions

3. **Regular Reviews**
   - Monthly review of usage patterns by team
   - Identify optimization opportunities
   - Adjust model access based on actual usage needs