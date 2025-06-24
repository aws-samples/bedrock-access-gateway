# Multi-Tenant API Key Design

## Overview
Extend the existing single API key authentication to support multiple users with individual API keys and Application Inference Profiles for cost tracking and usage monitoring.

## Requirements

### Functional Requirements
- Support multiple API keys with individual Application Inference Profile assignments
- Maintain backward compatibility with existing single key configuration
- Enable per-user cost tracking through Application Inference Profiles
- Restrict model access per user/API key
- Provide usage metrics and monitoring per user

### Non-Functional Requirements
- Minimal performance impact (< 5ms additional latency)
- Secure key management via AWS Secrets Manager
- Easy migration from single-key to multi-key setup
- CloudWatch integration for monitoring

## Architecture

### Secret Manager Structure

#### Current (Single Key)
```json
{
  "api_key": "bedrock-proxy-key"
}
```

#### New (Multi-Tenant with Backward Compatibility)
```json
{
  "api_key": "bedrock-proxy-key",  // Fallback for backward compatibility
  "multi_tenant_config": {
    "enabled": true,
    "users": {
      "user1-api-key-uuid": {
        "inference_profiles": [
          "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/user1-engineering"
        ],
        "allowed_models": [
          "anthropic.claude-3-sonnet-*",
          "anthropic.claude-3-haiku-*"
        ],
        "metadata": {
          "user_id": "user1",
          "department": "engineering",
          "email": "user1@company.com"
        }
      },
      "user2-api-key-uuid": {
        "inference_profiles": [
          "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/user2-marketing"
        ],
        "allowed_models": [
          "anthropic.claude-3-haiku-*"
        ],
        "metadata": {
          "user_id": "user2", 
          "department": "marketing",
          "email": "user2@company.com"
        }
      }
    }
  }
}
```

### Authentication Flow

1. **Request Reception**: Extract Bearer token from Authorization header
2. **Configuration Load**: Load configuration from Secrets Manager (with caching)
3. **Authentication**: 
   - If `multi_tenant_config.enabled = true`: Look up API key in users map
   - If not found or disabled: Fall back to single `api_key` validation
4. **Context Setting**: Set user context for downstream processing
5. **Authorization**: Validate model access permissions

### Model Access Control

#### Models API (`/models`)
- Return only models allowed for the authenticated user
- Include Application Inference Profiles associated with the user
- Maintain existing response format for compatibility

#### Chat/Embeddings APIs
- Validate requested model against user's `allowed_models` patterns
- Automatically use user's assigned Application Inference Profile
- Pass user metadata to Bedrock via `requestMetadata`

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Update `auth.py` for multi-tenant authentication
- [ ] Add user context management
- [ ] Implement configuration caching

### Phase 2: Model Access Control  
- [ ] Update `bedrock.py` for user-specific model filtering
- [ ] Implement Application Inference Profile selection
- [ ] Add model permission validation

### Phase 3: Integration & Testing
- [ ] Update CloudFormation templates
- [ ] Add comprehensive tests
- [ ] Validate backward compatibility

### Phase 4: Documentation & Deployment
- [ ] Update README with multi-tenant setup
- [ ] Create migration guide
- [ ] Deploy to production

## Cost Impact Analysis

### Secret Manager Costs
- Storage: $0.40/month/secret (no change)
- API calls: ~720 calls/month = $0.003/month (negligible)

### Application Inference Profiles
- No additional cost for profile creation
- Enable granular cost tracking and allocation
- Potential cost savings through usage optimization

### Performance Impact
- Configuration caching minimizes Secret Manager calls
- Expected additional latency: < 5ms per request
- Memory overhead: ~1KB per user configuration

## Migration Strategy

### Existing Deployments
1. Update CloudFormation stack with new parameters
2. Migrate existing secret to new format (with `multi_tenant_config.enabled = false`)
3. Gradually enable multi-tenant mode per environment

### New Deployments
- Deploy with multi-tenant configuration enabled by default
- Provide sample configuration in documentation
- Include CloudFormation parameters for easy setup

## Security Considerations

- API keys stored securely in AWS Secrets Manager
- User metadata not logged in application logs
- Application Inference Profile ARNs validated before use
- Fail-safe to single-key mode if multi-tenant config is invalid

## Monitoring & Observability

### CloudWatch Metrics
- Per-user request counts
- Per-user token usage
- Authentication failures
- Model access denials

### Application Inference Profile Benefits
- Automatic cost allocation per user/department
- Usage trends and patterns
- Budget alerts and controls