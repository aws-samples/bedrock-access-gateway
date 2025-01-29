# Troubleshooting Guide

This guide helps you troubleshoot common issues you might encounter when using the Bedrock Access Gateway.

## Common Issues

### 1. Parameter Store Access Error

To see errors, first you need to access the CloudWatch Logs of the Lambda/Fargate instance.

1. Go to the [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/home?#logsV2:log-groups/)
2. Search for `/aws/lambda/BedrockProxyAPI`
3. Click on the `Log Stream` to see the error details

```python
botocore.exceptions.ClientError: An error occurred (ParameterNotFound) when calling the GetParameter operation: Parameter /BedrockProxyAPIKey not found.
```

This error occurs when the Lambda function cannot access the API key parameter in Parameter Store.

**Possible solutions:**
- Verify that you created the parameter in Parameter Store with the correct name
- Check that the parameter name in the CloudFormation stack matches the one in Parameter Store
- Ensure the Lambda function's IAM role has permission to access Parameter Store
- If you didn't set up an API key, leave the `ApiKeyParam` field blank during deployment

### 2. Model Access Issues

If you receive an error about model access:

```
{"error": {"message": "User: arn:aws:iam::XXXX:role/XXX is not authorized to perform: bedrock:InvokeModel on resource: arn:aws:bedrock:REGION::foundation-model/XXX", "type": "auth_error", "code": 401}}
```

**Possible solutions:**
- Ensure you have requested access to the model in Amazon Bedrock
- Verify the Lambda/Fargate role has the necessary permissions to invoke Bedrock models
- Check that you're using the correct model ID
- Verify the model is available in your chosen region

### 3. API Key Authentication Failures

If you receive a 401 Unauthorized error:

```
{"detail": "Could not validate credentials"}
```

**Possible solutions:**
- Verify you're using the correct API key in your requests
- Check that the `Authorization` header is properly formatted (`Bearer YOUR-API-KEY`)
- If using environment variables, ensure `OPENAI_API_KEY` is set correctly

### 4. Cross-Region Access Issues

If you're trying to access models in a different region:

```
{"error": {"message": "Region 'us-east-1' is not enabled for your account", "type": "invalid_request_error", "code": 400}}
```

**Possible solutions:**
- Ensure the target region is enabled for your AWS account
- Verify the model you're trying to access is available in that region
- Check that your IAM roles have the necessary cross-region permissions

### 5. Rate Limiting and Quotas

If you're experiencing throttling or quota issues:

```
{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": 429}}
```

**Possible solutions:**
- Check your Bedrock service quotas in the AWS Console
- Consider implementing retry logic in your application
- Request a quota increase if needed

## Getting Help

If you're still experiencing issues:

1. Check the CloudWatch Logs for detailed error messages
2. Verify your AWS credentials and permissions
3. Review the [Usage Guide](./Usage.md) for correct API usage
4. Open a [GitHub issue](https://github.com/aws-samples/bedrock-access-gateway/issues/new?template=bug_report.md) with:
   - Detailed error message
   - Steps to reproduce
   - Your deployment configuration (region, model, etc.)
   - Any relevant CloudWatch logs

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS IAM Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/)
- [AWS Systems Manager Parameter Store](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html)
