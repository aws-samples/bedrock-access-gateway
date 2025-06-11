# Bedrock Access Gateway Terraform Resources

This document provides an overview of all resources created by the Terraform configuration for the Bedrock Access Gateway.

## IAM Resources

| Resource Type | Resource Name | Description |
|--------------|---------------|-------------|
| `aws_iam_role` | `proxy_api_handler_role` | IAM role for the Lambda function with basic execution permissions |
| `aws_iam_policy` | `proxy_api_handler_policy` | IAM policy for Bedrock and Secrets Manager access |
| `aws_iam_role_policy_attachment` | `proxy_api_handler_policy_attachment` | Attaches the policy to the role |

## Lambda Resources

| Resource Type | Resource Name | Description |
|--------------|---------------|-------------|
| `aws_lambda_function` | `proxy_api_handler` | Main Lambda function that handles API requests |
| `aws_lambda_permission` | `api_gateway_lambda_permission` | Permission for API Gateway to invoke the Lambda function |

## API Gateway Resources

| Resource Type | Resource Name | Description |
|--------------|---------------|-------------|
| `aws_api_gateway_rest_api` | `proxy_api` | The main API Gateway REST API |
| `aws_api_gateway_resource` | `api` | API resource for the `/api` path |
| `aws_api_gateway_resource` | `v1` | API resource for the `/api/v1` path |
| `aws_api_gateway_resource` | `proxy` | Proxy resource for all paths under `/api/v1` |
| `aws_api_gateway_method` | `proxy_any` | ANY method for the proxy resource |
| `aws_api_gateway_integration` | `lambda_integration` | Lambda integration for the proxy resource |
| `aws_api_gateway_method` | `v1_any` | ANY method for the `/api/v1` resource |
| `aws_api_gateway_integration` | `v1_lambda_integration` | Lambda integration for the `/api/v1` resource |
| `aws_api_gateway_deployment` | `api_deployment` | Deployment of the API Gateway |
| `aws_api_gateway_stage` | `api_stage` | Stage for the API Gateway deployment |

## Data Sources

| Data Source | Description |
|-------------|-------------|
| `aws_partition` | Used to get the current AWS partition for ARN construction |

## Variables

| Variable Name | Description | Default Value |
|--------------|-------------|---------------|
| `aws_region` | AWS region to deploy resources | `us-east-1` |
| `api_key_secret_arn` | Secret ARN in Secrets Manager for API Key | (Required) |
| `default_model_id` | Default Bedrock model ID | `anthropic.claude-3-sonnet-20240229-v1:0` |
| `ecr_account_id` | ECR account ID for container image | `366590864501` |
| `api_stage_name` | API Gateway stage name | `prod` |

## Outputs

| Output Name | Description |
|-------------|-------------|
| `api_base_url` | The base URL for the Bedrock Proxy API (OPENAI_API_BASE) |
