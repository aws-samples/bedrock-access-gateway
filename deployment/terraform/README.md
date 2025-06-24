# Bedrock Access Gateway Terraform Configuration

This directory contains Terraform configuration to deploy the Bedrock Access Gateway - an OpenAI-compatible RESTful API for Amazon Bedrock.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) (version 1.2.0 or later)
- AWS CLI configured with appropriate credentials
- Access to Amazon Bedrock models in your AWS account

## Configuration

1. Copy the example variables file and customize it for your environment:

```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` with your specific values:

```hcl
aws_region         = "us-east-1"  # The AWS region to deploy to
api_key_secret_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-api-key-secret"  # Your API key secret ARN
default_model_id   = "anthropic.claude-3-sonnet-20240229-v1:0"  # Default Bedrock model ID
ecr_account_id     = "366590864501"  # ECR account ID for the container image
```

## Service Quota Requirements

This deployment uses an API Gateway integration timeout of 59000ms (59 seconds), which exceeds the default AWS Service Quota of 29000ms (29 seconds). Before deploying, you should request a Service Quota increase for the "API Gateway integration timeout" in your AWS account.

To request a quota increase:

1. Navigate to the AWS Service Quotas console
2. Search for "Amazon API Gateway"
3. Find "Integration timeout" and request an increase to at least 59000ms
4. Provide a business justification (e.g., "Required for Bedrock model inference which can take longer than 29 seconds")

Deployment may still succeed without the quota increase, but API calls that take longer than 29 seconds will time out.


## Deployment

Initialize the Terraform configuration:

```bash
terraform init
```

Plan the deployment:

```bash
terraform plan
```

Apply the configuration:

```bash
terraform apply
```

## Outputs

After successful deployment, Terraform will output:

- `api_base_url`: The base URL for the Bedrock Proxy API (use this as your OPENAI_API_BASE)

## Cleanup

To remove all resources created by this Terraform configuration:

```bash
terraform destroy
```

## Resources Created

This Terraform configuration creates the following AWS resources:

- VPC with public subnets
- Internet Gateway and route tables
- IAM Role and Policy for Lambda
- Lambda Function for the API handler
- Application Load Balancer
- Security Groups
- Target Groups and Listeners
