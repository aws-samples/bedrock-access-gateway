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
