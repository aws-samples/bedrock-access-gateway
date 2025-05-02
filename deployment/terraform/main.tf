terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region = var.aws_region
}

# Lambda Function and IAM Role
resource "aws_iam_role" "proxy_api_handler_role" {
  name = "ProxyApiHandlerRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  ]
}

resource "aws_iam_policy" "proxy_api_handler_policy" {
  name = "ProxyApiHandlerPolicy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "bedrock:ListFoundationModels",
          "bedrock:ListInferenceProfiles"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Effect = "Allow"
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*",
          "arn:aws:bedrock:*:*:inference-profile/*"
        ]
      },
      {
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Effect   = "Allow"
        Resource = var.api_key_secret_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "proxy_api_handler_policy_attachment" {
  role       = aws_iam_role.proxy_api_handler_role.name
  policy_arn = aws_iam_policy.proxy_api_handler_policy.arn
}

resource "aws_lambda_function" "proxy_api_handler" {
  function_name = "ProxyApiHandler"
  description   = "Bedrock Proxy API Handler"

  image_uri = "${var.ecr_account_id}.dkr.ecr.${var.aws_region}.${data.aws_partition.current.dns_suffix}/bedrock-proxy-api:latest"

  package_type  = "Image"
  architectures = ["arm64"]
  memory_size   = 1024
  timeout       = 600

  role = aws_iam_role.proxy_api_handler_role.arn

  environment {
    variables = {
      DEBUG                         = "false"
      API_KEY_SECRET_ARN            = var.api_key_secret_arn
      DEFAULT_MODEL                 = var.default_model_id
      DEFAULT_EMBEDDING_MODEL       = "cohere.embed-multilingual-v3"
      ENABLE_CROSS_REGION_INFERENCE = "true"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.proxy_api_handler_policy_attachment
  ]
}

# API Gateway
resource "aws_api_gateway_rest_api" "proxy_api" {
  name        = "BedrockProxyAPI"
  description = "API Gateway for Bedrock Proxy"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# API Gateway Resources
resource "aws_api_gateway_resource" "api" {
  rest_api_id = aws_api_gateway_rest_api.proxy_api.id
  parent_id   = aws_api_gateway_rest_api.proxy_api.root_resource_id
  path_part   = "api"
}

resource "aws_api_gateway_resource" "v1" {
  rest_api_id = aws_api_gateway_rest_api.proxy_api.id
  parent_id   = aws_api_gateway_resource.api.id
  path_part   = "v1"
}

# Proxy resource to catch all paths under /api/v1
resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.proxy_api.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "{proxy+}"
}

# ANY method for the proxy resource
resource "aws_api_gateway_method" "proxy_any" {
  rest_api_id   = aws_api_gateway_rest_api.proxy_api.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "NONE"

  request_parameters = {
    "method.request.path.proxy" = true
  }
}

# Lambda integration for the ANY method
resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.proxy_api.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.proxy_any.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.proxy_api_handler.invoke_arn
}

# ANY method for the /api/v1 resource
resource "aws_api_gateway_method" "v1_any" {
  rest_api_id   = aws_api_gateway_rest_api.proxy_api.id
  resource_id   = aws_api_gateway_resource.v1.id
  http_method   = "ANY"
  authorization = "NONE"
}

# Lambda integration for the /api/v1 ANY method
resource "aws_api_gateway_integration" "v1_lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.proxy_api.id
  resource_id = aws_api_gateway_resource.v1.id
  http_method = aws_api_gateway_method.v1_any.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.proxy_api_handler.invoke_arn
}

# Deployment
resource "aws_api_gateway_deployment" "api_deployment" {
  depends_on = [
    aws_api_gateway_integration.lambda_integration,
    aws_api_gateway_integration.v1_lambda_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.proxy_api.id

  lifecycle {
    create_before_destroy = true
  }
}

# Stage
resource "aws_api_gateway_stage" "api_stage" {
  deployment_id = aws_api_gateway_deployment.api_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.proxy_api.id
  stage_name    = var.api_stage_name
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway_lambda_permission" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.proxy_api_handler.function_name
  principal     = "apigateway.amazonaws.com"

  # Allow invocation from any method on any resource within the API
  source_arn = "${aws_api_gateway_rest_api.proxy_api.execution_arn}/*/*"
}

# Data sources
data "aws_partition" "current" {}
