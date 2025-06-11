variable "aws_region" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "api_key_secret_arn" {
  description = "The secret ARN in Secrets Manager used to store the API Key"
  type        = string
  validation {
    condition     = can(regex("^arn:aws:secretsmanager:.*$", var.api_key_secret_arn))
    error_message = "The api_key_secret_arn must be a valid Secrets Manager ARN."
  }
}

variable "default_model_id" {
  description = "The default model ID, please make sure the model ID is supported in the current region"
  type        = string
  default     = "anthropic.claude-3-sonnet-20240229-v1:0"
}

variable "ecr_account_id" {
  description = "The ECR account ID where the container image is stored"
  type        = string
  default     = "366590864501"
}

variable "api_stage_name" {
  description = "The name of the API Gateway stage"
  type        = string
  default     = "prod"
}
