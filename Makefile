# Bedrock Access Gateway Makefile
# This Makefile provides commands for building, pushing to ECR, and deploying with Terraform

# Default AWS region
AWS_REGION ?= us-east-1

# Default Terraform directory
TF_DIR = deployment/terraform

# Default API Key Secret ARN (must be set when running terraform commands)
API_KEY_SECRET_ARN ?= 

# Default values
DEFAULT_MODEL_ID ?= anthropic.claude-3-sonnet-20240229-v1:0
ECR_ACCOUNT_ID ?= 366590864501

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build-and-push
build-and-push: ## Build and push Docker images to ECR
	@echo "Building and pushing Docker images to ECR..."
	@cd scripts && bash ./push-to-ecr.sh

.PHONY: tf-init
tf-init: ## Initialize Terraform
	@echo "Initializing Terraform..."
	@cd $(TF_DIR) && terraform init

.PHONY: tf-plan
tf-plan: ## Plan Terraform changes
	@if [ -z "$(API_KEY_SECRET_ARN)" ]; then \
		echo "Error: API_KEY_SECRET_ARN is required. Usage: make tf-plan API_KEY_SECRET_ARN=arn:aws:secretsmanager:..."; \
		exit 1; \
	fi
	@echo "Planning Terraform changes..."
	@cd $(TF_DIR) && terraform plan \
		-var="aws_region=$(AWS_REGION)" \
		-var="api_key_secret_arn=$(API_KEY_SECRET_ARN)" \
		-var="default_model_id=$(DEFAULT_MODEL_ID)" \
		-var="ecr_account_id=$(ECR_ACCOUNT_ID)"

.PHONY: tf-apply
tf-apply: ## Apply Terraform changes
	@if [ -z "$(API_KEY_SECRET_ARN)" ]; then \
		echo "Error: API_KEY_SECRET_ARN is required. Usage: make tf-apply API_KEY_SECRET_ARN=arn:aws:secretsmanager:..."; \
		exit 1; \
	fi
	@echo "Applying Terraform changes..."
	@cd $(TF_DIR) && terraform apply \
		-var="aws_region=$(AWS_REGION)" \
		-var="api_key_secret_arn=$(API_KEY_SECRET_ARN)" \
		-var="default_model_id=$(DEFAULT_MODEL_ID)" \
		-var="ecr_account_id=$(ECR_ACCOUNT_ID)"

.PHONY: tf-destroy
tf-destroy: ## Destroy Terraform resources
	@if [ -z "$(API_KEY_SECRET_ARN)" ]; then \
		echo "Error: API_KEY_SECRET_ARN is required. Usage: make tf-destroy API_KEY_SECRET_ARN=arn:aws:secretsmanager:..."; \
		exit 1; \
	fi
	@echo "Destroying Terraform resources..."
	@cd $(TF_DIR) && terraform destroy \
		-var="aws_region=$(AWS_REGION)" \
		-var="api_key_secret_arn=$(API_KEY_SECRET_ARN)" \
		-var="default_model_id=$(DEFAULT_MODEL_ID)" \
		-var="ecr_account_id=$(ECR_ACCOUNT_ID)"

.PHONY: tf-output
tf-output: ## Show Terraform outputs
	@echo "Showing Terraform outputs..."
	@cd $(TF_DIR) && terraform output

.PHONY: deploy
deploy: build-and-push tf-apply ## Build, push to ECR, and deploy with Terraform
	@echo "Deployment complete!"
	@echo "API Base URL:"
	@cd $(TF_DIR) && terraform output api_base_url

.PHONY: clean
clean: ## Clean local Docker images
	@echo "Cleaning local Docker images..."
	@docker rmi -f $$(docker images 'bedrock-proxy-api*' -q) 2>/dev/null || true
	@docker rmi -f $$(docker images 'bedrock-proxy-api-ecs*' -q) 2>/dev/null || true
