# NOTE: The script will try to create the ECR repository if it doesn't exist. Please grant the necessary permissions to the IAM user or role.
# Usage:
#    cd scripts
#    bash ./push-to-ecr.sh

set -o errexit  # exit on first error
set -o nounset  # exit on using unset variables
set -o pipefail # exit on any error in a pipeline

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Prompt user for inputs
echo "================================================"
echo "Bedrock Access Gateway - Build and Push to ECR"
echo "================================================"
echo ""

# Get repository name for Lambda version
read -p "Enter ECR repository name for Lambda (default: bedrock-proxy-api): " LAMBDA_REPO
LAMBDA_REPO=${LAMBDA_REPO:-bedrock-proxy-api}

# Get repository name for ECS/Fargate version
read -p "Enter ECR repository name for ECS/Fargate (default: bedrock-proxy-api-ecs): " ECS_REPO
ECS_REPO=${ECS_REPO:-bedrock-proxy-api-ecs}

# Get image tag
read -p "Enter image tag (default: latest): " TAG
TAG=${TAG:-latest}

# Get AWS region
read -p "Enter AWS region (default: us-east-1): " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

echo ""
echo "Configuration:"
echo "  Lambda Repository: $LAMBDA_REPO"
echo "  ECS/Fargate Repository: $ECS_REPO"
echo "  Image Tag: $TAG"
echo "  AWS Region: $AWS_REGION"
echo ""
read -p "Continue with these settings? (y/n): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

# Acknowledgment about ECR repository creation
echo "ℹ️  NOTICE: This script will automatically create ECR repositories if they don't exist."
echo "   The repositories will be created with the following default settings:"
echo "   - Image tag mutability: MUTABLE (allows overwriting tags)"
echo "   - Image scanning: Disabled"
echo "   - Encryption: AES256 (AWS managed encryption)"
echo ""
echo "   You can modify these settings later in the AWS ECR Console if needed."
echo "   Required IAM permissions: ecr:CreateRepository, ecr:GetAuthorizationToken,"
echo "   ecr:BatchCheckLayerAvailability, ecr:InitiateLayerUpload, ecr:UploadLayerPart,"
echo "   ecr:CompleteLayerUpload, ecr:PutImage"
echo ""
read -p "Do you acknowledge and want to proceed? (y/n): " ACK_CONFIRM
if [[ ! "$ACK_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

# Define variables
ARCHS=("arm64")  # Single architecture for simplicity

build_and_push_image() {
    local IMAGE_NAME=$1
    local TAG=$2
    local DOCKERFILE_PATH=$3
    local REGION=$AWS_REGION
    local ARCH=${ARCHS[0]}

    echo "Building $IMAGE_NAME:$TAG..."

    # Build Docker image
    # Note: --provenance=false and --sbom=false are required for Lambda compatibility
    # Without these flags, Docker BuildKit (especially with docker-container driver) may create
    # OCI image manifests with attestations that AWS Lambda does not support.
    # Lambda requires Docker V2 Schema 2 format without multi-manifest index.
    # See: https://github.com/aws-samples/bedrock-access-gateway/issues/206
    docker buildx build \
        --platform linux/$ARCH \
        --provenance=false \
        --sbom=false \
        -t $IMAGE_NAME:$TAG \
        -f $DOCKERFILE_PATH \
        --load \
        ../src/

    # Get the account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --region $REGION --query Account --output text)

    # Create repository URI
    REPOSITORY_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

    echo "Creating ECR repository if it doesn't exist..."
    # Create ECR repository if it doesn't exist
    aws ecr create-repository --repository-name "${IMAGE_NAME}" --region $REGION || true

    echo "Logging in to ECR..."
    # Log in to ECR
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REPOSITORY_URI

    echo "Pushing image to ECR..."
    # Tag the image for ECR
    docker tag $IMAGE_NAME:$TAG $REPOSITORY_URI:$TAG

    # Push the image to ECR
    docker push $REPOSITORY_URI:$TAG

    echo "✅ Successfully pushed $IMAGE_NAME:$TAG to $REPOSITORY_URI"
    echo ""
}

echo "Building and pushing Lambda image..."
build_and_push_image "$LAMBDA_REPO" "$TAG" "../src/Dockerfile"

echo "Building and pushing ECS/Fargate image..."
build_and_push_image "$ECS_REPO" "$TAG" "../src/Dockerfile_ecs"

echo "================================================"
echo "✅ All images successfully pushed!"
echo "================================================"
echo ""
echo "Your container image URIs:"
ACCOUNT_ID=$(aws sts get-caller-identity --region $AWS_REGION --query Account --output text)
echo "  Lambda: ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${LAMBDA_REPO}:${TAG}"
echo "  ECS/Fargate: ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECS_REPO}:${TAG}"
echo ""
echo "Next steps:"
echo "  1. Download the CloudFormation templates from deployment/ folder"
echo "  2. Update the ContainerImageUri parameter with your image URI above"
echo "  3. Deploy the stack via AWS CloudFormation Console"
echo ""
