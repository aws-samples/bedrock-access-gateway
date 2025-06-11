# NOTE: The script will try to create the ECR repository if it doesn't exist. Please grant the necessary permissions to the IAM user or role.
# Usage:
#    cd scripts
#    bash ./push-to-ecr.sh

set -o errexit  # exit on first error
set -o nounset  # exit on using unset variables
set -o pipefail # exit on any error in a pipeline

# Define variables
TAG="latest"
ARCHS=("arm64" "amd64")
AWS_REGIONS=("us-east-1") # List of AWS region, use below liest if you don't enable ECR repository replication
# AWS_REGIONS=("us-east-1" "us-west-2" "eu-central-1" "ap-southeast-1" "ap-southeast-2" "ap-northeast-1" "eu-central-1" "eu-west-3") # List of supported AWS regions

build_and_push_images() {
    local IMAGE_NAME=$1
    local TAG=$2
    local ENABLE_MULTI_ARCH=${3:-true}  # Parameter for enabling multi-arch build, default is true
    local podmanFILE_PATH=${4:-"../src/podmanfile_ecs"}  # Parameter for podmanfile path, default is "../src/podmanfile_ecs"

    # Build podman image for each architecture
    if [ "$ENABLE_MULTI_ARCH" == "true" ]; then
        for ARCH in "${ARCHS[@]}"
        do
            # Build multi-architecture podman image
            podman buildx build --platform linux/$ARCH -t $IMAGE_NAME:$TAG-$ARCH -f $podmanFILE_PATH --load ../src/
        done
    else
        # Build single architecture podman image
        podman buildx build --platform linux/${ARCHS[0]} -t $IMAGE_NAME:$TAG -f $podmanFILE_PATH --load ../src/
    fi

    # Push podman image to ECR for each architecture in each AWS region
    for REGION in "${AWS_REGIONS[@]}"
    do
        # Get the account ID for the current region
        ACCOUNT_ID=$(aws sts get-caller-identity --region $REGION --query Account --output text)

        # Create repository URI
        REPOSITORY_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

        # Create ECR repository if it doesn't exist
        aws ecr create-repository --repository-name "${IMAGE_NAME}" --region $REGION || true

        # Log in to ECR
        aws ecr get-login-password --region $REGION | podman login --username AWS --password-stdin $REPOSITORY_URI

        # Push the image to ECR for each architecture
        if [ "$ENABLE_MULTI_ARCH" == "true" ]; then
            for ARCH in "${ARCHS[@]}"
            do
                # Tag the image for the current region
                podman tag $IMAGE_NAME:$TAG-$ARCH $REPOSITORY_URI:$TAG-$ARCH
                # Push the image to ECR
                podman push $REPOSITORY_URI:$TAG-$ARCH
                # Create a manifest for the image
                podman manifest create $REPOSITORY_URI:$TAG $REPOSITORY_URI:$TAG-$ARCH --amend
                # Annotate the manifest with architecture information
                podman manifest annotate $REPOSITORY_URI:$TAG "$REPOSITORY_URI:$TAG-$ARCH" --os linux --arch $ARCH
            done

            # Push the manifest to ECR
            podman manifest push $REPOSITORY_URI:$TAG
        else
            # Tag the image for the current region
            podman tag $IMAGE_NAME:$TAG $REPOSITORY_URI:$TAG
            # Push the image to ECR
            podman push $REPOSITORY_URI:$TAG
        fi

        echo "Pushed $IMAGE_NAME:$TAG to $REPOSITORY_URI"
    done
}

build_and_push_images "bedrock-proxy-api" "$TAG" "false" "../src/podmanfile"
build_and_push_images "bedrock-proxy-api-ecs" "$TAG"
