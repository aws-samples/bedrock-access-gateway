#!/bin/bash

export IMAGE_NAME=bedrock_gateway

docker build . -f Dockerfile_ecs -t $IMAGE_NAME

