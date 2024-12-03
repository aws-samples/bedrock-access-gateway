#!/bin/bash

echo "Starting Bedrock Gateway..."

docker run -v $HOME/.aws:/root/.aws \
--add-host=host.docker.internal:host-gateway \
--restart always \
-e AWS_REGION=us-west-2 \
-e DEBUG=true \
-d -p 8000:80 \
--name bedrock-gateway \
bedrock_gateway