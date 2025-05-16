#!/usr/bin/env python3
import json
import re

import boto3
import requests
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform_v1beta1


def get_json_from_url(url):
    """Fetch JSON data from a URL"""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


def get_docker_hub(repo):
    """Get Docker Hub repository information"""
    url = f"https://hub.docker.com/v2/repositories/{repo}/?page_size=100"
    return get_json_from_url(url)


def get_bedrock_models():
    """Get a list of available AWS Bedrock models using boto3"""
    # Create a Bedrock client
    bedrock_client = boto3.client('bedrock')

    # Get the list of foundation models
    response = bedrock_client.list_foundation_models()

    # Extract and return model IDs
    return [model["modelId"] for model in response.get("modelSummaries", [])]


def get_vertexai_models():
    # FIXME: this currently uses a service account key
    options = ClientOptions(api_endpoint="us-central1-aiplatform.googleapis.com", credentials_file="/Users/llunesu/Downloads/liotest-443018-130306ecbcbd.json")
    client = aiplatform_v1beta1.ModelGardenServiceClient(client_options=options)

    # This list was handcrafted based on the Vertex AI model garden UI
    publishers = ["ai21", "meta", "google", "anthropic", "sesame", "microsoft", "openai", "qwen", "black-forest-labs", "bytedance", "liuhaotian", "openlm-research", "lmsys", "salesforce", "stability-ai", "mistralai", "cambai", "hidream-ai"]
    # List all models for all publishers
    return [model.name for publisher in publishers for model in client.list_publisher_models(parent=f"publishers/{publisher}")]


def get_dockerhub_models():
    # Get all repositories in the 'ai' namespace
    ai_repos = [repo.get("name") for repo in get_docker_hub("ai").get("results", [])]

    # Process all repositories and their tags
    return [repo+":"+tag.get("name") for repo in ai_repos for tag in get_docker_hub(f"ai/{repo}/tags").get("results", [])]

def main():

    bedrock_models = get_bedrock_models()
    print("Available AWS Bedrock models:")
    for model_id in bedrock_models:
        print(f"  - {model_id}")
    print("\n")

    vertexai_models = get_vertexai_models()
    print("Available Vertex AI models:")
    for model_id in vertexai_models:
        print(f"  - {model_id}")
    print("\n")

    dockerhub_models = get_dockerhub_models()
    print("Available Docker Hub models:")
    for model_id in dockerhub_models:
        print(f"  - {model_id}")
    print("\n")

    # Model mappings from Docker Hub repo to AWS Bedrock model ID
    docker_to_aws = {
        # "deepcoder-preview": "N/A",
        "deepseek-r1-distill-llama": "deepseek.r1-v1:0",
        # "gemma3": "N/A (Marketplace)",
        # "gemma3-qat": "N/A (Marketplace)",
        "llama3.1": "meta.llama3-1-8b-instruct-v1:0",
        "llama3.2": "meta.llama3-2-1b-instruct-v1:0",
        "llama3.3": "meta.llama3-3-70b-instruct-v1:0",
        "mistral": "mistral.mistral-7b-instruct-v0:2",
        # "mistral-nemo": "N/A",
        # "mxbai-embed-large": "N/A",
        # "phi4": "N/A (Marketplace)",
        # "qwen2.5": "N/A (Marketplace)",
        # "qwen3": "N/A",
        # "qwq": "N/A",
        # "smollm2": "N/A"
    }

    docker_to_bedrock_mapping = {}
    for model_id in dockerhub_models:
        try:
            # Add image:tag to mapping
            image_name, tag = model_id.split(":")
            bedrock_model = docker_to_aws[image_name]
            # Check if the tag starts with regex /\d+B-/ and extract the size
            if "B-" in tag:
                size = tag.split("B-")[0]
                # Replace any /-\d+b-/ with the size
                bedrock_model = re.sub(r'-\d+b-', f'-{size}b-', bedrock_model)
            if bedrock_models and bedrock_model not in bedrock_models:
                print(f"Warning: {bedrock_model} is not a valid Bedrock model ID.")
            docker_to_bedrock_mapping[f"{model_id}"] = bedrock_model
        except KeyError:
            print(f"Warning: {model_id} not found in mapping.")

    # dump the mapping to a JSON file
    print(json.dumps(docker_to_bedrock_mapping, indent=4))
    print("\n")


if __name__ == "__main__":
    main()
