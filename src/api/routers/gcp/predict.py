import requests
import os
from fastapi import APIRouter, Depends
from api.setting import PROVIDER
from google import genai

from api.auth import api_key_auth
from api.modelmapper import get_model
from api.routers.gcp.chat_types import ChatRequest, ChatCompletionResponse

from vertexai.preview.generative_models import GenerativeModel
import vertexai

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    responses={404: {"description": "Not found"}},
)

client = None

def get_project_and_location():
    from google.auth import default

    # Try ADC for project
    _, project_id = default()

    # Try metadata server for region
    try:
        zone = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        ).text
        location = zone.split("/")[-1].rsplit("-", 1)[0]
    except Exception:
        location = os.getenv("GCP_LOCATION", "us-central1")

    return project_id, location

def to_vertex_content(messages):
    return [
        {
            "role": msg.role,
            "parts": [{"text": msg.content}]
        }
        for msg in messages
    ]

def aggregate_parts(response):
    generated_texts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                generated_texts.append(part.text)
    return "\n".join(generated_texts)

project_id, location = get_project_and_location()
vertexai.init(
    project=project_id,
    location=location,
)

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatRequest):
    content = ""
    try:
        modelId = get_model(PROVIDER, request.model) 
        model_name = modelId.split("/")[-1]
        model = GenerativeModel(model_name)

        content = to_vertex_content(request.messages)
        response = model.generate_content(content)

        content = aggregate_parts(response)
    except Exception as e:
        content = f"ProjectID({project_id} - Location({location}))Error: {str(e)}"

    return {
        "id": "chat-response",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
    }
