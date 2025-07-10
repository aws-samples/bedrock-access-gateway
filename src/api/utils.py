import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CONTENT_FILTER_API_KEY")
ENDPOINT = os.getenv("CONTENT_FILTER_ENDPOINT")

if not API_KEY or not ENDPOINT:
    raise RuntimeError("CONTENT_FILTER_API_KEY and CONTENT_FILTER_ENDPOINT must be set in environment variables")

def check_content_safety(text: str) -> bool:
    """
    Check if the given text is safe according to Azure Content Safety.
    Args:
        text (str): The text to analyze for safety
    Returns:
        bool: True if the text is safe, False if it contains unsafe content
    """
    credential = AzureKeyCredential(API_KEY)
    client = ContentSafetyClient(endpoint=ENDPOINT, credential=credential)
    try:
        request = {"text": text}
        response = client.analyze_text(request)
        for category_result in response.categories_analysis:
            if category_result.severity > 0:
                return False
        return True
    except HttpResponseError:
        return True  # fallback safe response
    except Exception:
        return True  # fallback safe response

def get_last_user_message(messages: list) -> str:
    """
    Get the content of the last user message from the messages array.
    Args:
        messages (list): List of message objects (Pydantic models)
    Returns:
        str: Content of the last user message, or None if no user message found
    """
    for message in reversed(messages):
        if hasattr(message, "role") and message.role == "user":
            return message.content
    return None 