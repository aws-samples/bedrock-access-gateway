from typing import Optional
from fastapi import Request

from api.multi_tenant_auth import get_authenticator, UserContext


def get_user_context(request: Request) -> Optional[UserContext]:
    """Get user context from request state."""
    return getattr(request.state, 'user_context', None)


def validate_model_access(request: Request, model_id: str) -> bool:
    """Validate if the authenticated user has access to the specified model."""
    user_context = get_user_context(request)
    authenticator = get_authenticator()
    return authenticator.validate_model_access(user_context, model_id)


def get_user_inference_profile(request: Request, model_id: str) -> Optional[str]:
    """Get the appropriate inference profile for the authenticated user."""
    user_context = get_user_context(request)
    authenticator = get_authenticator()
    return authenticator.get_user_inference_profile(user_context, model_id)