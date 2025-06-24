import json
import os
from typing import Annotated, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.setting import DEFAULT_API_KEYS
from api.multi_tenant_auth import get_authenticator, UserContext

api_key_param = os.environ.get("API_KEY_PARAM_NAME")
api_key_secret_arn = os.environ.get("API_KEY_SECRET_ARN")
api_key_env = os.environ.get("API_KEY")
if api_key_param:
    # For backward compatibility.
    # Please now use secrets manager instead.
    ssm = boto3.client("ssm")
    api_key = ssm.get_parameter(Name=api_key_param, WithDecryption=True)["Parameter"]["Value"]
elif api_key_secret_arn:
    sm = boto3.client("secretsmanager")
    try:
        response = sm.get_secret_value(SecretId=api_key_secret_arn)
        if "SecretString" in response:
            secret = json.loads(response["SecretString"])
            api_key = secret["api_key"]
    except ClientError:
        raise RuntimeError("Unable to retrieve API KEY, please ensure the secret ARN is correct")
    except KeyError:
        raise RuntimeError('Please ensure the secret contains a "api_key" field')
elif api_key_env:
    api_key = api_key_env
else:
    # For local use only.
    api_key = DEFAULT_API_KEYS

security = HTTPBearer()


def api_key_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    request: Request,
) -> Optional[UserContext]:
    """Authenticate API key and return user context if multi-tenant, None if single-key."""
    api_key_value = credentials.credentials
    
    # Try multi-tenant authentication first
    authenticator = get_authenticator()
    try:
        user_context = authenticator.authenticate(api_key_value)
        
        # Store user context in request state for downstream use
        request.state.user_context = user_context
        
        return user_context
        
    except RuntimeError as e:
        # If multi-tenant auth fails, try fallback single-key auth
        if api_key_value == api_key:
            request.state.user_context = None
            return None
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid API Key"
        )
