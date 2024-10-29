import os
import logging
from typing import Annotated

import boto3
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.setting import DEFAULT_API_KEYS

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

api_key_param = os.environ.get("API_KEY_PARAM_NAME")
if api_key_param:
    ssm = boto3.client("ssm")
    api_key = ssm.get_parameter(Name=api_key_param, WithDecryption=True)["Parameter"][
        "Value"
    ]
    log.info("API key fetched from AWS SSM.")
else:
    api_key = DEFAULT_API_KEYS
    log.info("Using default API keys.")

security = HTTPBearer()


def api_key_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
):
    if credentials.credentials != api_key:
        log.exception(f"Invalid API key attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    log.info("API key authentication successful.")