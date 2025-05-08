import os
from typing import Annotated, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.setting import DEFAULT_API_KEYS

# api_key_param = os.environ.get("API_KEY_PARAM_NAME")
# api_key_secret_arn = os.environ.get("API_KEY_SECRET_ARN")
# api_key_env = os.environ.get("API_KEY")

DDB_TABLE = os.environ["API_KEYS_TABLE_NAME"]

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DDB_TABLE)

_key_cache: dict[str, str] = {}
_cache_initialized: bool = False

security = HTTPBearer()

def load_all_api_keys() -> None:

    global _cache_initialized
    try: 
        resp = table.scan()
    except Exception:
        return
    
    items = resp.get("Items", []) or []
    for item in items:
        if "UserID" in item and "ARNKey" in item:
            _key_cache[item["UserID"]] = item["ARNKey"]

    _cache_initialized = True

def get_api_key_for_user(user_id: str) ->  Optional[str]:
    global _cache_initialized
    if not _cache_initialized:
        load_all_api_keys()
    return _key_cache.get(user_id)

def api_key_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    token = credentials.credentials

    prefix = "key:"
    if not token.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Header Format")

    body = token[len(prefix):]
    user_id, sep, candidate_key = body.partition(":")
    if not sep or not user_id or not candidate_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Header Format")

    expected_key = get_api_key_for_user(user_id)
    if expected_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Unknown User {user_id}")


    if candidate_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
