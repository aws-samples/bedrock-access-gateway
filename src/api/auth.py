import os
import json
from typing import Annotated, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.setting import DEFAULT_API_KEYS

DDB_TABLE = os.environ["API_KEYS_TABLE_NAME"]
api_key_env = os.environ.get("API_KEY")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DDB_TABLE)
sm = boto3.client("secretsmanager")

_key_cache: dict[str, str] = {}
security = HTTPBearer()


try: 
    resp = table.scan()
except Exception:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unable to retrieve ARN table")

items = resp.get("Items", []) or []
for item in items:
    if "UserID" in item and "ARNKey" in item:
        _key_cache[item["UserID"]] = item["ARNKey"]

def get_arn_for_user(user_id: str) ->  Optional[str]:
    return _key_cache.get(user_id)

def get_secret_value(secret_arn: str) -> Optional[str]:
    try:
        response = sm.get_secret_value(SecretId=secret_arn)
        if "SecretString" in response:
            secret = json.loads(response["SecretString"])
            api_key = secret["api_key"]
    except ClientError:
        raise RuntimeError("Unable to retrieve API KEY, please ensure the secret ARN is correct")
    except KeyError:
        raise RuntimeError('Please ensure the secret contains a "api_key" field')
    
    return api_key

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

    secret_arn = get_arn_for_user(user_id)
    if secret_arn is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Unknown User {user_id}")
    
    expected_key = get_secret_value(secret_arn)
    if expected_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unable to retrieve API key")


    if candidate_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
