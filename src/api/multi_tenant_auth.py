import json
import logging
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class UserConfig:
    """User configuration for multi-tenant setup."""
    user_id: str
    api_key: str
    inference_profiles: List[str]
    allowed_models: List[str]
    metadata: Dict[str, Any]


@dataclass
class UserContext:
    """User context for authenticated requests."""
    user_id: str
    inference_profiles: List[str]
    allowed_models: List[str]
    metadata: Dict[str, Any]


class MultiTenantConfig:
    """Manages multi-tenant configuration from AWS Secrets Manager."""
    
    def __init__(self, secret_arn: str):
        self.secret_arn = secret_arn
        self.sm = boto3.client("secretsmanager")
        self._config_cache = None
        self._fallback_key = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from Secrets Manager."""
        try:
            response = self.sm.get_secret_value(SecretId=self.secret_arn)
            if "SecretString" in response:
                return json.loads(response["SecretString"])
            else:
                raise RuntimeError("Secret does not contain SecretString")
        except ClientError as e:
            logger.error(f"Unable to retrieve secret: {e}")
            raise RuntimeError("Unable to retrieve API KEY, please ensure the secret ARN is correct")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in secret: {e}")
            raise RuntimeError("Secret contains invalid JSON")
    
    def get_config(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get configuration with caching."""
        if self._config_cache is None or force_refresh:
            self._config_cache = self._load_config()
        return self._config_cache
    
    def is_multi_tenant_enabled(self) -> bool:
        """Check if multi-tenant mode is enabled."""
        config = self.get_config()
        return (
            "multi_tenant_config" in config and 
            config["multi_tenant_config"].get("enabled", False)
        )
    
    def get_fallback_key(self) -> Optional[str]:
        """Get the fallback single API key for backward compatibility."""
        if self._fallback_key is None:
            config = self.get_config()
            self._fallback_key = config.get("api_key")
        return self._fallback_key
    
    def get_user_config(self, api_key: str) -> Optional[UserConfig]:
        """Get user configuration for the given API key."""
        if not self.is_multi_tenant_enabled():
            return None
            
        config = self.get_config()
        users = config["multi_tenant_config"].get("users", {})
        
        if api_key not in users:
            return None
            
        user_data = users[api_key]
        return UserConfig(
            user_id=user_data.get("metadata", {}).get("user_id", "unknown"),
            api_key=api_key,
            inference_profiles=user_data.get("inference_profiles", []),
            allowed_models=user_data.get("allowed_models", []),
            metadata=user_data.get("metadata", {})
        )
    
    def list_users(self) -> List[UserConfig]:
        """List all configured users."""
        if not self.is_multi_tenant_enabled():
            return []
            
        config = self.get_config()
        users = config["multi_tenant_config"].get("users", {})
        
        user_configs = []
        for api_key, user_data in users.items():
            user_configs.append(UserConfig(
                user_id=user_data.get("metadata", {}).get("user_id", "unknown"),
                api_key=api_key,
                inference_profiles=user_data.get("inference_profiles", []),
                allowed_models=user_data.get("allowed_models", []),
                metadata=user_data.get("metadata", {})
            ))
        
        return user_configs


class MultiTenantAuthenticator:
    """Handles multi-tenant authentication."""
    
    def __init__(self, secret_arn: Optional[str] = None, fallback_key: Optional[str] = None):
        self.config = MultiTenantConfig(secret_arn) if secret_arn else None
        self.fallback_key = fallback_key
        
    def authenticate(self, api_key: str) -> Optional[UserContext]:
        """
        Authenticate API key and return user context.
        
        Returns:
            UserContext if multi-tenant auth succeeds
            None if should fall back to single-key auth
            
        Raises:
            RuntimeError if authentication fails
        """
        if not self.config:
            return None
            
        try:
            # Try multi-tenant authentication first
            user_config = self.config.get_user_config(api_key)
            if user_config:
                return UserContext(
                    user_id=user_config.user_id,
                    inference_profiles=user_config.inference_profiles,
                    allowed_models=user_config.allowed_models,
                    metadata=user_config.metadata
                )
            
            # If multi-tenant is enabled but key not found, fail
            if self.config.is_multi_tenant_enabled():
                raise RuntimeError("Invalid API Key")
                
            # Fall back to single-key validation
            fallback = self.config.get_fallback_key()
            if fallback and api_key == fallback:
                return None  # Indicates successful single-key auth
                
            raise RuntimeError("Invalid API Key")
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise RuntimeError("Authentication failed")
    
    def validate_model_access(self, user_context: Optional[UserContext], model_id: str) -> bool:
        """Validate if user has access to the specified model."""
        if not user_context:
            return True  # Single-key mode - allow all models
            
        # Check against allowed model patterns
        for pattern in user_context.allowed_models:
            if self._match_pattern(pattern, model_id):
                return True
                
        return False
    
    def get_user_inference_profile(self, user_context: Optional[UserContext], model_id: str) -> Optional[str]:
        """Get the appropriate inference profile for the user and model."""
        if not user_context or not user_context.inference_profiles:
            return None
            
        # For now, return the first profile
        # TODO: Add logic to select profile based on model or other criteria
        return user_context.inference_profiles[0]
    
    def _match_pattern(self, pattern: str, model_id: str) -> bool:
        """Check if model_id matches the pattern (supports wildcards)."""
        if pattern == "*":
            return True
            
        if pattern.endswith("*"):
            return model_id.startswith(pattern[:-1])
            
        return pattern == model_id


# Global authenticator instance
_authenticator: Optional[MultiTenantAuthenticator] = None


def get_authenticator() -> MultiTenantAuthenticator:
    """Get the global authenticator instance."""
    global _authenticator
    if _authenticator is None:
        secret_arn = os.environ.get("API_KEY_SECRET_ARN")
        fallback_key = os.environ.get("API_KEY")
        _authenticator = MultiTenantAuthenticator(secret_arn, fallback_key)
    return _authenticator


def reset_authenticator():
    """Reset the global authenticator (for testing)."""
    global _authenticator
    _authenticator = None