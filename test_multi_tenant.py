#!/usr/bin/env python3
"""
Test script for multi-tenant authentication functionality.
This script tests the multi-tenant auth system with various configurations.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import Request

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from api.multi_tenant_auth import MultiTenantConfig, MultiTenantAuthenticator, UserConfig, UserContext
from api.auth_utils import validate_model_access, get_user_inference_profile


class TestMultiTenantConfig:
    """Test multi-tenant configuration management."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_config = {
            "api_key": "fallback-key",
            "multi_tenant_config": {
                "enabled": True,
                "users": {
                    "user1-api-key": {
                        "inference_profiles": [
                            "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/user1-engineering"
                        ],
                        "allowed_models": [
                            "anthropic.claude-3-sonnet-*",
                            "anthropic.claude-3-haiku-*"
                        ],
                        "metadata": {
                            "user_id": "user1",
                            "department": "engineering"
                        }
                    },
                    "user2-api-key": {
                        "inference_profiles": [
                            "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/user2-marketing"
                        ],
                        "allowed_models": [
                            "anthropic.claude-3-haiku-*"
                        ],
                        "metadata": {
                            "user_id": "user2",
                            "department": "marketing"
                        }
                    }
                }
            }
        }
    
    @patch('boto3.client')
    def test_multi_tenant_enabled(self, mock_boto_client):
        """Test multi-tenant mode detection."""
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(self.test_config)
        }
        
        config = MultiTenantConfig("test-arn")
        assert config.is_multi_tenant_enabled() == True
    
    @patch('boto3.client')
    def test_user_config_retrieval(self, mock_boto_client):
        """Test user configuration retrieval."""
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(self.test_config)
        }
        
        config = MultiTenantConfig("test-arn")
        user_config = config.get_user_config("user1-api-key")
        
        assert user_config is not None
        assert user_config.user_id == "user1"
        assert user_config.api_key == "user1-api-key"
        assert len(user_config.inference_profiles) == 1
        assert len(user_config.allowed_models) == 2
    
    @patch('boto3.client')
    def test_invalid_api_key(self, mock_boto_client):
        """Test invalid API key handling."""
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(self.test_config)
        }
        
        config = MultiTenantConfig("test-arn")
        user_config = config.get_user_config("invalid-key")
        
        assert user_config is None


class TestMultiTenantAuthenticator:
    """Test multi-tenant authenticator."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_config = {
            "api_key": "fallback-key",
            "multi_tenant_config": {
                "enabled": True,
                "users": {
                    "user1-api-key": {
                        "inference_profiles": [
                            "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/user1-engineering"
                        ],
                        "allowed_models": [
                            "anthropic.claude-3-sonnet-*",
                            "anthropic.claude-3-haiku-*"
                        ],
                        "metadata": {
                            "user_id": "user1",
                            "department": "engineering"
                        }
                    }
                }
            }
        }
    
    @patch('boto3.client')
    def test_multi_tenant_authentication(self, mock_boto_client):
        """Test multi-tenant authentication."""
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(self.test_config)
        }
        
        authenticator = MultiTenantAuthenticator("test-arn")
        user_context = authenticator.authenticate("user1-api-key")
        
        assert user_context is not None
        assert user_context.user_id == "user1"
        assert len(user_context.inference_profiles) == 1
    
    @patch('boto3.client')
    def test_fallback_authentication(self, mock_boto_client):
        """Test fallback to single-key authentication."""
        single_key_config = {"api_key": "fallback-key"}
        
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(single_key_config)
        }
        
        authenticator = MultiTenantAuthenticator("test-arn")
        user_context = authenticator.authenticate("fallback-key")
        
        # Should return None for single-key mode
        assert user_context is None
    
    def test_model_access_validation(self):
        """Test model access validation."""
        user_context = UserContext(
            user_id="user1",
            inference_profiles=["test-profile"],
            allowed_models=["anthropic.claude-3-sonnet-*", "anthropic.claude-3-haiku-*"],
            metadata={}
        )
        
        authenticator = MultiTenantAuthenticator()
        
        # Test allowed models
        assert authenticator.validate_model_access(user_context, "anthropic.claude-3-sonnet-20240229-v1:0") == True
        assert authenticator.validate_model_access(user_context, "anthropic.claude-3-haiku-20240307-v1:0") == True
        
        # Test denied model
        assert authenticator.validate_model_access(user_context, "anthropic.claude-3-opus-20240229-v1:0") == False
        
        # Test single-key mode (None user_context)
        assert authenticator.validate_model_access(None, "any-model") == True
    
    def test_inference_profile_selection(self):
        """Test inference profile selection."""
        user_context = UserContext(
            user_id="user1",
            inference_profiles=["profile1", "profile2"],
            allowed_models=["*"],
            metadata={}
        )
        
        authenticator = MultiTenantAuthenticator()
        profile = authenticator.get_user_inference_profile(user_context, "test-model")
        
        # Should return first profile
        assert profile == "profile1"
        
        # Test single-key mode
        assert authenticator.get_user_inference_profile(None, "test-model") is None


def test_pattern_matching():
    """Test model pattern matching."""
    authenticator = MultiTenantAuthenticator()
    
    # Test exact match
    assert authenticator._match_pattern("exact-model", "exact-model") == True
    assert authenticator._match_pattern("exact-model", "different-model") == False
    
    # Test wildcard match
    assert authenticator._match_pattern("anthropic.claude-3-sonnet-*", "anthropic.claude-3-sonnet-20240229-v1:0") == True
    assert authenticator._match_pattern("anthropic.claude-3-sonnet-*", "anthropic.claude-3-haiku-20240307-v1:0") == False
    
    # Test global wildcard
    assert authenticator._match_pattern("*", "any-model") == True


def create_mock_request(user_context=None):
    """Create a mock request with user context."""
    request = Mock(spec=Request)
    request.state = Mock()
    request.state.user_context = user_context
    return request


def test_validate_model_access_utils():
    """Test validate_model_access utility function."""
    # Test with user context
    user_context = UserContext(
        user_id="user1",
        inference_profiles=["test-profile"],
        allowed_models=["anthropic.claude-3-sonnet-*"],
        metadata={}
    )
    
    request = create_mock_request(user_context)
    
    with patch('api.auth_utils.get_authenticator') as mock_get_auth:
        mock_auth = Mock()
        mock_auth.validate_model_access.return_value = True
        mock_get_auth.return_value = mock_auth
        
        result = validate_model_access(request, "anthropic.claude-3-sonnet-20240229-v1:0")
        assert result == True
        
        mock_auth.validate_model_access.assert_called_once_with(user_context, "anthropic.claude-3-sonnet-20240229-v1:0")


def test_get_user_inference_profile_utils():
    """Test get_user_inference_profile utility function."""
    user_context = UserContext(
        user_id="user1",
        inference_profiles=["test-profile"],
        allowed_models=["*"],
        metadata={}
    )
    
    request = create_mock_request(user_context)
    
    with patch('api.auth_utils.get_authenticator') as mock_get_auth:
        mock_auth = Mock()
        mock_auth.get_user_inference_profile.return_value = "test-profile"
        mock_get_auth.return_value = mock_auth
        
        result = get_user_inference_profile(request, "test-model")
        assert result == "test-profile"
        
        mock_auth.get_user_inference_profile.assert_called_once_with(user_context, "test-model")


if __name__ == "__main__":
    # Run basic tests
    print("Running multi-tenant authentication tests...")
    
    # Test pattern matching
    test_pattern_matching()
    print("✓ Pattern matching tests passed")
    
    # Test utility functions
    test_validate_model_access_utils()
    print("✓ Model access validation tests passed")
    
    test_get_user_inference_profile_utils()
    print("✓ Inference profile selection tests passed")
    
    print("All basic tests passed!")
    print("\nTo run full test suite with mocking, use: pytest test_multi_tenant.py")