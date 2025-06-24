#!/usr/bin/env python3
"""
Simple test script for multi-tenant authentication functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

from unittest.mock import Mock, patch
from api.multi_tenant_auth import MultiTenantAuthenticator, UserContext


def test_pattern_matching():
    """Test model pattern matching."""
    print("Testing pattern matching...")
    authenticator = MultiTenantAuthenticator()
    
    # Test exact match
    assert authenticator._match_pattern("exact-model", "exact-model") == True
    assert authenticator._match_pattern("exact-model", "different-model") == False
    
    # Test wildcard match
    assert authenticator._match_pattern("anthropic.claude-3-sonnet-*", "anthropic.claude-3-sonnet-20240229-v1:0") == True
    assert authenticator._match_pattern("anthropic.claude-3-sonnet-*", "anthropic.claude-3-haiku-20240307-v1:0") == False
    
    # Test global wildcard
    assert authenticator._match_pattern("*", "any-model") == True
    
    print("✓ Pattern matching tests passed")


def test_model_access_validation():
    """Test model access validation."""
    print("Testing model access validation...")
    
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
    
    print("✓ Model access validation tests passed")


def test_inference_profile_selection():
    """Test inference profile selection."""
    print("Testing inference profile selection...")
    
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
    
    print("✓ Inference profile selection tests passed")


def test_config_structure():
    """Test configuration structure validation."""
    print("Testing configuration structure...")
    
    # Test UserContext creation
    user_context = UserContext(
        user_id="test_user",
        inference_profiles=["arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/test"],
        allowed_models=["anthropic.claude-3-*"],
        metadata={"department": "engineering"}
    )
    
    assert user_context.user_id == "test_user"
    assert len(user_context.inference_profiles) == 1
    assert len(user_context.allowed_models) == 1
    assert user_context.metadata["department"] == "engineering"
    
    print("✓ Configuration structure tests passed")


if __name__ == "__main__":
    print("Running multi-tenant authentication tests...")
    print("=" * 50)
    
    try:
        test_pattern_matching()
        test_model_access_validation()
        test_inference_profile_selection()
        test_config_structure()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print("\nThe multi-tenant authentication system is ready for integration.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)