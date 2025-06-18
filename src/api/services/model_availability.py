"""
Model availability service with Parameter Store caching and in-memory optimization.

This service checks which Bedrock models have been granted access via the AWS console
and caches the results in Parameter Store with in-memory optimization for Lambda.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Configuration
MODEL_AVAILABILITY_TTL_MINUTES = int(os.getenv("MODEL_AVAILABILITY_TTL_MINUTES", "60"))
MODEL_AVAILABILITY_CHECK_ENABLED = os.getenv("MODEL_AVAILABILITY_CHECK_ENABLED", "true").lower() == "true"


def get_bedrock_model_availability(model_id: str, region: str = "us-east-1") -> Optional[Dict[str, Any]]:
    """
    Check the availability and entitlement status of a Bedrock model.

    Args:
        model_id: The ID of the Bedrock model (e.g., 'anthropic.claude-3-5-haiku-20241022-v1:0')
        region: The AWS region of the Bedrock service

    Returns:
        Dict containing the JSON response from the endpoint, or None if an error occurs
    """
    try:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        signer = SigV4Auth(credentials, "bedrock", region)

        endpoint_url = f"https://bedrock.{region}.amazonaws.com"
        path = f"/foundation-model-availability/{model_id}"
        url = f"{endpoint_url}{path}"

        headers = {
            "Accept": "*/*",
            "User-Agent": "bedrock-access-gateway/1.0",
            "x-amz-content-sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "host": f"bedrock.{region}.amazonaws.com",
        }

        request = AWSRequest(method="GET", url=url, headers=headers, data={})
        request.headers["x-amz-date"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        signer.add_auth(request)

        response = requests.get(url, headers=request.headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            logger.debug(f"Model {model_id} not supported by availability endpoint (400 Bad Request)")
        elif e.response.status_code == 403:
            logger.warning(f"Access denied for model availability check on {model_id} (403 Forbidden)")
        else:
            logger.warning(f"HTTP error checking model availability for {model_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error checking model availability for {model_id}: {e}")
        return None


class ModelAvailabilityService:
    """Service for checking and caching model availability."""

    def __init__(self, ssm_client=None, bedrock_client=None):
        """
        Initialize the service with optional clients for testing.

        Args:
            ssm_client: Optional SSM client for Parameter Store operations
            bedrock_client: Optional Bedrock client for model listing
        """
        self.ssm_client = ssm_client or boto3.client("ssm")
        self.bedrock_client = bedrock_client or boto3.client("bedrock")

        # In-memory cache (shared across Lambda invocations)
        self._in_memory_cache: Dict[str, Any] = {}
        self._cache_loaded_at: Optional[datetime] = None
        self._background_refresh_task: Optional[asyncio.Task] = None

    async def get_available_models(self, region: str) -> Set[str]:
        """
        Get the set of available models for the given region.

        Args:
            region: AWS region to check models for

        Returns:
            Set of available model IDs
        """
        if not MODEL_AVAILABILITY_CHECK_ENABLED:
            logger.info("Model availability checking disabled, returning empty set")
            return set()

        # Check if we have fresh in-memory cache
        if self._has_fresh_cache(region):
            logger.debug(f"Using fresh in-memory cache for region {region}")
            return set(self._in_memory_cache.get("available_models", []))

        # Load from Parameter Store
        cache_data = await self._load_from_parameter_store(region)
        if cache_data:
            self._in_memory_cache = cache_data
            self._cache_loaded_at = datetime.now(timezone.utc)

            # Trigger background refresh if cache is stale
            if self._is_cache_stale():
                await self._maybe_trigger_background_refresh(region)

            return set(cache_data.get("available_models", []))

        # No cache available, trigger immediate refresh
        logger.info(f"No cache found for region {region}, triggering immediate refresh")
        await self._refresh_availability_cache(region)
        return set(self._in_memory_cache.get("available_models", []))

    def _has_fresh_cache(self, region: str) -> bool:
        """Check if we have fresh in-memory cache data."""
        if not self._in_memory_cache or not self._cache_loaded_at:
            return False

        # Check if cache is for the same region (simple check)
        if not self._in_memory_cache.get("available_models"):
            return False

        # Check if cache is still fresh based on load time
        cache_age = datetime.now(timezone.utc) - self._cache_loaded_at
        return cache_age.total_seconds() < 300  # 5 minutes max for in-memory cache

    def _is_cache_stale(self) -> bool:
        """Check if the cache data is stale based on TTL."""
        if not self._in_memory_cache:
            return True

        next_refresh_str = self._in_memory_cache.get("next_refresh_after")
        if not next_refresh_str:
            return True

        try:
            next_refresh = datetime.fromisoformat(next_refresh_str.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) >= next_refresh
        except (ValueError, TypeError):
            return True

    async def _load_from_parameter_store(self, region: str) -> Optional[Dict[str, Any]]:
        """Load cache data from Parameter Store."""
        parameter_name = f"/bedrock-gateway/model-availability/{region}"

        try:
            if asyncio.iscoroutinefunction(self.ssm_client.get_parameter):
                response = await self.ssm_client.get_parameter(Name=parameter_name)
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.ssm_client.get_parameter(Name=parameter_name)
                )

            cache_data = json.loads(response["Parameter"]["Value"])
            logger.debug(f"Loaded cache from Parameter Store for region {region}")
            return cache_data

        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                logger.info(f"No cache parameter found for region {region}")
                return None
            else:
                logger.error(f"Error loading cache from Parameter Store: {e}")
                return None
        except Exception as e:
            logger.error(f"Error parsing cache data from Parameter Store: {e}")
            return None

    async def _maybe_trigger_background_refresh(self, region: str) -> None:
        """Trigger background refresh if not already running."""
        if self._background_refresh_task and not self._background_refresh_task.done():
            logger.debug("Background refresh already running, skipping")
            return

        logger.info(f"Triggering background refresh for region {region}")
        self._background_refresh_task = asyncio.create_task(self._refresh_availability_cache(region))

    async def _refresh_availability_cache(self, region: str) -> None:
        """Refresh the availability cache by checking all models."""
        logger.info(f"Refreshing model availability cache for region {region}")

        try:
            # Get all foundation models
            if asyncio.iscoroutinefunction(self.bedrock_client.list_foundation_models):
                response = await self.bedrock_client.list_foundation_models()
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.bedrock_client.list_foundation_models()
                )

            models = response.get("modelSummaries", [])
            logger.info(f"Found {len(models)} models to check for region {region}")

            # Check availability for each model
            checked_models = {}
            available_models = []

            for model in models:
                model_id = model["modelId"]

                # Determine the correct model ID format for availability checking
                model_id_for_check = self._normalize_model_id_for_availability_check(model_id)

                # Skip models that are known to not work with the availability endpoint
                if not model_id_for_check:
                    logger.debug(f"Skipping availability check for unsupported model: {model_id}")
                    continue

                availability_info = get_bedrock_model_availability(model_id_for_check, region)

                if availability_info and availability_info.get("entitlementAvailability") == "AVAILABLE":
                    # Add both the original model ID and potential cross-region inference profile ID
                    available_models.append(model_id)  # Original model ID
                    
                    # Also add cross-region inference profile ID if applicable
                    region_prefix = self._get_inference_region_prefix(region)
                    if region_prefix:
                        cr_model_id = f"{region_prefix}.{model_id}"
                        available_models.append(cr_model_id)
                    
                    checked_models[model_id_for_check] = {
                        "available": True,
                        "last_checked": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    checked_models[model_id_for_check] = {
                        "available": False,
                        "last_checked": datetime.now(timezone.utc).isoformat(),
                    }

            # Create cache data - optimized for size
            now = datetime.now(timezone.utc)
            next_refresh = now + timedelta(minutes=MODEL_AVAILABILITY_TTL_MINUTES)

            # Only store essential data to minimize parameter size
            cache_data = {
                "last_updated": now.isoformat(),
                "ttl_minutes": MODEL_AVAILABILITY_TTL_MINUTES,
                "next_refresh_after": next_refresh.isoformat(),
                "available_models": available_models,
                # Store minimal checked_models data - just availability status
                "checked_models": {
                    model_id: {"available": model_data["available"]} for model_id, model_data in checked_models.items()
                },
            }

            # Update Parameter Store - use advanced parameter for larger data
            parameter_name = f"/bedrock-gateway/model-availability/{region}"
            cache_json = json.dumps(cache_data, separators=(",", ":"))  # Compact JSON

            # Use advanced parameter tier if data is too large for standard tier
            parameter_type = "String"
            tier = "Standard"
            if len(cache_json) > 4000:  # Leave some buffer under 4KB limit
                tier = "Advanced"
                logger.info(f"Using advanced parameter tier for large cache data ({len(cache_json)} chars)")

            if asyncio.iscoroutinefunction(self.ssm_client.put_parameter):
                await self.ssm_client.put_parameter(
                    Name=parameter_name, Value=cache_json, Type=parameter_type, Tier=tier, Overwrite=True
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ssm_client.put_parameter(
                        Name=parameter_name, Value=cache_json, Type=parameter_type, Tier=tier, Overwrite=True
                    ),
                )

            # Update in-memory cache
            self._in_memory_cache = cache_data
            self._cache_loaded_at = now

            logger.info(f"Cache refresh completed for region {region}. Found {len(available_models)} available models")

        except Exception as e:
            logger.error(f"Error refreshing availability cache for region {region}: {e}")
            raise

    def _get_inference_region_prefix(self, region: str) -> str:
        """Get the cross-region inference prefix for a given region."""
        if region.startswith("ap-"):
            return "apac"
        return region[:2]  # us-east-1 -> us, eu-west-1 -> eu, etc.

    def _normalize_model_id_for_availability_check(self, model_id: str) -> str:
        """
        Normalize model ID for the foundation-model-availability endpoint.

        Args:
            model_id: The original model ID from list_foundation_models

        Returns:
            Normalized model ID for availability checking, or None if unsupported
        """
        # Skip legacy models that don't work with the availability endpoint
        legacy_models = [
            "amazon.titan-tg1-large",  # Legacy Titan model
            "amazon.titan-e1t-medium",  # Legacy Titan model
        ]

        if model_id in legacy_models:
            logger.debug(f"Skipping legacy model: {model_id}")
            return None

        # Skip models that already have complex version suffixes that cause 400 errors
        if model_id.count(":") > 1:
            logger.debug(f"Skipping model with complex versioning: {model_id}")
            return None

        # For models without version suffix, add :0 for specific providers
        if ":" not in model_id:
            supported_providers = ["anthropic", "amazon", "ai21", "cohere", "meta", "mistral"]
            if any(provider in model_id for provider in supported_providers):
                return f"{model_id}:0"
            else:
                # Unknown provider, try as-is first
                return model_id

        # Model already has version suffix, use as-is
        return model_id


# Global service instance for Lambda
_service_instance: Optional[ModelAvailabilityService] = None


def get_model_availability_service() -> ModelAvailabilityService:
    """Get the global ModelAvailabilityService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelAvailabilityService()
    return _service_instance
