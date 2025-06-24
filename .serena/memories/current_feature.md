# Current Feature: Model Access Filtering

## Feature Overview
The current branch `feature/add-model-access-filtering` is implementing a feature to filter model results based on model availability, with optimization for Lambda cold starts.

## Key Components

1. **Model Availability Service**
   - Located in `/src/api/services/model_availability.py`
   - Provides mechanism to check which Bedrock models have been granted access
   - Uses AWS Parameter Store for caching results
   - Implements in-memory optimization for faster Lambda cold starts

2. **Changes to API Router**
   - Updated `/src/api/routers/model.py` to filter models based on availability
   - Preserves custom models in results regardless of availability status

3. **Lazy Model Loading**
   - Modified `/src/api/models/bedrock.py` to initialize model list lazily
   - Added `get_bedrock_model_list()` function to retrieve models on demand
   - Removed eager initialization that was slowing Lambda cold start

4. **IAM Policy Updates**
   - Added broader permissions for Bedrock APIs
   - Added SSM Parameter Store permissions for model availability cache
   - Updated CloudFormation templates and Terraform configurations

## Implementation Details
- Uses the Bedrock foundation-model-availability endpoint to check model access
- Caches results in Parameter Store with TTL-based expiry (default 60 minutes)
- Keeps in-memory cache for rapid access during Lambda execution
- Background refresh mechanism for cache updates
- Handles cross-region inference profiles