#!/usr/bin/env python3
import boto3
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Custom model ID and region
CUSTOM_MODEL_ID = "arn:aws:bedrock:us-east-1:529237317113:imported-model/okp1jkklp8kk"
REGION = "us-east-1"
MAX_TOKENS = 100

# Create a Bedrock Runtime client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

# Test direct invocation
def test_direct_invocation():
    """Test direct invocation of the custom model."""
    logger.info(f"Testing direct invocation of {CUSTOM_MODEL_ID}")

    # Create a request body
    body = {
        "prompt": "Human: Hello, how are you?\nAssistant:",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=CUSTOM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        # Parse the response
        response_body = json.loads(response["body"].read())
        logger.info(f"Response: {json.dumps(response_body, indent=2)}")

        # Check for content
        if "completion" in response_body:
            logger.info(f"Completion: {response_body['completion']}")
        elif "generation" in response_body:
            logger.info(f"Generation: {response_body['generation']}")
        elif "generations" in response_body and len(response_body["generations"]) > 0:
            logger.info(f"Generation: {response_body['generations'][0].get('text', '')}")
        elif "generated_text" in response_body:
            logger.info(f"Generated text: {response_body['generated_text']}")
        elif "choices" in response_body and len(response_body["choices"]) > 0:
            logger.info(f"Choice: {response_body['choices'][0].get('text', '')}")
        else:
            logger.warning(f"Unknown response format: {response_body}")

        return True

    except Exception as e:
        logger.error(f"Error invoking model: {str(e)}")
        return False

# Test streaming invocation
def test_streaming_invocation():
    """Test streaming invocation of the custom model."""
    logger.info(f"Testing streaming invocation of {CUSTOM_MODEL_ID}")

    # Create a request body
    body = {
        "prompt": "Human: Tell me about quantum computing in simple terms.\nAssistant:",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    try:
        # Invoke the model with streaming
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId=CUSTOM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        # Process the streaming response
        stream = response.get("body")
        full_response = ""
        
        logger.info("Streaming chunks:")
        for chunk in stream:
            # Log raw chunk info (only first few to avoid flooding logs)
            if full_response == "":
                logger.info(f"Raw chunk: {chunk}")
                
            # For the imported models, the contentType might be missing from each chunk
            # but we can still process the chunks
                
            try:
                chunk_bytes = chunk.get("chunk", {}).get("bytes")
                if not chunk_bytes:
                    logger.warning("No bytes in chunk")
                    continue
                    
                chunk_body = json.loads(chunk_bytes.decode())
                logger.info(f"Chunk body: {json.dumps(chunk_body)}")
                
                # Extract text based on common response formats
                text = None
                if "completion" in chunk_body:
                    text = chunk_body["completion"]
                elif "generation" in chunk_body:
                    text = chunk_body["generation"]
                    logger.info(f"Found generation field: {text}")
                elif "delta" in chunk_body:
                    text = chunk_body["delta"]
                elif "content" in chunk_body:
                    text = chunk_body["content"]
                elif "text" in chunk_body:
                    text = chunk_body["text"]
                
                if text:
                    full_response += text
                    logger.info(f"Extracted text: {text}")
                else:
                    logger.warning(f"No text extracted from chunk: {chunk_body}")
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
        
        logger.info(f"Full response: {full_response}")
        return len(full_response) > 0

    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        return False

if __name__ == "__main__":
    # Run both tests
    direct_result = test_direct_invocation()
    streaming_result = test_streaming_invocation()
    
    if direct_result and streaming_result:
        logger.info("All tests passed")
    elif direct_result:
        logger.info("Direct invocation test passed, streaming test failed")
    elif streaming_result:
        logger.info("Streaming test passed, direct invocation test failed")
    else:
        logger.info("All tests failed")