#!/usr/bin/env python3

import requests
import json

# Configuration
BASE_URL = "http://localhost:8080/api/v1"
API_KEY = "bedrock"

def test_models_endpoint():
    """Test the models endpoint and print results"""
    url = f"{BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        models = response.json()
        print(f"✅ Successfully retrieved {len(models.get('data', []))} models")
        print("\n📋 Available Models:")
        print("-" * 50)
        
        for model in models.get('data', []):
            # print(f"• {model.get('id', 'Unknown')}")
            print(model)
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the container running on port 8080?")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Bedrock Access Gateway...")
    test_models_endpoint()
