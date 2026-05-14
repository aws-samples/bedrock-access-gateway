#!/bin/bash
#
# Prerequisite: Authenticate to AWS via access keys or profile. 
# 
# Runs the uv app with OPENAI_BASE_URL set to http://127.0.0.1:8080/api/v1, and OPENAI_API_KEY set to arbitrary string without validating the key.
#
# Tests with: 
# curl $OPENAI_BASE_URL/chat/completions \             
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "messages": [
#       {
#         "role": "user",
#         "content": "Hello!"
#       }
#     ]
#   }'
#

# Model selection
export DEFAULT_MODEL=${DEFAULT_MODEL:-"us.anthropic.claude-sonnet-4-5-20250929-v1:0"}

export OPENAI_BASE_URL=http://127.0.0.1:8080/api/v1
export OPENAI_API_KEY=accept_all_keys

export ACCEPT_ALL_OPENAI_KEY=true

python -c 'import tiktoken_ext.openai_public as tke; tke.cl100k_base()'

# Runs the uv app
python -m uvicorn api.app:app --host 127.0.0.1 --port 8080
