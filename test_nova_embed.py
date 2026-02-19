#!/usr/bin/env python3
"""
Minimal test script for Nova 2 multimodal embeddings via the bedrock-access-gateway.

Usage:
    # Start the gateway first:
    #   cd src && uvicorn api.app:app --port 8000
    #
    # Then run:
    #   python test_nova_embed.py
    #
    # Or against a deployed gateway:
    #   BASE_URL=https://your-gateway.example.com/api/v1 python test_nova_embed.py

Requires: openai>=1.0.0
    pip install openai
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.environ.get("AWS_BEDROCK_BEARER_TOKEN", "dummy")
MODEL = "amazon.nova-2-multimodal-embeddings-v1:0"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def test_single_text():
    resp = client.embeddings.create(model=MODEL, input="Hello, world!")
    assert len(resp.data) == 1
    assert len(resp.data[0].embedding) == 3072
    print(f"[PASS] single text -> dim={len(resp.data[0].embedding)}")


def test_batch_texts():
    texts = ["semantic search", "vector database", "RAG pipeline"]
    resp = client.embeddings.create(model=MODEL, input=texts)
    assert len(resp.data) == 3
    assert all(len(e.embedding) == 3072 for e in resp.data)
    print(f"[PASS] batch ({len(texts)} texts) -> dim={len(resp.data[0].embedding)}")


def test_custom_dimensions():
    resp = client.embeddings.create(model=MODEL, input="test", dimensions=256)
    assert len(resp.data[0].embedding) == 256
    print(f"[PASS] custom dimensions -> dim={len(resp.data[0].embedding)}")


def test_base64_encoding():
    resp = client.embeddings.create(
        model=MODEL, input="base64 test", encoding_format="base64"
    )
    assert isinstance(resp.data[0].embedding, str)
    print(f"[PASS] base64 encoding -> type={type(resp.data[0].embedding).__name__}")


def test_model_listed():
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert MODEL in ids, f"{MODEL} not found in models list: {ids[:5]}..."
    print(f"[PASS] model listed in /models")


if __name__ == "__main__":
    print(f"Testing against: {BASE_URL}")
    print(f"Model: {MODEL}\n")
    tests = [test_single_text, test_batch_texts, test_custom_dimensions, test_base64_encoding, test_model_listed]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
