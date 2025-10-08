#!/bin/bash
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8080}"