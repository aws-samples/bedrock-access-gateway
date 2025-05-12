import os
import json
from pathlib import Path

USE_MODEL_MAPPING = os.getenv("USE_MODEL_MAPPING", "true").lower() == "true"

_model_map = None

def load_model_map():
    global _model_map
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    modelmap_path = os.path.join(BASE_DIR, "../data/modelmap.json")
    with open(modelmap_path, "r") as f:
        _model_map = json.load(f)

def get_model(key):
    if _model_map and key in _model_map:
        return _model_map[key]

    return key