import os
import json
from pathlib import Path
from api.setting import USE_FALLBACK_MODEL

_model_map = None

def load_model_map():
    global _model_map
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    modelmap_path = os.path.join(BASE_DIR, "../data/modelmap.json")
    with open(modelmap_path, "r") as f:
        _model_map = json.load(f)

def get_model(provider, region, model):
    provider = provider.lower()
    region = region.lower()
    model = model.lower().removesuffix(":latest")

    available_models = _model_map.get(provider, {}).get(region, {})
    fallback = available_models.get("fallback", "") if USE_FALLBACK_MODEL else model
    return available_models.get(model, fallback)
