import os
import json
from pathlib import Path
from api.setting import FALLBACK_MODEL

_model_map = None

def load_model_map():
    global _model_map
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    modelmap_path = os.path.join(BASE_DIR, "../data/modelmap.json")
    with open(modelmap_path, "r") as f:
        _model_map = json.load(f)

def get_model(provider, model):
    provider = provider.lower()
    model = model.lower().removesuffix(":latest")

    available_models = _model_map.get(provider, {})
    if FALLBACK_MODEL == None or FALLBACK_MODEL.lower() == model:
        return available_models.get(model, model)
    else:
        return available_models.get(model, get_model(provider, FALLBACK_MODEL))
