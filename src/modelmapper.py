import json
from pathlib import Path

_model_map = None

def load_model_map():
    with open("./data/model_map.json", "r") as f:
        _model_map = json.load(f)

def get_model(key):
    if key in _model_map:
        return _model_map[key]
    
    return key