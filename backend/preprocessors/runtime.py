import os

import cv2
import numpy as np
import torch

from backend import resources
from backend import utils as backend_utils

DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_MODEL_CACHE = {
    "Depth": {"path": None, "model": None},
}


def _offload_model(model):
    if model is None:
        return
    try:
        model.to(resources.unet_offload_device())
    except Exception:
        pass


def offload_cached_preprocessors():
    for entry in _MODEL_CACHE.values():
        _offload_model(entry["model"])
    resources.soft_empty_cache()


def apply_residency_policy(mode='offload'):
    loaded_entries = [entry for entry in _MODEL_CACHE.values() if entry['model'] is not None]
    actions = {'mode': mode, 'count': len(loaded_entries)}
    for entry in loaded_entries:
        _offload_model(entry['model'])
        if mode == 'destroy':
            entry['model'] = None
            entry['path'] = None
    if loaded_entries and mode in ('offload', 'destroy'):
        resources.soft_empty_cache(force=(mode == 'destroy'))
    return actions


def _prepare_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def _get_depth_config(model_path):
    name = os.path.basename(model_path).lower()
    for key, cfg in DEPTH_MODEL_CONFIGS.items():
        if key in name:
            return cfg
    return DEPTH_MODEL_CONFIGS["vitl"]


def _get_cached_model(method, model_path, loader):
    entry = _MODEL_CACHE[method]
    if entry["model"] is None or entry["path"] != model_path:
        _offload_model(entry["model"])
        entry["model"] = loader(model_path)
        entry["path"] = model_path
    return entry["model"]


def _load_depth_model(model_path):
    from .depth_anything_v2 import DepthAnythingV2

    model = DepthAnythingV2(**_get_depth_config(model_path))
    state_dict = _prepare_state_dict(backend_utils.load_torch_file(model_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model








def _normalize_depth_input(image):
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def preprocess_depth(image, model_path, input_size=518, max_depth=20.0):
    model = _get_cached_model("Depth", model_path, _load_depth_model)
    device = resources.get_torch_device()
    model = model.to(device)

    with torch.no_grad():
        depth_np = model.infer_image(image, input_size=input_size, max_depth=max_depth)

    depth_np = depth_np.astype(np.float32)
    depth_min = float(depth_np.min())
    depth_max = float(depth_np.max())
    if depth_max > depth_min:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_np = np.zeros_like(depth_np, dtype=np.float32)

    _offload_model(model)
    result = np.repeat((depth_np.clip(0, 1) * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    return result


def _safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y








def run_structural_preprocessor(method, image, model_path=None):
    if method == "Depth":
        if not model_path:
            raise FileNotFoundError("Depth preprocessor path is missing")
        return preprocess_depth(image, model_path)
    raise KeyError(f"Unsupported structural preprocessor method: {method}")
