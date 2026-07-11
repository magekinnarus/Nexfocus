import os
import re
from pathlib import Path

from modules.config import path_upscale_models

_MODEL_SCALE_METADATA_CACHE: dict[tuple[str, int, int], int] = {}

def list_available_models():
    """Scan models/upscale_models/ for .pth files."""
    if not path_upscale_models:
        return []
    
    models = []
    for folder in path_upscale_models:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.pth') or f.lower().endswith('.safetensors'):
                    models.append(f)
    return sorted(list(set(models)))

def get_model_scale(model):
    """Auto-detect scale factor from model architecture."""
    # Priority 1: Model attribute
    for attr in ['scale', 'upscale', 'upscale_factor', 'upsampler_scale']:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (int, float)):
                return int(val)
    
    # Priority 2: Inferred from architecture if possible
    # (Some architectures like ESRGAN-2c2 might have complex logic in their __init__)
    
    # Priority 3: Fallback to name-based if it's a wrapper or if attribute missing
    return 4  # Default fallback if unknown


def _resolve_model_path(model_name: str) -> Path:
    for folder in path_upscale_models:
        candidate = Path(folder) / model_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Upscale model not found: {model_name}")


def get_model_scale_for_name(model_name: str) -> int:
    """Return scalar scale metadata without loading or attaching the model.

    This function is called by the live UI while another task may be sampling.
    Loading a GAN only to render the scale label can borrow VRAM outside the
    auxiliary lease and overlap Flux/SDXL execution, so unknown names use the
    established scale-4 fallback.
    """
    model_path = _resolve_model_path(model_name)
    stat = model_path.stat()
    cache_key = (str(model_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached_scale = _MODEL_SCALE_METADATA_CACHE.get(cache_key)
    if cached_scale is not None:
        return cached_scale

    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)x", model_path.name, flags=re.IGNORECASE)
    native_scale = int(round(float(match.group(1)))) if match else 4
    native_scale = max(1, native_scale)

    _MODEL_SCALE_METADATA_CACHE.clear()
    _MODEL_SCALE_METADATA_CACHE[cache_key] = int(native_scale)
    return int(native_scale)


def clear_model_cache():
    """Compatibility name retained for clearing metadata-only scale entries."""
    _MODEL_SCALE_METADATA_CACHE.clear()


def load_model(model_name):
    """Reject the retired direct-load API so this bridge cannot own live state."""
    raise RuntimeError(
        "modules.upscaler.load_model no longer owns GAN model state; "
        "use GanUpscaleWorker or get_model_scale_for_name()."
    )


def perform_upscale(img, model_name=None, scale_override=None, retain_warm=False):
    """Compatibility bridge into the worker-owned ephemeral GAN path.

    ``retain_warm`` remains in the signature for callers but never grants this
    bridge authority to clear or retain another assembly's artifacts.
    """
    if img is None:
        return None

    from backend.auxiliary_workers.gan_upscale_worker import run_gan_upscale

    return run_gan_upscale(
        img,
        model_name=model_name,
        scale_override=scale_override,
    )
