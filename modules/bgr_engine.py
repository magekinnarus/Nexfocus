"""Compatibility bridge for the backend-owned background-removal worker.

This module retains file and temporary-output compatibility for older callers.
It deliberately owns no Remover instance or JIT/model cache.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import modules.mask_processing as mask_processing
from backend.auxiliary_workers.background_removal_worker import run_background_removal
from modules.util import HWC3


def load_model(jit: bool = True):
    """Retired compatibility entry point; model lifetime belongs to the worker."""
    raise RuntimeError(
        "modules.bgr_engine.load_model is retired; use "
        "BackgroundRemovalWorker or run_background_removal()."
    )


def remove_background(
    image: np.ndarray,
    threshold: float = 0.5,
    jit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility bridge returning the worker's neutral RGBA/mask arrays."""
    result = run_background_removal(image, threshold=threshold, jit=jit)
    if result is None:
        raise ValueError("Background removal requires an image.")
    return result


def remove_background_from_file(
    filepath: str,
    threshold: float = 0.5,
    jit: bool = True,
) -> tuple[str, str]:
    """Load and persist files around the backend worker without owning a model."""
    with Image.open(filepath) as image:
        image_np = HWC3(np.array(image.convert("RGBA")))

    rgba, mask = remove_background(image_np, threshold=threshold, jit=jit)
    return mask_processing.save_to_temp_png(rgba), mask_processing.save_to_temp_png(mask)


def unload_model() -> None:
    """No-op compatibility hook; every worker tears itself down per request."""
    return None
