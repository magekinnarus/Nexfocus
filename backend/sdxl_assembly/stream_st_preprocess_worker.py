from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

import modules.flags as flags
from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    StructuralHintArtifact,
    SDXLAssemblyValidationError,
)
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

def get_preprocess_cache_key(desc: Any) -> str:
    import hashlib
    import json
    # Cache key depends on control image fingerprint, type, preprocessor ID, params, and resolution
    payload = {
        "image_fingerprint": desc.image_fingerprint,
        "control_type": desc.control_type,
        "preprocessor_id": desc.preprocessor_id,
        "preprocessor_path": str(desc.preprocessor_path) if desc.preprocessor_path is not None else None,
        "preprocessor_params": desc.preprocessor_params,
        "target_width": desc.target_width,
        "target_height": desc.target_height,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

class StreamingStructuralPreprocessWorker:
    """Worker managing the lifecycle of structural ControlNet preprocessors and CPU hint caching."""

    # Class-level cache to persist cached preprocessed hints across requests/workers
    _PREPROCESS_CACHE: OrderedDict[str, StructuralHintArtifact] = OrderedDict()
    _PREPROCESS_CACHE_LIMIT = 8

    @classmethod
    def clear_preprocess_cache(cls) -> None:
        cls._PREPROCESS_CACHE.clear()

    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request

    def preprocess(self) -> Dict[int, StructuralHintArtifact]:
        """Preprocesses structural control inputs, returning CPU-parked hint artifacts."""
        prepared_hints: Dict[int, StructuralHintArtifact] = {}
        
        for desc in self.request.structural_controls:
            # 1. Validation
            if len(desc.unsupported_mode_errors) > 0:
                raise SDXLAssemblyValidationError(
                    f"Structural control descriptor in slot {desc.slot_index} has unsupported modes: "
                    f"{desc.unsupported_mode_errors}"
                )

            cache_key = get_preprocess_cache_key(desc)
            cached = self._PREPROCESS_CACHE.get(cache_key)
            if cached is not None:
                self._PREPROCESS_CACHE.move_to_end(cache_key)
                log_telemetry("structural_preprocess_hit", f"slot={desc.slot_index} key={cache_key[:12]}")
                from dataclasses import replace
                prepared_hints[desc.slot_index] = replace(cached, cache_hit=True)
                continue

            log_telemetry("structural_preprocess_miss", f"slot={desc.slot_index} key={cache_key[:12]}")
            log_telemetry("structural_preprocess_begin", f"slot={desc.slot_index}")

            start_time = time.perf_counter()

            # Convert descriptor's image_pixels (original) back to HWC numpy format for preprocessor
            pixels_numpy = desc.image_pixels.numpy()
            if pixels_numpy.ndim == 4:
                pixels_numpy = pixels_numpy[0]
            # Convert float32 [0.0, 1.0] back to uint8 [0, 255]
            pixels_uint8 = (pixels_numpy * 255.0).round().astype(np.uint8)

            # Resize source image to target width/height
            from modules.util import resize_image
            resized_pixels = resize_image(pixels_uint8, width=desc.target_width, height=desc.target_height)

            # Run preprocessor if not skipped
            cn_img = resized_pixels
            if desc.preprocessor_id != "None" and desc.preprocessor_id is not None:
                from backend.preprocessors.runtime import run_structural_preprocessor
                from extras import preprocessors

                try:
                    if desc.control_type == flags.cn_canny:
                        low = desc.preprocessor_params.get("low_threshold", 64)
                        high = desc.preprocessor_params.get("high_threshold", 128)
                        cn_img = preprocessors.canny_pyramid(resized_pixels, low, high)
                    elif desc.control_type == flags.cn_cpds:
                        cn_img = preprocessors.cpds(resized_pixels)
                    elif desc.control_type == flags.cn_depth:
                        cn_img = run_structural_preprocessor(flags.cn_depth, resized_pixels, str(desc.preprocessor_path))
                    else:
                        raise KeyError(f"Unknown structural type: {desc.control_type}")
                finally:
                    # In streaming posture, preprocessor models are not warm-retained: apply residency destroy
                    from backend.preprocessors.runtime import apply_residency_policy
                    apply_residency_policy("destroy")

            # Normalize and create CPU tensor
            from modules.core import numpy_to_pytorch
            hint_tensor = numpy_to_pytorch(cn_img)  # [1, H, W, C]
            # Move dim to [1, C, H, W]
            hint_tensor = hint_tensor.movedim(-1, 1).cpu()

            # Compute hash of the preprocessed hint
            import hashlib
            hint_hash = hashlib.sha256(hint_tensor.numpy().tobytes()).hexdigest()

            duration = time.perf_counter() - start_time
            log_telemetry("structural_preprocess_complete", f"slot={desc.slot_index} duration={duration:.3f}s")

            artifact = StructuralHintArtifact(
                slot_index=desc.slot_index,
                control_type=desc.control_type,
                hint_tensor=hint_tensor,
                hint_fingerprint=hint_hash,
                cache_hit=False,
                preprocess_wall=duration,
            )

            self._PREPROCESS_CACHE[cache_key] = artifact
            self._PREPROCESS_CACHE.move_to_end(cache_key)
            while len(self._PREPROCESS_CACHE) > self._PREPROCESS_CACHE_LIMIT:
                self._PREPROCESS_CACHE.popitem(last=False)

            prepared_hints[desc.slot_index] = artifact

        return prepared_hints

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        pass
