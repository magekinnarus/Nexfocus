from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import torch

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest, StructuralHintArtifact
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

class StreamingStructuralControlWorker:
    """Worker managing the lifecycle, copy-on-request, and CPU offloading of structural ControlNet models."""

    # Class-level cache to persist loaded support models across requests
    _SUPPORT_MODEL_CACHE: Dict[str, Any] = {}

    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.active_copied_controls: List[Any] = []

    @classmethod
    def clear_support_cache(cls) -> None:
        stale = list(cls._SUPPORT_MODEL_CACHE.values())
        cls._SUPPORT_MODEL_CACHE.clear()
        for model in stale:
            if hasattr(model, "cleanup"):
                try:
                    model.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up controlnet in shared cache clear: {e}")
            patcher = getattr(model, 'control_model_wrapped', None)
            if patcher is not None:
                try:
                    patcher.detach()
                except Exception as e:
                    logger.warning(f"Error detaching controlnet patcher in shared cache clear: {e}")

    def attach_conditioning(
        self,
        conditioning: Dict[str, Any],
        prepared_hints: Dict[int, StructuralHintArtifact]
    ) -> Dict[str, Any]:
        """Loads, detaches/parks, copies, and chains structural ControlNet models onto the conditioning."""
        if not self.request.structural_controls:
            return conditioning

        log_telemetry("structural_control_attach_begin")
        start_time = time.perf_counter()

        # 1. Invalidate and evict stale support models not requested in this request
        requested_sha256s = {desc.checkpoint_sha256 for desc in self.request.structural_controls}
        stale_keys = [k for k in self._SUPPORT_MODEL_CACHE.keys() if k not in requested_sha256s]
        for k in stale_keys:
            log_telemetry("structural_control_evict", f"sha256={k[:12]}")
            model = self._SUPPORT_MODEL_CACHE.pop(k)
            if hasattr(model, "cleanup"):
                try:
                    model.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up stale controlnet: {e}")
            patcher = getattr(model, 'control_model_wrapped', None)
            if patcher is not None:
                try:
                    patcher.detach()
                except Exception as e:
                    logger.warning(f"Error detaching stale controlnet patcher: {e}")

        # 2. Retrieve or load active control models
        loaded_controls_in_order: List[Tuple[Any, Any, StructuralHintArtifact]] = []
        for desc in self.request.structural_controls:
            model = self._SUPPORT_MODEL_CACHE.get(desc.checkpoint_sha256)
            if model is None:
                log_telemetry("structural_control_load_begin", f"path={desc.checkpoint_path.name}")
                from backend.controlnet import load_controlnet
                try:
                    model = load_controlnet(str(desc.checkpoint_path))
                except Exception as e:
                    logger.error(f"Failed to load controlnet checkpoint: {e}")
                    raise RuntimeError(f"Failed to load controlnet checkpoint: {e}") from e

                # In streaming posture, reusable structural model/support state is parked on CPU/offload.
                patcher = getattr(model, 'control_model_wrapped', None)
                if patcher is not None:
                    try:
                        patcher.detach()
                    except Exception as e:
                        logger.warning(f"Failed to detach freshly loaded controlnet patcher: {e}")

                self._SUPPORT_MODEL_CACHE[desc.checkpoint_sha256] = model
                log_telemetry("structural_control_load_complete", f"sha256={desc.checkpoint_sha256[:12]}")
            else:
                log_telemetry("structural_control_hit", f"sha256={desc.checkpoint_sha256[:12]}")

            # Retrieve hint artifact
            hint_art = prepared_hints.get(desc.slot_index)
            if hint_art is None:
                raise RuntimeError(f"Missing prepared hint artifact for slot {desc.slot_index}")

            loaded_controls_in_order.append((model, desc, hint_art))

        # 3. Chain and attach to conditioning payloads (positive and negative)
        new_positive = []
        new_negative = []
        chain_caches: List[Dict[Any, Any]] = [dict() for _ in loaded_controls_in_order]

        for cond_name, cond_list in [("positive", conditioning.get("positive", [])), ("negative", conditioning.get("negative", []))]:
            c = []
            for t in cond_list:
                d = t[1].copy()
                current_cnet = d.get('control', None)

                for idx, (model, desc, hint_art) in enumerate(loaded_controls_in_order):
                    chain_cache = chain_caches[idx]
                    if current_cnet in chain_cache:
                        copied_control = chain_cache[current_cnet]
                    else:
                        copied_control = model.copy()
                        copied_control.set_cond_hint(
                            hint_art.hint_tensor,
                            desc.weight,
                            (desc.start_percent, desc.end_percent)
                        )
                        copied_control.set_previous_controlnet(current_cnet)
                        chain_cache[current_cnet] = copied_control
                        self.active_copied_controls.append(copied_control)
                    current_cnet = copied_control

                d['control'] = current_cnet
                d['control_apply_to_uncond'] = False
                c.append([t[0], d])

            if cond_name == "positive":
                new_positive = c
            else:
                new_negative = c

        duration = time.perf_counter() - start_time
        log_telemetry("structural_control_attach_complete", f"duration={duration:.3f}s")

        return {
            "positive": new_positive,
            "negative": new_negative,
        }

    def end(self) -> None:
        """Cleans up request-local run-bound states and detaches cached model patchers."""
        log_telemetry("structural_control_end_begin")
        for copied_control in self.active_copied_controls:
            if hasattr(copied_control, "cleanup"):
                try:
                    copied_control.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up copied control: {e}")
        self.active_copied_controls.clear()

        # Detach support models in cache to ensure they are offloaded to CPU
        for sha, model in self._SUPPORT_MODEL_CACHE.items():
            patcher = getattr(model, 'control_model_wrapped', None)
            if patcher is not None:
                try:
                    patcher.detach()
                except Exception as e:
                    logger.warning(f"Error detaching cached control model patcher: {e}")

        log_telemetry("structural_control_end_complete")

    def release_owned_resources(self) -> None:
        """Tears down all cached support models."""
        log_telemetry("structural_control_release_begin")
        self.end()
        self.clear_support_cache()
        log_telemetry("structural_control_release_complete")
