from __future__ import annotations

import time
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    PreparedSpatialContext,
    SpatialAssemblyArtifacts
)
from backend.sdxl_assembly.progress import log_telemetry
from backend.sdxl_assembly.runtime_state import acquire_vae_component

logger = logging.getLogger(__name__)

def _build_denoise_mask(mask: torch.Tensor, latent_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"Expected a [B, H, W] mask when building the denoise mask, got shape {tuple(mask.shape)}.")
    latent_h = int(latent_shape[-2])
    latent_w = int(latent_shape[-1])
    pooled = torch.nn.functional.max_pool2d(mask[:, None, :, :], kernel_size=8, stride=8)
    if pooled.shape[-2] != latent_h or pooled.shape[-1] != latent_w:
        pooled = torch.nn.functional.interpolate(
            pooled,
            size=(latent_h, latent_w),
            mode="nearest",
        )
    return (pooled > 0.5).to(dtype=torch.float32).detach().cpu()


def _clone_cpu_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()

class VaeEncodeWorker:
    """Worker representing VaeEncodeWorker (loads/unloads VAE transiently for encoding)."""
    
    # Class-level cache to persist cached latents across requests/workers
    _ENCODE_CACHE: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    _ENCODE_CACHE_LIMIT = 8
    
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.vae = None

    def encode(self, prepared: PreparedSpatialContext) -> SpatialAssemblyArtifacts:
        """Encodes prepared spatial pixels to latents under worker ownership, returning CPU-parked artifacts."""
        # 1. Input validation & limit checks
        if prepared.bb_pixels.shape[0] != 1:
            raise ValueError(
                f"VAE encode worker currently supports batch_size=1 only, got {prepared.bb_pixels.shape[0]}."
            )
            
        # 2. Resolve VAE identity
        vae_identity = self.request.vae.sha256 if self.request.vae else f"embedded:{self.request.checkpoint.sha256}"
        
        # 3. Cache lookup
        cache_key = prepared.get_cache_key(vae_identity)
        
        cached = self._ENCODE_CACHE.get(cache_key)
        if cached is not None:
            self._ENCODE_CACHE.move_to_end(cache_key)
            log_telemetry("spatial_prepare_hit", f"key={cache_key[:12]}")
            
            # Deep clone cached tensors to CPU to prevent mutation of cache entries
            route_latent = _clone_cpu_tensor(cached["route_latent"])
            masked_latent = _clone_cpu_tensor(cached.get("masked_latent"))
            bb_latent = _clone_cpu_tensor(cached.get("bb_latent"))
            denoise_mask = _clone_cpu_tensor(cached.get("denoise_mask"))
            blend_mask = _clone_cpu_tensor(cached.get("blend_mask"))
            
            return SpatialAssemblyArtifacts(
                route_latent=route_latent,
                masked_latent=masked_latent,
                bb_latent=bb_latent,
                denoise_mask=denoise_mask,
                blend_mask=blend_mask,
                source_fingerprint=cached["source_fingerprint"],
                image_fingerprint=prepared.image_fingerprint,
                mask_fingerprint=prepared.mask_fingerprint,
                route_latent_fingerprint=cached["route_latent_fingerprint"],
                masked_latent_fingerprint=cached.get("masked_latent_fingerprint"),
                bb_latent_fingerprint=cached.get("bb_latent_fingerprint"),
                denoise_mask_fingerprint=cached.get("denoise_mask_fingerprint"),
                blend_mask_fingerprint=cached.get("blend_mask_fingerprint"),
                bbox=prepared.bbox,
                bbox_area_ratio=prepared.bbox_area_ratio,
                mask_coverage=prepared.mask_coverage,
                cache_hit=True,
                encode_wall=0.0,
            )

        log_telemetry("spatial_prepare_miss", f"key={cache_key[:12]}")
        log_telemetry("vae_encode_begin")
        
        # 4. Acquire VAE
        self.vae = acquire_vae_component(self.request)
        if self.vae is None:
            raise RuntimeError("VAE encode worker failed to acquire base VAE component.")
            
        encode_start = time.perf_counter()
        device = torch.device(self.request.device)
        
        route_latent = None
        masked_latent = None
        bb_latent = None
        denoise_mask = None
        
        try:
            # 5. Attach VAE
            self.vae.patcher.patch_model(device_to=device, lowvram_model_memory=0)
            if hasattr(self.vae, "first_stage_model"):
                self.vae.first_stage_model.to(device=device)
                
            # Helper to encode pixels on device
            def _encode_pixels(pixels_cpu: torch.Tensor) -> torch.Tensor:
                pixels_gpu = pixels_cpu.to(device=device)
                return self.vae.encode(pixels_gpu)["samples"].detach().cpu()
            
            with torch.inference_mode():
                # Perform VAE encoding based on the mode
                mode = prepared.mode
                
                if mode == "image":
                    if prepared.original_mask is not None:
                        # Encode full original pixels
                        route_latent = _encode_pixels(prepared.original_pixels)
                        
                        # Encode full masked pixels
                        mask_unsqueezed = prepared.original_mask.unsqueeze(-1)
                        masked_pixels = prepared.original_pixels * (1.0 - mask_unsqueezed) + 0.5 * mask_unsqueezed
                        masked_latent = _encode_pixels(masked_pixels)
                        
                        # Encode cropped region
                        bb_latent = _encode_pixels(prepared.bb_pixels)
                    else:
                        route_latent = _encode_pixels(prepared.bb_pixels)
                        masked_latent = None
                        bb_latent = None
                elif mode in ("inpaint", "outpaint"):
                    # For inpaint and outpaint, the latent is computed from bb_pixels
                    route_latent = _encode_pixels(prepared.bb_pixels)
                    masked_latent = None
                    bb_latent = route_latent

                # Compute denoise mask if bb_mask is present
                if prepared.bb_mask is not None:
                    denoise_mask = _build_denoise_mask(prepared.bb_mask, route_latent.shape)
                else:
                    denoise_mask = None
                    
        finally:
            # 6. Deterministic eject/release VAE
            from backend import resources
            resources.eject_model(getattr(self.vae, "patcher", None))
            log_telemetry("vae_encode_eject")
            self.vae = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        encode_wall = time.perf_counter() - encode_start
        log_telemetry("vae_encode_complete", f"duration={encode_wall:.3f}s")

        # 7. Compute fingerprints
        import hashlib
        def _hash_tensor(t: torch.Tensor | None) -> str:
            if t is None:
                return ""
            return hashlib.sha256(t.numpy().tobytes()).hexdigest()

        route_latent_fingerprint = _hash_tensor(route_latent)
        masked_latent_fingerprint = _hash_tensor(masked_latent) if masked_latent is not None else None
        bb_latent_fingerprint = _hash_tensor(bb_latent) if bb_latent is not None else None
        denoise_mask_fingerprint = _hash_tensor(denoise_mask) if denoise_mask is not None else None
        blend_mask_fingerprint = _hash_tensor(prepared.blend_mask) if prepared.blend_mask is not None else None

        # Store in cache
        cache_entry = {
            "route_latent": _clone_cpu_tensor(route_latent),
            "masked_latent": _clone_cpu_tensor(masked_latent),
            "bb_latent": _clone_cpu_tensor(bb_latent),
            "denoise_mask": _clone_cpu_tensor(denoise_mask),
            "blend_mask": _clone_cpu_tensor(prepared.blend_mask),
            "source_fingerprint": prepared.image_fingerprint,
            "route_latent_fingerprint": route_latent_fingerprint,
            "masked_latent_fingerprint": masked_latent_fingerprint,
            "bb_latent_fingerprint": bb_latent_fingerprint,
            "denoise_mask_fingerprint": denoise_mask_fingerprint,
            "blend_mask_fingerprint": blend_mask_fingerprint,
        }
        self._ENCODE_CACHE[cache_key] = cache_entry
        self._ENCODE_CACHE.move_to_end(cache_key)
        while len(self._ENCODE_CACHE) > self._ENCODE_CACHE_LIMIT:
            self._ENCODE_CACHE.popitem(last=False)

        return SpatialAssemblyArtifacts(
            route_latent=route_latent,
            masked_latent=masked_latent,
            bb_latent=bb_latent,
            denoise_mask=denoise_mask,
            blend_mask=prepared.blend_mask,
            source_fingerprint=prepared.image_fingerprint,
            image_fingerprint=prepared.image_fingerprint,
            mask_fingerprint=prepared.mask_fingerprint,
            route_latent_fingerprint=route_latent_fingerprint,
            masked_latent_fingerprint=masked_latent_fingerprint,
            bb_latent_fingerprint=bb_latent_fingerprint,
            denoise_mask_fingerprint=denoise_mask_fingerprint,
            blend_mask_fingerprint=blend_mask_fingerprint,
            bbox=prepared.bbox,
            bbox_area_ratio=prepared.bbox_area_ratio,
            mask_coverage=prepared.mask_coverage,
            cache_hit=False,
            encode_wall=float(encode_wall),
        )

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        log_telemetry("vae_encode_release")
        self.vae = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
