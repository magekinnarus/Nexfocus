from __future__ import annotations

import time
import logging
from typing import Any, Tuple
from types import SimpleNamespace
import numpy as np
import torch

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.sdxl_assembly.runtime_state import acquire_vae_component

logger = logging.getLogger(__name__)

class TransientVaeDecodeWorker:
    """Worker representing TransientVaeDecodeWorker (loads/unloads VAE transiently for decoding)."""
    
    def __init__(self, request: SDXLAssemblyRequest) -> None:
        self.request = request
        self.vae = None

    def prepare_latents(self, device: torch.device) -> Any:
        """Creates the initial zero latent tensor for txt2img on the execution device."""
        latent_h = max(1, self.request.height // 8)
        latent_w = max(1, self.request.width // 8)
        samples = torch.zeros((1, 4, latent_h, latent_w), dtype=torch.float16, device=device)
        return SimpleNamespace(samples=samples, fingerprint="initial_zero_latent")

    def decode(self, latent: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, float, float]:
        """Decodes latents to HWC RGB numpy image.
        
        Loads/attaches VAE transiently, then releases/ejects the VAE state from GPU.
        """
        if not torch.isfinite(latent).all():
            raise RuntimeError("VAE decode rejected a non-finite SDXL latent.")

        # 1. Acquire VAE from CPU-pinned components
        self.vae = acquire_vae_component(self.request)
        if self.vae is None:
            raise RuntimeError("Transient VAE decode worker failed to acquire base VAE component.")
            
        log_telemetry("vae_decode_begin")
        
        # 2. Attach VAE to execution device
        attach_start = time.perf_counter()
        self.vae.patcher.patch_model(device_to=device, lowvram_model_memory=0)
        if hasattr(self.vae, "first_stage_model"):
            # Decode already has a downstream fp32 safeguard, but making the
            # attach contract explicit keeps the live dtype truthful in logs
            # and avoids ambiguous mixed-precision regressions.
            self.vae.first_stage_model.to(device=device, dtype=torch.float32)
        live_param = next(self.vae.first_stage_model.parameters(), None)
        live_device = live_param.device if isinstance(live_param, torch.Tensor) else device
        live_dtype = live_param.dtype if isinstance(live_param, torch.Tensor) else torch.float32
        attach_time = time.perf_counter() - attach_start
        log_telemetry(
            "vae_decode_attached",
            f"route={self.request.route_id} live_device={live_device} live_dtype={live_dtype}",
        )

        from backend import decode
        import modules.core as core
        
        # 3. Decode
        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                decoded_patch = decode.decode_preloaded_vae(self.vae, latent, tiled=self.request.tiled)
                if not torch.isfinite(decoded_patch).all():
                    log_telemetry("vae_decode_nonfinite", f"route={self.request.route_id}")
                    raise RuntimeError(
                        "VAE decode produced non-finite pixels; the invalid image was not converted or saved."
                    )
                output_image = core.pytorch_to_numpy(decoded_patch)[0]
        finally:
            # 4. Release/eject under worker ownership
            from backend import resources
            resources.eject_model(getattr(self.vae, "patcher", None))
            self.vae = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                 
        decode_time = time.perf_counter() - decode_start
        log_telemetry("vae_decode_complete", f"attach={attach_time:.3f}s decode={decode_time:.3f}s")
        return output_image, attach_time, decode_time

    def teardown_assembly_order(self) -> None:
        """Extension point for tracking teardown order."""
        self.vae = None
