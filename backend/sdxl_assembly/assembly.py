from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional
import numpy as np
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLAssemblyResult,
    SDXLRuntimeIdentity,
)
from backend.sdxl_assembly.progress import log_telemetry

logger = logging.getLogger(__name__)

class SDXLAssembly:
    """Coordinates worker execution steps, execution order, and strict teardown."""

    def __init__(
        self,
        unet_spine: Any,
        text_worker: Any,
        vae_worker: Any,
        lora_worker: Any,
    ) -> None:
        self.unet_spine = unet_spine
        self.text_worker = text_worker
        self.vae_worker = vae_worker
        self.lora_worker = lora_worker

    def execute(self, request: SDXLAssemblyRequest, callback: Optional[Any] = None) -> SDXLAssemblyResult:
        """Executes the pipeline steps in strict chronological order."""
        # 1. Static and posture validation
        request.validate()
        
        timings: Dict[str, float] = {}
        device = torch.device(request.device)

        # 2. Materialize LoRA patches first
        lora_start = time.perf_counter()
        patches = self.lora_worker.materialize_patches()
        timings["lora_patch"] = time.perf_counter() - lora_start

        # 3. Resolve text conditioning (avoid holding extra latent memory)
        text_start = time.perf_counter()
        conditioning = self.text_worker.get_conditioning()
        timings["text_encode"] = time.perf_counter() - text_start
        if bool(request.metadata.get("release_text_encoder_after_task", False)):
            text_release_start = time.perf_counter()
            if hasattr(self.text_worker, "teardown_assembly_order"):
                self.text_worker.teardown_assembly_order()
            timings["text_release"] = time.perf_counter() - text_release_start

        # 4. Coordinate VAE latent preparation
        vae_start = time.perf_counter()
        latent_bundle = self.vae_worker.prepare_latents(device)
        timings["vae_prep"] = time.perf_counter() - vae_start

        # Extension Point: Pre-diffusion ControlNet preprocessing artifacts & application
        self._prepare_controlnet_artifacts(request)

        # 5. Coordinate UNet spine denoise
        unet_start = time.perf_counter()
        self.unet_spine.start()
        timings["unet_start"] = time.perf_counter() - unet_start

        try:
            denoise_start = time.perf_counter()
            samples = self.unet_spine.denoise(
                latent_bundle.samples, conditioning, callback=callback
            )
            timings["unet_denoise"] = time.perf_counter() - denoise_start
        finally:
            self.unet_spine.end()

        # 6. Decode results using VAE worker
        decode_start = time.perf_counter()
        output_image, load_time, decode_time = self.vae_worker.decode(samples, device)
        timings["vae_decode"] = time.perf_counter() - decode_start

        runtime_identity = SDXLRuntimeIdentity(
            checkpoint=request.checkpoint,
            vae=request.vae,
            unet_posture=request.unet_posture,
            clip_posture=request.clip_posture,
            vae_posture=request.vae_posture,
            lora_posture=request.lora_posture,
        )

        return SDXLAssemblyResult(
            output_image=output_image,
            seed=request.seed,
            width=output_image.shape[1],
            height=output_image.shape[0],
            runtime_identity=runtime_identity,
            timings=timings,
            metadata={
                "runtime_identity": runtime_identity.as_dict(),
            },
        )

    # Extension Points for ControlNet
    def _prepare_controlnet_artifacts(self, request: SDXLAssemblyRequest) -> None:
        """Extension point for structural ControlNet preprocessing artifacts and control-model application."""
        pass

    def close(self) -> None:
        """Deterministically tears down the assembly and its components in strict order."""
        log_telemetry("cleanup_begin", "reason=assembly_close")
        
        # 1. TransientVaeWorker unloads VAE tensors first
        if hasattr(self.vae_worker, "teardown_assembly_order"):
            self.vae_worker.teardown_assembly_order()
            
        # 2. LoraPatchWorker rolls back or detaches patch weights
        if hasattr(self.lora_worker, "teardown_assembly_order"):
            self.lora_worker.teardown_assembly_order()
            
        # 3. TextEncoderWorker releases CPU/GPU pinned models
        if hasattr(self.text_worker, "teardown_assembly_order"):
            self.text_worker.teardown_assembly_order()
            
        # 4. UNetSpine unloads or deallocates weights
        if hasattr(self.unet_spine, "teardown_assembly_order"):
            self.unet_spine.teardown_assembly_order()
            
        log_telemetry("cleanup_complete", "reason=assembly_close")
