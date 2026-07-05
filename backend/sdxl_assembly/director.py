from __future__ import annotations

import logging
from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
)
from backend.sdxl_assembly.assembler import SDXLAssemblyAssembler
from backend.sdxl_assembly.assembly import SDXLAssembly

logger = logging.getLogger(__name__)

class SDXLAssemblyDirector:
    """Authoritative director for SDXL Assembly selection and validation."""

    @staticmethod
    def select_assembly(request: SDXLAssemblyRequest) -> SDXLAssembly:
        # Enforce validation of the only supported posture combination in W02
        if request.unet_posture != UNetPostureKind.STREAMING:
            raise NotImplementedError(
                f"UNet posture '{request.unet_posture}' is not supported in W02. "
                f"Only '{UNetPostureKind.STREAMING}' is supported."
            )
        if request.clip_posture != TextEncoderPostureKind.CPU_PINNED:
            raise NotImplementedError(
                f"CLIP posture '{request.clip_posture}' is not supported in W02. "
                f"Only '{TextEncoderPostureKind.CPU_PINNED}' is supported."
            )
        if request.vae_posture != VAEPostureKind.TRANSIENT:
            raise NotImplementedError(
                f"VAE posture '{request.vae_posture}' is not supported in W02. "
                f"Only '{VAEPostureKind.TRANSIENT}' is supported."
            )
        if request.lora_posture != LoraPatchPostureKind.STREAMING:
            raise NotImplementedError(
                f"LoRA posture '{request.lora_posture}' is not supported in W02. "
                f"Only '{LoraPatchPostureKind.STREAMING}' is supported."
            )

        # Retrieve/instantiate posture-specific workers via Assembler.
        cpu_lora_worker = SDXLAssemblyAssembler.acquire_cpu_lora_worker(request)
        unet_spine = SDXLAssemblyAssembler.acquire_unet_spine(request, lora_worker=cpu_lora_worker)
        text_encode_worker = SDXLAssemblyAssembler.acquire_text_encode_worker(request, lora_worker=cpu_lora_worker)
        vae_decode_worker = SDXLAssemblyAssembler.acquire_vae_decode_worker(request)

        spatial_context_worker = None
        vae_encode_worker = None
        if request.spatial_context is not None:
            spatial_context_worker = SDXLAssemblyAssembler.acquire_spatial_context_worker(request)
            vae_encode_worker = SDXLAssemblyAssembler.acquire_vae_encode_worker(request)

        # Acquire structural preprocessor and control workers
        st_preprocess_worker = SDXLAssemblyAssembler.acquire_st_preprocess_worker(request)
        st_control_worker = SDXLAssemblyAssembler.acquire_st_control_worker(request)

        # Acquire contextual control worker
        ctx_control_worker = SDXLAssemblyAssembler.acquire_ctx_control_worker(request)

        # Log selection
        logger.debug(
            "[SDXL Telemetry] assembly_select | unet_posture=%s clip_posture=%s vae_posture=%s lora_posture=%s",
            request.unet_posture,
            request.clip_posture,
            request.vae_posture,
            request.lora_posture,
        )

        return SDXLAssembly(
            unet_spine=unet_spine,
            text_encode_worker=text_encode_worker,
            vae_decode_worker=vae_decode_worker,
            lora_worker=cpu_lora_worker,
            spatial_context_worker=spatial_context_worker,
            vae_encode_worker=vae_encode_worker,
            st_preprocess_worker=st_preprocess_worker,
            st_control_worker=st_control_worker,
            ctx_control_worker=ctx_control_worker,
        )

