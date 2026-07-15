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
        # Enforce validation of the only supported production posture combinations.
        is_w02_streaming = (
            request.unet_posture == UNetPostureKind.STREAMING
            and request.clip_posture == TextEncoderPostureKind.CPU_PINNED
            and request.vae_posture == VAEPostureKind.TRANSIENT
            and request.lora_posture == LoraPatchPostureKind.STREAMING
        )
        is_w12b_resident = (
            request.unet_posture == UNetPostureKind.RESIDENT
            and request.clip_posture == TextEncoderPostureKind.CPU_PINNED
            and request.vae_posture == VAEPostureKind.TRANSIENT
            and request.lora_posture == LoraPatchPostureKind.RESIDENT
        )
        is_w12c_gpu_text = (
            request.unet_posture == UNetPostureKind.RESIDENT
            and request.clip_posture == TextEncoderPostureKind.GPU_PINNED
            and request.vae_posture == VAEPostureKind.TRANSIENT
            and request.lora_posture == LoraPatchPostureKind.RESIDENT
        )

        if not is_w02_streaming and not is_w12b_resident and not is_w12c_gpu_text:
            raise NotImplementedError(
                f"Posture combination (unet={request.unet_posture}, clip={request.clip_posture}, "
                f"vae={request.vae_posture}, lora={request.lora_posture}) is not supported."
            )

        # Retrieve/instantiate posture-specific workers via Assembler.  The
        # resident W12b composition is side-specific: UNet LoRAs compile on GPU,
        # while CPU text keeps CPU CLIP LoRA ownership.
        if request.lora_posture == LoraPatchPostureKind.RESIDENT:
            lora_worker = SDXLAssemblyAssembler.acquire_gpu_lora_worker(request)
            if request.clip_posture == TextEncoderPostureKind.GPU_PINNED:
                text_lora_worker = lora_worker
            else:
                text_lora_worker = SDXLAssemblyAssembler.acquire_cpu_lora_worker(request)
        else:
            lora_worker = SDXLAssemblyAssembler.acquire_cpu_lora_worker(request)
            text_lora_worker = lora_worker

        unet_spine = SDXLAssemblyAssembler.acquire_unet_spine(request, lora_worker=lora_worker)
        text_encode_worker = SDXLAssemblyAssembler.acquire_text_encode_worker(request, lora_worker=text_lora_worker)
        vae_decode_worker = SDXLAssemblyAssembler.acquire_vae_decode_worker(request)

        spatial_context_worker = None
        vae_encode_worker = None
        if request.spatial_context is not None:
            spatial_context_worker = SDXLAssemblyAssembler.acquire_spatial_context_worker(request)
            vae_encode_worker = SDXLAssemblyAssembler.acquire_vae_encode_worker(request)
        elif request.tiled_refinement is not None and request.tiled_refinement.enabled:
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
            lora_worker=lora_worker,
            spatial_context_worker=spatial_context_worker,
            vae_encode_worker=vae_encode_worker,
            st_preprocess_worker=st_preprocess_worker,
            st_control_worker=st_control_worker,
            ctx_control_worker=ctx_control_worker,
        )
