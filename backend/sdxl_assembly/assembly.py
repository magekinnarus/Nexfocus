from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional
import numpy as np
import torch

from backend import resources
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
        text_encode_worker: Any,
        vae_decode_worker: Any,
        lora_worker: Any,
        spatial_context_worker: Optional[Any] = None,
        vae_encode_worker: Optional[Any] = None,
        st_preprocess_worker: Optional[Any] = None,
        st_control_worker: Optional[Any] = None,
        ctx_control_worker: Optional[Any] = None,
    ) -> None:
        self.unet_spine = unet_spine
        self.text_encode_worker = text_encode_worker
        self.vae_decode_worker = vae_decode_worker
        self.lora_worker = lora_worker
        self.spatial_context_worker = spatial_context_worker
        self.vae_encode_worker = vae_encode_worker
        self.st_preprocess_worker = st_preprocess_worker
        self.st_control_worker = st_control_worker
        self.ctx_control_worker = ctx_control_worker

    def execute(self, request: SDXLAssemblyRequest, callback: Optional[Any] = None) -> SDXLAssemblyResult:
        """Executes the pipeline steps in strict chronological order."""
        # 1. Static and posture validation
        request.validate()
        
        timings: Dict[str, float] = {}
        result_metadata: Dict[str, Any] = {}
        device = torch.device(request.device)
        unet_started = False
        structural_control_session_active = False
        contextual_session_active = False

        try:
            # 2. Materialize LoRA patches first
            lora_start = time.perf_counter()
            patches = self.lora_worker.materialize_patches()
            timings["lora_patch"] = time.perf_counter() - lora_start

            # 3. Resolve text conditioning (avoid holding extra latent memory)
            text_start = time.perf_counter()
            conditioning = self.text_encode_worker.get_conditioning()
            timings["text_encode"] = time.perf_counter() - text_start
            if bool(request.metadata.get("release_text_encoder_after_task", False)):
                text_release_start = time.perf_counter()
                if hasattr(self.text_encode_worker, "teardown_assembly_order"):
                    self.text_encode_worker.teardown_assembly_order()
                timings["text_release"] = time.perf_counter() - text_release_start

            # 4. Coordinate spatial preparation and VAE encode
            spatial_artifacts = None
            denoise_mask_cpu = None
            if request.spatial_context is not None:
                spatial_prep_start = time.perf_counter()
                prepared_context = self.spatial_context_worker.prepare()
                timings["spatial_prep"] = time.perf_counter() - spatial_prep_start
                
                vae_encode_start = time.perf_counter()
                spatial_artifacts = self.vae_encode_worker.encode(prepared_context)
                timings["vae_encode"] = time.perf_counter() - vae_encode_start
                
                latent_samples_cpu, denoise_mask_cpu = self._resolve_spatial_inputs(request, spatial_artifacts)
                result_metadata["spatial_contract"] = self._build_spatial_metadata(request, spatial_artifacts)
            else:
                vae_start = time.perf_counter()
                latent_bundle = self.vae_decode_worker.prepare_latents(torch.device("cpu"))
                timings["vae_prep"] = time.perf_counter() - vae_start
                
                latent_samples_cpu = latent_bundle.samples

            # 4.5. Streaming structural preprocess
            prepared_hints = {}
            if self.st_preprocess_worker is not None and len(request.structural_controls) > 0:
                preprocess_start = time.perf_counter()
                prepared_hints = self.st_preprocess_worker.preprocess()
                timings["structural_preprocess"] = time.perf_counter() - preprocess_start

            # 4.6. Streaming structural control attach
            if self.st_control_worker is not None and len(request.structural_controls) > 0:
                structural_control_session_active = True
                control_start = time.perf_counter()
                conditioning = self.st_control_worker.attach_conditioning(conditioning, prepared_hints)
                timings["structural_control_attach"] = time.perf_counter() - control_start

            # 4.7. Streaming contextual control preprocess
            if self.ctx_control_worker is not None and len(request.contextual_controls) > 0:
                preprocess_start = time.perf_counter()
                self.ctx_control_worker.preprocess()
                timings["contextual_preprocess"] = time.perf_counter() - preprocess_start

            # 5. Coordinate UNet spine denoise
            unet_start = time.perf_counter()
            self.unet_spine.start()
            unet_started = True
            timings["unet_start"] = time.perf_counter() - unet_start

            # 5.1. Streaming contextual control attach
            if self.ctx_control_worker is not None and len(request.contextual_controls) > 0:
                contextual_session_active = True
                self.ctx_control_worker.attach_unet_patches(self.unet_spine)

            # Materialize latents on target GPU device immediately before UNet denoise
            materialize_start = time.perf_counter()
            latent_samples_gpu = latent_samples_cpu.to(device, dtype=torch.float16)
            denoise_mask_gpu = (
                denoise_mask_cpu.to(device=device, dtype=torch.float32)
                if denoise_mask_cpu is not None
                else None
            )
            timings["latent_materialize"] = time.perf_counter() - materialize_start
            
            denoise_start = time.perf_counter()
            samples = self.unet_spine.denoise(
                latent_samples_gpu,
                conditioning,
                callback=callback,
                denoise_mask=denoise_mask_gpu,
            )
            timings["unet_denoise"] = time.perf_counter() - denoise_start
        except resources.InterruptProcessingException:
            raise
        except Exception as e:
            logger.error(f"[SDXL Assembly] Worker execution failed: {e}")
            raise RuntimeError(f"Worker execution failed: {e}") from e
        finally:
            if unet_started:
                self.unet_spine.end()
            if structural_control_session_active and self.st_control_worker is not None:
                self.st_control_worker.end()
            if contextual_session_active and self.ctx_control_worker is not None:
                self.ctx_control_worker.end()

        # 6. Decode results using VAE worker
        try:
            decode_start = time.perf_counter()
            output_image, load_time, decode_time = self.vae_decode_worker.decode(samples, device)
            timings["vae_decode"] = time.perf_counter() - decode_start
        except resources.InterruptProcessingException:
            raise
        except Exception as e:
            logger.error(f"[SDXL Assembly] VAE decode failed: {e}")
            raise RuntimeError(f"VAE decode failed: {e}") from e

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
                **result_metadata,
            },
        )

    def _resolve_spatial_inputs(
        self,
        request: SDXLAssemblyRequest,
        spatial_artifacts: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        spatial_mode = str(request.spatial_context.mode if request.spatial_context is not None else "image").strip().lower()
        if spatial_mode in {"inpaint", "outpaint"}:
            latent = (
                spatial_artifacts.bb_latent
                if spatial_artifacts.bb_latent is not None
                else spatial_artifacts.masked_latent
            )
            if latent is None:
                latent = spatial_artifacts.route_latent
        else:
            latent = spatial_artifacts.route_latent
            if latent is None:
                latent = spatial_artifacts.bb_latent

        if latent is None:
            raise RuntimeError(f"Spatial artifacts did not produce a usable latent for mode={spatial_mode}.")
        return latent, spatial_artifacts.denoise_mask

    def _build_spatial_metadata(self, request: SDXLAssemblyRequest, spatial_artifacts: Any) -> Dict[str, Any]:
        spatial_mode = str(request.spatial_context.mode if request.spatial_context is not None else "image").strip().lower()
        return {
            "mode": spatial_mode,
            "cache_hit": bool(spatial_artifacts.cache_hit),
            "source_fingerprint": spatial_artifacts.source_fingerprint,
            "image_fingerprint": spatial_artifacts.image_fingerprint,
            "mask_fingerprint": spatial_artifacts.mask_fingerprint,
            "route_latent_fingerprint": spatial_artifacts.route_latent_fingerprint,
            "masked_latent_fingerprint": spatial_artifacts.masked_latent_fingerprint,
            "bb_latent_fingerprint": spatial_artifacts.bb_latent_fingerprint,
            "denoise_mask_fingerprint": spatial_artifacts.denoise_mask_fingerprint,
            "blend_mask_fingerprint": spatial_artifacts.blend_mask_fingerprint,
            "bbox": tuple(int(v) for v in spatial_artifacts.bbox),
            "bbox_area_ratio": float(spatial_artifacts.bbox_area_ratio),
            "mask_coverage": float(spatial_artifacts.mask_coverage),
        }

    # Extension Points for ControlNet
    def _prepare_controlnet_artifacts(self, request: SDXLAssemblyRequest) -> None:
        """Extension point for structural ControlNet preprocessing artifacts and control-model application."""
        pass

    def close(self) -> None:
        """Detach request-local state without destroying reusable warm worker domains."""
        log_telemetry("cleanup_begin", "reason=assembly_close")
        
        # 0. ControlNet workers keep support/payload caches warm across safe SDXL
        # request closes. Full support teardown is reserved for explicit domain release.
        if self.st_control_worker is not None and hasattr(self.st_control_worker, "end"):
            try:
                self.st_control_worker.end()
            except Exception as e:
                logger.warning(f"Error detaching structural control worker: {e}")

        if self.ctx_control_worker is not None and hasattr(self.ctx_control_worker, "end"):
            try:
                self.ctx_control_worker.end()
            except Exception as e:
                logger.warning(f"Error detaching contextual control worker: {e}")
            
        # 1. vae_decode_worker unloads VAE tensors first
        if hasattr(self.vae_decode_worker, "teardown_assembly_order"):
            try:
                self.vae_decode_worker.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing vae_decode_worker: {e}")
            
        # 1.2. vae_encode_worker unloads VAE tensors
        if self.vae_encode_worker is not None and hasattr(self.vae_encode_worker, "teardown_assembly_order"):
            try:
                self.vae_encode_worker.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing vae_encode_worker: {e}")
            
        # 1.3. spatial_context_worker cleanup
        if self.spatial_context_worker is not None and hasattr(self.spatial_context_worker, "teardown_assembly_order"):
            try:
                self.spatial_context_worker.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing spatial_context_worker: {e}")
            
        # 2. LoraWorker rolls back or detaches patch weights
        if hasattr(self.lora_worker, "teardown_assembly_order"):
            try:
                self.lora_worker.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing lora_worker: {e}")
            
        # 3. text_encode_worker releases CPU/GPU pinned models
        if hasattr(self.text_encode_worker, "teardown_assembly_order"):
            try:
                self.text_encode_worker.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing text_encode_worker: {e}")
            
        # 4. UNetSpine unloads or deallocates weights
        if hasattr(self.unet_spine, "teardown_assembly_order"):
            try:
                self.unet_spine.teardown_assembly_order()
            except Exception as e:
                logger.warning(f"Error closing unet_spine: {e}")
            
        log_telemetry("cleanup_complete", "reason=assembly_close")
