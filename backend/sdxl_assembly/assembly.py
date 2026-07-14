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
from modules import blending

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

        if request.tiled_refinement is not None and request.tiled_refinement.enabled:
            return self._execute_tiled_refinement(request, callback)
        
        timings: Dict[str, float] = {}
        result_metadata: Dict[str, Any] = {}
        device = torch.device(request.device)
        unet_started = False
        structural_control_session_active = False
        contextual_session_active = False
        prepared_context = None
        workflow_contract = request.metadata.get("workflow_contract")
        if isinstance(workflow_contract, dict):
            result_metadata["workflow_contract"] = dict(workflow_contract)

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
            output_image = self._compose_spatial_output(
                request=request,
                prepared_context=prepared_context,
                spatial_artifacts=spatial_artifacts,
                output_image=output_image,
            )
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

    def _compose_spatial_output(
        self,
        *,
        request: SDXLAssemblyRequest,
        prepared_context: Any,
        spatial_artifacts: Any,
        output_image: np.ndarray,
    ) -> np.ndarray:
        if request.spatial_context is None or prepared_context is None or spatial_artifacts is None:
            return output_image

        spatial_mode = str(request.spatial_context.mode or "image").strip().lower()
        if spatial_mode not in {"inpaint", "outpaint"}:
            return output_image

        bbox = getattr(prepared_context, "bbox", None) or getattr(spatial_artifacts, "bbox", None)
        if bbox is None:
            return output_image
        y1, y2, x1, x2 = [int(v) for v in bbox]
        if y2 <= y1 or x2 <= x1:
            return output_image

        if spatial_mode == "outpaint" and getattr(prepared_context, "working_pixels", None) is not None:
            base_pixels = prepared_context.working_pixels
        else:
            base_pixels = getattr(prepared_context, "original_pixels", None)
        if base_pixels is None:
            return output_image

        blend_mask = getattr(prepared_context, "blend_mask", None)
        if blend_mask is None:
            blend_mask = getattr(spatial_artifacts, "blend_mask", None)
        if blend_mask is None:
            log_telemetry("spatial_compose_skipped", f"mode={spatial_mode} reason=missing_blend_mask")
            return output_image

        log_telemetry(
            "spatial_compose_begin",
            f"mode={spatial_mode} bbox={tuple(int(v) for v in bbox)} blend=morphological_sin2",
        )

        base_batch = self._ensure_image_batch_tensor(base_pixels)
        patch_batch = self._ensure_image_batch_tensor(output_image)
        blend_batch = self._ensure_mask_batch_tensor(blend_mask)
        if base_batch is None or patch_batch is None or blend_batch is None:
            return output_image

        if patch_batch.shape[0] != base_batch.shape[0]:
            if patch_batch.shape[0] == 1 and base_batch.shape[0] > 1:
                patch_batch = patch_batch.repeat(base_batch.shape[0], 1, 1, 1)
            else:
                raise ValueError("Decoded patch batch size does not match compose base batch size.")
        if blend_batch.shape[0] != base_batch.shape[0]:
            if blend_batch.shape[0] == 1 and base_batch.shape[0] > 1:
                blend_batch = blend_batch.repeat(base_batch.shape[0], 1, 1)
            else:
                raise ValueError("Blend mask batch size does not match compose base batch size.")

        patch_h = max(1, y2 - y1)
        patch_w = max(1, x2 - x1)
        patch_resized = torch.nn.functional.interpolate(
            patch_batch.movedim(-1, 1),
            size=(patch_h, patch_w),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

        result = base_batch.clone()
        base_h = int(base_batch.shape[1])
        base_w = int(base_batch.shape[2])
        iy1, iy2 = max(0, y1), min(base_h, y2)
        ix1, ix2 = max(0, x1), min(base_w, x2)
        cy1, cy2 = iy1 - y1, iy2 - y1
        cx1, cx2 = ix1 - x1, ix2 - x1
        if iy2 > iy1 and ix2 > ix1:
            result[:, iy1:iy2, ix1:ix2, :] = patch_resized[:, cy1:cy2, cx1:cx2, :]

        weight = blending.apply_sin2_curve(blend_batch[..., None].to(dtype=torch.float32))
        composed = torch.clamp(result * weight + base_batch * (1.0 - weight), min=0.0, max=1.0)
        final_output = self._tensor_to_output_image(composed, output_image)
        log_telemetry(
            "spatial_compose_complete",
            f"mode={spatial_mode} output={final_output.shape[1]}x{final_output.shape[0]} blend=morphological_sin2",
        )
        return final_output

    def _ensure_image_batch_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] < 3:
            raise ValueError(f"Expected an image tensor shaped [B, H, W, C], got {tuple(tensor.shape)}.")
        tensor = tensor[..., :3].to(dtype=torch.float32).contiguous()
        if tensor.numel() and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _ensure_mask_batch_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4:
            tensor = tensor.amax(dim=-1)
        if tensor.ndim != 3:
            raise ValueError(f"Expected a mask tensor shaped [B, H, W], got {tuple(tensor.shape)}.")
        tensor = tensor.to(dtype=torch.float32).contiguous()
        if tensor.numel() and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _tensor_to_output_image(self, tensor: torch.Tensor, reference: np.ndarray) -> np.ndarray:
        image = tensor.detach().cpu()[0, ..., :3].clamp(0.0, 1.0).contiguous().numpy()
        reference_array = np.asarray(reference)
        if np.issubdtype(reference_array.dtype, np.integer):
            return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(reference_array.dtype, copy=False)
        return image.astype(reference_array.dtype if reference_array.dtype != np.dtype("O") else np.float32, copy=False)

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

    def _execute_tiled_refinement(self, request: SDXLAssemblyRequest, callback: Optional[Any] = None) -> SDXLAssemblyResult:
        """Runs the tiled refinement loop under backend assembly ownership."""
        from backend.sdxl_assembly.contracts import PreparedSpatialContext
        from modules.pipeline.tiled_refinement import select_tile_resolution, split_into_tiles, stitch_tiles
        import modules.core as core

        target_tensor = request.tiled_refinement.target_image.pixels[0]  # shape [H, W, C], float32 [0.0, 1.0]
        target_img_np = (target_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        H, W, C = target_img_np.shape

        timings: Dict[str, float] = {}
        device = torch.device(request.device)
        unet_started = False

        # Pre-flight warm-state handling matching W11c residency contract
        progress_state = callback.progress_state if callback is not None else None
        retain_warm = True
        if progress_state is not None:
            from modules.pipeline.tiled_refinement import should_retain_sdxl_warm_state
            retain_warm = should_retain_sdxl_warm_state(progress_state)

        if not retain_warm:
            resources.teardown_active_runtime("upscale_preflight")
        else:
            resources.cleanup_memory('tiled_refine_preflight', unload_models=False, force_cache=True, trim_host=True, target_phase=resources.MemoryPhase.TILED_REFINE)

        # Build prompt conditioning once (it is frozen for the request)
        lora_start = time.perf_counter()
        patches = self.lora_worker.materialize_patches()
        timings["lora_patch"] = time.perf_counter() - lora_start

        text_start = time.perf_counter()
        conditioning = self.text_encode_worker.get_conditioning()
        timings["text_encode"] = time.perf_counter() - text_start

        if bool(request.metadata.get("release_text_encoder_after_task", False)):
            text_release_start = time.perf_counter()
            if hasattr(self.text_encode_worker, "teardown_assembly_order"):
                self.text_encode_worker.teardown_assembly_order()
            timings["text_release"] = time.perf_counter() - text_release_start

        # Tile resolution setup
        min_overlap = request.tiled_refinement.overlap
        bucket, nx, ny, overlap_w, overlap_h = select_tile_resolution(W, H, min_overlap)
        bucket_w, bucket_h = bucket

        tiles = split_into_tiles(target_img_np, bucket_w, bucket_h, nx, ny, overlap_w, overlap_h)
        print(f'[Tiled Refinement (Assembly)] Processing {len(tiles)} tiles...')

        # Start UNet spine
        unet_start = time.perf_counter()
        self.unet_spine.start()
        unet_started = True
        timings["unet_start"] = time.perf_counter() - unet_start

        refined_tiles = []
        denose_start = time.perf_counter()
        try:
            for i, t in enumerate(tiles):
                # Check for interrupts
                resources.throw_exception_if_processing_interrupted()

                # Report progress if callback is available
                if callback is not None and callback.raw_callback is not None:
                    progress_state = callback.progress_state
                    if progress_state is not None:
                        current_progress = int(getattr(progress_state, "current_progress", 0) or 0)
                        percent = int(current_progress + (i / len(tiles)) * 10)
                        callback.raw_callback(progress_state, percent, f'Refining tile {i+1}/{len(tiles)} ...')

                # VAE encode tile in VAE_ENCODE memory phase scope
                with resources.memory_phase_scope(
                    resources.MemoryPhase.VAE_ENCODE,
                    task=None,
                    notes={'route': 'tiled_refine', 'tile_size': [t.w, t.h], 'denoise': float(request.tiled_refinement.denoise_strength)},
                    end_notes={'completed': True},
                ):
                    pixels_cpu = torch.from_numpy(t.tile_image).unsqueeze(0).float() / 255.0
                    prepared = PreparedSpatialContext(
                        mode="image",
                        original_pixels=pixels_cpu,
                        bb_pixels=pixels_cpu,
                        image_fingerprint=f"tile_{i}",
                        bb_pixels_fingerprint=f"tile_{i}",
                    )
                    artifacts = self.vae_encode_worker.encode(prepared)
                    latent_samples = artifacts.route_latent

                # Update callback in progress_state or use dummy callback
                def tile_callback(step, temp_latent, x, total_steps, denoised=None):
                    resources.throw_exception_if_processing_interrupted()

                latent_samples_gpu = latent_samples.to(device, dtype=torch.float16)

                # Denoise tile
                denoise_result = self.unet_spine.denoise(
                    latent_samples_gpu,
                    conditioning,
                    callback=tile_callback,
                )

                # Decode tile (non-tiled) using transient VAE decode worker
                decoded_img, _, _ = self.vae_decode_worker.decode(denoise_result, device)

                refined_tiles.append(t._replace(tile_image=decoded_img))

                # Post-tile cleanup
                resources.cleanup_memory('tiled_refine_tile_complete', notes={'tile_index': i}, trim_host=False, target_phase=resources.MemoryPhase.TILED_REFINE)

        except resources.InterruptProcessingException:
            # Handle Skip vs Stop semantics
            progress_state = callback.progress_state if callback is not None else None
            if progress_state is not None and getattr(progress_state, 'last_stop', False) == 'skip':
                print('[Tiled Refinement (Assembly)] User skipped tiled refinement. Stitching partially completed tiles...')
                progress_state.last_stop = False
                for j in range(len(refined_tiles), len(tiles)):
                    refined_tiles.append(tiles[j])
            else:
                raise
        finally:
            if unet_started:
                self.unet_spine.end()

        timings["unet_denoise"] = time.perf_counter() - denose_start

        # Stitch
        if callback is not None and callback.raw_callback is not None:
            progress_state = callback.progress_state
            if progress_state is not None:
                current_progress = int(getattr(progress_state, "current_progress", 0) or 0)
                callback.raw_callback(progress_state, current_progress + 10, 'Stitching tiles ...')

        stitch_start = time.perf_counter()
        result = stitch_tiles(refined_tiles, (H, W, C), bucket_w, bucket_h)
        timings["stitch"] = time.perf_counter() - stitch_start

        # Final sweep matching W11c residency contract
        if not retain_warm:
            resources.teardown_active_runtime("upscale_finalization")
        else:
            resources.cleanup_memory('tiled_refine_finalize', unload_models=False, force_cache=True, target_phase=resources.MemoryPhase.FINALIZE)

        runtime_identity = SDXLRuntimeIdentity(
            checkpoint=request.checkpoint,
            vae=request.vae,
            unet_posture=request.unet_posture,
            clip_posture=request.clip_posture,
            vae_posture=request.vae_posture,
            lora_posture=request.lora_posture,
        )

        result_metadata = {}
        workflow_contract = request.metadata.get("workflow_contract")
        if isinstance(workflow_contract, dict):
            result_metadata["workflow_contract"] = dict(workflow_contract)

        return SDXLAssemblyResult(
            output_image=result,
            seed=request.seed,
            width=result.shape[1],
            height=result.shape[0],
            runtime_identity=runtime_identity,
            timings=timings,
            metadata={
                "runtime_identity": runtime_identity.as_dict(),
                **result_metadata,
            },
        )
