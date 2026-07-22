from __future__ import annotations

import math
import time
import logging
from typing import Any, Optional
import torch

from backend.sdxl_assembly.contracts import SDXLAssemblyRequest
from backend.sdxl_assembly.progress import log_telemetry
from backend.sdxl_assembly.runtime_state import (
    acquire_resident_unet_component,
)
from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker

logger = logging.getLogger(__name__)

class ResidentUnetSpine:
    """Worker representing ResidentUnetSpine (GPU-pinned resident weights)."""

    def __init__(self, request: SDXLAssemblyRequest, lora_worker: GpuLoraWorker | None = None) -> None:
        self.request = request
        self.lora_worker = lora_worker or GpuLoraWorker(request)
        self.unet = None
        self.is_active = False
        self.x_center = None
        self._active_color_worker = None

    def materialize_scheduler_and_loras(
        self,
        request: SDXLAssemblyRequest,
        lora_worker: GpuLoraWorker | None = None,
    ) -> bool:
        """Explicitly reloads clean weights (if needed), applies LoRA patches, and compiles.
        Returns True if there was a warm hit, False otherwise.
        """
        # Determine the desired LoRA signature.
        from backend.sdxl_assembly.runtime_state import _unet_lora_signature
        desired_lora_signature = _unet_lora_signature(request)

        # Get current markers from model
        model = self.unet.model
        current_lora_signature = getattr(model, "_nex_resident_lora_signature", None)

        # Get compile metrics
        compile_metrics = getattr(model, "_nex_resident_compile_metrics", None)
        has_compiled_state = (
            compile_metrics is not None
            or not desired_lora_signature
        )

        # 1. Warm-hit check: everything matches and we have compiled state
        if (
            current_lora_signature == desired_lora_signature
            and has_compiled_state
        ):
            logger.debug("[SDXL Telemetry] Warm resident UNet spine hit for materialize.")
            return True

        # 2. Coordinated restore check
        needs_restore = (
            current_lora_signature is not None  # Not initial load
            and (
                current_lora_signature != desired_lora_signature
                or bool(getattr(model, "_patched_marker", False))
                or not has_compiled_state
            )
        )

        # 3. Perform reload if needs_restore
        if needs_restore:
            logger.info("[SDXL Telemetry] Rebuilding resident state due to signature mismatch.")
            runtime_reload = getattr(self.unet, "runtime_reload", None)
            if callable(runtime_reload):
                import torch
                target_device = torch.device(request.device)
                runtime_reload(model, target_device)
                model.device = target_device
            else:
                raise RuntimeError("Resident UNet spine component missing runtime_reload callable.")

            # Clear patcher state
            self.unet.patches = {}
            self.unet.weight_wrapper_patches = {}
            self.unet.backup.clear()
            self.unet.object_patches_backup.clear()
            model.current_weight_patches_uuid = None
            if hasattr(model, "_patched_marker"):
                delattr(model, "_patched_marker")

        # Update requests and workers
        self.request = request
        from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker
        if lora_worker is not None:
            self.lora_worker = lora_worker
        else:
            self.lora_worker = GpuLoraWorker(request)

        # 4. Apply UNet LoRA patches
        self.lora_worker.apply_unet_patches(self.unet)

        # 5. Compile patches on GPU using compile_unet_patches
        compile_metrics = self.lora_worker.compile_unet_patches(self.unet)

        # 6. Record markers on model
        self.unet.model._nex_resident_lora_signature = desired_lora_signature
        self.unet.model._nex_resident_compile_metrics = compile_metrics or {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }
        return False

    def start(self, status_callback: Any = None, progress_state: Any = None) -> None:
        """Acquires the base UNet (if not already loaded), applies UNet-side LoRAs
        (GPU patch compilation), and registers residency.
        """
        if self.unet is None:
            try:
                # 1. Acquire the owned UNet for this resident spine.
                if status_callback is not None and progress_state is not None:
                    status_callback(
                        progress_state,
                        int(getattr(progress_state, 'current_progress', 0) or 0),
                        'Loading UNet spine ...',
                    )
                self.unet = acquire_resident_unet_component(self.request)

                # 2. Materialize the LoRA stack.
                unet_patches = int(getattr(self.lora_worker, 'unet_patch_count', 0) or 0)
                if unet_patches > 0 and status_callback is not None and progress_state is not None:
                    status_callback(
                        progress_state,
                        int(getattr(progress_state, 'current_progress', 0) or 0),
                        f'Compiling {unet_patches} UNet LoRA patches ...',
                    )
                self.materialize_scheduler_and_loras(self.request, self.lora_worker)

                # Since compile_unet_patches cleared the patches, we mark it.
                self.unet.runtime_release_to_meta = False
            except Exception:
                self.release_owned_resources()
                raise
        else:
            logger.debug("[SDXL Telemetry] Reusing warm owned UNet in resident spine.")
            log_telemetry("unet_spine_owned_reuse")

        self.is_active = True

    def denoise(
        self,
        latent: torch.Tensor,
        conditioning: Any,
        callback: Optional[Any] = None,
        denoise_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the denoise loop directly on GPU (no CPU offloading for resident spine)."""
        self.x_center = None
        self._active_color_worker = None

        device = torch.device(self.request.device)

        log_telemetry("resident_spine_stream_begin")

        try:
            # Attach the compiled UNet to device (no Low VRAM memory offloading)
            self.unet.patch_model(device_to=device, lowvram_model_memory=0)

            dtype = self._infer_unet_dtype()
            latent = latent.to(device=device, dtype=dtype)
            if self.request.color_extraction and self.request.color_extraction.enabled:
                self.x_center = latent.detach().clone()

            # Setup noise
            generator = torch.Generator(device=device).manual_seed(self.request.seed)
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)

            # 1. Convert positive and negative conditions using _convert_sampler_cond
            import uuid
            from backend import cond_utils

            def convert_sampler_cond(cond_list):
                out = []
                for cross_attn, payload in cond_list:
                    converted = payload.copy()
                    if cross_attn is not None:
                        converted["cross_attn"] = cross_attn
                    converted["model_conds"] = converted.get("model_conds", {})
                    converted["uuid"] = uuid.uuid4()
                    out.append(converted)
                return out

            converted_conds = {
                "positive": convert_sampler_cond(conditioning.get("positive")),
                "negative": convert_sampler_cond(conditioning.get("negative")),
            }

            # Ensure model has a valid dtype attribute (especially for mocks)
            if hasattr(self.unet, "model"):
                model_dtype = getattr(self.unet.model, "dtype", None)
                if model_dtype is None or not isinstance(model_dtype, torch.dtype):
                    try:
                        self.unet.model.dtype = dtype
                    except AttributeError:
                        pass

            # 2. Process conditions using cond_utils.process_conds
            processed_conds = cond_utils.process_conds(
                self.unet.model,
                noise,
                converted_conds,
                device,
                latent_image=latent,
                denoise_mask=denoise_mask,
                seed=self.request.seed,
            )

            # Calculate sigmas
            from backend import sampling
            denoise_val = float(self.request.denoise_strength if self.request.denoise_strength is not None else 1.0)
            sampler_instance = sampling.KSampler(
                self.unet,
                self.request.steps,
                device,
                self.request.sampler,
                self.request.scheduler,
                denoise_val,
                model_options={"quality": {"sharpness": self.request.sharpness, "adaptive_cfg": self.request.adaptive_cfg}},
            )
            sigmas = sampler_instance.sigmas

            if sigmas.shape[-1] == 0:
                return latent

            sampler_function = self._resolve_sampler_function()
            model_sampling = self.unet.model.model_sampling
            max_sigma = float(model_sampling.sigma_max)
            sigma = float(sigmas[0])
            max_denoise = math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

            scaled_noise = self._noise_scaling(
                model_sampling,
                sigmas[0],
                noise,
                latent,
                max_denoise=max_denoise,
            )

            total_steps = len(sigmas) - 1
            k_callback = None
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps, x.get("denoised", None))

            from backend import precision
            with torch.inference_mode(), precision.autocast_context(device):
                samples = sampler_function(
                    self._build_direct_model_callable(
                        self.unet,
                        processed_conds,
                        latent_image=latent,
                        reference_noise=noise,
                        denoise_mask=denoise_mask,
                    ),
                    scaled_noise,
                    sigmas,
                    extra_args={"denoise_mask": denoise_mask},
                    callback=k_callback,
                    disable=True,
                )

            output_latent = model_sampling.inverse_noise_scaling(sigmas[-1], samples)
            finite_mask = torch.isfinite(output_latent)
            if not bool(finite_mask.all()):
                nonfinite = int((~finite_mask).sum().item())
                log_telemetry(
                    "spine_resident_nonfinite",
                    f"route={self.request.route_id} nonfinite={nonfinite} total={output_latent.numel()}",
                )
                try:
                    from backend.sdxl_assembly.runtime_state import release_active_sdxl_resident_spine
                    lora_sig = getattr(self.unet.model, "_nex_resident_lora_signature", ())
                    metrics = getattr(self.unet.model, "_nex_resident_compile_metrics", None)
                    logger.error(
                        "[SDXL Telemetry] Non-finite latent detected in resident spine! "
                        "Evicting resident state. LoRA Signature: %s, Compile Metrics: %s",
                        lora_sig, metrics
                    )
                    release_active_sdxl_resident_spine("non-finite resident latent")
                except Exception as evict_err:
                    logger.error("Failed to release active resident spine on non-finite latent: %s", evict_err)
                raise RuntimeError(
                    "SDXL resident sampling produced a non-finite latent; "
                    "the invalid image was not decoded or saved."
                )
            log_telemetry(
                "spine_resident_latent_finite",
                f"route={self.request.route_id} min={float(output_latent.min().item()):.4f} "
                f"max={float(output_latent.max().item()):.4f}",
            )
        finally:
            # Park and release unet parameters from active GPU stream (stays on offload GPU)
            self._park_compiled_unet_before_decode()
            self._close_color_extraction_state()

        log_telemetry("spine_resident_complete")
        return output_latent

    def _close_color_extraction_state(self) -> None:
        """Close the run-bound overlay and release its latent on every path."""
        worker = self._active_color_worker
        self._active_color_worker = None
        try:
            if worker is not None:
                worker.close()
        finally:
            self.x_center = None

    def end(self) -> None:
        self._park_compiled_unet_before_decode()
        self.is_active = False

    def teardown_assembly_order(self) -> None:
        if bool(self.request.metadata.get("release_warm_unet_after_task", False)):
            self.release_owned_resources()
            return
        self.end()
        log_telemetry("resident_spine_retain", f"checkpoint={self.request.checkpoint.path.name}")

    def release_owned_resources(self) -> None:
        self.end()
        self.unet = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stateless UNet helper methods copied from sdxl_unified_runtime_execution
    def _infer_unet_dtype(self) -> torch.dtype:
        model = getattr(self.unet, "model", None)
        if model is None:
            return torch.float16
        for tensor in list(model.parameters()):
            if isinstance(tensor, torch.Tensor):
                return tensor.dtype
        return torch.float16

    def _resolve_sampler_function(self):
        sampler_name = self.request.sampler
        from backend import k_diffusion
        func_name = f"sample_{sampler_name}"
        if sampler_name.endswith("_cfg_pp"):
            func_name = f"sample_{sampler_name[:-7]}"

        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {sampler_name} not implemented in k_diffusion as {func_name}")
        return sampler_function

    def _noise_scaling(
        self,
        model_sampling: Any,
        sigma: Any,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        *,
        max_denoise: bool | None = None,
    ) -> torch.Tensor:
        if max_denoise is not None:
            try:
                return model_sampling.noise_scaling(sigma, noise, latent_image, max_denoise)
            except TypeError:
                return model_sampling.noise_scaling(sigma, noise, latent_image)
        try:
            return model_sampling.noise_scaling(sigma, noise, latent_image)
        except TypeError:
            return model_sampling.noise_scaling(sigma, noise, latent_image, False)

    def _diffusion_progress(self, model_sampling: Any, sigma: Any) -> float:
        try:
            timestep = model_sampling.timestep(sigma)
            if isinstance(timestep, torch.Tensor):
                timestep_value = float(timestep.detach().reshape(-1)[0].item())
            else:
                timestep_value = float(timestep)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, 1.0 - timestep_value / 999.0))

    def _apply_sharpness_quality(
        self,
        x_input: torch.Tensor,
        cond_pred: torch.Tensor,
        *,
        sharpness: float,
        diffusion_progress: float,
    ) -> torch.Tensor:
        if sharpness <= 0.0:
            return cond_pred

        alpha = 0.001 * sharpness * diffusion_progress
        if alpha < 0.01:
            return cond_pred

        from backend import sampling
        positive_eps = x_input - cond_pred
        degraded_eps = sampling.anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=cond_pred)
        positive_eps_weighted = degraded_eps * alpha + positive_eps * (1.0 - alpha)
        return x_input - positive_eps_weighted

    def _build_direct_model_callable(
        self,
        execution_unet: Any,
        processed_conds: dict[str, Any],
        *,
        latent_image: torch.Tensor,
        reference_noise: torch.Tensor,
        denoise_mask: torch.Tensor | None = None,
    ):
        model_options = dict(getattr(execution_unet, "model_options", {}) or {})
        model_options["quality"] = {
            "sharpness": self.request.sharpness,
            "adaptive_cfg": self.request.adaptive_cfg,
        }
        disable_cfg1_optimization = bool(model_options.get("disable_cfg1_optimization", False))
        cfg_pp = "_cfg_pp" in self.request.sampler
        model_sampling = execution_unet.model.model_sampling

        def model_fn(x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            active_mask = kwargs.get("denoise_mask", denoise_mask)
            x_input = x
            latent_mask = None
            if active_mask is not None:
                active_mask = active_mask.to(device=x.device, dtype=x.dtype)
                latent_mask = 1.0 - active_mask
                preserved_latent = self._noise_scaling(
                    model_sampling,
                    sigma,
                    reference_noise,
                    latent_image,
                ).to(dtype=x.dtype, device=x.device)
                x_input = x * active_mask + preserved_latent * latent_mask

            positive_conds = processed_conds.get("positive")
            negative_conds = processed_conds.get("negative")
            if math.isclose(self.request.cfg, 1.0) and not disable_cfg1_optimization:
                negative_conds = None

            cond_pred, uncond_pred = self._calc_fullframe_cond_batch(
                execution_unet,
                [positive_conds, negative_conds],
                x_input,
                sigma,
            )

            diffusion_progress = self._diffusion_progress(model_sampling, sigma)
            cond_pred = self._apply_sharpness_quality(
                x_input,
                cond_pred,
                sharpness=self.request.sharpness,
                diffusion_progress=diffusion_progress,
            )

            from backend import sampling
            out = sampling.cfg_function(
                execution_unet.model,
                cond_pred,
                uncond_pred,
                self.request.cfg,
                x_input,
                sigma,
                model_options=model_options,
                cfg_pp=cfg_pp,
                adaptive_cfg=self.request.adaptive_cfg,
                diffusion_progress=diffusion_progress,
            )
            if active_mask is not None and latent_mask is not None:
                latent_ref = latent_image.to(device=out.device, dtype=out.dtype)
                out = out * active_mask + latent_ref * latent_mask
            return out

        # Wrap model_fn locally with ColorExtractionWorker if enabled
        if self.request.color_extraction and self.request.color_extraction.enabled and getattr(self, "x_center", None) is not None:
            from backend.sdxl_assembly.color_extraction_worker import ColorExtractionWorker
            color_worker = ColorExtractionWorker(self.request.color_extraction)
            color_worker.prepare(x_center=self.x_center, sigma_max=float(model_sampling.sigma_max))
            self._active_color_worker = color_worker

            original_model_fn = model_fn
            def wrapped_model_fn(x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                out = original_model_fn(x, sigma, **kwargs)
                out = color_worker.correct_denoised(out, sigma)
                return out
            model_fn = wrapped_model_fn

        class _DirectModelInner:
            def __init__(self, inner_model: Any):
                self.inner_model = inner_model

        class _DirectModelCallable:
            def __init__(self, fn, inner_model: Any):
                self._fn = fn
                self.inner_model = _DirectModelInner(inner_model)

            def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                return self._fn(x, sigma, **kwargs)

        return _DirectModelCallable(model_fn, execution_unet.model)

    def _calc_fullframe_cond_batch(
        self,
        execution_unet: Any,
        conds: list[Any],
        x_in: torch.Tensor,
        timestep: torch.Tensor,
    ) -> list[torch.Tensor]:
        from backend import cond_utils
        out_conds = [torch.zeros_like(x_in) for _ in conds]
        out_counts = [torch.ones_like(x_in) * 1e-37 for _ in conds]
        to_run: list[tuple[Any, int]] = []

        for cond_index, cond in enumerate(conds):
            if cond is None:
                continue
            for cond_entry in cond:
                prepared = cond_utils.get_area_and_mult(cond_entry, x_in, timestep)
                if prepared is None:
                    continue
                if prepared.area is not None or prepared.input_x.shape != x_in.shape:
                    raise ValueError("Unified SDXL direct denoise only supports full-frame txt2img conditions.")
                to_run.append((prepared, cond_index))

        while len(to_run) > 0:
            first = to_run[0]
            to_batch = []
            for index in range(len(to_run)):
                if cond_utils.can_concat_cond(to_run[index][0], first[0]):
                    to_batch.append(index)

            batch_items = [to_run[index] for index in to_batch]
            for index in sorted(to_batch, reverse=True):
                to_run.pop(index)

            batch_input_x = [prepared.input_x for prepared, _ in batch_items]
            batch_mult = [prepared.mult for prepared, _ in batch_items]
            batch_conditioning = [prepared.conditioning for prepared, _ in batch_items]
            batch_cond_indices = [cond_index for _, cond_index in batch_items]
            input_x = torch.cat(batch_input_x)
            conditioning_batch = cond_utils.cond_cat(batch_conditioning)
            timestep_batch = torch.cat([timestep] * len(batch_cond_indices))
            outputs = execution_unet.model.apply_model(input_x, timestep_batch, **conditioning_batch).chunk(len(batch_cond_indices))

            for output, cond_index, mult in zip(outputs, batch_cond_indices, batch_mult):
                out_conds[cond_index] += output * mult
                out_counts[cond_index] += mult

        for index in range(len(out_conds)):
            out_conds[index] /= out_counts[index]
        return out_conds

    def _park_compiled_unet_before_decode(self) -> None:
        if self.unet is not None and hasattr(self.unet, "detach"):
            try:
                self.unet.detach()
            except Exception:
                pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
