from __future__ import annotations

import math
import time
from typing import Any

import torch

from backend import cond_utils, k_diffusion, sampling
from backend.sdxl_runtime_contract import GpuAttachedExecutionState
from modules import blending


class UnifiedSDXLRuntimeExecutionMixin:
    def _validate_prepared_inputs(self, prepared_inputs: Any) -> None:
        if prepared_inputs.base_model is None or prepared_inputs.compiled_unet is None or prepared_inputs.conditioning is None:
            raise RuntimeError("Unified SDXL runtime requires prepared base model, compiled UNet, and conditioning artifacts.")
        if self.base_model is not None and prepared_inputs.base_model.fingerprint != self.base_model.fingerprint:
            raise RuntimeError("Prepared base model does not match the loaded runtime base model.")
        if self.compiled_unet is not None and prepared_inputs.compiled_unet.artifact_fingerprint != self.compiled_unet.artifact_fingerprint:
            raise RuntimeError("Prepared compiled UNet does not match the loaded runtime compiled artifact.")
        if self.conditioning is not None and prepared_inputs.conditioning.prompt_fingerprint != self.conditioning.prompt_fingerprint:
            raise RuntimeError("Prepared conditioning does not match the loaded runtime conditioning artifact.")
        if (
            self.structural_conditioning is not None
            and prepared_inputs.structural_conditioning is not None
            and prepared_inputs.structural_conditioning.artifact_fingerprint != self.structural_conditioning.artifact_fingerprint
        ):
            raise RuntimeError("Prepared structural conditioning does not match the loaded runtime structural artifact.")
        if (
            self.spatial_conditioning is not None
            and prepared_inputs.spatial_conditioning is not None
            and prepared_inputs.spatial_conditioning.artifact_fingerprint != self.spatial_conditioning.artifact_fingerprint
        ):
            raise RuntimeError("Prepared spatial conditioning does not match the loaded runtime spatial artifact.")
        for feature_name, feature in prepared_inputs.injected_features.items():
            runtime_feature = self.injected_features.get(feature_name)
            if runtime_feature is None:
                raise RuntimeError(f"Prepared injected feature {feature_name!r} is not loaded in the runtime.")
            if runtime_feature.feature_fingerprint != feature.feature_fingerprint:
                raise RuntimeError(f"Prepared injected feature {feature_name!r} does not match the loaded runtime artifact.")

    def _transition_execution_state(
        self,
        prepared_inputs: Any | None,
        *,
        active_phase: str,
        attached_component_ids: tuple[str, ...],
        device: torch.device,
        stream_budget_mb: float = 0.0,
        headroom_mb: float = 0.0,
    ) -> GpuAttachedExecutionState:
        state = GpuAttachedExecutionState(
            execution_class=self._execution_class_label(),
            device=str(device),
            active_phase=active_phase,
            attached_component_ids=attached_component_ids,
            stream_budget_mb=float(stream_budget_mb),
            headroom_mb=float(headroom_mb),
        )
        self.execution_state = state
        if prepared_inputs is not None:
            prepared_inputs.gpu_attached_execution_state = state
        return state

    def _build_attached_component_ids(self, prepared_inputs: Any) -> tuple[str, ...]:
        component_ids: list[str] = []
        if prepared_inputs.base_model is not None:
            component_ids.append(
                f"base_model:{prepared_inputs.base_model.fingerprint or prepared_inputs.base_model.source_path or self.config.checkpoint_path}"
            )
        if prepared_inputs.compiled_unet is not None:
            component_ids.append(
                f"compiled_unet:{prepared_inputs.compiled_unet.artifact_fingerprint or prepared_inputs.compiled_unet.source_fingerprint or self.config.checkpoint_path}"
            )
        if prepared_inputs.conditioning is not None:
            component_ids.append(f"conditioning:{prepared_inputs.conditioning.prompt_fingerprint}")
        if prepared_inputs.structural_conditioning is not None:
            component_ids.append(
                f"structural_conditioning:{prepared_inputs.structural_conditioning.artifact_fingerprint or prepared_inputs.structural_conditioning.source_fingerprint}"
            )
        if prepared_inputs.spatial_conditioning is not None:
            component_ids.append(
                f"spatial_conditioning:{prepared_inputs.spatial_conditioning.spatial_mode}:"
                f"{prepared_inputs.spatial_conditioning.artifact_fingerprint or prepared_inputs.spatial_conditioning.source_fingerprint}"
            )
        for feature_name, feature in sorted(prepared_inputs.injected_features.items(), key=lambda item: str(item[0])):
            component_ids.append(
                f"injected_feature:{feature_name}:{feature.feature_fingerprint or feature.context_key}"
            )
        return tuple(component_ids)

    def _build_decode_component_ids(self) -> tuple[str, ...]:
        component_ids: list[str] = []
        if self.base_model is not None:
            component_ids.append(f"base_model:{self.base_model.fingerprint or self.base_model.source_path or self.config.checkpoint_path}")
        if self.conditioning is not None:
            component_ids.append(f"conditioning:{self.conditioning.prompt_fingerprint}")
        if self.structural_conditioning is not None:
            component_ids.append(
                f"structural_conditioning:{self.structural_conditioning.artifact_fingerprint or self.structural_conditioning.source_fingerprint}"
            )
        if self.spatial_conditioning is not None:
            component_ids.append(
                f"spatial_conditioning:{self.spatial_conditioning.spatial_mode}:"
                f"{self.spatial_conditioning.artifact_fingerprint or self.spatial_conditioning.source_fingerprint}"
            )
        component_ids.append(f"vae:{self._checkpoint_fingerprint or self.config.checkpoint_path}")
        return tuple(component_ids)

    def _move_nested_tensors_to_device(self, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._move_nested_tensors_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_nested_tensors_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_nested_tensors_to_device(item, device) for item in value)
        return value

    def _compose_decoded_images(self, decoded_images: torch.Tensor) -> torch.Tensor:
        prepared = self.prepared_inputs
        spatial = prepared.spatial_conditioning if prepared is not None else None
        payload = (prepared.payload or {}) if prepared is not None else {}
        if spatial is None:
            return decoded_images

        spatial_mode = str(spatial.spatial_mode or "image").strip().lower()
        if spatial_mode not in {"inpaint", "outpaint"}:
            return decoded_images

        bbox = payload.get("bbox") or spatial.bbox
        if bbox is None:
            return decoded_images
        y1, y2, x1, x2 = [int(v) for v in bbox]
        if y2 <= y1 or x2 <= x1:
            return decoded_images

        base_key = "outpaint_working_pixels" if spatial_mode == "outpaint" else "source_pixels"
        base_pixels = self._ensure_image_batch_tensor(payload.get(base_key))
        if base_pixels is None:
            return decoded_images

        blend_mask = self._ensure_mask_batch_tensor(payload.get("blend_mask"))
        if blend_mask is None:
            return decoded_images

        patch = self._ensure_image_batch_tensor(decoded_images)
        if patch is None:
            return decoded_images

        if base_pixels.shape[0] != patch.shape[0]:
            if patch.shape[0] == 1 and base_pixels.shape[0] > 1:
                patch = patch.repeat(base_pixels.shape[0], 1, 1, 1)
            else:
                raise ValueError("Decoded patch batch size does not match compose base batch size.")
        if blend_mask.shape[0] != base_pixels.shape[0]:
            if blend_mask.shape[0] == 1 and base_pixels.shape[0] > 1:
                blend_mask = blend_mask.repeat(base_pixels.shape[0], 1, 1)
            else:
                raise ValueError("Blend mask batch size does not match compose base batch size.")

        patch_h = max(1, y2 - y1)
        patch_w = max(1, x2 - x1)
        patch_resized = torch.nn.functional.interpolate(
            patch.movedim(-1, 1),
            size=(patch_h, patch_w),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

        result = base_pixels.clone()
        base_h = int(base_pixels.shape[1])
        base_w = int(base_pixels.shape[2])
        iy1, iy2 = max(0, y1), min(base_h, y2)
        ix1, ix2 = max(0, x1), min(base_w, x2)
        cy1, cy2 = iy1 - y1, iy2 - y1
        cx1, cx2 = ix1 - x1, ix2 - x1
        if iy2 > iy1 and ix2 > ix1:
            result[:, iy1:iy2, ix1:ix2, :] = patch_resized[:, cy1:cy2, cx1:cx2, :]

        weight = blending.apply_sin2_curve(blend_mask[..., None].to(dtype=torch.float32))
        return torch.clamp(result * weight + base_pixels * (1.0 - weight), min=0.0, max=1.0)

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

    def _convert_sampler_cond(self, cond: Any) -> list[dict[str, Any]]:
        import uuid

        out: list[dict[str, Any]] = []
        if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
            for entry in cond:
                converted = entry.copy()
                converted["uuid"] = converted.get("uuid", uuid.uuid4())
                out.append(converted)
            return out

        for cross_attn, payload in cond:
            converted = payload.copy()
            if cross_attn is not None:
                converted["cross_attn"] = cross_attn
            converted["model_conds"] = converted.get("model_conds", {})
            converted["uuid"] = uuid.uuid4()
            out.append(converted)
        return out

    def _prepare_direct_conds(
        self,
        *,
        execution_unet: Any,
        noise: torch.Tensor,
        positive: Any,
        negative: Any,
        latent_image: torch.Tensor,
        denoise_mask: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[dict[str, Any], float]:
        processed_conds = {
            "positive": self._convert_sampler_cond(positive),
            "negative": self._convert_sampler_cond(negative),
        }
        cond_start = time.perf_counter()
        processed_conds = cond_utils.process_conds(
            execution_unet.model,
            noise,
            processed_conds,
            device,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            seed=self.config.seed,
        )
        return processed_conds, time.perf_counter() - cond_start

    def _build_attached_payload(self, prepared_inputs: Any, device: torch.device) -> dict[str, Any]:
        payload = prepared_inputs.payload or {}
        encoded_prompt_pair = self._move_nested_tensors_to_device(payload.get("encoded_prompt_pair"), device)
        adm_pair = self._move_nested_tensors_to_device(payload.get("adm_pair"), device)
        execution_unet = self._build_contextual_execution_unet(prepared_inputs)
        latent, denoise_mask = self._resolve_initial_latent_and_mask(prepared_inputs, device)
        latent, noise = self._create_latent_and_noise(device, initial_latent=latent)
        positive, negative = self._build_sampler_conditioning(
            encoded_prompt_pair=encoded_prompt_pair,
            adm_pair=adm_pair,
        )
        positive, negative = self._apply_structural_controlnets_to_conditioning(
            prepared_inputs,
            positive,
            negative,
        )
        processed_conds, cond_prepare_duration = self._prepare_direct_conds(
            execution_unet=execution_unet,
            noise=noise,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise_mask=denoise_mask,
            device=device,
        )
        attached_payload = {
            **payload,
            "encoded_prompt_pair": encoded_prompt_pair,
            "adm_pair": adm_pair,
            "positive": positive,
            "negative": negative,
            "processed_conds": processed_conds,
            "execution_unet": execution_unet,
            "latent": latent,
            "noise": noise,
            "denoise_mask": denoise_mask,
            "sigmas": self._calculate_sigmas(device, execution_unet=execution_unet),
            "cond_prepare_duration": float(cond_prepare_duration),
        }
        return attached_payload

    def _infer_unet_dtype(self) -> torch.dtype:
        model = getattr(self.unet, "model", None)
        if model is None:
            return torch.float16
        dtype_getter = getattr(model, "get_dtype", None)
        if callable(dtype_getter):
            try:
                dtype = dtype_getter()
                if isinstance(dtype, torch.dtype):
                    return dtype
            except Exception:
                pass
        for tensor in list(model.parameters()):
            if isinstance(tensor, torch.Tensor):
                return tensor.dtype
        return torch.float16

    def _create_latent_and_noise(
        self,
        device: torch.device,
        *,
        initial_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self._infer_unet_dtype()
        if initial_latent is not None:
            latent = initial_latent.to(device=device, dtype=dtype)
            latent_shape = tuple(int(dim) for dim in latent.shape)
        else:
            latent_h = max(1, int(self.config.height) // 8)
            latent_w = max(1, int(self.config.width) // 8)
            latent_shape = (int(self.config.batch_size or 1), 4, latent_h, latent_w)
            latent = torch.zeros(
                latent_shape,
                device=device,
                dtype=dtype,
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.config.seed))
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return latent, noise

    def _resolve_initial_latent_and_mask(
        self,
        prepared_inputs: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        payload = prepared_inputs.payload or {}
        spatial = prepared_inputs.spatial_conditioning
        spatial_mode = str(getattr(spatial, "spatial_mode", "image") or "image").strip().lower()
        disable_initial_latent = bool(getattr(self.config, "disable_initial_latent", False))

        if "initial_latent" in payload and payload["initial_latent"] is not None and not (disable_initial_latent and spatial_mode == "inpaint"):
            latent = payload["initial_latent"]
            denoise_mask = payload.get("denoise_mask")
            if isinstance(latent, dict):
                denoise_mask = latent.get("noise_mask", denoise_mask)
                latent = latent.get("samples")
            latent = self._move_nested_tensors_to_device(latent, device)
            denoise_mask = self._move_nested_tensors_to_device(denoise_mask, device)
            return latent, denoise_mask

        if spatial is None:
            return None, None

        latent_key = "source_latent"
        denoise_mask_key = None
        if spatial_mode in {"inpaint", "outpaint"}:
            latent_key = "bb_latent"
            denoise_mask_key = "bb_denoise_mask"

        latent = payload.get(latent_key)
        if latent is None and spatial_mode == "inpaint":
            latent = payload.get("masked_source_latent")
            if latent is None:
                latent = payload.get("source_latent")
        elif latent is None:
            latent = payload.get("source_latent")
        denoise_mask = payload.get(denoise_mask_key) if denoise_mask_key is not None else None

        if disable_initial_latent and spatial_mode == "inpaint":
            latent = None

        latent = self._move_nested_tensors_to_device(latent, device)
        denoise_mask = self._move_nested_tensors_to_device(denoise_mask, device)
        return latent, denoise_mask

    def _build_sampler_conditioning(
        self,
        *,
        encoded_prompt_pair: dict[str, dict[str, torch.Tensor]] | None,
        adm_pair: dict[str, torch.Tensor] | None,
    ) -> tuple[list[list[Any]], list[list[Any]]]:
        if not encoded_prompt_pair or not adm_pair:
            raise RuntimeError("Unified SDXL runtime requires encoded prompt and ADM artifacts for denoise.")
        positive = [[
            encoded_prompt_pair["positive"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["positive"]["pooled"],
                "model_conds": {"y": adm_pair["positive"]},
            },
        ]]
        negative = [[
            encoded_prompt_pair["negative"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["negative"]["pooled"],
                "model_conds": {"y": adm_pair["negative"]},
            },
        ]]
        return positive, negative

    def _calculate_sigmas(self, device: torch.device, *, execution_unet: Any | None = None) -> torch.Tensor:
        sampler_model = execution_unet or self.unet
        quality = dict(getattr(self.config, "quality", {}) or {})
        denoise_val = float(getattr(self.config, "denoise_strength", None) or getattr(self.config, "denoise", None) or 1.0)
        sampler_instance = sampling.KSampler(
            sampler_model,
            int(self.config.steps),
            device,
            self.config.sampler,
            self.config.scheduler,
            denoise_val,
            model_options={"quality": quality} if quality else {},
        )
        return sampler_instance.sigmas

    def _resolve_sampler_function(self):
        sampler_name = self.config.sampler
        if sampler_name == "dpm_fast":
            def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_fast(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    len(sigmas) - 1,
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_fast_function

        if sampler_name == "dpm_adaptive":
            def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_adaptive(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_adaptive_function

        func_name = f"sample_{self.config.sampler.replace('_cfg_pp', '')}"
        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {self.config.sampler} not implemented in k_diffusion as {func_name}")
        return sampler_function

    def _calc_fullframe_cond_batch(
        self,
        execution_unet: Any,
        conds: list[Any],
        x_in: torch.Tensor,
        timestep: torch.Tensor,
    ) -> list[torch.Tensor]:
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

    def _noise_scaling(
        self,
        model_sampling: Any,
        sigma: Any,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        *,
        max_denoise: bool | None,
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
        denoise_mask: torch.Tensor | None,
    ):
        model_options = dict(getattr(execution_unet, "model_options", {}) or {})
        quality = dict(getattr(self.config, "quality", {}) or {})
        if quality and "quality" not in model_options:
            model_options["quality"] = quality
        disable_cfg1_optimization = bool(model_options.get("disable_cfg1_optimization", False))
        cfg_pp = "_cfg_pp" in self.config.sampler
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
                    max_denoise=None,
                ).to(dtype=x.dtype, device=x.device)
                x_input = x * active_mask + preserved_latent * latent_mask
            negative_conds = processed_conds.get("negative")
            if math.isclose(self.config.cfg, 1.0) and not disable_cfg1_optimization:
                negative_conds = None
            cond_pred, uncond_pred = self._calc_fullframe_cond_batch(
                execution_unet,
                [processed_conds.get("positive"), negative_conds],
                x_input,
                sigma,
            )
            diffusion_progress = self._diffusion_progress(model_sampling, sigma)
            cond_pred = self._apply_sharpness_quality(
                x_input,
                cond_pred,
                sharpness=float(quality.get("sharpness", 0.0)),
                diffusion_progress=diffusion_progress,
            )
            out = sampling.cfg_function(
                execution_unet.model,
                cond_pred,
                uncond_pred,
                self.config.cfg,
                x_input,
                sigma,
                model_options=model_options,
                cfg_pp=cfg_pp,
                adaptive_cfg=float(quality.get("adaptive_cfg", 0.0)),
                diffusion_progress=diffusion_progress,
            )
            if active_mask is not None and latent_mask is not None:
                latent_ref = latent_image.to(device=out.device, dtype=out.dtype)
                out = out * active_mask + latent_ref * latent_mask
            return out

        return model_fn

    def _run_prepared_denoise(
        self,
        attached_payload: dict[str, Any],
        *,
        device: torch.device,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> torch.Tensor:
        _ = device
        sigmas = attached_payload.get("sigmas")
        latent = attached_payload.get("latent")
        noise = attached_payload.get("noise")
        denoise_mask = attached_payload.get("denoise_mask")
        processed_conds = attached_payload.get("processed_conds")
        execution_unet = attached_payload.get("execution_unet") or self.unet
        if sigmas is None or latent is None or noise is None or processed_conds is None:
            raise RuntimeError("Unified SDXL runtime attached payload is missing denoise inputs.")
        if sigmas.shape[-1] == 0:
            return latent
        sampler_function = self._resolve_sampler_function()
        model_sampling = execution_unet.model.model_sampling
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
        samples = sampler_function(
            self._build_direct_model_callable(
                execution_unet,
                processed_conds,
                latent_image=latent,
                reference_noise=noise,
                denoise_mask=denoise_mask,
            ),
            scaled_noise,
            sigmas,
            extra_args={"denoise_mask": denoise_mask},
            callback=k_callback,
            disable=disable_pbar,
        )
        return model_sampling.inverse_noise_scaling(sigmas[-1], samples)
