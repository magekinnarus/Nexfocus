from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from backend.sdxl_runtime_contract import (
    InjectedFeatureArtifact,
    SpatialConditioningArtifact,
    StructuralConditioningArtifact,
)
from backend import resources


_SPATIAL_LATENT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SPATIAL_LATENT_CACHE_LIMIT = 8


def _clone_spatial_cache_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_spatial_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_spatial_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_spatial_cache_value(item) for item in value)
    return value


def load_spatial_latent_cache(cache_key: str | None) -> dict[str, Any] | None:
    if not cache_key:
        return None

    cached = _SPATIAL_LATENT_CACHE.get(cache_key)
    if cached is None:
        return None

    _SPATIAL_LATENT_CACHE.move_to_end(cache_key)
    return _clone_spatial_cache_value(cached)


def remember_spatial_latent_cache(cache_key: str | None, payload: dict[str, Any] | None) -> None:
    if not cache_key or payload is None:
        return

    _SPATIAL_LATENT_CACHE[cache_key] = _clone_spatial_cache_value(payload)
    _SPATIAL_LATENT_CACHE.move_to_end(cache_key)
    while len(_SPATIAL_LATENT_CACHE) > _SPATIAL_LATENT_CACHE_LIMIT:
        _SPATIAL_LATENT_CACHE.popitem(last=False)


def clear_spatial_latent_cache() -> None:
    _SPATIAL_LATENT_CACHE.clear()


class UnifiedSDXLRuntimeArtifactMixin:
    def _load_structural_runtime_modules(self):
        import modules.core as core

        return core

    def _encode_spatial_pixels_for_artifacts(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Unified SDXL runtime requires a loaded VAE before encoding spatial artifacts.")
        decode_device = self._execution_device()
        self._attach_vae(decode_device)
        try:
            if hasattr(self.vae, "first_stage_model"):
                self.vae.first_stage_model.to(device=decode_device)
            return self.vae.encode(pixels)["samples"].detach().cpu()
        finally:
            resources.eject_model(getattr(self.vae, "patcher", None))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _normalize_structural_task_shape(self, task: Any) -> tuple[Any, float, float]:
        if not isinstance(task, (list, tuple)) or len(task) < 3:
            raise ValueError(f"Unexpected structural task payload: {task!r}")
        return task[0], float(task[1]), float(task[2])

    def _normalize_structural_hint(self, payload: Any) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            tensor = payload.detach().cpu()
        else:
            tensor = torch.as_tensor(payload).detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected structural hint shaped [B, H, W, C], got {tuple(tensor.shape)}.")
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 1, 3)
        if tensor.shape[-1] != 3:
            raise ValueError(f"Expected structural hint with 3 channels, got {tuple(tensor.shape)}.")
        tensor = tensor.to(dtype=torch.float32).contiguous()
        if tensor.numel() and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _prepare_structural_conditioning_artifacts(
        self,
    ) -> tuple[StructuralConditioningArtifact | None, dict[str, Any], dict[str, float]]:
        structural_tasks = self.config.structural_tasks or {}
        if not structural_tasks:
            return None, {}, {}

        prepared_tasks: dict[str, list[tuple[torch.Tensor, float, float]]] = {}
        task_count = 0
        for cn_type, tasks in sorted(structural_tasks.items(), key=lambda item: str(item[0])):
            normalized_tasks: list[tuple[torch.Tensor, float, float]] = []
            for raw_task in tasks or ():
                hint, cn_stop, cn_weight = self._normalize_structural_task_shape(raw_task)
                normalized_tasks.append((self._normalize_structural_hint(hint), cn_stop, cn_weight))
            if normalized_tasks:
                prepared_tasks[str(cn_type)] = normalized_tasks
                task_count += len(normalized_tasks)

        if not prepared_tasks:
            return None, {}, {}

        artifact_fingerprint = self._hash_payload(
            {
                "structural_tasks": prepared_tasks,
                "controlnet_paths": self.config.controlnet_paths,
            }
        )
        artifact = StructuralConditioningArtifact(
            family="sdxl",
            variant=self.config.model_variant,
            source_fingerprint=self.base_model.fingerprint or self.config.checkpoint_path,
            artifact_fingerprint=artifact_fingerprint,
            task_count=task_count,
            control_types=tuple(sorted(prepared_tasks.keys())),
            reusable=True,
        )
        return (
            artifact,
            {"structural_tasks": prepared_tasks},
            {
                "structural_task_count": float(task_count),
                "structural_control_type_count": float(len(prepared_tasks)),
            },
        )

    def _load_structural_controlnet(self, cn_type: str) -> Any:
        from backend import loader, resources

        model_path = (self.config.controlnet_paths or {}).get(cn_type)
        if not model_path:
            return None
        if model_path in self._loaded_controlnets:
            controlnet = self._loaded_controlnets[model_path]
            if self.config.controlnet_quality:
                loader.patch_controlnet_for_quality(controlnet, self.config.controlnet_quality)
            return controlnet

        controlnet = resources.query_loaded_controlnet(model_path)
        if controlnet is None and not getattr(self, "_structural_controlnets_prefetched", False):
            requested_paths = [
                path for path in dict.fromkeys((self.config.controlnet_paths or {}).values()) if path
            ]
            if requested_paths:
                resources.trigger_refresh_controlnets(requested_paths)
            self._structural_controlnets_prefetched = True
            controlnet = resources.query_loaded_controlnet(model_path)
        if controlnet is not None:
            if self.config.controlnet_quality:
                loader.patch_controlnet_for_quality(controlnet, self.config.controlnet_quality)
            self._loaded_controlnets[model_path] = controlnet
            self._borrowed_controlnet_paths.add(model_path)
            return controlnet

        core = self._load_structural_runtime_modules()
        controlnet = core.load_controlnet(model_path)
        if self.config.controlnet_quality:
            loader.patch_controlnet_for_quality(controlnet, self.config.controlnet_quality)
        self._loaded_controlnets[model_path] = controlnet
        self._borrowed_controlnet_paths.discard(model_path)
        return controlnet

    def _unload_controlnets(self) -> None:
        borrowed_paths = set(getattr(self, "_borrowed_controlnet_paths", set()) or set())
        for model_path, controlnet in self._loaded_controlnets.items():
            if model_path in borrowed_paths:
                continue
            cleanup = getattr(controlnet, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._loaded_controlnets = {}
        self._borrowed_controlnet_paths = set()
        self._structural_controlnets_prefetched = False

    def _apply_structural_controlnets_to_conditioning(
        self,
        prepared_inputs: Any,
        positive: list[list[Any]],
        negative: list[list[Any]],
    ) -> tuple[list[list[Any]], list[list[Any]]]:
        payload = prepared_inputs.payload or {}
        structural_tasks = payload.get("structural_tasks") or {}
        if not structural_tasks:
            return positive, negative

        core = self._load_structural_runtime_modules()
        positive_cond = positive
        negative_cond = negative
        for cn_type in sorted(structural_tasks.keys()):
            controlnet = self._load_structural_controlnet(cn_type)
            if controlnet is None:
                continue
            for hint, cn_stop, cn_weight in structural_tasks.get(cn_type, ()):
                positive_cond, negative_cond = core.apply_controlnet(
                    positive_cond,
                    negative_cond,
                    controlnet,
                    hint,
                    cn_weight,
                    0.0,
                    cn_stop,
                )
        return positive_cond, negative_cond

    def _prepare_spatial_conditioning_artifacts(
        self,
    ) -> tuple[SpatialConditioningArtifact | None, dict[str, Any], dict[str, float]]:
        pixels = self._normalize_source_pixels(self.config.source_pixels)
        if pixels is None:
            return None, {}, {}
        if self.vae is None:
            raise RuntimeError("Unified SDXL runtime requires a loaded VAE before preparing spatial artifacts.")

        spatial_mode = self._resolve_spatial_mode()
        if spatial_mode == "outpaint":
            return self._prepare_outpaint_spatial_conditioning_artifacts(pixels)
        if spatial_mode == "inpaint":
            return self._prepare_inpaint_spatial_conditioning_artifacts(pixels)

        metrics: dict[str, float] = {"spatial_cache_hit": 0.0}
        image_fingerprint = self._hash_payload(pixels)

        latent_artifacts = self._resolve_spatial_latent_artifacts(
            spatial_mode="image",
            bb_pixels=pixels,
            bb_mask=None,
        )
        source_latent = latent_artifacts["route_latent"]
        metrics["source_vae_encode_cpu"] = float(latent_artifacts["encode_wall"])
        metrics["spatial_cache_hit"] = 1.0 if latent_artifacts["cache_hit"] else 0.0

        source_fingerprint = self._hash_payload(
            {
                "checkpoint": self.base_model.fingerprint if self.base_model is not None else self.config.checkpoint_path,
                "image": image_fingerprint,
                "target_width": int(self.config.width),
                "target_height": int(self.config.height),
            }
        )

        payload: dict[str, Any] = {
            "source_pixels": pixels,
            "source_latent": source_latent,
        }
        mask = self._normalize_source_mask(self.config.source_mask, pixels)
        bbox = (0, int(pixels.shape[1]), 0, int(pixels.shape[2]))
        bbox_area_ratio = 1.0
        mask_coverage = 0.0
        mask_fingerprint = None
        masked_latent = None
        bb_latent = None
        denoise_mask = None
        blend_mask = None

        if mask is not None:
            mask_fingerprint = self._hash_payload(mask)
            bbox = self._mask_bbox(mask[0])
            mask_coverage = float(mask.mean().item()) if mask.numel() else 0.0
            bbox_area_ratio = self._bbox_area_ratio(bbox, pixels.shape[1], pixels.shape[2])
            blend_mask = self._build_fullres_blend_mask(mask)

            masked_pixels = pixels * (1.0 - mask[..., None]) + 0.5 * mask[..., None]
            masked_encode_start = time.perf_counter()
            masked_latent = self._encode_spatial_pixels_for_artifacts(masked_pixels)
            metrics["masked_vae_encode_cpu"] = float(time.perf_counter() - masked_encode_start)

            bb_pixels = self._crop_and_resize_pixels(pixels, bbox)
            bb_mask = self._crop_and_resize_mask(mask, bbox)
            bb_masked_pixels = bb_pixels * (1.0 - bb_mask[..., None]) + 0.5 * bb_mask[..., None]
            bb_encode_start = time.perf_counter()
            bb_latent = self._encode_spatial_pixels_for_artifacts(bb_masked_pixels)
            metrics["bb_vae_encode_cpu"] = float(time.perf_counter() - bb_encode_start)
            denoise_mask = self._build_denoise_mask(bb_mask, bb_latent.shape)

            payload.update(
                {
                    "source_mask": mask,
                    "masked_source_latent": masked_latent,
                    "bb_pixels": bb_pixels,
                    "bb_mask": bb_mask,
                    "bb_latent": bb_latent,
                    "bb_denoise_mask": denoise_mask,
                    "bbox": bbox,
                    "blend_mask": blend_mask,
                }
            )
        else:
            metrics["masked_vae_encode_cpu"] = 0.0
            metrics["bb_vae_encode_cpu"] = 0.0

        source_latent_fingerprint = self._hash_payload(source_latent)
        masked_latent_fingerprint = self._hash_payload(masked_latent) if masked_latent is not None else None
        bb_latent_fingerprint = self._hash_payload(bb_latent) if bb_latent is not None else None
        denoise_mask_fingerprint = self._hash_payload(denoise_mask) if denoise_mask is not None else None
        artifact_fingerprint = self._hash_payload(
            {
                "source_fingerprint": source_fingerprint,
                "mask_fingerprint": mask_fingerprint,
                "source_latent_fingerprint": source_latent_fingerprint,
                "masked_latent_fingerprint": masked_latent_fingerprint,
                "bb_latent_fingerprint": bb_latent_fingerprint,
                "denoise_mask_fingerprint": denoise_mask_fingerprint,
                "blend_mask_fingerprint": self._hash_payload(blend_mask) if blend_mask is not None else None,
                "bbox": bbox,
            }
        )

        metrics["spatial_mask_coverage"] = float(mask_coverage)
        metrics["spatial_bbox_area_ratio"] = float(bbox_area_ratio)

        return (
            SpatialConditioningArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                spatial_mode=spatial_mode,
                source_fingerprint=source_fingerprint,
                image_fingerprint=image_fingerprint,
                mask_fingerprint=mask_fingerprint,
                artifact_fingerprint=artifact_fingerprint,
                source_latent_fingerprint=source_latent_fingerprint,
                masked_latent_fingerprint=masked_latent_fingerprint,
                bb_latent_fingerprint=bb_latent_fingerprint,
                denoise_mask_fingerprint=denoise_mask_fingerprint,
                bbox=tuple(int(v) for v in bbox),
                target_width=int(self.config.width),
                target_height=int(self.config.height),
                mask_coverage=float(mask_coverage),
                bbox_area_ratio=float(bbox_area_ratio),
                reusable=True,
            ),
            payload,
            metrics,
        )

    def _resolve_prepared_spatial_context(
        self,
        pixels: torch.Tensor,
        *,
        spatial_mode: str,
    ) -> dict[str, Any] | None:
        context = getattr(self.config, "resolved_spatial_context", None)
        if context is None:
            return None

        required_attrs = ("bb", "bb_image", "bb_mask", "blend_mask", "target_w", "target_h")
        missing = [name for name in required_attrs if getattr(context, name, None) is None]
        if missing:
            raise ValueError(
                f"Unified SDXL runtime resolved {spatial_mode} context is missing required fields: {', '.join(missing)}."
            )

        original_pixels = pixels.detach().cpu().contiguous()
        bb_pixels = self._normalize_source_pixels(getattr(context, "bb_image", None))
        if bb_pixels is None:
            raise RuntimeError(f"Unified SDXL runtime resolved {spatial_mode} context is missing BB pixels.")
        bb_mask = self._normalize_source_mask(getattr(context, "bb_mask", None), bb_pixels)
        if bb_mask is None:
            raise RuntimeError(f"Unified SDXL runtime resolved {spatial_mode} context is missing BB mask.")
        blend_mask = self._normalize_source_mask(getattr(context, "blend_mask", None), original_pixels)
        bbox = tuple(int(v) for v in getattr(context, "bb"))

        resolved = {
            "original_pixels": original_pixels,
            "bb_pixels": bb_pixels,
            "bb_mask": bb_mask,
            "blend_mask": blend_mask,
            "bbox": bbox,
            "target_width": int(getattr(context, "target_w")),
            "target_height": int(getattr(context, "target_h")),
        }
        if spatial_mode == "outpaint":
            working_pixels = self._normalize_source_pixels(getattr(context, "original_image", None))
            if working_pixels is None:
                working_pixels = original_pixels
            resolved["working_pixels"] = working_pixels
            resolved["working_mask"] = self._normalize_source_mask(getattr(context, "original_mask", None), working_pixels)
        return resolved

    def _resolve_spatial_vae_identity(self) -> str:
        vae_path = str(getattr(self.config, "vae_path", "") or "").strip()
        if vae_path:
            return str(self._fingerprint_source_path(vae_path) or vae_path)
        if self.base_model is not None and self.base_model.fingerprint:
            return f"embedded:{self.base_model.fingerprint}"
        return f"embedded:{self.config.checkpoint_path}"

    def _resolve_spatial_latent_artifacts(
        self,
        *,
        spatial_mode: str,
        bb_pixels: torch.Tensor,
        bb_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        vae_identity = self._resolve_spatial_vae_identity()
        bb_pixels_fingerprint = self._hash_payload(bb_pixels)
        mask_fingerprint = self._hash_payload(bb_mask) if bb_mask is not None else None
        cache_key = self._hash_payload(
            {
                "spatial_mode": str(spatial_mode or "image").strip().lower(),
                "vae_identity": vae_identity,
                "bb_pixels_fingerprint": bb_pixels_fingerprint,
                "bb_mask_fingerprint": mask_fingerprint,
            }
        )

        cached = load_spatial_latent_cache(cache_key)
        if cached is not None:
            logging.info(
                "[Nex-SpatialCache] hit route=%s mode=%s fingerprint=%s",
                self.route_label,
                spatial_mode,
                cache_key[:12],
            )
            return {
                **cached,
                "cache_hit": True,
                "encode_wall": 0.0,
                "vae_identity": vae_identity,
                "bb_pixels_fingerprint": bb_pixels_fingerprint,
                "mask_fingerprint": mask_fingerprint,
            }

        encode_start = time.perf_counter()
        route_latent = self._encode_spatial_pixels_for_artifacts(bb_pixels)
        encode_wall = time.perf_counter() - encode_start
        
        if bb_mask is not None:
            denoise_mask = self._build_denoise_mask(bb_mask, route_latent.shape)
            denoise_mask_fingerprint = self._hash_payload(denoise_mask)
        else:
            denoise_mask = None
            denoise_mask_fingerprint = None
            
        source_latent_fingerprint = self._hash_payload(route_latent)

        remember_spatial_latent_cache(
            cache_key,
            {
                "route_latent": route_latent,
                "denoise_mask": denoise_mask,
                "source_latent_fingerprint": source_latent_fingerprint,
                "denoise_mask_fingerprint": denoise_mask_fingerprint,
            },
        )
        logging.info(
            "[Nex-SpatialCache] miss route=%s mode=%s fingerprint=%s",
            self.route_label,
            spatial_mode,
            cache_key[:12],
        )
        return {
            "route_latent": route_latent,
            "denoise_mask": denoise_mask,
            "source_latent_fingerprint": source_latent_fingerprint,
            "denoise_mask_fingerprint": denoise_mask_fingerprint,
            "cache_hit": False,
            "encode_wall": float(encode_wall),
            "vae_identity": vae_identity,
            "bb_pixels_fingerprint": bb_pixels_fingerprint,
            "mask_fingerprint": mask_fingerprint,
        }

    def _prepare_inpaint_spatial_conditioning_artifacts(
        self,
        pixels: torch.Tensor,
    ) -> tuple[SpatialConditioningArtifact, dict[str, Any], dict[str, float]]:
        if pixels.shape[0] != 1:
            raise ValueError("Unified SDXL runtime inpaint preparation currently supports batch_size=1 only.")
        if self.vae is None:
            raise RuntimeError("Unified SDXL runtime requires a loaded VAE before preparing inpaint artifacts.")

        original_pixels = pixels.detach().cpu().contiguous()
        base_mask = self._normalize_source_mask(self.config.source_mask, pixels)
        prepared_context = self._resolve_prepared_spatial_context(pixels, spatial_mode="inpaint")
        if prepared_context is None:
            base_mask_2d = (
                (base_mask[0] * 255.0).to(dtype=torch.uint8).cpu().numpy()
                if base_mask is not None
                else None
            )

            from modules.pipeline.inpaint import InpaintPipeline
            image_np = (original_pixels[0].clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())

            inpaint = InpaintPipeline()
            prepare_start = time.perf_counter()
            context = inpaint.prepare(
                image=image_np,
                mask=base_mask_2d,
                extend_factor=1.2,
            )
            prepare_wall = time.perf_counter() - prepare_start
            bb_pixels = self._normalize_source_pixels(context.bb_image)
            if bb_pixels is None:
                raise RuntimeError("Unified SDXL runtime failed to prepare inpaint BB pixels.")
            bb_mask = self._normalize_source_mask(context.bb_mask, bb_pixels)
            if bb_mask is None:
                raise RuntimeError("Unified SDXL runtime failed to prepare inpaint BB mask.")
            blend_mask = self._normalize_source_mask(context.blend_mask, original_pixels)
            bbox = tuple(int(v) for v in context.bb)
            target_width = int(context.target_w)
            target_height = int(context.target_h)
        else:
            prepare_wall = 0.0
            bb_pixels = prepared_context["bb_pixels"]
            bb_mask = prepared_context["bb_mask"]
            blend_mask = prepared_context["blend_mask"]
            bbox = prepared_context["bbox"]
            target_width = int(prepared_context["target_width"])
            target_height = int(prepared_context["target_height"])

        image_fingerprint = self._hash_payload(original_pixels)
        latent_artifacts = self._resolve_spatial_latent_artifacts(
            spatial_mode="inpaint",
            bb_pixels=bb_pixels,
            bb_mask=bb_mask,
        )
        route_latent = latent_artifacts["route_latent"]
        denoise_mask = latent_artifacts["denoise_mask"]
        encode_wall = float(latent_artifacts["encode_wall"])
        mask_fingerprint = latent_artifacts["mask_fingerprint"]
        bb_pixels_fingerprint = latent_artifacts["bb_pixels_fingerprint"]
        vae_identity = latent_artifacts["vae_identity"]
        source_fingerprint = self._hash_payload(
            {
                "checkpoint": self.base_model.fingerprint if self.base_model is not None else self.config.checkpoint_path,
                "vae_identity": vae_identity,
                "image": image_fingerprint,
                "base_mask": self._hash_payload(base_mask) if base_mask is not None else None,
                "bb_pixels_fingerprint": bb_pixels_fingerprint,
                "bbox": bbox,
                "target_width": target_width,
                "target_height": target_height,
            }
        )
        source_latent_fingerprint = latent_artifacts["source_latent_fingerprint"]
        denoise_mask_fingerprint = latent_artifacts["denoise_mask_fingerprint"]
        artifact_fingerprint = self._hash_payload(
            {
                "source_fingerprint": source_fingerprint,
                "mask_fingerprint": mask_fingerprint,
                "source_latent_fingerprint": source_latent_fingerprint,
                "denoise_mask_fingerprint": denoise_mask_fingerprint,
                "blend_mask_fingerprint": self._hash_payload(blend_mask) if blend_mask is not None else None,
                "bbox": bbox,
            }
        )
        mask_coverage = float(bb_mask.mean().item()) if bb_mask.numel() else 0.0
        bbox_area_ratio = self._bbox_area_ratio(bbox, original_pixels.shape[1], original_pixels.shape[2])
        payload = {
            "source_pixels": original_pixels,
            "source_mask": base_mask,
            "source_latent": route_latent,
            "bb_pixels": bb_pixels,
            "bb_mask": bb_mask,
            "bb_latent": route_latent,
            "bb_denoise_mask": denoise_mask,
            "bbox": bbox,
            "blend_mask": blend_mask,
        }
        metrics = {
            "source_vae_encode_cpu": 0.0,
            "masked_vae_encode_cpu": 0.0,
            "bb_vae_encode_cpu": float(encode_wall),
            "spatial_mask_coverage": float(mask_coverage),
            "spatial_bbox_area_ratio": float(bbox_area_ratio),
            "inpaint_prepare_cpu": float(prepare_wall),
            "spatial_cache_hit": 1.0 if latent_artifacts["cache_hit"] else 0.0,
        }
        return (
            SpatialConditioningArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                spatial_mode="inpaint",
                source_fingerprint=source_fingerprint,
                image_fingerprint=image_fingerprint,
                mask_fingerprint=mask_fingerprint,
                artifact_fingerprint=artifact_fingerprint,
                source_latent_fingerprint=source_latent_fingerprint,
                masked_latent_fingerprint=None,
                bb_latent_fingerprint=source_latent_fingerprint,
                denoise_mask_fingerprint=denoise_mask_fingerprint,
                bbox=bbox,
                target_width=target_width,
                target_height=target_height,
                mask_coverage=float(mask_coverage),
                bbox_area_ratio=float(bbox_area_ratio),
                reusable=True,
            ),
            payload,
            metrics,
        )

    def _prepare_outpaint_spatial_conditioning_artifacts(
        self,
        pixels: torch.Tensor,
    ) -> tuple[SpatialConditioningArtifact, dict[str, Any], dict[str, float]]:
        if pixels.shape[0] != 1:
            raise ValueError("Unified SDXL runtime outpaint preparation currently supports batch_size=1 only.")
        if self.vae is None:
            raise RuntimeError("Unified SDXL runtime requires a loaded VAE before preparing outpaint artifacts.")

        original_pixels = pixels.detach().cpu().contiguous()
        base_mask = self._normalize_source_mask(self.config.source_mask, pixels)
        prepared_context = self._resolve_prepared_spatial_context(pixels, spatial_mode="outpaint")
        if prepared_context is None:
            base_mask_2d = (
                (base_mask[0] * 255.0).to(dtype=torch.uint8).cpu().numpy()
                if base_mask is not None
                else None
            )
            context = self._build_outpaint_context(original_pixels[0], base_mask_2d)
            bb_pixels = self._normalize_source_pixels(context["bb_pixels"])
            if bb_pixels is None:
                raise RuntimeError("Unified SDXL runtime failed to prepare outpaint BB pixels.")
            bb_mask = self._normalize_source_mask(context["bb_mask"], bb_pixels)
            if bb_mask is None:
                raise RuntimeError("Unified SDXL runtime failed to prepare outpaint BB mask.")
            blend_mask = self._normalize_source_mask(context["blend_mask"], original_pixels)
            bbox = tuple(int(v) for v in context["bbox"])
            target_width = int(context["target_width"])
            target_height = int(context["target_height"])
            working_pixels = self._normalize_source_pixels(context["working_pixels"])
            working_mask = self._normalize_source_mask(context["working_mask"], working_pixels)
            prepare_wall = float(context["prepare_wall"])
        else:
            bb_pixels = prepared_context["bb_pixels"]
            bb_mask = prepared_context["bb_mask"]
            blend_mask = prepared_context["blend_mask"]
            bbox = prepared_context["bbox"]
            target_width = int(prepared_context["target_width"])
            target_height = int(prepared_context["target_height"])
            working_pixels = prepared_context["working_pixels"]
            working_mask = prepared_context["working_mask"]
            prepare_wall = 0.0

        image_fingerprint = self._hash_payload(original_pixels)
        latent_artifacts = self._resolve_spatial_latent_artifacts(
            spatial_mode="outpaint",
            bb_pixels=bb_pixels,
            bb_mask=bb_mask,
        )
        route_latent = latent_artifacts["route_latent"]
        denoise_mask = latent_artifacts["denoise_mask"]
        encode_wall = float(latent_artifacts["encode_wall"])
        mask_fingerprint = latent_artifacts["mask_fingerprint"]
        bb_pixels_fingerprint = latent_artifacts["bb_pixels_fingerprint"]
        vae_identity = latent_artifacts["vae_identity"]
        source_fingerprint = self._hash_payload(
            {
                "checkpoint": self.base_model.fingerprint if self.base_model is not None else self.config.checkpoint_path,
                "vae_identity": vae_identity,
                "image": image_fingerprint,
                "base_mask": self._hash_payload(base_mask) if base_mask is not None else None,
                "bb_pixels_fingerprint": bb_pixels_fingerprint,
                "direction": str(self.config.outpaint_direction or "").strip().lower() or None,
                "expansion": int(self.config.outpaint_expansion_size),
                "pixelate": bool(self.config.outpaint_pixelate),
                "bbox": bbox,
                "target_width": target_width,
                "target_height": target_height,
            }
        )
        source_latent_fingerprint = latent_artifacts["source_latent_fingerprint"]
        denoise_mask_fingerprint = latent_artifacts["denoise_mask_fingerprint"]
        artifact_fingerprint = self._hash_payload(
            {
                "source_fingerprint": source_fingerprint,
                "mask_fingerprint": mask_fingerprint,
                "source_latent_fingerprint": source_latent_fingerprint,
                "denoise_mask_fingerprint": denoise_mask_fingerprint,
                "blend_mask_fingerprint": self._hash_payload(blend_mask) if blend_mask is not None else None,
                "bbox": bbox,
            }
        )
        mask_coverage = float(bb_mask.mean().item()) if bb_mask.numel() else 0.0
        bbox_area_ratio = self._bbox_area_ratio(bbox, original_pixels.shape[1], original_pixels.shape[2])
        payload = {
            "source_pixels": original_pixels,
            "source_mask": base_mask,
            "source_latent": route_latent,
            "bb_pixels": bb_pixels,
            "bb_mask": bb_mask,
            "bb_latent": route_latent,
            "bb_denoise_mask": denoise_mask,
            "bbox": bbox,
            "blend_mask": blend_mask,
            "outpaint_direction": str(self.config.outpaint_direction or "").strip().lower() or None,
            "outpaint_working_pixels": working_pixels,
            "outpaint_working_mask": working_mask,
        }
        metrics = {
            "source_vae_encode_cpu": 0.0,
            "masked_vae_encode_cpu": 0.0,
            "bb_vae_encode_cpu": float(encode_wall),
            "spatial_mask_coverage": float(mask_coverage),
            "spatial_bbox_area_ratio": float(bbox_area_ratio),
            "outpaint_prepare_cpu": float(prepare_wall),
            "spatial_cache_hit": 1.0 if latent_artifacts["cache_hit"] else 0.0,
        }
        return (
            SpatialConditioningArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                spatial_mode="outpaint",
                source_fingerprint=source_fingerprint,
                image_fingerprint=image_fingerprint,
                mask_fingerprint=mask_fingerprint,
                artifact_fingerprint=artifact_fingerprint,
                source_latent_fingerprint=source_latent_fingerprint,
                masked_latent_fingerprint=None,
                bb_latent_fingerprint=source_latent_fingerprint,
                denoise_mask_fingerprint=denoise_mask_fingerprint,
                bbox=bbox,
                target_width=target_width,
                target_height=target_height,
                mask_coverage=float(mask_coverage),
                bbox_area_ratio=float(bbox_area_ratio),
                reusable=True,
            ),
            payload,
            metrics,
        )

    def _build_outpaint_context(self, pixels: torch.Tensor, mask_2d: Any) -> dict[str, Any]:
        from modules.pipeline.outpaint import OutpaintPipeline

        image_np = (pixels.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())
        outpaint = OutpaintPipeline()
        direction = str(self.config.outpaint_direction or "").strip().lower() or None
        if mask_2d is None:
            if direction is None:
                raise ValueError("Unified SDXL runtime outpaint preparation requires outpaint_direction when source_mask is absent.")
            prepare_start = time.perf_counter()
            working_image, working_mask = outpaint.prepare_outpaint_canvas_only(
                image_np,
                direction,
                expansion_size=int(self.config.outpaint_expansion_size),
                pixelate=bool(self.config.outpaint_pixelate),
            )
            context = outpaint.prepare(
                image=working_image,
                mask=working_mask,
                outpaint_direction=None,
                extend_factor=1.2,
            )
            prepare_wall = time.perf_counter() - prepare_start
        else:
            prepare_start = time.perf_counter()
            working_image = image_np
            working_mask = mask_2d
            context = outpaint.prepare(
                image=working_image,
                mask=working_mask,
                outpaint_direction=direction,
                extend_factor=1.2,
            )
            prepare_wall = time.perf_counter() - prepare_start
        return {
            "working_pixels": working_image,
            "working_mask": working_mask,
            "bb_pixels": context.bb_image,
            "bb_mask": context.bb_mask,
            "blend_mask": context.blend_mask,
            "bbox": context.bb,
            "target_width": context.target_w,
            "target_height": context.target_h,
            "direction": direction,
            "prepare_wall": float(prepare_wall),
        }

    def _resolve_spatial_mode(self) -> str:
        normalized = str(self.config.spatial_mode or "").strip().lower()
        if normalized:
            return normalized
        if str(self.config.outpaint_direction or "").strip():
            return "outpaint"
        if self.config.source_mask is not None:
            return "inpaint"
        return "image"

    def _normalize_source_pixels(self, pixels: Any) -> torch.Tensor | None:
        if pixels is None:
            return None
        tensor = torch.as_tensor(pixels).detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] < 3:
            raise ValueError(
                "Unified SDXL runtime source_pixels must have shape [B, H, W, C] or [H, W, C] with at least 3 channels."
            )
        tensor = tensor[..., :3].to(dtype=torch.float32).contiguous()
        if tensor.numel() and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _normalize_source_mask(self, mask: Any, pixels: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        tensor = torch.as_tensor(mask).detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[:2] == pixels.shape[1:3]:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4:
            tensor = tensor.amax(dim=-1)
        if tensor.ndim != 3:
            raise ValueError(
                "Unified SDXL runtime source_mask must have shape [B, H, W], [H, W], or [B, H, W, C]."
            )
        if tensor.shape[0] == 1 and pixels.shape[0] > 1:
            tensor = tensor.repeat(int(pixels.shape[0]), 1, 1)
        if tensor.shape[0] != pixels.shape[0] or tensor.shape[1] != pixels.shape[1] or tensor.shape[2] != pixels.shape[2]:
            raise ValueError("Unified SDXL runtime source_mask must match source_pixels spatial shape and batch.")
        tensor = tensor.to(dtype=torch.float32).contiguous()
        if tensor.numel() and float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return (tensor > 0.5).to(dtype=torch.float32)

    def _mask_bbox(self, mask: torch.Tensor) -> tuple[int, int, int, int]:
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask when building a bounding box, got shape {tuple(mask.shape)}.")
        active = torch.nonzero(mask > 0.5, as_tuple=False)
        if active.numel() == 0:
            return (0, int(mask.shape[0]), 0, int(mask.shape[1]))
        y1 = int(active[:, 0].min().item())
        y2 = int(active[:, 0].max().item()) + 1
        x1 = int(active[:, 1].min().item())
        x2 = int(active[:, 1].max().item()) + 1
        return (y1, y2, x1, x2)

    def _bbox_area_ratio(self, bbox: tuple[int, int, int, int], height: int, width: int) -> float:
        y1, y2, x1, x2 = bbox
        bbox_area = max(0, y2 - y1) * max(0, x2 - x1)
        full_area = max(1, int(height) * int(width))
        return float(bbox_area) / float(full_area)

    def _crop_and_resize_pixels(
        self,
        pixels: torch.Tensor,
        bbox: tuple[int, int, int, int],
    ) -> torch.Tensor:
        y1, y2, x1, x2 = bbox
        crop = pixels[:, y1:y2, x1:x2, :].movedim(-1, 1)
        resized = torch.nn.functional.interpolate(
            crop,
            size=(int(self.config.height), int(self.config.width)),
            mode="bilinear",
            align_corners=False,
        )
        return resized.movedim(1, -1).contiguous().cpu()

    def _crop_and_resize_mask(
        self,
        mask: torch.Tensor,
        bbox: tuple[int, int, int, int],
    ) -> torch.Tensor:
        y1, y2, x1, x2 = bbox
        crop = mask[:, None, y1:y2, x1:x2]
        resized = torch.nn.functional.interpolate(
            crop,
            size=(int(self.config.height), int(self.config.width)),
            mode="nearest",
        )
        return resized[:, 0, :, :].contiguous().cpu()

    def _build_denoise_mask(
        self,
        mask: torch.Tensor,
        latent_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
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

    def _build_fullres_blend_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 3:
            raise ValueError(f"Expected a [B, H, W] mask when building blend masks, got shape {tuple(mask.shape)}.")
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required to build unified SDXL inpaint/outpaint blend masks.") from exc

        outputs: list[torch.Tensor] = []
        for item in mask.detach().cpu():
            mask_np = np.asarray((item > 0.5).to(dtype=torch.uint8).numpy() * 255, dtype=np.uint8)
            x_int16 = np.zeros_like(mask_np, dtype=np.int16)
            x_int16[mask_np > 127] = 256
            kernel = np.ones((3, 3), dtype=np.int16)
            for _ in range(32):
                maxed = cv2.dilate(x_int16, kernel) - 8
                x_int16 = np.maximum(maxed, x_int16)
            outputs.append(torch.from_numpy(np.clip(x_int16, 0, 255).astype(np.float32) / 255.0))
        return torch.stack(outputs, dim=0).contiguous()

    def _build_feature_boundary_placeholder(self) -> dict[str, InjectedFeatureArtifact]:
        return {
            "feature_boundary_placeholder": InjectedFeatureArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                block_id="diffusion_boundary",
                timestep_key="unbound",
                context_key="not-prepared-in-w07c2",
                feature_fingerprint=None,
                reusable=True,
            )
        }

    def _load_contextual_runtime_modules(self):
        from backend import ip_adapter, pulid_runtime

        return ip_adapter, pulid_runtime

    def _normalize_contextual_task_shape(self, task: Any) -> tuple[Any, float, float, float]:
        if not isinstance(task, (list, tuple)):
            raise ValueError(f"Unexpected contextual task payload: {task!r}")
        if len(task) >= 4:
            return task[0], float(task[1]), float(task[2]), float(task[3])
        if len(task) == 3:
            return task[0], float(task[1]), float(task[2]), 0.0
        raise ValueError(f"Unexpected contextual task shape: {task!r}")

    def _normalize_contextual_preprocessed_payload(self, payload: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        conds, unconds = payload
        if not isinstance(conds, (list, tuple)) or not isinstance(unconds, (list, tuple)):
            return None
        return (
            [torch.as_tensor(item).detach().cpu() for item in conds],
            [torch.as_tensor(item).detach().cpu() for item in unconds],
        )

    def _prepare_contextual_tasks_for_type(
        self,
        cn_type: str,
        tasks: Any,
        contextual_assets: dict[str, Any],
    ) -> tuple[list[tuple[Any, float, float, float]], float]:
        contextual_ip_adapter, active_pulid_runtime = self._load_contextual_runtime_modules()
        prepared_tasks: list[tuple[Any, float, float, float]] = []
        preprocess_start = time.perf_counter()
        contextual_model_paths = contextual_assets.get("contextual_model_paths", {}) or {}
        clip_vision_path = contextual_assets.get("clip_vision_path")
        ip_negative_path = contextual_assets.get("ip_negative_path")
        insightface_model_names = contextual_assets.get("insightface_model_names") or ["antelopev2"]
        eva_clip_path = contextual_assets.get("eva_clip_path")

        for raw_task in tasks or ():
            payload, cn_stop, cn_weight, cn_start = self._normalize_contextual_task_shape(raw_task)
            prepared_payload = self._normalize_contextual_preprocessed_payload(payload)
            if prepared_payload is None:
                if cn_type == "PuLID":
                    prepared_payload = active_pulid_runtime.preprocess(
                        self._coerce_contextual_input_image(payload),
                        model_path=contextual_model_paths.get(cn_type),
                        eva_clip_path=eva_clip_path,
                        insightface_model_names=insightface_model_names,
                    )
                else:
                    prepared_payload = contextual_ip_adapter.preprocess(
                        self._coerce_contextual_input_image(payload),
                        model_path=contextual_model_paths.get(cn_type),
                        clip_vision_path=clip_vision_path,
                        ip_negative_path=ip_negative_path,
                        insightface_model_names=insightface_model_names,
                        cache_kind=cn_type,
                    )
                prepared_payload = self._normalize_contextual_preprocessed_payload(prepared_payload)
            if prepared_payload is None:
                continue
            prepared_tasks.append((prepared_payload, cn_stop, cn_weight, cn_start))
        return prepared_tasks, time.perf_counter() - preprocess_start

    def _coerce_contextual_input_image(self, payload: Any) -> np.ndarray:
        array = np.asarray(payload)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected contextual input image shaped [H, W, C], got {array.shape!r}.")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0) * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])

    def _prepare_injected_feature_artifacts(
        self,
    ) -> tuple[dict[str, InjectedFeatureArtifact], dict[str, Any], dict[str, float]]:
        contextual_tasks = self.config.contextual_tasks or {}
        if not contextual_tasks:
            return self._build_feature_boundary_placeholder(), {}, {}

        prepared_payloads: dict[str, list[tuple[Any, float, float, float]]] = {}
        artifacts: dict[str, InjectedFeatureArtifact] = {}
        metrics: dict[str, float] = {}
        total_preprocess_cpu = 0.0
        total_task_count = 0

        for cn_type, tasks in sorted(contextual_tasks.items(), key=lambda item: str(item[0])):
            prepared_tasks, preprocess_cpu = self._prepare_contextual_tasks_for_type(
                str(cn_type),
                tasks,
                self.config.contextual_assets,
            )
            total_preprocess_cpu += preprocess_cpu
            if not prepared_tasks:
                continue
            total_task_count += len(prepared_tasks)
            prepared_payloads[str(cn_type)] = prepared_tasks
            feature_fingerprint = self._hash_payload(
                {
                    "cn_type": str(cn_type),
                    "tasks": prepared_tasks,
                    "assets": self.config.contextual_assets,
                }
            )
            feature_name = f"contextual:{cn_type}"
            artifacts[feature_name] = InjectedFeatureArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                block_id="attn2",
                timestep_key="variable",
                context_key=f"{cn_type}:{len(prepared_tasks)}",
                feature_fingerprint=feature_fingerprint,
                reusable=True,
            )
            metrics[f"contextual_{str(cn_type).lower()}_task_count"] = float(len(prepared_tasks))

        if not artifacts:
            return self._build_feature_boundary_placeholder(), {}, {}

        metrics["contextual_prepare_cpu"] = float(total_preprocess_cpu)
        metrics["contextual_task_count"] = float(total_task_count)
        return artifacts, {"contextual_tasks": prepared_payloads}, metrics

    def _build_contextual_execution_unet(self, prepared_inputs: Any) -> Any:
        from modules import flags

        contextual_ip_adapter, active_pulid_runtime = self._load_contextual_runtime_modules()
        execution_unet = self.unet
        payload = prepared_inputs.payload or {}
        contextual_tasks = payload.get("contextual_tasks") or {}
        if not contextual_tasks:
            return execution_unet

        ip_face_tasks: list[Any] = []
        for cn_type in (flags.cn_ip,):
            ip_face_tasks.extend(list(contextual_tasks.get(cn_type, ())))
        pulid_tasks = list(contextual_tasks.get(flags.cn_pulid, ()))

        if ip_face_tasks:
            execution_unet = contextual_ip_adapter.patch_model(execution_unet, ip_face_tasks)
        if pulid_tasks:
            execution_unet = active_pulid_runtime.patch_model(execution_unet, pulid_tasks)
        return execution_unet
