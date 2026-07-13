from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend import resources
from backend import environment_profile as environment_profiles
from backend.flux_fill_v3.activation import (
    resolve_flux_fill_assets,
    resolve_flux_fill_process_key,
    resolve_flux_fill_request_t5_posture,
    resolve_flux_fill_spine_kind,
    sync_flux_fill_process_activation,
)
from backend.flux_fill_v3.contracts import (
    FluxFillCategory,
    FluxFillPreviewContext,
    FluxFillRequest,
)
from backend.flux_fill_v3.director import FluxAssemblyDirector
from modules.pipeline.inference import get_sampling_callback

logger = logging.getLogger(__name__)


def _shape_of_array(value) -> tuple[int, ...] | None:
    if isinstance(value, np.ndarray):
        return tuple(int(dim) for dim in value.shape)
    return None


def _mask_fill_ratio(mask) -> float | None:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    mask_np = mask[:, :, 0] if mask.ndim == 3 else mask
    if mask_np.ndim != 2:
        return None
    return float(np.count_nonzero(mask_np > 127)) / float(mask_np.size)


def _should_force_flux_host_cleanup() -> bool:
    try:
        profile = resources.active_memory_environment_profile()
        profile_name = getattr(profile, "name", None)
        return profile_name in (
            environment_profiles.PROFILE_COLAB_FREE,
            environment_profiles.PROFILE_LOCAL_LOW_VRAM,
        )
    except Exception:
        return False


def _publish_flux_removal_runtime(context, task_state) -> None:
    try:
        from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL

        requested_key = resolve_flux_fill_process_key(
            task_state,
            route_family="flux_fill",
            selected_engine=OBJR_ENGINE_FLUX_FILL,
        )
        route_id = str(getattr(context, "route_id", "") or "flux_removal")
        sync_flux_fill_process_activation(
            SimpleNamespace(route_id=route_id),
            task_state,
            requested_key,
        )
    except Exception:
        logger.debug("Failed to publish Flux removal runtime ownership.", exc_info=True)


def execute_flux_fill_removal(context, *, progress_percent_start: int = 10):
    import modules.objr_engine as objr_engine

    task_state = context.task_state
    force_host_cleanup = _should_force_flux_host_cleanup()

    with resources.memory_phase_scope(
        "diffusion",
        task=task_state,
        notes={"route": "flux_removal"},
        end_notes={"route": "flux_removal", "completed": True},
    ):
        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, progress_percent_start, "Object Removal Starting...")

        with Image.open(task_state.remove_base_image) as img_pil:
            image_np = np.array(img_pil.convert("RGB"))
        with Image.open(task_state.remove_mask_image) as mask_pil:
            mask_np = np.array(mask_pil.convert("L"))

        prepared_mask = objr_engine.prepare_flux_fill_mask(
            mask_np,
            grow=task_state.objr_mask_dilate,
            blur=task_state.objr_mask_blur,
        )

        assets = resolve_flux_fill_assets(task_state)
        spine_kind = resolve_flux_fill_spine_kind(task_state)
        t5_posture = resolve_flux_fill_request_t5_posture(task_state, spine_kind=spine_kind)

        logger.debug(
            "[Flux Telemetry] Removal route request image=%s mask=%s mask_fill=%.4f "
            "prompt_chars=%s preview_interval=%s force_host_cleanup=%s seed=%s steps=%s sampler=%s "
            "scheduler=%s blend=%s",
            _shape_of_array(image_np),
            _shape_of_array(prepared_mask),
            _mask_fill_ratio(prepared_mask) or 0.0,
            len(str(assets.prompt or "")),
            getattr(task_state, "preview_update_interval", None),
            force_host_cleanup,
            int(task_state.seed),
            int(task_state.steps),
            task_state.sampler_name,
            task_state.scheduler_name,
            getattr(task_state, "objr_blend_mode", None),
        )

        req = FluxFillRequest(
            unet_path=assets.unet_path,
            ae_path=assets.ae_path,
            conditioning_cache_path=assets.conditioning_cache_path,
            seed=int(task_state.seed),
            steps=int(task_state.steps),
            sampler=task_state.sampler_name,
            scheduler=task_state.scheduler_name,
            prefetch_depth=int(getattr(task_state, "prefetch_depth", 1)),
            prefetch_chunk_mb=int(getattr(task_state, "prefetch_chunk_mb", 64)),
            unet_spine=spine_kind,
            t5_posture=t5_posture,
            disk_paged_t5_gc_interval=getattr(task_state, "flux_fill_disk_paged_t5_gc_interval", "auto"),
            image=image_np,
            mask=prepared_mask,
            prompt=assets.prompt,
            blend_mode=task_state.objr_blend_mode,
            clip_l_path=assets.clip_l_path,
            t5_path=assets.t5_path,
            category=FluxFillCategory.REMOVAL,
        )

        preview_context = None

        def preview_transform(latent):
            nonlocal preview_context
            if preview_context is None:
                from ldm_patched.modules import latent_formats

                preview_context = FluxFillPreviewContext(latent_formats.Flux(), latent.device)
            return preview_context.decode(latent)

        callback = get_sampling_callback(
            task_state,
            context.progressbar_callback,
            0,
            1,
            0,
            int(task_state.steps),
            preview_transform=preview_transform,
        )

        director = FluxAssemblyDirector()
        assembly = director.select_assembly(req)
        _publish_flux_removal_runtime(context, task_state)
        result = assembly.execute(req, callback=callback)

        resources.cleanup_memory(
            "flux_removal_image_complete",
            gc_collect=force_host_cleanup,
            trim_host=force_host_cleanup,
            notes={"route_id": "flux_removal"},
            target_phase=resources.MemoryPhase.DIFFUSION,
            task=task_state,
        )
        return result
