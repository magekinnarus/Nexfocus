from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from backend.process_transition import (
    PROCESS_CLASS_FLUX_FILL,
    PROCESS_FAMILY_FLUX_FILL,
    ProcessKey,
    build_process_key,
    clear_active_runtime,
    set_active_runtime,
)
from backend.flux_fill_v3.contracts import UNetSpineKind, T5PostureKind

logger = logging.getLogger(__name__)

FLUX_FILL_TIER_FP8 = "fp8"
FLUX_FILL_AE_ASSET_ID = "inpaint.flux_fill.ae"
FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID = "inpaint.flux_fill.empty_conditioning"
FLUX_FILL_CLIP_L_ASSET_ID = "inpaint.flux_fill.text_encoder.clip_l"
FLUX_FILL_T5XXL_FP16_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.fp16"

FLUX_FILL_UNET_ASSET_BY_TIER = {
    FLUX_FILL_TIER_FP8: "inpaint.flux_fill.unet.fp8",
}
FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT = {
    "flux_fill_fp8": FLUX_FILL_UNET_ASSET_BY_TIER[FLUX_FILL_TIER_FP8],
}
FLUX_FILL_MODEL_VARIANT_BY_TIER = {
    FLUX_FILL_TIER_FP8: "flux_fill_fp8",
}


@dataclass(frozen=True)
class FluxFillActivationAssets:
    unet_path: str
    ae_path: str
    conditioning_cache_path: str
    model_variant: str
    conditioning_kind: str
    clip_l_path: str
    t5_path: str
    prompt: str


def _assign_task_state_attr(task_state: Any, name: str, value: Any) -> None:
    if task_state is None:
        return
    try:
        setattr(task_state, name, value)
    except Exception:
        pass


def _normalize_flux_fill_conditioning(value: Any) -> str:
    normalized = str(value or "empty").strip().lower().replace("-", "_").replace(" ", "_")
    return "empty" if normalized in {"", "empty"} else normalized


def _normalize_flux_fill_tier(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in FLUX_FILL_MODEL_VARIANT_BY_TIER:
        return normalized
    return None


def _coerce_positive_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if resolved <= 0.0:
        return None
    return resolved


def _merge_flux_fill_prompt_text(prompt: Any, additional_prompt: Any) -> str:
    prompt_text = str(prompt or "").strip()
    additional_prompt_text = str(additional_prompt or "").strip()
    if additional_prompt_text == "":
        return prompt_text
    if prompt_text == "":
        return additional_prompt_text
    return additional_prompt_text + "\n" + prompt_text


def _resolve_flux_fill_prompt_text(task_state: Any) -> str:
    if task_state is None:
        return ""

    remove_prompt = str(getattr(task_state, "remove_prompt", "") or "").strip()
    merged_inpaint_prompt = _merge_flux_fill_prompt_text(
        getattr(task_state, "prompt", ""),
        getattr(task_state, "inpaint_additional_prompt", ""),
    )
    goal_tokens = {str(goal or "").strip().lower() for goal in getattr(task_state, "goals", [])}
    current_tab = str(getattr(task_state, "current_tab", "") or "").strip().lower().replace("-", "_").replace(" ", "_")
    runtime_route_id = str(getattr(task_state, "runtime_route_id", "") or "").strip().lower()

    try:
        from modules.route_intent import resolve_route_intent
        intent = resolve_route_intent(task_state, prefer_runtime_route=True)
        if intent.wants_removal:
            return remove_prompt
        if intent.wants_inpaint:
            return merged_inpaint_prompt
    except Exception:
        pass

    if runtime_route_id in {"removal", "flux_removal"}:
        return remove_prompt
    if runtime_route_id == "flux_inpaint":
        return merged_inpaint_prompt
    if current_tab == "inpaint":
        return merged_inpaint_prompt
    if current_tab == "remove" and bool(getattr(task_state, "remove_obj_enabled", False)):
        return remove_prompt
    if {"remove_obj", "removal"} & goal_tokens:
        return remove_prompt
    return merged_inpaint_prompt


def resolve_flux_fill_total_ram_gb(source: Any | None = None) -> float:
    if source is not None:
        for attr_name in ("flux_fill_total_ram_gb", "total_ram_gb"):
            resolved = _coerce_positive_float(getattr(source, attr_name, None))
            if resolved is not None:
                return resolved

        for attr_name in ("flux_fill_total_ram_mb", "total_ram_mb", "hardware_total_ram_mb", "runtime_total_ram_mb"):
            resolved_mb = _coerce_positive_float(getattr(source, attr_name, None))
            if resolved_mb is not None:
                return resolved_mb / 1024.0

    try:
        from backend import resources
        profile = resources.active_memory_environment_profile()
        resolved_mb = _coerce_positive_float(getattr(profile, "total_ram_mb", None))
        if resolved_mb is not None:
            return resolved_mb / 1024.0
    except Exception:
        pass

    try:
        from backend.environment_profile import detect_total_ram_mb
        resolved_mb = _coerce_positive_float(detect_total_ram_mb())
        if resolved_mb is not None:
            return resolved_mb / 1024.0
    except Exception:
        pass

    return 0.0


def resolve_flux_fill_total_vram_mb(source: Any | None = None) -> float:
    if source is not None:
        for attr_name in ("flux_fill_total_vram_gb", "total_vram_gb"):
            resolved = _coerce_positive_float(getattr(source, attr_name, None))
            if resolved is not None:
                return resolved * 1024.0

        for attr_name in ("flux_fill_total_vram_mb", "total_vram_mb", "hardware_total_vram_mb", "runtime_total_vram_mb"):
            resolved_mb = _coerce_positive_float(getattr(source, attr_name, None))
            if resolved_mb is not None:
                return resolved_mb

    try:
        from backend import resources
        profile = resources.active_memory_environment_profile()
        resolved_mb = _coerce_positive_float(getattr(profile, "total_vram_mb", None))
        if resolved_mb is not None:
            return resolved_mb
    except Exception:
        pass

    try:
        from backend.environment_profile import detect_total_vram_mb
        resolved_mb = _coerce_positive_float(detect_total_vram_mb())
        if resolved_mb is not None:
            return resolved_mb
    except Exception:
        pass

    return 0.0


def _resolve_flux_fill_model_variant(task_state: Any) -> str:
    return "flux_fill_fp8"


def resolve_flux_fill_assets(task_state: Any) -> FluxFillActivationAssets | None:
    from modules import model_registry

    conditioning_kind = _normalize_flux_fill_conditioning(getattr(task_state, "flux_fill_conditioning", None))

    direct_unet_path = str(
        getattr(task_state, "flux_fill_unet_path", None) or getattr(task_state, "unet_path", None) or ""
    ).strip()
    direct_ae_path = str(
        getattr(task_state, "flux_fill_ae_path", None) or getattr(task_state, "ae_path", None) or ""
    ).strip()
    direct_conditioning_path = str(
        getattr(task_state, "flux_fill_conditioning_cache_path", None)
        or getattr(task_state, "conditioning_cache_path", None)
        or ""
    ).strip()

    model_variant = _resolve_flux_fill_model_variant(task_state)
    unet_asset_id = FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT.get(
        model_variant,
        FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT["flux_fill_fp8"],
    )
    if direct_unet_path:
        unet_path = direct_unet_path
    else:
        unet_path = model_registry.ensure_asset(unet_asset_id)

    if direct_ae_path:
        ae_path = direct_ae_path
    else:
        ae_path = model_registry.ensure_asset(FLUX_FILL_AE_ASSET_ID)

    prompt_text = _resolve_flux_fill_prompt_text(task_state)

    if prompt_text != "":
        conditioning_kind = "prompt"

    direct_clip_l_path = str(
        getattr(task_state, "flux_fill_clip_l_path", None) or getattr(task_state, "clip_l_path", None) or ""
    ).strip()
    direct_t5_path = str(
        getattr(task_state, "flux_fill_t5_path", None) or getattr(task_state, "t5_path", None) or ""
    ).strip()

    if direct_clip_l_path:
        clip_l_path = direct_clip_l_path
    else:
        clip_l_path = model_registry.ensure_asset(FLUX_FILL_CLIP_L_ASSET_ID)

    if direct_t5_path:
        t5_path = direct_t5_path
    else:
        t5_path = model_registry.ensure_asset(FLUX_FILL_T5XXL_FP16_ASSET_ID)

    if direct_conditioning_path:
        conditioning_cache_path = direct_conditioning_path
    elif conditioning_kind == "empty":
        conditioning_cache_path = model_registry.ensure_asset(FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID)
    else:
        # Prompt-conditioned caching path
        from backend.flux_fill_v3.t5_worker import get_prompt_cache_path
        conditioning_cache_path = get_prompt_cache_path(
            prompt_text,
            clip_l_path=clip_l_path,
            t5_path=t5_path,
            cache_mode=getattr(task_state, "flux_fill_prompt_cache", "temp"),
        )

    assets = FluxFillActivationAssets(
        unet_path=str(unet_path),
        ae_path=str(ae_path),
        conditioning_cache_path=str(conditioning_cache_path),
        model_variant=model_variant,
        conditioning_kind=conditioning_kind,
        clip_l_path=str(clip_l_path),
        t5_path=str(t5_path),
        prompt=prompt_text,
    )
    _assign_task_state_attr(task_state, "flux_fill_model_variant", assets.model_variant)
    _assign_task_state_attr(task_state, "flux_fill_unet_path", assets.unet_path)
    _assign_task_state_attr(task_state, "flux_fill_ae_path", assets.ae_path)
    _assign_task_state_attr(task_state, "flux_fill_conditioning_cache_path", assets.conditioning_cache_path)
    _assign_task_state_attr(task_state, "flux_fill_clip_l_path", assets.clip_l_path)
    _assign_task_state_attr(task_state, "flux_fill_t5_path", assets.t5_path)
    return assets


def resolve_flux_fill_spine_kind(task_state: Any) -> UNetSpineKind:
    """ Resolves the greenfield UNetSpineKind from task state parameters. """
    if task_state is None:
        return UNetSpineKind.STREAMING

    # Greenfield runtime posture option
    posture = str(
        getattr(task_state, "flux_fill_runtime_posture", None)
        or getattr(task_state, "flux_fill_unet_spine", None)
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if posture == "resident":
        return UNetSpineKind.RESIDENT
    if posture == "streaming":
        return UNetSpineKind.STREAMING

    try:
        from backend.staging_manager import (
            FLUX_RUNTIME_POSTURE_RESIDENT,
            PlacementSolver,
        )

        total_vram_mb = resolve_flux_fill_total_vram_mb(task_state)
        total_ram_mb = resolve_flux_fill_total_ram_gb(task_state) * 1024.0
        requested_variant = _resolve_flux_fill_model_variant(task_state)
        plan = PlacementSolver.solve(total_vram_mb, total_ram_mb, requested_variant)
        runtime_posture = str(getattr(plan, "runtime_posture", "") or "").strip().lower()
        if runtime_posture == FLUX_RUNTIME_POSTURE_RESIDENT:
            return UNetSpineKind.RESIDENT
    except Exception:
        pass

    return UNetSpineKind.STREAMING


def resolve_flux_fill_request_t5_posture(
    task_state: Any,
    *,
    spine_kind: UNetSpineKind | None = None,
) -> T5PostureKind:
    if task_state is not None:
        requested = str(getattr(task_state, "flux_fill_t5_posture", "disk_paged") or "").strip().lower()
        if requested == "cpu_resident":
            total_ram_gb = resolve_flux_fill_total_ram_gb(task_state)
            if total_ram_gb >= 31.0:
                return T5PostureKind.CPU_RESIDENT
    return T5PostureKind.DISK_PAGED


def resolve_flux_fill_process_key(
    task_state: Any,
    *,
    route_family: str | None = None,
    selected_engine: str | None = None,
) -> ProcessKey | None:
    from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL, normalize_objr_engine

    # Resolve selected engine
    if selected_engine is None and task_state is not None:
        selected_engine = normalize_objr_engine(getattr(task_state, "objr_engine", None))

    is_flux_fill = False
    if str(route_family or "").strip().lower() == "flux_fill":
        inpaint_route = getattr(task_state, "inpaint_route", None)
        if inpaint_route == "flux" or selected_engine == OBJR_ENGINE_FLUX_FILL:
            is_flux_fill = True
    elif selected_engine == OBJR_ENGINE_FLUX_FILL:
        is_flux_fill = True

    if not is_flux_fill:
        return None

    spine_kind = resolve_flux_fill_spine_kind(task_state)
    _assign_task_state_attr(task_state, "flux_fill_unet_spine", spine_kind.value)

    assets = resolve_flux_fill_assets(task_state)
    if assets is None:
        return None

    t5_posture_kind = resolve_flux_fill_request_t5_posture(task_state, spine_kind=spine_kind)
    _assign_task_state_attr(task_state, "flux_fill_t5_posture", t5_posture_kind.value)

    # Normalize cache path for identity comparisons to avoid prompt resets
    identity_conditioning_path = assets.conditioning_cache_path
    if assets.conditioning_kind == "prompt":
        identity_conditioning_path = "prompt_conditioning"

    identity = tuple(
        sorted(
            (
                ("ae_path", assets.ae_path),
                ("conditioning_cache_path", identity_conditioning_path),
                ("model_variant", assets.model_variant),
                ("unet_path", assets.unet_path),
                ("unet_spine", spine_kind.value),
                ("clip_l_path", assets.clip_l_path),
                ("t5_path", assets.t5_path),
            )
        )
    )

    return build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=identity,
        residency_class="resident" if spine_kind == UNetSpineKind.RESIDENT else "streaming",
        route_family="flux_fill",
    )


def sync_flux_fill_process_activation(
    route: Any,
    task_state: Any,
    requested_process_key: ProcessKey | None,
) -> Any:
    if (
        requested_process_key is not None
        and requested_process_key.family == PROCESS_FAMILY_FLUX_FILL
    ):
        spine_kind = resolve_flux_fill_spine_kind(task_state)
        safe_to_retain = (spine_kind == UNetSpineKind.RESIDENT)
        set_active_runtime(
            family=PROCESS_FAMILY_FLUX_FILL,
            key=requested_process_key,
            route_owner=route.route_id,
            safe_to_retain=safe_to_retain,
        )
    else:
        clear_active_runtime()
    return None


def resolve_flux_fill_t5_posture(unet_spine: UNetSpineKind, total_ram_gb: float | None = None) -> T5PostureKind:
    return T5PostureKind.DISK_PAGED


