from __future__ import annotations

from dataclasses import dataclass
from modules.flux_fill_surface import normalize_flux_fill_inpaint_route

_DISABLED_VALUE = "Disabled"
_REMOVE_BG_GOAL = "remove_bg"
_REMOVE_OBJ_GOAL = "remove_obj"

_KNOWN_ROUTE_FAMILIES = {
    "txt2img": "txt2img",
    "upscale": "upscale",
    "super_upscale": "upscale",
    "color_enhanced_upscale": "upscale",
    "inpaint": "image_input",
    "outpaint": "image_input",
    "flux_inpaint": "flux_fill",
    "removal": "removal",
}


def normalize_current_tab(value) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "generate"}:
        return "txt2img"
    return normalized


def route_family_for_route_id(route_id: str | None) -> str | None:
    normalized = str(route_id or "").strip().lower()
    return _KNOWN_ROUTE_FAMILIES.get(normalized)


def _resolve_frozen_route_snapshot(state) -> tuple[str, str | None]:
    requested_route_id = str(getattr(state, "requested_route_id", "") or "").strip().lower()
    if requested_route_id == "":
        return "", None

    requested_route_family = str(getattr(state, "requested_route_family", "") or "").strip().lower()
    normalized_family = route_family_for_route_id(requested_route_id) or requested_route_family or None
    return requested_route_id, normalized_family


def _has_controlnet_tasks(state) -> bool:
    cn_tasks = getattr(state, "cn_tasks", {}) or {}
    if not isinstance(cn_tasks, dict):
        return False
    return any(len(tasks) > 0 for tasks in cn_tasks.values())


def _has_outpaint_signal(state) -> bool:
    return (
        bool(getattr(state, "outpaint_step2_checkbox", False))
        or bool(getattr(state, "outpaint_selections", []))
        or getattr(state, "outpaint_mask_image", None) is not None
    )


def _has_inpaint_mix_signal(state) -> bool:
    return (
        bool(getattr(state, "inpaint_step2_checkbox", False))
        or getattr(state, "inpaint_mask_image", None) is not None
        or getattr(state, "inpaint_context_mask_image", None) is not None
        or getattr(state, "inpaint_bb_image", None) is not None
    )


@dataclass(frozen=True)
class RouteIntent:
    current_tab: str
    input_image_active: bool
    has_controlnet_tasks: bool
    expects_controlnet: bool
    wants_removal: bool
    wants_upscale: bool
    wants_outpaint: bool
    wants_inpaint: bool
    wants_flux_inpaint: bool
    mixed_inpaint_request: bool
    mixed_outpaint_request: bool
    route_id: str
    route_family: str


def resolve_route_intent(state, *, prefer_runtime_route: bool = False) -> RouteIntent:
    current_tab = normalize_current_tab(getattr(state, "current_tab", ""))
    input_image_active = bool(getattr(state, "input_image_checkbox", False))
    has_controlnet_tasks = _has_controlnet_tasks(state)
    goals = set(getattr(state, "goals", []) or [])

    remove_bg_enabled = bool(getattr(state, "remove_bg_enabled", False) or (_REMOVE_BG_GOAL in goals))
    remove_obj_enabled = bool(getattr(state, "remove_obj_enabled", False) or (_REMOVE_OBJ_GOAL in goals))
    wants_removal = input_image_active and current_tab == "remove" and (remove_bg_enabled or remove_obj_enabled)

    uov_method = str(getattr(state, "uov_method", "") or "").strip().lower()
    wants_color_enhancement = uov_method in {"color enhancement", "color-enhanced-upscale"}
    wants_upscale = (
        input_image_active
        and current_tab == "uov"
        and getattr(state, "uov_input_image", None) is not None
        and uov_method not in {"", _DISABLED_VALUE.casefold()}
        and ("upscale" in uov_method or wants_color_enhancement)
    )

    mixed_outpaint_request = (
        input_image_active
        and current_tab == "ip"
        and bool(getattr(state, "mixing_image_prompt_and_outpaint", False))
        and has_controlnet_tasks
        and getattr(state, "outpaint_input_image", None) is not None
        and _has_outpaint_signal(state)
    )
    wants_outpaint = input_image_active and (
        (current_tab == "outpaint" and getattr(state, "outpaint_input_image", None) is not None)
        or mixed_outpaint_request
    )

    mixed_inpaint_request = (
        input_image_active
        and current_tab == "ip"
        and bool(getattr(state, "mixing_image_prompt_and_inpaint", False))
        and has_controlnet_tasks
        and getattr(state, "inpaint_input_image", None) is not None
        and _has_inpaint_mix_signal(state)
    )
    wants_inpaint = input_image_active and not wants_outpaint and (
        (current_tab == "inpaint" and getattr(state, "inpaint_input_image", None) is not None)
        or (mixed_inpaint_request and not mixed_outpaint_request)
    )
    wants_flux_inpaint = wants_inpaint and current_tab == "inpaint" and normalize_flux_fill_inpaint_route(
        getattr(state, "inpaint_route", None)
    ) == "flux"

    expects_controlnet = False
    if input_image_active:
        if wants_outpaint:
            expects_controlnet = has_controlnet_tasks and bool(getattr(state, "mixing_image_prompt_and_outpaint", False))
        elif wants_inpaint:
            expects_controlnet = has_controlnet_tasks and bool(getattr(state, "mixing_image_prompt_and_inpaint", False))
        elif wants_removal or wants_upscale:
            expects_controlnet = False
        else:
            expects_controlnet = has_controlnet_tasks

    route_id = "txt2img"
    route_family = "txt2img"
    if wants_removal:
        route_id = "removal"
        route_family = "removal"
    elif wants_upscale:
        if wants_color_enhancement:
            route_id = "color_enhanced_upscale"
        elif "super-upscale" in uov_method:
            route_id = "super_upscale"
        else:
            route_id = "upscale"
        route_family = "upscale"
    elif wants_outpaint:
        route_id = "outpaint"
        route_family = "image_input"
    elif wants_flux_inpaint:
        route_id = "flux_inpaint"
        route_family = "flux_fill"
    elif wants_inpaint:
        route_id = "inpaint"
        route_family = "image_input"

    requested_route_id, requested_route_family = _resolve_frozen_route_snapshot(state)
    if requested_route_id and requested_route_family is not None:
        route_id = requested_route_id
        route_family = requested_route_family
        wants_removal = route_id == "removal"
        wants_upscale = route_id in {"upscale", "super_upscale", "color_enhanced_upscale"}
        wants_outpaint = route_id == "outpaint"
        wants_flux_inpaint = route_id == "flux_inpaint"
        wants_inpaint = route_id in {"inpaint", "flux_inpaint"}

    if prefer_runtime_route:
        runtime_route_id = str(getattr(state, "runtime_route_id", "") or "").strip().lower()
        runtime_route_family = route_family_for_route_id(runtime_route_id)
        if runtime_route_id and runtime_route_family is not None:
            route_id = runtime_route_id
            route_family = runtime_route_family
            wants_removal = route_id == "removal"
            wants_upscale = route_id in {"upscale", "super_upscale", "color_enhanced_upscale"}
            wants_outpaint = route_id == "outpaint"
            wants_flux_inpaint = route_id == "flux_inpaint"
            wants_inpaint = route_id in {"inpaint", "flux_inpaint"}

    return RouteIntent(
        current_tab=current_tab,
        input_image_active=input_image_active,
        has_controlnet_tasks=has_controlnet_tasks,
        expects_controlnet=expects_controlnet,
        wants_removal=wants_removal,
        wants_upscale=wants_upscale,
        wants_outpaint=wants_outpaint,
        wants_inpaint=wants_inpaint,
        wants_flux_inpaint=wants_flux_inpaint,
        mixed_inpaint_request=mixed_inpaint_request,
        mixed_outpaint_request=mixed_outpaint_request,
        route_id=route_id,
        route_family=route_family,
    )
