"""Named compatibility translation into the authoritative workflow compiler.

Production queue entry may use ``capture_workflow_selection`` and
``capture_controlnet_slot_inputs``.  Late compilation helpers in this module
exist only for legacy/direct probes and must not be used by accepted queued
execution paths.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

from modules.flux_fill_surface import normalize_objr_engine
from modules.pipeline.workflow_compiler import compile_workflow_plan
from modules.pipeline.workflow_contracts import (
    FrozenControlNetSlotInput,
    FrozenWorkflowPlan,
    FrozenWorkflowSelection,
    VALID_WORKFLOW_SURFACES,
    normalize_workflow_token,
    require_workflow_plan,
)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _contains(source: Any, name: str) -> bool:
    if isinstance(source, Mapping):
        return name in source
    return hasattr(source, name)


def _legacy_route_surface(source: Any) -> str:
    requested = normalize_workflow_token(_value(source, "requested_route_id", ""))
    if requested in {"txt2img", "generate"}:
        return "normal_generate"
    if requested in {
        "inpaint", "outpaint", "removal", "upscale",
        "super_upscale", "color_enhanced_upscale",
    }:
        return requested
    if requested == "flux_inpaint":
        return "inpaint"
    if requested == "flux_removal":
        return "removal"

    goals = {normalize_workflow_token(item) for item in (_value(source, "goals", ()) or ())}
    if "removal" in goals or "remove_bg" in goals or "remove_obj" in goals:
        return "removal"
    if "outpaint" in goals:
        return "outpaint"
    if "inpaint" in goals:
        return "inpaint"
    if "cn" in goals:
        return "controlnet"
    return ""


def derive_source_surface(source: Any, *, queue_capture: bool = False) -> str:
    """Translate app/legacy fields into one frozen Generate surface."""
    input_image_known = _contains(source, "input_image_checkbox")
    input_image_enabled = bool(_value(source, "input_image_checkbox", False))
    if queue_capture and input_image_known and not input_image_enabled:
        return "normal_generate"

    explicit = normalize_workflow_token(_value(source, "requested_source_surface", ""))
    if explicit in VALID_WORKFLOW_SURFACES:
        return explicit

    legacy = _legacy_route_surface(source)
    if legacy:
        return legacy

    if input_image_known and not input_image_enabled:
        return "normal_generate"

    tab = normalize_workflow_token(_value(source, "current_tab", ""))
    if tab in {"ip", "controlnet", "control_net", "image_prompt"}:
        return "controlnet"
    if tab == "inpaint":
        return "inpaint"
    if tab == "outpaint":
        return "outpaint"
    if tab in {"remove", "removal"}:
        return "removal"
    if tab in {"uov", "upscale", "super_upscale"}:
        method = normalize_workflow_token(_value(source, "uov_method", ""))
        if "color" in method and "enhance" in method:
            return "color_enhanced_upscale"
        if "super" in method and "upscale" in method:
            return "super_upscale"
        if method not in {"", "disabled", "none"}:
            return "upscale"
    return "normal_generate"


def capture_workflow_selection(source: Any, *, queue_capture: bool = False) -> FrozenWorkflowSelection:
    """Freeze the Layer 0 values required by the pure compiler."""
    surface = derive_source_surface(source, queue_capture=queue_capture)
    goals = {normalize_workflow_token(item) for item in (_value(source, "goals", ()) or ())}
    remove_background = bool(_value(source, "remove_bg_enabled", False) or "remove_bg" in goals)
    remove_object = bool(_value(source, "remove_obj_enabled", False) or "remove_obj" in goals)
    selection = FrozenWorkflowSelection(
        source_surface=surface,
        input_image_enabled=bool(_value(source, "input_image_checkbox", False)),
        inpaint_route=str(_value(source, "inpaint_route", "sdxl") or "sdxl"),
        object_removal_engine=normalize_objr_engine(_value(source, "objr_engine", None)),
        remove_background=remove_background,
        remove_object=remove_object,
        upscale_method=str(_value(source, "uov_method", "") or ""),
        allow_inpaint_controlnet=bool(_value(source, "mixing_image_prompt_and_inpaint", False)),
        allow_outpaint_controlnet=bool(_value(source, "mixing_image_prompt_and_outpaint", False)),
    )
    selection.validate()
    return selection


def capture_controlnet_slot_inputs(raw_maps: Any) -> tuple[FrozenControlNetSlotInput, ...]:
    """Translate parsed raw slot maps without deciding whether they are active."""
    if not isinstance(raw_maps, Mapping):
        return ()
    slots = []
    for raw_type, tasks in raw_maps.items():
        for ordinal, task in enumerate(tasks or ()):
            values = list(task) if isinstance(task, (list, tuple)) else []
            slots.append(FrozenControlNetSlotInput(
                ui_slot_index=values[4] if len(values) > 4 else ordinal,
                control_type=str(raw_type or ""),
                input_image=values[0] if len(values) > 0 else None,
                end_percent=values[1] if len(values) > 1 else 1.0,
                weight=values[2] if len(values) > 2 else 1.0,
                start_percent=values[3] if len(values) > 3 else 0.0,
            ))
    return tuple(slots)


def compile_legacy_workflow_plan(
    state: Any,
    *,
    source_surface: Optional[str] = None,
) -> FrozenWorkflowPlan:
    """Explicit legacy/direct-probe adapter; never a production fallback."""
    selection = capture_workflow_selection(state, queue_capture=False)
    if source_surface is not None:
        selection = replace(selection, source_surface=normalize_workflow_token(source_surface))
    return compile_workflow_plan(selection, capture_controlnet_slot_inputs(_value(state, "cn_tasks", {})))


def bind_legacy_workflow_plan(
    state: Any,
    *,
    source_surface: Optional[str] = None,
) -> FrozenWorkflowPlan:
    existing = getattr(state, "workflow_plan", None)
    if existing is not None:
        return require_workflow_plan(state)
    plan = compile_legacy_workflow_plan(state, source_surface=source_surface)
    setter = getattr(state, "set_workflow_plan", None)
    if callable(setter):
        setter(plan)
    else:
        setattr(state, "workflow_plan", plan)
    bound = require_workflow_plan(state)
    if bound is not plan:
        raise RuntimeError("Legacy workflow adapter failed to bind the compiled plan exactly once")
    return bound


__all__ = [
    "bind_legacy_workflow_plan",
    "capture_controlnet_slot_inputs",
    "capture_workflow_selection",
    "compile_legacy_workflow_plan",
    "derive_source_surface",
]
