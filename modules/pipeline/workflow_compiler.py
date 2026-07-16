"""Pure Layer 1 workflow compiler.

The compiler consumes only immutable queue selection and parsed slot inputs.
It never reads TaskState, UI state, goals, prepared assets, or runtime state.
"""

from __future__ import annotations

from typing import Iterable

from modules import flags
from modules.flux_fill_surface import (
    OBJR_ENGINE_FLUX_FILL,
    is_flux_fill_inpaint_route,
    normalize_objr_engine,
)
from modules.pipeline.workflow_contracts import (
    AUXILIARY_BACKGROUND_REMOVAL,
    AUXILIARY_GAN_UPSCALE,
    AUXILIARY_MAT_INPAINT,
    ControlNetOverlayPlan,
    EXECUTION_KIND_AUXILIARY,
    EXECUTION_KIND_MAIN,
    FrozenControlNetSlotDescriptor,
    FrozenControlNetSlotInput,
    FrozenExecutionDeclaration,
    FrozenExecutionStep,
    FrozenWorkflowPlan,
    FrozenWorkflowSelection,
    MAIN_FAMILY_FLUX_FILL,
    MAIN_FAMILY_SDXL,
    freeze_workflow_payload,
    workflow_payload_fingerprint,
)


def _base_route(selection: FrozenWorkflowSelection) -> tuple[str, str]:
    surface = selection.source_surface
    if surface == "inpaint":
        if is_flux_fill_inpaint_route(selection.inpaint_route):
            return "flux_inpaint", "flux_fill"
        return "inpaint", "image_input"
    if surface == "outpaint":
        return "outpaint", "image_input"
    if surface == "removal":
        engine = normalize_objr_engine(selection.object_removal_engine)
        if selection.remove_object and engine == OBJR_ENGINE_FLUX_FILL:
            return "flux_removal", "flux_fill"
        return "removal", "removal"
    if surface in {"upscale", "super_upscale", "color_enhanced_upscale"}:
        return surface, "upscale"
    return "txt2img", "txt2img"


def _execution_declaration(
    selection: FrozenWorkflowSelection,
    route_id: str,
) -> FrozenExecutionDeclaration:
    if route_id in {"flux_inpaint", "flux_removal"}:
        return FrozenExecutionDeclaration(
            main_family=MAIN_FAMILY_FLUX_FILL,
            ordered_steps=(
                FrozenExecutionStep("flux_fill", EXECUTION_KIND_MAIN, MAIN_FAMILY_FLUX_FILL),
            ),
        )
    if route_id == "upscale":
        return FrozenExecutionDeclaration(
            ordered_auxiliary_requirements=(AUXILIARY_GAN_UPSCALE,),
            ordered_steps=(
                FrozenExecutionStep("gan_upscale", EXECUTION_KIND_AUXILIARY, AUXILIARY_GAN_UPSCALE),
            ),
        )
    if route_id == "removal":
        auxiliary = []
        steps = []
        if selection.remove_background:
            auxiliary.append(AUXILIARY_BACKGROUND_REMOVAL)
            steps.append(FrozenExecutionStep(
                "background_removal",
                EXECUTION_KIND_AUXILIARY,
                AUXILIARY_BACKGROUND_REMOVAL,
            ))
        if selection.remove_object:
            auxiliary.append(AUXILIARY_MAT_INPAINT)
            steps.append(FrozenExecutionStep(
                "mat_inpaint",
                EXECUTION_KIND_AUXILIARY,
                AUXILIARY_MAT_INPAINT,
            ))
        return FrozenExecutionDeclaration(
            ordered_auxiliary_requirements=tuple(auxiliary),
            ordered_steps=tuple(steps),
        )
    if route_id == "color_enhanced_upscale":
        return FrozenExecutionDeclaration(
            main_family=MAIN_FAMILY_SDXL,
            ordered_auxiliary_requirements=(AUXILIARY_GAN_UPSCALE,),
            ordered_steps=(
                FrozenExecutionStep("gan_upscale", EXECUTION_KIND_AUXILIARY, AUXILIARY_GAN_UPSCALE),
                FrozenExecutionStep("sdxl_color_pass", EXECUTION_KIND_MAIN, MAIN_FAMILY_SDXL),
            ),
        )
    return FrozenExecutionDeclaration(
        main_family=MAIN_FAMILY_SDXL,
        ordered_steps=(
            FrozenExecutionStep("sdxl", EXECUTION_KIND_MAIN, MAIN_FAMILY_SDXL),
        ),
    )


def _overlay_permission(selection: FrozenWorkflowSelection) -> tuple[bool, str]:
    surface = selection.source_surface
    if surface == "controlnet":
        return True, "controlnet_tab"
    if surface == "inpaint" and selection.allow_inpaint_controlnet:
        return True, "inpaint_mixing"
    if surface == "outpaint" and selection.allow_outpaint_controlnet:
        return True, "outpaint_mixing"
    return False, "none"


def _admit_descriptors(
    slot_inputs: Iterable[FrozenControlNetSlotInput],
) -> tuple[FrozenControlNetSlotDescriptor, ...]:
    descriptors = []
    for slot in slot_inputs:
        control_type = flags.resolve_cn_type(slot.control_type, default=None)
        if control_type is None:
            raise ValueError(
                f"Unsupported active ControlNet type {slot.control_type!r} cannot be admitted into the frozen plan"
            )
        try:
            slot_index = int(slot.ui_slot_index)
            end_percent = float(slot.end_percent)
            weight = float(slot.weight)
            start_percent = float(slot.start_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frozen ControlNet slot {slot.control_type!r}: {slot!r}") from exc
        frozen_image = freeze_workflow_payload(slot.input_image)
        descriptors.append(FrozenControlNetSlotDescriptor(
            ui_slot_index=slot_index,
            control_type=control_type,
            input_image=frozen_image,
            end_percent=end_percent,
            weight=weight,
            start_percent=start_percent,
            payload_fingerprint=workflow_payload_fingerprint(frozen_image),
        ))
    return tuple(sorted(descriptors, key=lambda item: item.ui_slot_index))


def expected_stage_ids(plan: FrozenWorkflowPlan) -> tuple[str, ...]:
    route_id = plan.route_id
    overlay = plan.controlnet_overlay
    stages: list[str] = []

    if route_id in {"removal", "flux_removal"}:
        return ("removal",)
    if route_id == "color_enhanced_upscale":
        return ("image_input_prepare", "color_enhanced_upscale")
    if route_id in {"super_upscale", "upscale"}:
        return ("image_input_prepare", "prompt_encode", "upscale")
    if route_id == "flux_inpaint":
        return ("image_input_prepare", "flux_inpaint")

    if route_id in {"inpaint", "outpaint"}:
        stages.append("image_input_prepare")
        if overlay.enabled:
            stages.append("controlnet_support_load")
        stages.append("outpaint_prepare" if route_id == "outpaint" else "inpaint_prepare")
    elif overlay.enabled:
        stages.extend(("image_input_prepare", "controlnet_support_load"))

    stages.append("prompt_encode")
    if overlay.structural_descriptors:
        stages.append("structural_controlnet")
    if overlay.contextual_descriptors:
        stages.append("contextual_controlnet")
    stages.append("diffusion_batch")
    return tuple(stages)


def compile_workflow_plan(
    selection: FrozenWorkflowSelection,
    slot_inputs: Iterable[FrozenControlNetSlotInput] = (),
) -> FrozenWorkflowPlan:
    """Compile the final immutable plan without consulting mutable state."""
    if not isinstance(selection, FrozenWorkflowSelection):
        raise TypeError("compile_workflow_plan requires FrozenWorkflowSelection")
    selection.validate()

    route_id, route_family = _base_route(selection)
    execution = _execution_declaration(selection, route_id)
    overlay_allowed, activation_source = _overlay_permission(selection)
    parsed_slots = tuple(slot_inputs)

    if route_id == "flux_inpaint" and overlay_allowed and parsed_slots:
        raise ValueError(
            "ControlNet overlay is not supported for Flux Fill inpaint; select SDXL inpaint or disable mixing"
        )

    active_descriptors = _admit_descriptors(parsed_slots) if overlay_allowed else ()
    overlay = ControlNetOverlayPlan(
        enabled=bool(active_descriptors),
        activation_source=activation_source if active_descriptors else "none",
        active_slot_descriptors=active_descriptors,
    )
    provisional = FrozenWorkflowPlan(
        route_id=route_id,
        route_family=route_family,
        source_surface=selection.source_surface,
        execution_declaration=execution,
        controlnet_overlay=overlay,
        ordered_stage_ids=(),
    )
    plan = FrozenWorkflowPlan(
        route_id=route_id,
        route_family=route_family,
        source_surface=selection.source_surface,
        execution_declaration=execution,
        controlnet_overlay=overlay,
        ordered_stage_ids=expected_stage_ids(provisional),
    )
    plan.validate()
    return plan


__all__ = ["compile_workflow_plan", "expected_stage_ids"]
