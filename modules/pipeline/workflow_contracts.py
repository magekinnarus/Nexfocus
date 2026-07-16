"""Immutable Layer 1 workflow and execution contracts.

These contracts contain no TaskState/UI inference.  Queue capture and legacy
translation live in ``workflow_legacy_adapter``; the pure compiler lives in
``workflow_compiler``.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from modules import flags


VALID_WORKFLOW_SURFACES = frozenset({
    "normal_generate",
    "controlnet",
    "inpaint",
    "outpaint",
    "removal",
    "upscale",
    "super_upscale",
    "color_enhanced_upscale",
})

EXECUTION_KIND_MAIN = "main_family"
EXECUTION_KIND_AUXILIARY = "auxiliary"
MAIN_FAMILY_SDXL = "sdxl"
MAIN_FAMILY_FLUX_FILL = "flux_fill"

AUXILIARY_GAN_UPSCALE = "gan_upscale"
AUXILIARY_BACKGROUND_REMOVAL = "background_removal"
AUXILIARY_MAT_INPAINT = "mat_inpaint"


def normalize_workflow_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def freeze_workflow_payload(value: Any) -> Any:
    """Copy descriptor payloads so later task/UI mutation cannot rewrite them."""
    if isinstance(value, np.ndarray):
        frozen = value.copy()
        frozen.setflags(write=False)
        return frozen
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(value, dict):
        return {copy.deepcopy(key): freeze_workflow_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [freeze_workflow_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_workflow_payload(item) for item in value)
    return copy.deepcopy(value)


def thaw_workflow_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def workflow_payload_fingerprint(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            payload = value.detach().cpu().contiguous().numpy().tobytes()
            return hashlib.sha256(payload).hexdigest()
    except Exception:
        pass
    return hashlib.sha256(repr(value).encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class FrozenWorkflowSelection:
    """Layer 0 queue-captured inputs needed for authoritative planning."""

    source_surface: str
    input_image_enabled: bool = False
    inpaint_route: str = "sdxl"
    object_removal_engine: str = ""
    remove_background: bool = False
    remove_object: bool = False
    upscale_method: str = ""
    allow_inpaint_controlnet: bool = False
    allow_outpaint_controlnet: bool = False

    def validate(self) -> None:
        if self.source_surface not in VALID_WORKFLOW_SURFACES:
            raise ValueError(f"Unsupported frozen workflow surface: {self.source_surface!r}")


@dataclass(frozen=True)
class FrozenControlNetSlotInput:
    """Queue-parsed slot input before active-overlay admission."""

    ui_slot_index: int
    control_type: str
    input_image: Any
    end_percent: float
    weight: float
    start_percent: float


@dataclass(frozen=True)
class FrozenControlNetSlotDescriptor:
    """One literal ControlNet slot admitted by the frozen workflow plan."""

    ui_slot_index: int
    control_type: str
    input_image: Any
    end_percent: float
    weight: float
    start_percent: float
    payload_fingerprint: str = ""

    def materialize_task(self) -> list[Any]:
        return [
            thaw_workflow_payload(self.input_image),
            float(self.end_percent),
            float(self.weight),
            float(self.start_percent),
            int(self.ui_slot_index),
        ]


@dataclass(frozen=True)
class ControlNetOverlayPlan:
    enabled: bool
    activation_source: str
    active_slot_descriptors: tuple[FrozenControlNetSlotDescriptor, ...] = field(default_factory=tuple)

    @property
    def active_types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.control_type for item in self.active_slot_descriptors))

    @property
    def structural_descriptors(self) -> tuple[FrozenControlNetSlotDescriptor, ...]:
        return tuple(
            item for item in self.active_slot_descriptors
            if flags.get_cn_channel(item.control_type) == flags.cn_structural
        )

    @property
    def contextual_descriptors(self) -> tuple[FrozenControlNetSlotDescriptor, ...]:
        return tuple(
            item for item in self.active_slot_descriptors
            if flags.get_cn_channel(item.control_type) == flags.cn_contextual
        )


@dataclass(frozen=True)
class FrozenExecutionStep:
    step_id: str
    execution_kind: str
    owner: str

    def validate(self) -> None:
        if not self.step_id or not self.owner:
            raise ValueError("Frozen execution steps require a step_id and owner")
        if self.execution_kind not in {EXECUTION_KIND_MAIN, EXECUTION_KIND_AUXILIARY}:
            raise ValueError(f"Unsupported execution-step kind: {self.execution_kind!r}")


@dataclass(frozen=True)
class FrozenExecutionDeclaration:
    """Layer 1 declaration consumed by transition and family admission layers."""

    main_family: Optional[str] = None
    ordered_auxiliary_requirements: tuple[str, ...] = field(default_factory=tuple)
    ordered_steps: tuple[FrozenExecutionStep, ...] = field(default_factory=tuple)

    @property
    def auxiliary_only(self) -> bool:
        return self.main_family is None and bool(self.ordered_auxiliary_requirements)

    def validate(self) -> None:
        if self.main_family not in {None, MAIN_FAMILY_SDXL, MAIN_FAMILY_FLUX_FILL}:
            raise ValueError(f"Unsupported main execution family: {self.main_family!r}")
        for step in self.ordered_steps:
            step.validate()
        declared_aux = tuple(
            step.owner for step in self.ordered_steps
            if step.execution_kind == EXECUTION_KIND_AUXILIARY
        )
        if declared_aux != tuple(self.ordered_auxiliary_requirements):
            raise ValueError(
                "Execution declaration auxiliary order is inconsistent: "
                f"expected {self.ordered_auxiliary_requirements!r}, got {declared_aux!r}"
            )
        main_steps = tuple(
            step.owner for step in self.ordered_steps
            if step.execution_kind == EXECUTION_KIND_MAIN
        )
        expected_main = () if self.main_family is None else (self.main_family,)
        if main_steps != expected_main:
            raise ValueError(
                "Execution declaration main-family steps are inconsistent: "
                f"expected {expected_main!r}, got {main_steps!r}"
            )

    def identity(self) -> tuple[Any, ...]:
        return (
            self.main_family,
            tuple(self.ordered_auxiliary_requirements),
            tuple((step.step_id, step.execution_kind, step.owner) for step in self.ordered_steps),
        )


@dataclass(frozen=True)
class FrozenWorkflowPlan:
    """Immutable Layer 1 execution truth for one queued task."""

    route_id: str
    route_family: str
    source_surface: str
    execution_declaration: FrozenExecutionDeclaration
    controlnet_overlay: ControlNetOverlayPlan
    ordered_stage_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.route_id or not self.route_family:
            raise ValueError("Frozen workflow plan requires route_id and route_family")
        if self.source_surface not in VALID_WORKFLOW_SURFACES:
            raise ValueError(f"Unsupported frozen workflow surface: {self.source_surface}")
        self.execution_declaration.validate()

        descriptors = self.controlnet_overlay.active_slot_descriptors
        slots = [int(item.ui_slot_index) for item in descriptors]
        if len(slots) != len(set(slots)):
            raise ValueError(f"Frozen workflow plan contains duplicate ControlNet slots: {slots}")
        for item in descriptors:
            if item.ui_slot_index < 0:
                raise ValueError(f"ControlNet slot index must be non-negative: {item.ui_slot_index}")
            if flags.resolve_cn_type(item.control_type, default=None) is None:
                raise ValueError(f"Frozen workflow plan contains unsupported ControlNet type: {item.control_type}")
        if self.controlnet_overlay.enabled != bool(descriptors):
            raise ValueError("Frozen workflow plan overlay state does not match active descriptors")
        if self.controlnet_overlay.enabled and self.controlnet_overlay.activation_source == "none":
            raise ValueError("Enabled ControlNet overlay must identify its activation source")

        from modules.pipeline.workflow_compiler import expected_stage_ids

        expected = expected_stage_ids(self)
        if tuple(self.ordered_stage_ids) != expected:
            raise ValueError(
                "Frozen workflow plan stage sequence is inconsistent: "
                f"expected {expected!r}, got {tuple(self.ordered_stage_ids)!r}"
            )

    def materialize_cn_tasks(self) -> Dict[str, list[list[Any]]]:
        tasks: Dict[str, list[list[Any]]] = {cn_type: [] for cn_type in flags.cn_all_types}
        for item in self.controlnet_overlay.active_slot_descriptors:
            tasks.setdefault(item.control_type, []).append(item.materialize_task())
        return tasks

    def identity(self) -> tuple[Any, ...]:
        overlay = self.controlnet_overlay
        return (
            self.route_id,
            self.route_family,
            self.source_surface,
            self.execution_declaration.identity(),
            bool(overlay.enabled),
            overlay.activation_source,
            tuple(
                (
                    item.ui_slot_index,
                    item.control_type,
                    round(float(item.weight), 8),
                    round(float(item.start_percent), 8),
                    round(float(item.end_percent), 8),
                    item.payload_fingerprint,
                )
                for item in overlay.active_slot_descriptors
            ),
            tuple(self.ordered_stage_ids),
        )

    def telemetry_record(self) -> Mapping[str, Any]:
        overlay = self.controlnet_overlay
        execution = self.execution_declaration
        return {
            "route": self.route_id,
            "family": self.route_family,
            "source_surface": self.source_surface,
            "main_family": execution.main_family,
            "auxiliary_requirements": tuple(execution.ordered_auxiliary_requirements),
            "execution_steps": tuple(step.step_id for step in execution.ordered_steps),
            "overlay_enabled": bool(overlay.enabled),
            "overlay_source": overlay.activation_source,
            "active_slots": tuple(item.ui_slot_index for item in overlay.active_slot_descriptors),
            "active_types": tuple(item.control_type for item in overlay.active_slot_descriptors),
            "ordered_stages": tuple(self.ordered_stage_ids),
        }


def require_workflow_plan(state: Any) -> FrozenWorkflowPlan:
    plan = getattr(state, "workflow_plan", None)
    if plan is None:
        raise RuntimeError(
            "Queue-bound workflow plan is missing; production consumers may not infer or compile it late"
        )
    if not isinstance(plan, FrozenWorkflowPlan):
        raise ValueError("Task carries an invalid workflow plan object")
    plan.validate()
    return plan


def planned_tasks_for_channel(plan: FrozenWorkflowPlan, channel: str) -> Dict[str, list[list[Any]]]:
    tasks = plan.materialize_cn_tasks()
    return {
        control_type: values
        for control_type, values in tasks.items()
        if flags.get_cn_channel(control_type) == channel
    }


__all__ = [
    "AUXILIARY_BACKGROUND_REMOVAL",
    "AUXILIARY_GAN_UPSCALE",
    "AUXILIARY_MAT_INPAINT",
    "ControlNetOverlayPlan",
    "EXECUTION_KIND_AUXILIARY",
    "EXECUTION_KIND_MAIN",
    "FrozenControlNetSlotDescriptor",
    "FrozenControlNetSlotInput",
    "FrozenExecutionDeclaration",
    "FrozenExecutionStep",
    "FrozenWorkflowPlan",
    "FrozenWorkflowSelection",
    "MAIN_FAMILY_FLUX_FILL",
    "MAIN_FAMILY_SDXL",
    "VALID_WORKFLOW_SURFACES",
    "freeze_workflow_payload",
    "normalize_workflow_token",
    "planned_tasks_for_channel",
    "require_workflow_plan",
    "workflow_payload_fingerprint",
]
