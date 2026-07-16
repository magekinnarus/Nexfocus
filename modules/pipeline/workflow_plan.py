"""Deprecated compatibility facade for W12d workflow planning.

Authoritative production code imports ``workflow_contracts`` and
``workflow_compiler`` directly.  This module preserves older direct/probe
imports and makes their legacy adaptation explicit.
"""

from modules.pipeline.workflow_compiler import expected_stage_ids
from modules.pipeline.workflow_contracts import (
    ControlNetOverlayPlan,
    FrozenControlNetSlotDescriptor,
    FrozenControlNetSlotInput,
    FrozenExecutionDeclaration,
    FrozenExecutionStep,
    FrozenWorkflowPlan,
    FrozenWorkflowSelection,
    planned_tasks_for_channel,
    require_workflow_plan,
)
from modules.pipeline.workflow_legacy_adapter import (
    bind_legacy_workflow_plan,
    compile_legacy_workflow_plan,
    derive_source_surface,
)


# Historical names retained only for direct/probe compatibility.
compile_workflow_plan = compile_legacy_workflow_plan
ensure_workflow_plan = bind_legacy_workflow_plan


__all__ = [
    "ControlNetOverlayPlan",
    "FrozenControlNetSlotDescriptor",
    "FrozenControlNetSlotInput",
    "FrozenExecutionDeclaration",
    "FrozenExecutionStep",
    "FrozenWorkflowPlan",
    "FrozenWorkflowSelection",
    "compile_workflow_plan",
    "derive_source_surface",
    "ensure_workflow_plan",
    "expected_stage_ids",
    "planned_tasks_for_channel",
    "require_workflow_plan",
]
