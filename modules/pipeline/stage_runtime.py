from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from backend import memory_governor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PipelineResourceRequirement:
    resource_id: str
    description: str
    resource_type: str = 'model'
    optional: bool = False
    owner: Optional[str] = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageMemoryEstimate:
    ram_mb: Optional[float] = None
    vram_mb: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStageResult:
    route_complete: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)
    output: Any = None


@dataclass
class StageExecutionRecord:
    stage_id: str
    phase_name: str
    resources: tuple[PipelineResourceRequirement, ...] = ()
    memory_estimate: Optional[StageMemoryEstimate] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRoute:
    route_id: str
    family: str
    display_name: str
    stages: Sequence['PipelineStage']
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRouteContext:
    async_task: Any
    task_state: Any
    route_id: str
    route_family: str
    workflow_plan: Any = None
    execution_family: Optional[str] = None
    residency_class: Optional[str] = None
    sdxl_policy: Any = None
    progressbar_callback: Optional[Callable[[Any, int, str], None]] = None
    yield_result_callback: Optional[Callable[..., None]] = None
    base_model_additional_loras: List[Any] = field(default_factory=list)
    image_input_result: Dict[str, Any] = field(default_factory=dict)
    prompt_tasks: List[Dict[str, Any]] = field(default_factory=list)
    final_scheduler_name: Optional[str] = None
    all_steps: int = 1
    preparation_steps: int = 0
    processing_start_time: Optional[float] = None
    route_artifacts: Dict[str, Any] = field(default_factory=dict)
    executed_stages: List[StageExecutionRecord] = field(default_factory=list)
    route_complete: bool = False
    route_notes: Dict[str, Any] = field(default_factory=dict)

    def has_goal(self, goal: str) -> bool:
        return goal in self.task_state.goals

    def has_controlnet_overlay(self) -> bool:
        plan = self.workflow_plan or getattr(self.task_state, "workflow_plan", None)
        return bool(plan is not None and plan.controlnet_overlay.enabled)

    def note_stage(
        self,
        stage: 'PipelineStage',
        *,
        resources: Sequence[PipelineResourceRequirement] = (),
        memory_estimate: Optional[StageMemoryEstimate] = None,
        notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.executed_stages.append(
            StageExecutionRecord(
                stage_id=stage.stage_id,
                phase_name=stage.phase_name,
                resources=tuple(resources),
                memory_estimate=memory_estimate,
                notes=dict(notes or {}),
            )
        )

    def update_image_input_result(self, payload: Dict[str, Any]) -> None:
        self.image_input_result = dict(payload)
        self.base_model_additional_loras = list(payload.get('base_model_additional_loras', self.base_model_additional_loras))
        self.task_state.base_model_additional_loras = list(self.base_model_additional_loras)

    def set_route_artifact(self, stage_id: str, payload: Any, *, fingerprint: Any | None = None) -> None:
        self.route_artifacts[stage_id] = {
            'fingerprint': fingerprint,
            'payload': payload,
        }

    def get_route_artifact(self, stage_id: str) -> Any:
        artifact = self.route_artifacts.get(stage_id)
        if isinstance(artifact, dict) and 'payload' in artifact:
            return artifact['payload']
        return artifact

    def complete_route(self, **notes: Any) -> None:
        self.route_complete = True
        self.route_notes.update(notes)


class PipelineStage(ABC):
    stage_id = 'stage'
    phase_name = 'task'

    def describe_resources(self, context: PipelineRouteContext) -> Sequence[PipelineResourceRequirement]:
        return ()

    def estimate_memory(self, context: PipelineRouteContext) -> Optional[StageMemoryEstimate]:
        return None

    def prepare(self, context: PipelineRouteContext) -> None:
        return None

    @abstractmethod
    def execute(self, context: PipelineRouteContext) -> Optional[PipelineStageResult]:
        raise NotImplementedError

    def finalize(
        self,
        context: PipelineRouteContext,
        *,
        result: Optional[PipelineStageResult] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        return None


class PipelineStageRunner:
    def run(self, route: PipelineRoute, context: PipelineRouteContext) -> PipelineRouteContext:
        for stage in route.stages:
            resources = tuple(stage.describe_resources(context))
            memory_estimate = stage.estimate_memory(context)
            residency_plan = memory_governor.plan_for_task(task=context.task_state, phase=stage.phase_name)
            logging.getLogger(__name__).debug(
                f"[Residency] phase={stage.phase_name} profile={residency_plan.notes.get('profile')} "
                f"required={','.join(residency_plan.pinned) or '-'} "
                f"warm={','.join(residency_plan.warm) or '-'} "
                f"evictable={','.join(residency_plan.evictable) or '-'}"
            )
            context.note_stage(
                stage,
                resources=resources,
                memory_estimate=memory_estimate,
                notes={
                    'residency_profile': residency_plan.notes.get('profile'),
                    'residency_phase': residency_plan.notes.get('phase'),
                    'residency_required': list(residency_plan.pinned),
                    # Pre-W13 compatibility for stage-note consumers.
                    'residency_pinned': list(residency_plan.pinned),
                    'residency_warm': list(residency_plan.warm),
                    'residency_evictable': list(residency_plan.evictable),
                },
            )

            result: Optional[PipelineStageResult] = None
            error: Optional[BaseException] = None
            try:
                stage.prepare(context)
                result = stage.execute(context) or PipelineStageResult()
            except BaseException as exc:
                error = exc
                raise
            finally:
                stage.finalize(context, result=result, error=error)

            if result is not None and result.notes:
                context.route_notes.update(result.notes)
            if (result is not None and result.route_complete) or context.route_complete:
                context.complete_route(**context.route_notes)
                break

        return context
