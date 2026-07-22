import os
import time
import traceback
import threading
import re
import logging

import torch

import backend.resources as resources
from backend import process_transition
from backend import sdxl_runtime_policy
import modules.config
import modules.flags as flags
from modules.task_state import TaskState
from modules.pipeline.output import build_image_wall, yield_result
from modules.pipeline.routes import build_generation_route, describe_route
from modules.pipeline.stage_runtime import PipelineRouteContext, PipelineStageRunner
from modules.pipeline.workflow_compiler import compile_workflow_plan
from modules.pipeline.workflow_contracts import FrozenWorkflowSelection, require_workflow_plan
from modules.pipeline.workflow_legacy_adapter import (
    capture_controlnet_slot_inputs,
    capture_workflow_selection,
)
from modules.lora_channel_policy import (
    merge_lora_channel_overrides,
)


logger = logging.getLogger(__name__)


def discard_inactive_controlnet_tasks(task_state):
    """Defensive legacy normalization after the immutable plan is compiled.

    This is deliberately not planning authority.  Route/stage/admission code
    consumes ``workflow_plan`` and remains correct if this bridge is removed.
    """
    plan = require_workflow_plan(task_state)
    if plan.controlnet_overlay.enabled:
        return 0

    discarded = sum(len(tasks) for tasks in task_state.cn_tasks.values())
    if discarded:
        for cn_type in list(task_state.cn_tasks.keys()):
            task_state.cn_tasks[cn_type] = []
        task_state.ensure_cn_task_maps()
        print(
            f'[ControlNet] Ignoring {discarded} inactive hidden slot input(s) '
            f'for route {getattr(task_state, "requested_route_id", "") or "unknown"}.'
        )
    return discarded


class AsyncTask:
    callback_steps: float = 0.0

    def __init__(self, args):
        import uuid
        self.task_id = str(uuid.uuid4())[:8]
        self.enqueue_time = time.time()
        self.ui_delivered_result_count = 0

        from modules.flags import MetadataScheme
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.state = TaskState()
        self.yields = self.state.yields
        self.results = self.state.results # Shared reference
        self.is_valid = len(args) > 0
        
        if not self.is_valid:
            return

        if isinstance(args, list):
            raise TypeError("AsyncTask received a positional args list instead of a named dictionary. Clear your browser cache and restart.")

        s = self.state

        import modules.parameter_registry as registry
        for param in registry.PARAM_REGISTRY:
            if param.task_field is None:
                continue
            
            val = args.get(param.name, param.default)
            if param.transform and val is not None:
                try:
                    val = param.transform(val)
                except (ValueError, TypeError):
                    val = param.default
            setattr(s, param.task_field, val)

        requested_route_id = str(args.get("requested_route_id", "") or "").strip().lower()
        requested_route_family = str(args.get("requested_route_family", "") or "").strip().lower()
        if requested_route_id:
            s.requested_route_id = requested_route_id
        if requested_route_family:
            s.requested_route_family = requested_route_family
        selection = args.get("workflow_selection")
        if not isinstance(selection, FrozenWorkflowSelection):
            # Layer 0 capture for API/direct callers that did not pass through
            # ui_logic.get_tasks. This is still pre-slot queue capture, not a
            # downstream execution fallback.
            selection = capture_workflow_selection(args, queue_capture=True)
        s.workflow_selection = selection
        s.requested_source_surface = selection.source_surface

        frozen_goals = args.get("goals", None)
        if isinstance(frozen_goals, (list, tuple, set)):
            s.goals = [str(goal) for goal in frozen_goals if str(goal).strip()]

        s.original_steps = s.steps

        lora_data = []
        for i in range(default_max_lora_number):
            enabled = bool(args.get(f'lora_{i}_enabled', False))
            name = str(args.get(f'lora_{i}_model', 'None'))
            weight = float(args.get(f'lora_{i}_weight', 1.0))
            lora_data.append((enabled, name, weight))
        s.loras = get_enabled_loras(lora_data)
        s.lora_channel_overrides = merge_lora_channel_overrides(args.get("lora_channel_overrides"))

        if not getattr(args_manager.args, 'disable_metadata', False):
            s.save_metadata_to_images = args.get('save_metadata_to_images', False)
            scheme_val = args.get('metadata_scheme', 'fooocus_nex')
            try:
                s.metadata_scheme = MetadataScheme(scheme_val)
            except ValueError:
                s.metadata_scheme = MetadataScheme.FOOOCUS
        else:
            s.save_metadata_to_images = False
            s.metadata_scheme = MetadataScheme.FOOOCUS

        def has_controlnet_input(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ''
            if isinstance(value, dict):
                for key in ['image', 'mask', 'background']:
                    item = value.get(key)
                    if isinstance(item, str) and item.strip() != '':
                        return True
                    if item is not None and not isinstance(item, str):
                        return True
                return False
            return True

        from modules.config import default_controlnet_image_count
        for i in range(default_controlnet_image_count):
            cn_img = args.get(f'cn_{i}_image')
            cn_stop = args.get(f'cn_{i}_stop', 1.0)
            cn_weight = args.get(f'cn_{i}_weight', 1.0)
            raw_cn_type = args.get(f'cn_{i}_type')
            if not has_controlnet_input(cn_img):
                continue

            cn_type = flags.resolve_cn_type(raw_cn_type, default=None)
            cn_start = args.get(f'cn_{i}_start', 0.0)
            if cn_type is None:
                # Preserve an explicitly populated unknown slot long enough
                # for the frozen-plan compiler to reject it when the selected
                # surface actually activates the overlay.  Hidden unknown
                # slots remain harmless and are discarded by the defensive
                # compatibility bridge after planning.
                raw_key = str(raw_cn_type or "<unknown>").strip()
                s.cn_tasks.setdefault(raw_key, []).append([cn_img, cn_stop, cn_weight, cn_start, i])
                print(f'[ControlNet] Preserving unsupported guidance type for plan validation: {raw_cn_type!r}')
            elif not s.add_cn_task(cn_type, [cn_img, cn_stop, cn_weight, cn_start, i]):
                print(f'[ControlNet] Skipping unsupported guidance type: {raw_cn_type!r}')

        # Compile only after every raw UI CN slot has been parsed.  The plan,
        # not the defensive discard below, is the durable execution boundary.
        plan = compile_workflow_plan(
            selection,
            capture_controlnet_slot_inputs(s.cn_tasks),
        )
        s.set_workflow_plan(plan)
        s.requested_route_id = plan.route_id
        s.requested_route_family = plan.route_family
        logger.info("[Workflow Plan] %s", dict(plan.telemetry_record()))
        discard_inactive_controlnet_tasks(s)

    @property
    def generate_image_grid(self): return self.state.generate_image_grid
    @property
    def last_stop(self): return self.state.last_stop
    @last_stop.setter
    def last_stop(self, value): self.state.last_stop = value
    @property
    def processing(self): return self.state.processing
    @processing.setter
    def processing(self, value): self.state.processing = value


async_tasks = []
_active_task = None
_active_task_mutex = threading.RLock()


def set_active_task(task):
    global _active_task
    with _active_task_mutex:
        _active_task = task


def get_active_task():
    with _active_task_mutex:
        return _active_task

def cancel_task(task_id: str) -> bool:
    global async_tasks
    with _active_task_mutex:
        active = _active_task
        if active and getattr(active, 'task_id', None) == task_id:
            request_interrupt('stop', active)
            return True
        for i, task in enumerate(async_tasks):
            if getattr(task, 'task_id', None) == task_id:
                async_tasks.pop(i)
                task.yields.append(['finish', []])
                return True
    return False


def request_interrupt(action, task=None):
    # Flux stop/skip interrupts are intentionally non-destructive.
    # Route-entry reconciliation decides whether a later route switch should tear residency down.
    target = get_active_task()
    if target is None:
        target = task
    if target is not None:
        target.last_stop = action
    resources.interrupt_current_processing()
    return target if target is not None else task


def progressbar(task_state, number, text):
    resources.throw_exception_if_processing_interrupted()
    task_state.current_progress = int(number)
    task_state.current_status_text = str(text or '')
    print(f'[Fooocus] {text}')
    task_state.yields.append(['preview', (number, text, None)])


@torch.no_grad()
@torch.inference_mode()
def _release_route_runtime_state(task_state):
    task_state.initial_latent = None
    task_state.positive_cond = None
    task_state.negative_cond = None
    task_state.uov_input_image = None
    task_state.inpaint_input_image = None
    task_state.inpaint_mask_image = None
    task_state.inpaint_context = None
    task_state.context_mask = None
    task_state.outpaint_input_image = None
    task_state.outpaint_mask_image = None
    for cn_type in list(task_state.cn_tasks.keys()):
        task_state.cn_tasks[cn_type] = []
    task_state.ensure_cn_task_maps()
    task_state.planned_cn_tasks = {}
    task_state.planned_cn_tasks_by_channel = {}



@torch.no_grad()
@torch.inference_mode()
def handler(async_task: AsyncTask):
    async_task.last_stop = False
    task_state = async_task.state

    if getattr(async_task, "is_utility", False):
        task_state.processing = True
        task_state.current_progress = 0
        action = getattr(async_task, "utility_action", "")
        if action == "release_controlnet_cache":
            print("[ControlNet] Manual cache release request received. Executing...")
            task_state.current_status_text = "Releasing ControlNet Caches..."
            task_state.yields.append(['preview', (10, "Releasing ControlNet Caches...", None)])

            from backend.sdxl_assembly.lifecycle_coordinator import release_domains
            from backend.sdxl_assembly.runtime_state import LifecycleDomain

            release_domains(
                (LifecycleDomain.STRUCTURAL_CN, LifecycleDomain.CONTEXTUAL_CN),
                reason="manual_release"
            )

            task_state.current_progress = 100
            task_state.current_status_text = "ControlNet Caches Released."
            task_state.yields.append(['preview', (100, "ControlNet Caches Released.", None)])

        task_state.processing = False
        return

    import backend.resources as resources_backend
    resources_backend.interrupt_current_processing(False)
    task_state.processing = True
    task_state.current_progress = 0
    resources.begin_memory_phase('task', notes={'goals': list(task_state.goals)})

    print(f'[Parameters] Seed = {task_state.seed}')
    dims = re.findall(r'\d+', str(task_state.aspect_ratios_selection))
    if len(dims) < 2:
        raise ValueError(f'Invalid aspect ratio selection: {task_state.aspect_ratios_selection!r}')
    task_state.width, task_state.height = int(dims[0]), int(dims[1])

    # Resolve model taxonomy first
    resolved_taxonomy = modules.config.resolve_model_taxonomy(task_state.base_model_name)
    if str(task_state.base_model_name).lower().endswith('.gguf'):
        message = (
            'GGUF model checkpoints are not supported. '
            'Select an SDXL checkpoint instead.'
        )
        print(f'[Nex Error] {message}')
        task_state.yields.append(['preview', (0, message, None)])
        raise ValueError(message)

    # Resolve execution policy
    active_profile = resources.active_memory_environment_profile()
    task_state.sdxl_execution_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture=getattr(resolved_taxonomy, 'architecture', None),
        base_model_name=task_state.base_model_name,
        profile=active_profile,
        requested_residency_class=getattr(task_state, 'sdxl_residency_class', None) or None,
    )
    task_state.sdxl_execution_family = str(getattr(task_state.sdxl_execution_policy, 'execution_family', '') or '')
    task_state.sdxl_residency_class = str(getattr(task_state.sdxl_execution_policy, 'residency_class', '') or '')

    with resources.memory_phase_scope(
        resources.MemoryPhase.ROUTE_SELECT,
        task=task_state,
        notes={
            'current_tab': task_state.current_tab,
            'input_image_checkbox': bool(task_state.input_image_checkbox),
            'requested_route_id': task_state.requested_route_id,
        },
        end_notes={'completed': True},
    ):
        route = build_generation_route(task_state)

    task_state.runtime_route_id = route.route_id
    task_state.runtime_route_family = route.family
    task_state.runtime_route_display_name = route.display_name

    print(f"[Route] {route.route_id}: {' -> '.join(describe_route(route))}")

    transition_decision = process_transition.reconcile_runtime_state(route, task_state)
    if transition_decision is not None:
        task_state.process_transition_action = str(getattr(transition_decision, "action", "") or "")
        task_state.process_transition_reason = str(getattr(transition_decision, "reason", "") or "")
        task_state.process_transition_previous_family = str(getattr(getattr(transition_decision, "current_key", None), "family", "") or "")
        task_state.process_transition_requested_family = str(getattr(getattr(transition_decision, "requested_key", None), "family", "") or "")
        task_state.process_transition_reuse_allowed = bool(getattr(transition_decision, "reuse_allowed", False))
    else:
        task_state.process_transition_action = ""
        task_state.process_transition_reason = ""
        task_state.process_transition_previous_family = ""
        task_state.process_transition_requested_family = ""
        task_state.process_transition_reuse_allowed = False

    transition_status = process_transition.user_facing_transition_status(transition_decision)
    if transition_status:
        progressbar(task_state, task_state.current_progress, transition_status)

    route_context = PipelineRouteContext(
        async_task=async_task,
        task_state=task_state,
        route_id=route.route_id,
        route_family=route.family,
        workflow_plan=require_workflow_plan(task_state),
        execution_family=getattr(task_state.sdxl_execution_policy, 'execution_family', None),
        residency_class=resources.normalize_sdxl_residency_class(getattr(task_state, 'sdxl_residency_class', None)),
        sdxl_policy=task_state.sdxl_execution_policy,
        progressbar_callback=progressbar,
        yield_result_callback=yield_result,
        base_model_additional_loras=list(task_state.base_model_additional_loras),
    )
    PipelineStageRunner().run(route, route_context)

    task_state.processing = False
    _release_route_runtime_state(task_state)


def worker():
    pid = os.getpid()
    print(f'Started worker with PID {pid}')
    
    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            set_active_task(task)
            try:
                handler(task)
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': bool(task.state.generate_image_grid)},
                    end_notes={'completed': True, 'success': True},
                ):
                    if task.state.generate_image_grid:
                        build_image_wall(task.state)
                    task.yields.append(['finish', task.results])
            except resources.InterruptProcessingException:
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': False},
                    end_notes={'completed': True, 'success': False, 'interrupted': True},
                ):
                    task.yields.append(['finish', task.results])
            except:
                traceback.print_exc()
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': False},
                    end_notes={'completed': True, 'success': False},
                ):
                    task.yields.append(['finish', task.results])
            finally:
                set_active_task(None)
                resources.cleanup_memory('task_finalize', force_cache=True, notes={'completed': True}, target_phase=resources.MemoryPhase.FINALIZE)
                resources.end_memory_phase('task', notes={'completed': True})


threading.Thread(target=worker, daemon=True).start()
