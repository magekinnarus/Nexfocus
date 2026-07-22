from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from PIL import Image

from backend import conditioning
from backend import environment_profile as environment_profiles
from backend import sdxl_runtime_policy
import modules.flags as flags
from modules.util import HWC3
from modules.pipeline.workflow_contracts import FrozenWorkflowPlan, require_workflow_plan
from modules.pipeline.stage_runtime import (
    PipelineResourceRequirement,
    PipelineRoute,
    PipelineRouteContext,
    PipelineStage,
    PipelineStageResult,
    StageMemoryEstimate,
)

logger = logging.getLogger(__name__)


def _describe_route_resources(*requirements: PipelineResourceRequirement) -> Sequence[PipelineResourceRequirement]:
    return requirements


def _estimated_megapixels(task_state) -> float:
    width = max(int(getattr(task_state, 'width', 0) or 0), 1)
    height = max(int(getattr(task_state, 'height', 0) or 0), 1)
    return float(width * height) / 1_000_000.0


def has_color_gan_override(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def select_nearest_sdxl_bucket(
    width: int,
    height: int,
    inventory: Sequence[str] | None = None,
) -> tuple[int, int]:
    """Select the nearest SDXL bucket with stable inventory-order tie breaking."""
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"Source dimensions must be positive, got {width}x{height}.")
    entries = inventory if inventory is not None else flags.sdxl_aspect_ratios
    target_ratio = float(width) / float(height)
    buckets = [(int(value.split('*')[0]), int(value.split('*')[1])) for value in entries]
    if not buckets:
        raise ValueError("No SDXL aspect-ratio buckets are configured.")
    return min(
        buckets,
        key=lambda bucket: abs(bucket[0] / bucket[1] - target_ratio),
    )


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


def describe_route(route: PipelineRoute) -> list[str]:
    return [stage.stage_id for stage in route.stages]


def _save_step1_result(context: PipelineRouteContext, payload, description: str) -> None:
    if payload is None:
        return

    images_to_save = payload if isinstance(payload, (list, tuple)) else [payload]

    task_state = context.task_state
    if context.progressbar_callback is not None:
        context.progressbar_callback(task_state, 100, f'Saving {description} ...')

    img_paths = []
    for image in images_to_save:
        saved_path = _save_logged_output(
            context,
            image,
            description,
            prompt_text=getattr(task_state, 'prompt', ''),
            negative_prompt=getattr(task_state, 'negative_prompt', ''),
            seed=getattr(task_state, 'seed', None),
        )
        if saved_path:
            img_paths.append(saved_path)
    if context.yield_result_callback is not None:
        context.yield_result_callback(task_state, img_paths, 100, do_not_show_finished_images=True)


def _load_logged_image_payload(payload):
    if isinstance(payload, str):
        with Image.open(payload) as image:
            return np.array(image)
    return payload


def _load_removal_array(filepath: str, *, mode: str) -> np.ndarray:
    """Load a removal-stage transport file into a neutral workflow array."""
    with Image.open(filepath) as image:
        if str(mode).upper() == 'RGB':
            # Preserve the legacy removal-shell contract: transparent source
            # pixels are composited onto white instead of exposing hidden RGB.
            return HWC3(np.array(image.convert('RGBA')))
        return np.array(image.convert(mode))


def _save_removal_temp(payload) -> str | None:
    """Persist a neutral removal output without giving persistence model ownership."""
    from modules import mask_processing

    return mask_processing.save_to_temp_png(payload)


def _resolve_logged_image_dimensions(payload, *, fallback_height: int, fallback_width: int) -> tuple[int, int]:
    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            height, width = payload.shape
            return int(height), int(width)
        if payload.ndim >= 3:
            height, width = payload.shape[:2]
            return int(height), int(width)
    return int(fallback_height), int(fallback_width)


def _save_logged_output(
    context: PipelineRouteContext,
    payload,
    description: str,
    *,
    prompt_text: str = "",
    negative_prompt: str = "",
    seed=None,
):
    from modules.pipeline.output import save_and_log

    if payload is None:
        return None

    task_state = context.task_state
    image_payload = _load_logged_image_payload(payload)
    if image_payload is None:
        return None

    height, width = _resolve_logged_image_dimensions(
        image_payload,
        fallback_height=getattr(task_state, 'height', 0) or 0,
        fallback_width=getattr(task_state, 'width', 0) or 0,
    )
    img_paths = save_and_log(
        task_state,
        height,
        width,
        [image_payload],
        {
            'log_positive_prompt': str(prompt_text or ''),
            'log_negative_prompt': str(negative_prompt or ''),
            'positive': [],
            'negative': [],
            'styles': list(getattr(task_state, 'style_selections', []) or []),
            'task_seed': getattr(task_state, 'seed', 0) if seed is None else seed,
            'description': description,
        },
        False,
        list(getattr(task_state, 'loras', []) or []),
    )
    if not img_paths:
        return None
    return img_paths[0]


def _record_prepared_route_artifact(context: PipelineRouteContext, stage_name: str, payload, **extra):
    fingerprint = conditioning.build_sdxl_prepared_payload_fingerprint(
        stage_name,
        residency_class=context.residency_class,
        model_identity=getattr(context.task_state, 'base_model_name', None),
        route_family_reconciliation_signature=context.route_family,
        prepared_artifact_signature=payload,
        execution_family=context.execution_family,
        route_id=context.route_id,
        **extra,
    )
    context.set_route_artifact(stage_name, payload, fingerprint=fingerprint)
    return fingerprint


def _resolve_inpaint_prompt(task_state) -> str:
    prompt = str(getattr(task_state, 'prompt', '') or '').strip()
    additional_prompt = str(getattr(task_state, 'inpaint_additional_prompt', '') or '').strip()
    if additional_prompt == '':
        return prompt
    if prompt == '':
        return additional_prompt
    return additional_prompt + '\n' + prompt


def _should_force_flux_host_cleanup() -> bool:
    try:
        from backend import resources

        profile = resources.active_memory_environment_profile()
        profile_name = getattr(profile, "name", None)
        return profile_name in (
            environment_profiles.PROFILE_COLAB_FREE,
            environment_profiles.PROFILE_LOCAL_LOW_VRAM,
        )
    except Exception:
        return False


def _should_aggressively_cleanup_flux_remove(task_state) -> bool:
    if flags.remove_bg in getattr(task_state, "goals", ()):
        return True

    previous_family = str(getattr(task_state, "process_transition_previous_family", "") or "").strip().lower()
    reuse_allowed = bool(getattr(task_state, "process_transition_reuse_allowed", False))

    try:
        from backend.process_transition import PROCESS_FAMILY_FLUX_FILL
    except Exception:
        return True

    if previous_family != PROCESS_FAMILY_FLUX_FILL:
        return True
    return not reuse_allowed


def sync_flux_fill_route_session(route: PipelineRoute, task_state, *, progress: bool = False):
    # Legacy Flux Fill is archived during the greenfield rebuild.
    # Shared route code keeps the symbol as a compatibility shim, but it no
    # longer owns or synchronizes any Flux runtime/session state.
    return None


class ImageInputPreparationStage(PipelineStage):
    stage_id = 'image_input_prepare'
    phase_name = 'image_input_prepare'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='image_input',
                description='User-provided image inputs, masks, and route assets resolved for the active family.',
                resource_type='input',
                owner='modules.pipeline.image_input',
                tags=('route-entry',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'input-shape-dependent'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.image_input import apply_image_input

        task_state = context.task_state
        with resources.memory_phase_scope(
            resources.MemoryPhase.IMAGE_INPUT_PREPARE,
            task=task_state,
            notes={'current_tab': task_state.current_tab},
            end_notes={'completed': True},
        ):
            payload = apply_image_input(task_state, context.base_model_additional_loras, context.progressbar_callback)
        context.update_image_input_result(payload)
        _record_prepared_route_artifact(
            context,
            'image_input_prepare',
            payload,
            current_tab=task_state.current_tab,
            goals=tuple(task_state.goals),
        )
        return PipelineStageResult(notes={'goals': list(task_state.goals)})


class ControlNetSupportLoadStage(PipelineStage):
    stage_id = 'controlnet_support_load'
    phase_name = 'model_refresh'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='structural_controlnet_paths',
                description='Resolved structural ControlNet asset paths retained for later unified-runtime loading.',
                resource_type='artifact',
                owner='modules.pipeline.image_input',
                tags=('controlnet', 'structural'),
                optional=True,
            ),
            PipelineResourceRequirement(
                resource_id='contextual_support_models',
                description='Contextual adapter support assets such as CLIP vision and insightface loaded for active guidance.',
                owner='backend.ip_adapter',
                tags=('controlnet', 'contextual'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'support-model-load'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import load_controlnet_support_models

        task_state = context.task_state
        if not context.has_controlnet_overlay():
            return PipelineStageResult()

        task_state.current_progress = max(task_state.current_progress, 1)
        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, task_state.current_progress, 'Loading ControlNets ...')

        load_controlnet_support_models(context.image_input_result)
        return PipelineStageResult()


class InpaintPreparationStage(PipelineStage):
    stage_id = 'inpaint_prepare'
    phase_name = 'inpaint_prepare'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='inpaint_assets',
                description='Prepared inpaint image, context mask, BB image, and retained inpaint context.',
                resource_type='artifact',
                owner='modules.pipeline.inpaint',
                tags=('inpaint', 'latent'),
            ),
            PipelineResourceRequirement(
                resource_id='candidate_vae',
                description='VAE selected for inpaint latent encoding.',
                owner='backend.sdxl_unified_runtime',
                tags=('vae',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(ram_mb=round(max(128.0, megapixels * 96.0), 1), notes={'basis': 'image-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import EarlyReturnException, apply_inpaint

        task_state = context.task_state
        try:
            apply_inpaint(
                task_state,
                context.image_input_result.get('inpaint_image'),
                context.image_input_result.get('inpaint_mask'),
                context.progressbar_callback,
                context.yield_result_callback,
            )
        except EarlyReturnException as exc:
            _save_step1_result(context, exc.payload, 'Phase 1 Inpaint BB')
            return PipelineStageResult(route_complete=True, notes={'early_return': True, 'route': 'inpaint'})
        _record_prepared_route_artifact(
            context,
            'inpaint_prepare',
            {
                'inpaint_context': getattr(task_state, 'inpaint_context', None),
                'initial_latent': getattr(task_state, 'initial_latent', None),
                'width': task_state.width,
                'height': task_state.height,
                'denoising_strength': getattr(task_state, 'denoising_strength', None),
            },
            current_tab=task_state.current_tab,
            goals=tuple(task_state.goals),
        )
        return PipelineStageResult()


class OutpaintPreparationStage(PipelineStage):
    stage_id = 'outpaint_prepare'
    phase_name = 'outpaint_prepare'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='outpaint_assets',
                description='Prepared outpaint canvas, mask, and retained outpaint context.',
                resource_type='artifact',
                owner='modules.pipeline.outpaint',
                tags=('outpaint', 'latent'),
            ),
            PipelineResourceRequirement(
                resource_id='candidate_vae',
                description='VAE selected for outpaint latent encoding.',
                owner='backend.sdxl_unified_runtime',
                tags=('vae',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(ram_mb=round(max(128.0, megapixels * 96.0), 1), notes={'basis': 'expanded-canvas-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import apply_outpaint_inference_setup

        task_state = context.task_state
        apply_outpaint_inference_setup(
            task_state,
            context.image_input_result.get('outpaint_image'),
            context.image_input_result.get('outpaint_mask'),
            context.progressbar_callback,
            context.yield_result_callback,
        )
        _record_prepared_route_artifact(
            context,
            'outpaint_prepare',
            {
                'outpaint_context': getattr(task_state, 'inpaint_context', None),
                'initial_latent': getattr(task_state, 'initial_latent', None),
                'width': task_state.width,
                'height': task_state.height,
                'denoising_strength': getattr(task_state, 'denoising_strength', None),
            },
            current_tab=task_state.current_tab,
            goals=tuple(task_state.goals),
        )
        return PipelineStageResult()


class PromptEncodingStage(PipelineStage):
    stage_id = 'prompt_encode'
    phase_name = 'prompt_encode'

    def describe_resources(self, context: PipelineRouteContext):
        task_state = context.task_state
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='base_model',
                description=f'Base model {task_state.base_model_name!r} prepared for prompt encoding.',
                owner='backend.sdxl_unified_runtime',
                tags=('checkpoint', 'clip', 'vae'),
            ),
            PipelineResourceRequirement(
                resource_id='prompt_conditions',
                description='Positive and negative conditioning retained for downstream stages.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('conditioning',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'conditioning-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.preprocessing import apply_overrides, process_prompt

        task_state = context.task_state
        apply_overrides(task_state)

        if context.image_input_result.get('skip_prompt_processing', False):
            context.prompt_tasks = []
            return PipelineStageResult(notes={'prompt_processing': 'skipped'})

        context.prompt_tasks = process_prompt(
            task_state,
            context.base_model_additional_loras,
            context.progressbar_callback,
            route_context=context,
            route_family=context.route_family,
            residency_class=context.residency_class,
        )
        resources.cleanup_memory('encoding_to_diffusion', notes={'goals': list(task_state.goals)}, target_phase=resources.MemoryPhase.DIFFUSION, task=task_state)
        return PipelineStageResult(notes={'task_count': len(context.prompt_tasks)})


class StructuralControlNetStage(PipelineStage):
    stage_id = 'structural_controlnet'
    phase_name = 'structural_preprocess'

    def finalize(self, context: PipelineRouteContext, *, result=None, error=None):
        from backend import resources

        if error is not None:
            return

        contextual_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_contextual)
        next_phase = resources.MemoryPhase.CONTEXTUAL_PREPROCESS if sum(len(tasks) for tasks in contextual_tasks.values()) > 0 else resources.MemoryPhase.CONTROL_APPLY
        resources.cleanup_memory(
            'structural_preprocess_complete',
            gc_collect=False,
            target_phase=next_phase,
            notes={'route_id': context.route_id},
            task=context.task_state,
        )

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='structural_preprocessors',
                description='Structural preprocessors for PyraCanny, CPDS, and Depth guidance.',
                owner='backend.preprocessors.runtime',
                tags=('controlnet', 'structural'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'preprocessor-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import preprocess_structural_controlnets
        from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly

        if not context.has_controlnet_overlay():
            return PipelineStageResult(notes={'status': 'skipped'})

        structural_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_structural)
        if sum(len(tasks) for tasks in structural_tasks.values()) == 0:
            return PipelineStageResult(notes={'status': 'skipped'})

        eligible, _ = is_eligible_for_sdxl_assembly(
            task_state=context.task_state,
            loras=context.task_state.loras,
            controlnet_paths=context.image_input_result.get('controlnet_paths', {}),
            contextual_assets=context.image_input_result.get('contextual_assets', {}),
            image_input_result=context.image_input_result,
        )
        if eligible:
            return PipelineStageResult(notes={'status': 'assembly_delegated'})

        if context.progressbar_callback is not None:
            for cn_type, status in (
                (flags.cn_canny, 'Running canny preprocessor ...'),
                (flags.cn_depth, 'Running depth preprocessor ...'),
            ):
                if structural_tasks.get(cn_type):
                    context.progressbar_callback(
                        context.task_state,
                        context.task_state.current_progress,
                        status,
                    )

        preprocess_structural_controlnets(
            context.task_state,
            structural_preprocessor_paths=context.image_input_result.get('structural_preprocessor_paths'),
        )
        return PipelineStageResult()


class ContextualControlNetStage(PipelineStage):
    stage_id = 'contextual_controlnet'
    phase_name = 'contextual_preprocess'

    def finalize(self, context: PipelineRouteContext, *, result=None, error=None):
        from backend import resources

        if error is not None:
            return

        resources.cleanup_memory(
            'contextual_preprocess_complete',
            gc_collect=False,
            target_phase=resources.MemoryPhase.DIFFUSION,
            notes={'route_id': context.route_id},
            task=context.task_state,
        )

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='contextual_adapters',
                description='Contextual guidance assets such as IP-Adapter and PuLID support models.',
                owner='backend.ip_adapter',
                tags=('controlnet', 'contextual'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'adapter-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import preprocess_contextual_controlnets
        from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly

        if not context.has_controlnet_overlay():
            return PipelineStageResult(notes={'status': 'skipped'})

        contextual_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_contextual)
        if sum(len(tasks) for tasks in contextual_tasks.values()) == 0:
            return PipelineStageResult(notes={'status': 'skipped'})

        eligible, _ = is_eligible_for_sdxl_assembly(
            task_state=context.task_state,
            loras=context.task_state.loras,
            controlnet_paths=context.image_input_result.get('controlnet_paths', {}),
            contextual_assets=context.image_input_result.get('contextual_assets', {}),
            image_input_result=context.image_input_result,
        )
        if eligible:
            return PipelineStageResult(notes={'status': 'assembly_delegated'})

        if contextual_tasks.get(flags.cn_ip) and context.progressbar_callback is not None:
            context.progressbar_callback(
                context.task_state,
                context.task_state.current_progress,
                'Running IP-Adapter preprocessor ...',
            )

        preprocess_contextual_controlnets(
            context.task_state,
            contextual_assets=context.image_input_result.get('contextual_assets'),
        )
        return PipelineStageResult()


class DiffusionTaskStage(PipelineStage):
    stage_id = 'diffusion_batch'
    phase_name = 'diffusion'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='prompt_tasks',
                description='Per-image prompt task dictionaries with retained conditioning.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('conditioning', 'tasks'),
            ),
            PipelineResourceRequirement(
                resource_id='diffusion_models',
                description='UNet, VAE, and optional ControlNet state used during iterative task execution.',
                owner='backend.sdxl_unified_runtime',
                tags=('unet', 'vae', 'diffusion'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(512.0, megapixels * 768.0), 1), notes={'basis': 'route-resolution'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.preprocessing import apply_overrides, patch_samplers
        from modules.pipeline.inference import process_task

        task_state = context.task_state
        steps, _, _ = apply_overrides(task_state)
        context.all_steps = max(steps * task_state.image_number, 1)
        context.preparation_steps = task_state.current_progress
        context.final_scheduler_name = patch_samplers(task_state)

        task_state.yields.append(['preview', (task_state.current_progress, 'Moving model to GPU ...', None)])
        context.processing_start_time = time.perf_counter()

        for i, task_dict in enumerate(context.prompt_tasks):
            execution_start_time = time.perf_counter()
            interrupted_action = None

            try:
                process_task(
                    task_state,
                    task_dict,
                    i,
                    task_state.image_number,
                    context.all_steps,
                    context.preparation_steps,
                    task_state.denoising_strength,
                    context.final_scheduler_name,
                    task_state.loras,
                    context.image_input_result.get('controlnet_paths', {}),
                    context.progressbar_callback,
                    context.yield_result_callback,
                    route_family=context.route_family,
                    contextual_assets=context.image_input_result.get('contextual_assets', {}),
                    base_model_additional_loras=context.base_model_additional_loras,
                    image_input_result=context.image_input_result,
                )
            except resources.InterruptProcessingException:
                if task_state.last_stop == 'skip':
                    print('User skipped')
                    task_state.last_stop = False
                    interrupted_action = 'skip'
                else:
                    print('User stopped')
                    interrupted_action = 'stop'
            finally:
                if 'c' in task_dict:
                    del task_dict['c']
                if 'uc' in task_dict:
                    del task_dict['uc']
                resources.cleanup_memory('task_image_complete', gc_collect=False, notes={'task_index': i}, target_phase=resources.MemoryPhase.DECODE, task=task_state)

            if interrupted_action == 'skip':
                continue
            if interrupted_action == 'stop':
                break

            print(f'Task {i + 1} time: {time.perf_counter() - execution_start_time:.2f}s')

        print(f'Total processing time: {time.perf_counter() - context.processing_start_time:.2f}s')
        return PipelineStageResult(route_complete=True, notes={'completed': True, 'tasks_processed': len(context.prompt_tasks)})


class FluxFillInpaintStage(PipelineStage):
    stage_id = 'flux_inpaint'
    phase_name = 'diffusion'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='flux_fill_runtime',
                description='Greenfield Flux Fill runtime assembled on demand through backend.flux_fill_v3.',
                owner='backend.flux_fill_v3',
                tags=('flux', 'inpaint', 'runtime'),
            ),
            PipelineResourceRequirement(
                resource_id='inpaint_context',
                description='Prepared Inpaint tab context and blend mask carried into Flux Fill.',
                resource_type='artifact',
                owner='modules.pipeline.image_input',
                tags=('inpaint', 'flux'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(512.0, megapixels * 640.0), 1), notes={'basis': 'flux-fill-resolution'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules import objr_engine
        from modules.pipeline.image_input import prepare_flux_inpaint_context
        from modules.pipeline.inference import get_sampling_callback
        from modules.pipeline.inpaint import InpaintPipeline
        from modules.pipeline.output import save_and_log
        from backend.flux_fill_v3.activation import (
            resolve_flux_fill_assets,
            resolve_flux_fill_request_t5_posture,
            resolve_flux_fill_spine_kind,
        )
        from backend.flux_fill_v3.contracts import FluxFillRequest, FluxFillPreviewContext, FluxFillCategory, UNetSpineKind
        from backend.flux_fill_v3.director import FluxAssemblyDirector

        task_state = context.task_state
        resources.begin_memory_phase('diffusion', notes={'route': 'flux_inpaint'})
        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')
                context.progressbar_callback(task_state, task_state.current_progress, 'Preparing Flux Fill Inpaint ...')

        ctx = task_state.inpaint_context
        if ctx is None:
            inpaint_image = context.image_input_result.get('inpaint_image')
            inpaint_mask = context.image_input_result.get('inpaint_mask')
            ctx = prepare_flux_inpaint_context(task_state, inpaint_image, inpaint_mask)

        # Resolve prompt and assets using greenfield helpers
        prompt_text = _resolve_inpaint_prompt(task_state)
        assets = resolve_flux_fill_assets(task_state)
        spine_kind = resolve_flux_fill_spine_kind(task_state)
        t5_posture = resolve_flux_fill_request_t5_posture(task_state, spine_kind=spine_kind)

        stitcher = InpaintPipeline()
        processed_count = 0
        total_count = max(1, int(getattr(task_state, 'image_number', 1) or 1))
        base_seed = int(task_state.seed)
        output_height, output_width = ctx.original_image.shape[:2]
        task_state.width = output_width
        task_state.height = output_height
        all_steps = max(int(task_state.steps) * total_count, 1)
        preparation_steps = task_state.current_progress
        force_host_cleanup = _should_force_flux_host_cleanup()

        for image_index in range(total_count):
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, f'Flux Fill Inpaint {image_index + 1}/{total_count} ...')

            seed = base_seed if getattr(task_state, 'disable_seed_increment', False) else base_seed + image_index

            preview_context = None
            preview_transform = None

            if not getattr(task_state, 'disable_preview', False):
                def preview_transform(latent):
                    nonlocal preview_context
                    if preview_context is None:
                        from ldm_patched.modules import latent_formats
                        preview_context = FluxFillPreviewContext(latent_formats.Flux(), latent.device)
                    return preview_context.decode(latent)

            callback = get_sampling_callback(
                task_state,
                context.progressbar_callback,
                image_index,
                total_count,
                preparation_steps,
                all_steps,
                preview_transform=preview_transform,
                preview_stitch_context=ctx,
            )

            interrupted_action = None
            try:
                resources.throw_exception_if_processing_interrupted()

                logger.debug(
                    "[Flux Telemetry] Inpaint route request original=%s bb=%s bb_mask=%s bb_box=%s "
                    "target=%sx%s mask_fill=%.4f prompt_chars=%s preview_interval=%s "
                    "force_host_cleanup=%s seed=%s steps=%s sampler=%s scheduler=%s",
                    _shape_of_array(getattr(ctx, "original_image", None)),
                    _shape_of_array(getattr(ctx, "bb_image", None)),
                    _shape_of_array(getattr(ctx, "bb_mask", None)),
                    getattr(ctx, "bb", None),
                    getattr(ctx, "target_w", None),
                    getattr(ctx, "target_h", None),
                    _mask_fill_ratio(getattr(ctx, "bb_mask", None)) or 0.0,
                    len(str(assets.prompt or "")),
                    getattr(task_state, "preview_update_interval", None),
                    force_host_cleanup,
                    seed,
                    int(task_state.steps),
                    task_state.sampler_name,
                    task_state.scheduler_name,
                )

                req = FluxFillRequest(
                    unet_path=assets.unet_path,
                    ae_path=assets.ae_path,
                    conditioning_cache_path=assets.conditioning_cache_path,
                    seed=seed,
                    steps=int(task_state.steps),
                    sampler=task_state.sampler_name,
                    scheduler=task_state.scheduler_name,
                    prefetch_depth=int(getattr(task_state, 'prefetch_depth', 1)),
                    prefetch_chunk_mb=int(getattr(task_state, 'prefetch_chunk_mb', 64)),
                    unet_spine=spine_kind,
                    t5_posture=t5_posture,
                    disk_paged_t5_gc_interval=getattr(task_state, 'flux_fill_disk_paged_t5_gc_interval', 'auto'),
                    image=ctx.bb_image,
                    mask=ctx.bb_mask,
                    prompt=assets.prompt,
                    blend_mode="none",  # Do blend and stitching manually below
                    clip_l_path=assets.clip_l_path,
                    t5_path=assets.t5_path,
                    category=FluxFillCategory.INPAINT,
                )

                director = FluxAssemblyDirector()
                assembly = director.select_assembly(
                    req,
                    status_callback=context.progressbar_callback,
                    progress_state=task_state,
                )
                result = assembly.execute(req, callback=callback)

            except resources.InterruptProcessingException:
                if task_state.last_stop == 'skip':
                    print('User skipped')
                    task_state.last_stop = False
                    interrupted_action = 'skip'
                else:
                    print('User stopped')
                    interrupted_action = 'stop'

            if interrupted_action == 'skip':
                continue
            if interrupted_action == 'stop':
                break

            import logging
            logging.getLogger(__name__).debug(f"[Flux Telemetry] Applying final morphological blending and stitch-back for image {image_index + 1}/{total_count}")
            stitched_image = stitcher.stitch(ctx, np.asarray(result.output_image))

            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, 100, f'Saving Flux Fill Inpaint {image_index + 1}/{total_count} to system ...')

            task_dict = {
                'log_positive_prompt': prompt_text,
                'log_negative_prompt': task_state.negative_prompt,
                'positive': [],
                'negative': [],
                'styles': task_state.style_selections,
                'task_seed': seed,
                'description': 'Flux Fill Inpaint',
            }
            current_img_paths = save_and_log(task_state, output_height, output_width, [stitched_image], task_dict, False, task_state.loras)
            if context.yield_result_callback is not None:
                context.yield_result_callback(
                    task_state,
                    current_img_paths,
                    100,
                    do_not_show_finished_images=task_state.disable_intermediate_results,
                )
            processed_count += 1
            result = None
            stitched_image = None
            preview_context = None
            resources.cleanup_memory(
                'flux_inpaint_image_complete',
                gc_collect=force_host_cleanup,
                trim_host=force_host_cleanup,
                notes={'task_index': image_index, 'route_id': getattr(context, 'route_id', 'flux_inpaint')},
                target_phase=resources.MemoryPhase.DIFFUSION,
                task=task_state,
            )

        return PipelineStageResult(route_complete=True, notes={'completed': True, 'route': 'flux_inpaint', 'tasks_processed': processed_count})


class UpscaleStage(PipelineStage):
    stage_id = 'upscale'
    phase_name = 'upscale'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='upscaler_model',
                description='GAN upscaler model for light upscale only; Super-Upscale now consumes a provided refinement target.',
                owner='backend.auxiliary_workers.gan_upscale_worker',
                tags=('upscale',),
                optional=True,
            ),
            PipelineResourceRequirement(
                resource_id='upscale_target_image',
                description='Provided pre-upscaled target image for Color Enhancement donor use or Super-Upscale tiled refinement.',
                resource_type='artifact',
                owner='modules.task_state',
                tags=('upscale', 'artifact'),
                optional=True,
            ),
            PipelineResourceRequirement(
                resource_id='retained_conditions',
                description='Prompt conditioning retained for super-upscale tiled refinement.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('upscale', 'conditioning'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(256.0, megapixels * 384.0), 1), notes={'basis': 'upscale-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import apply_upscale
        from modules.pipeline.output import save_and_log
        from modules.pipeline.tiled_refinement import apply_tiled_diffusion_refinement

        task_state = context.task_state
        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, 'Image processing ...')

        direct_return = apply_upscale(task_state, context.progressbar_callback)
        if not direct_return:
            from modules.pipeline.tiled_refinement import apply_tiled_diffusion_refinement
            prompt_task = context.prompt_tasks[0] if len(context.prompt_tasks) > 0 else None
            task_state.uov_input_image = apply_tiled_diffusion_refinement(
                task_state,
                task_state.uov_input_image,
                context.progressbar_callback,
                prompt_task=prompt_task,
            )

        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, 100, 'Saving image to system ...')

        img_paths = save_and_log(
            task_state,
            task_state.height,
            task_state.width,
            [task_state.uov_input_image],
            {
                'log_positive_prompt': task_state.prompt,
                'log_negative_prompt': task_state.negative_prompt,
                'positive': [],
                'negative': [],
                'styles': task_state.style_selections,
                'task_seed': task_state.seed,
            },
            task_state.use_expansion,
            task_state.loras,
        )
        if context.yield_result_callback is not None:
            context.yield_result_callback(task_state, img_paths, 100, do_not_show_finished_images=True)
        return PipelineStageResult(route_complete=True, notes={'completed': True})


class ColorEnhancedUpscaleStage(PipelineStage):
    stage_id = 'color_enhanced_upscale'
    phase_name = 'diffusion'

    def describe_resources(self, context: PipelineRouteContext):
        requirements = (
            PipelineResourceRequirement(
                resource_id='sdxl_assembly',
                description='SDXL assembly execution for color pass.',
                owner='backend.sdxl_assembly',
                tags=('diffusion',),
            ),
            PipelineResourceRequirement(
                resource_id='color_extraction_overlay',
                description='Run-bound SDXL color-extraction parameter overlay.',
                owner='backend.sdxl_assembly.color_extraction_worker',
                tags=('diffusion', 'overlay'),
            ),
            PipelineResourceRequirement(
                resource_id='wavelet_color_utility',
                description='Stateless GAN-detail/SDXL-color wavelet transplant.',
                resource_type='utility',
                owner='backend.sdxl_assembly.wavelet_color',
                tags=('upscale', 'stateless'),
            ),
        )
        return _describe_route_resources(*requirements)

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(256.0, megapixels * 384.0), 1), notes={'basis': 'upscale-resolution'})

    def execute(self, context: PipelineRouteContext):
        from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
        from backend.sdxl_assembly.progress import log_telemetry
        from backend.sdxl_assembly.wavelet_color import wavelet_reconstruction
        from modules.pipeline.output import save_and_log
        import modules.pipeline.preprocessing as preprocessing
        import modules.mask_processing as mask_proc
        import modules.flags as flags
        import copy
        import numpy as np
        import torch
        import cv2

        task_state = context.task_state
        uov_input_image = task_state.uov_input_image
        uov_input_image = mask_proc.ensure_numpy(uov_input_image)
        if uov_input_image is None:
            raise ValueError('Color Enhancement requires a readable source image.')
        uov_input_image = np.ascontiguousarray(HWC3(uov_input_image), dtype=np.uint8)
        orig_h, orig_w = uov_input_image.shape[:2]

        # 1. The GAN detail donor is a required frozen neutral image artifact.
        # This route never admits or executes a GAN model.
        provided_gan_input = getattr(task_state, 'upscale_gan_output_image', None)
        if isinstance(provided_gan_input, str):
            provided_gan_input = provided_gan_input.strip()
        if not has_color_gan_override(provided_gan_input):
            raise ValueError(
                'Color Enhancement requires a color enhancement target. '
                'Generate it with Upscale first, then provide it in the target input.'
            )

        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(
                    task_state,
                    task_state.current_progress,
                    'Color Enhancement: Using target image ...',
                )

        gan_output = mask_proc.ensure_numpy(provided_gan_input)
        if gan_output is None:
            raise ValueError('The color enhancement target could not be read.')
        gan_output = np.ascontiguousarray(HWC3(gan_output), dtype=np.uint8)
        gan_h, gan_w = gan_output.shape[:2]
        if gan_h < orig_h or gan_w < orig_w:
            raise ValueError(
                'The color enhancement target must not be smaller than the source image '
                f'(source={orig_w}x{orig_h}, provided={gan_w}x{gan_h}).'
            )
        log_telemetry('color_enhancement_target', f'provided_dims={gan_w}x{gan_h}')
        print(f"[Color Enhancement] Using target image: {gan_w}x{gan_h}.")

        # 2. SDXL always uses the original source. The GAN image is reserved
        # exclusively for the final high-frequency detail donor.
        source_branch = 'original'
        log_telemetry(
            "color_source_policy",
            f"source_area={orig_w * orig_h} branch=original policy=strict_original",
        )

        # 3. Deterministic nearest-bucket selection
        bucket_w, bucket_h = select_nearest_sdxl_bucket(orig_w, orig_h)

        print(f"[Color Enhancement] Resizing original to SDXL bucket {bucket_w}x{bucket_h}.")

        # Resize the original source to the SDXL bucket.
        color_pass_source_resized = cv2.resize(uov_input_image, (bucket_w, bucket_h), interpolation=cv2.INTER_LANCZOS4)

        # 4. Invoke the SDXL color pass on the resized chosen source image using derived task state
        derived_state = copy.copy(task_state)
        derived_state.width = bucket_w
        derived_state.height = bucket_h
        derived_state.cfg_scale = 1.5
        derived_state.sampler_name = 'dpmpp_2m'
        derived_state.scheduler_name = str(getattr(task_state, 'scheduler_name', '') or '')
        # Color extraction does not need a prompt. When supplied, use only the
        # tab-local upscale prompt; the main negative prompt remains shared with
        # the task, matching the other image-input tabs' prompt ownership.
        derived_state.prompt = str(getattr(task_state, 'upscale_prompt', '') or '').strip()
        derived_state.negative_prompt = str(getattr(task_state, 'negative_prompt', '') or '')
        derived_state.source_pixels = color_pass_source_resized
        derived_state.uov_method = 'Color Enhancement'
        derived_state.goals = []  # Empty goals to avoid recursive upscale loop
        derived_state.loras = list(getattr(task_state, 'loras', []) or [])
        color_final_scheduler_name = preprocessing.patch_samplers(derived_state)
        color_all_steps = max(1, int(getattr(derived_state, 'steps', 1) or 1))
        color_workflow_contract = {
            'workflow_id': 'color_enhanced_upscale',
            'workflow_name': 'Color Enhancement',
            'workflow_family': 'upscale',
            'assembly_route_id': 'color_enhancement',
            'assembly_variant': 'sdxl_color_enhancement',
            'source_policy': 'strict_original',
            'donor_policy': 'provided_gan_detail',
            'sampler_policy': 'forced_dpmpp_2m',
            'scheduler_policy': 'inherit_user_selection',
            'steps_policy': 'inherit_user_selection',
            'cfg_policy': 'fixed_1_5',
        }

        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, task_state.current_progress + 5, 'Color Enhancement: Running SDXL color pass ...')

        log_telemetry(
            "color_pass_begin",
            f"workflow={color_workflow_contract['assembly_variant']} "
            f"branch={source_branch} bucket={bucket_w}*{bucket_h} "
            f"sampler=dpmpp_2m scheduler={derived_state.scheduler_name} "
            f"final_scheduler={color_final_scheduler_name} steps={color_all_steps} cfg=1.5",
        )
        try:
            sdxl_output = run_sdxl_assembly_task(
                task_state=derived_state,
                task_dict={'task_seed': task_state.seed},
                current_task_id=0,
                total_count=1,
                all_steps=color_all_steps,
                preparation_steps=0,
                denoising_strength=0.35,
                final_scheduler_name=color_final_scheduler_name,
                loras=list(getattr(derived_state, 'loras', []) or []),
                controlnet_paths={},
                contextual_assets={},
                base_model_additional_loras=list(getattr(context, 'base_model_additional_loras', []) or []),
                image_input_result=None,
                progressbar_callback=context.progressbar_callback,
            )
        except BaseException as exc:
            log_telemetry("color_pass_failure", f"error_type={type(exc).__name__}")
            raise
        else:
            log_telemetry("color_pass_complete", f"bucket={bucket_w}*{bucket_h}")

        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, task_state.current_progress + 15, 'Color Enhancement: Transplanting color ...')

        # 5. Resize SDXL output to match the GAN output dimensions
        # And perform the wavelet transplant
        gan_output_t = torch.from_numpy(gan_output).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        sdxl_output_t = torch.from_numpy(sdxl_output).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Perform wavelet color transplant using Content (GAN) and Color (SDXL)
        log_telemetry("wavelet_transplant_begin", f"levels=5 target={gan_output.shape[1]}x{gan_output.shape[0]}")
        transplanted_t = wavelet_reconstruction(gan_output_t, sdxl_output_t, levels=5)
        log_telemetry("wavelet_transplant_complete", f"target={gan_output.shape[1]}x{gan_output.shape[0]}")

        # Convert back to HWC uint8 numpy array
        final_image = np.ascontiguousarray(
            (transplanted_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
            .clip(0.0, 255.0)
            .astype(np.uint8)
        )

        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, 95, 'Saving image to system ...')

        # Save and log output
        task_state.uov_input_image = final_image
        task_state.width = final_image.shape[1]
        task_state.height = final_image.shape[0]

        output_task_dict = {
            'log_positive_prompt': task_state.prompt,
            'log_negative_prompt': task_state.negative_prompt,
            'positive': [],
            'negative': [],
            'styles': task_state.style_selections,
            'task_seed': task_state.seed,
        }
        enhanced_paths = save_and_log(
            task_state,
            task_state.height,
            task_state.width,
            [final_image],
            {**output_task_dict, 'description': 'Color Enhancement'},
            task_state.use_expansion,
            task_state.loras,
        )
        img_paths = list(enhanced_paths or [])
        if context.yield_result_callback is not None:
            context.yield_result_callback(task_state, img_paths, 100, do_not_show_finished_images=True)

        return PipelineStageResult(
            route_complete=True,
            notes={
                'completed': True,
                'source_area': orig_w * orig_h,
                'source_branch': source_branch,
                'gan_source': 'provided',
                'bucket': f"{bucket_w}*{bucket_h}",
                'workflow_id': color_workflow_contract['workflow_id'],
                'workflow_name': color_workflow_contract['workflow_name'],
                'workflow_family': color_workflow_contract['workflow_family'],
                'assembly_route_id': color_workflow_contract['assembly_route_id'],
                'assembly_variant': color_workflow_contract['assembly_variant'],
                'source_policy': color_workflow_contract['source_policy'],
                'donor_policy': color_workflow_contract['donor_policy'],
                'sampler_policy': color_workflow_contract['sampler_policy'],
                'scheduler_policy': color_workflow_contract['scheduler_policy'],
                'steps_policy': color_workflow_contract['steps_policy'],
                'cfg_policy': color_workflow_contract['cfg_policy'],
                'sampler': 'dpmpp_2m',
                'scheduler': str(getattr(derived_state, 'scheduler_name', '') or ''),
                'final_scheduler': color_final_scheduler_name,
                'steps': color_all_steps,
                'cfg': float(derived_state.cfg_scale),
                'gan_dims': f"{gan_output.shape[1]}x{gan_output.shape[0]}",
                'final_dims': f"{final_image.shape[1]}x{final_image.shape[0]}",
                'final_output_dimensions': (int(final_image.shape[1]), int(final_image.shape[0])),
                'output_labels': ('Color Enhancement',),
            }
        )


class RemovalStage(PipelineStage):
    stage_id = 'removal'
    phase_name = 'removal'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='background_removal_worker',
                description='Ephemeral backend-owned InSPyReNet background-removal worker.',
                owner='backend.auxiliary_workers.background_removal_worker',
                tags=('removal',),
                optional=True,
            ),
            PipelineResourceRequirement(
                resource_id='mat_inpaint_worker',
                description='Ephemeral backend-owned MAT object-inpaint worker.',
                owner='backend.auxiliary_workers.mat_inpaint_worker',
                tags=('removal',),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(vram_mb=2048.0, notes={'basis': 'engine-load-headroom'})

    def execute(self, context: PipelineRouteContext):
        from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL, normalize_objr_engine
        from backend import resources

        task_state = context.task_state
        task_state.inpaint_context = None
        selected_engine = normalize_objr_engine(task_state.objr_engine)
        use_flux_fill_removal_adapter = (
            getattr(context, 'route_id', None) == 'flux_removal'
            or (selected_engine == OBJR_ENGINE_FLUX_FILL and flags.remove_obj in task_state.goals)
        )
        aggressive_flux_remove_cleanup = (
            use_flux_fill_removal_adapter and _should_aggressively_cleanup_flux_remove(task_state)
        )
        resources.begin_memory_phase('removal', notes={'goals': list(task_state.goals)})
        try:
            # BGR and MAT are standalone auxiliary workers. Their admission
            # must not trigger broad memory-governor cleanup. Flux removal is
            # intentionally left on its separate main-family transition path.
            if use_flux_fill_removal_adapter:
                if aggressive_flux_remove_cleanup:
                    if context.progressbar_callback is not None:
                        context.progressbar_callback(task_state, 5, 'Clearing VRAM for Removal Models...')
                    resources.cleanup_memory('removal_preflight', unload_models=True, force_cache=True, trim_host=True, notes={'goals': list(task_state.goals)}, target_phase=resources.MemoryPhase.REMOVAL)
                elif context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, 5, 'Preparing Flux Fill Removal...')
            elif context.progressbar_callback is not None:
                context.progressbar_callback(task_state, 5, 'Preparing Auxiliary Removal...')

            if flags.remove_bg in task_state.goals:
                from backend.auxiliary_workers import run_background_removal

                if context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, 10, 'Background Removal Starting...')
                image_np = _load_removal_array(task_state.remove_base_image, mode='RGB')
                rgba_image, binary_mask = run_background_removal(
                    image_np,
                    threshold=task_state.bgr_threshold,
                    jit=task_state.bgr_jit,
                )
                char_path = _save_removal_temp(rgba_image)
                mask_path = _save_removal_temp(binary_mask)
                persisted_char_path = _save_logged_output(
                    context,
                    char_path,
                    'Background Removal Subject',
                    seed=getattr(task_state, 'seed', None),
                )
                persisted_mask_path = _save_logged_output(
                    context,
                    mask_path,
                    'Background Removal Mask',
                    seed=getattr(task_state, 'seed', None),
                )
                if context.yield_result_callback is not None:
                    context.yield_result_callback(
                        task_state,
                        [
                            persisted_char_path or char_path,
                            persisted_mask_path or mask_path,
                        ],
                        50 if flags.remove_obj in task_state.goals else 100,
                        do_not_show_finished_images=True,
                    )
                if flags.remove_obj in task_state.goals:
                    task_state.remove_mask_image = mask_path

            if flags.remove_obj in task_state.goals:
                if selected_engine == OBJR_ENGINE_FLUX_FILL:
                    from backend.flux_fill_v3.removal_adapter import execute_flux_fill_removal

                    result = execute_flux_fill_removal(
                        context,
                        progress_percent_start=60 if flags.remove_bg in task_state.goals else 10,
                    )

                    persisted_res_path = _save_logged_output(
                        context,
                        result.output_image,
                        'Object Removal',
                        prompt_text=getattr(task_state, 'remove_prompt', ''),
                        negative_prompt=getattr(task_state, 'negative_prompt', ''),
                        seed=getattr(task_state, 'seed', None),
                    )

                    if context.yield_result_callback is not None:
                        context.yield_result_callback(
                            task_state,
                            [persisted_res_path],
                            100,
                            do_not_show_finished_images=True,
                        )
                else:
                    from backend.auxiliary_workers import run_mat_inpaint

                    if context.progressbar_callback is not None:
                        context.progressbar_callback(task_state, 60 if flags.remove_bg in task_state.goals else 10, 'Object Removal Starting...')
                    image_np = _load_removal_array(task_state.remove_base_image, mode='RGB')
                    mask_np = _load_removal_array(task_state.remove_mask_image, mode='L')
                    result_np = run_mat_inpaint(
                        image_np,
                        mask_np,
                        seed=task_state.seed,
                        mask_dilate=task_state.objr_mask_dilate,
                    )
                    res_path = _save_removal_temp(result_np)
                    persisted_res_path = _save_logged_output(
                        context,
                        res_path,
                        'Object Removal',
                        prompt_text=getattr(task_state, 'remove_prompt', ''),
                        negative_prompt=getattr(task_state, 'negative_prompt', ''),
                        seed=getattr(task_state, 'seed', None),
                    )
                    if context.yield_result_callback is not None:
                        context.yield_result_callback(
                            task_state,
                            [persisted_res_path or res_path],
                            100,
                            do_not_show_finished_images=True,
                        )

            return PipelineStageResult(route_complete=True, notes={'completed': True})
        finally:
            resources.end_memory_phase('removal', notes={'completed': True})


def build_generation_route_from_plan(plan: FrozenWorkflowPlan) -> PipelineRoute:
    """Build the exact route/stage sequence from immutable Layer 1 truth."""
    if not isinstance(plan, FrozenWorkflowPlan):
        raise TypeError("build_generation_route_from_plan requires FrozenWorkflowPlan")
    plan.validate()
    route_id = plan.route_id
    family = plan.route_family

    stage_factories = {
        'image_input_prepare': ImageInputPreparationStage,
        'controlnet_support_load': ControlNetSupportLoadStage,
        'inpaint_prepare': InpaintPreparationStage,
        'outpaint_prepare': OutpaintPreparationStage,
        'prompt_encode': PromptEncodingStage,
        'structural_controlnet': StructuralControlNetStage,
        'contextual_controlnet': ContextualControlNetStage,
        'diffusion_batch': DiffusionTaskStage,
        'flux_inpaint': FluxFillInpaintStage,
        'removal': RemovalStage,
        'color_enhanced_upscale': ColorEnhancedUpscaleStage,
        'upscale': UpscaleStage,
    }
    try:
        stages = [stage_factories[stage_id]() for stage_id in plan.ordered_stage_ids]
    except KeyError as exc:
        raise ValueError(f"Frozen workflow plan contains an unknown stage: {exc.args[0]}") from exc

    display_names = {
        'txt2img': 'Txt2Img',
        'inpaint': 'Inpaint',
        'outpaint': 'Outpaint',
        'flux_inpaint': 'Flux Inpaint',
        'removal': 'Removal',
        'flux_removal': 'Flux Remove',
        'upscale': 'Upscale',
        'super_upscale': 'Upscale',
        'color_enhanced_upscale': 'Color Enhancement',
    }
    return PipelineRoute(
        route_id=route_id,
        family=family,
        display_name=display_names.get(route_id, route_id),
        stages=stages,
        notes={
            'workflow_plan': dict(plan.telemetry_record()),
            'ordered_stage_ids': tuple(plan.ordered_stage_ids),
        },
    )


def build_generation_route(task_state) -> PipelineRoute:
    """Task-shell bridge that requires an already-bound queue plan."""
    return build_generation_route_from_plan(require_workflow_plan(task_state))
