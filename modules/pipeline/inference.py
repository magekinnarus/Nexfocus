import os
import time
import logging
import torch
import numpy as np
import modules.core as core
import modules.flags as flags
import modules.config as config
import modules.model_taxonomy as model_taxonomy
import backend.resources as resources
import backend.loader as loader
from backend import sdxl_runtime_policy
from modules.util import get_file_from_folder_list
from modules.pipeline.output import resolve_workflow_identity, save_and_log, yield_result
from modules.pipeline.workflow_contracts import require_workflow_plan


def _is_debug_console_logging_enabled() -> bool:
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def _resolve_preview_update_interval(task_state) -> int:
    try:
        return max(1, int(getattr(task_state, 'preview_update_interval', 1) or 1))
    except Exception:
        return 1


def _resolve_text_only_progress_update_interval(total_steps) -> int:
    try:
        resolved_total_steps = max(1, int(total_steps or 1))
    except Exception:
        resolved_total_steps = 1
    return max(5, resolved_total_steps // 10 or 1)


def _resolve_completed_global_steps(current_task_id, completed_steps, total_steps, all_steps) -> int:
    try:
        resolved_total_steps = max(1, int(total_steps or 1))
    except Exception:
        resolved_total_steps = 1
    try:
        resolved_all_steps = max(1, int(all_steps or resolved_total_steps))
    except Exception:
        resolved_all_steps = resolved_total_steps
    try:
        resolved_task_id = max(0, int(current_task_id or 0))
    except Exception:
        resolved_task_id = 0

    completed_global_steps = resolved_task_id * resolved_total_steps + max(1, int(completed_steps))
    return max(1, min(resolved_all_steps, completed_global_steps))


def _format_sampling_bar(completed_steps: int, total_steps: int, *, width: int = 20) -> tuple[str, int]:
    """Return the fixed-width sampling bar and its integer percentage."""
    resolved_total_steps = max(1, int(total_steps or 1))
    completed = max(0, min(resolved_total_steps, int(completed_steps or 0)))
    percent = int(round(completed * 100.0 / resolved_total_steps))
    filled = int(round(width * completed / resolved_total_steps))
    if completed > 0 and filled == 0:
        filled = 1
    bar = '█' * filled + ' ' * (width - filled)
    return bar[:width], percent


def _resolve_preview_stitch_context(task_state, workflow_plan):
    """Keep Inpaint previews crop-local while Outpaint previews show the canvas."""
    route_id = str(getattr(workflow_plan, 'route_id', '') or '').strip().lower()
    if route_id == 'inpaint':
        return None
    return getattr(task_state, 'inpaint_context', None)


def get_sampling_callback(
    task_state,
    progressbar_callback,
    current_task_id,
    total_count,
    preparation_steps,
    all_steps,
    preview_transform=None,
    disable_pbar=True,
    preview_stitch_context=None,
):
    """
    Returns a callback function for the diffusion sampler to report progress.
    """
    sampling_started_at = time.perf_counter()
    last_step_at = sampling_started_at
    debug_mode = _is_debug_console_logging_enabled()

    def callback(step, x0, x, total_steps, y):
        nonlocal last_step_at
        resources.throw_exception_if_processing_interrupted()
        if step == 0:
            task_state.callback_steps = 0
            if debug_mode:
                logging.getLogger(__name__).debug(
                    "[Flux Telemetry] Sampling callback config preview_interval=%s "
                    "preview_transform=%s preview_stitch=%s total_steps=%s image=%s/%s",
                    _resolve_preview_update_interval(task_state),
                    preview_transform is not None,
                    preview_stitch_context is not None,
                    total_steps,
                    current_task_id + 1,
                    total_count,
                )
        completed_steps = max(step + 1, 1)
        completed_global_steps = _resolve_completed_global_steps(
            current_task_id,
            completed_steps,
            total_steps,
            all_steps,
        )
        task_state.callback_steps = completed_global_steps * (100 - preparation_steps) / float(all_steps)
        progress_val = int(preparation_steps + task_state.callback_steps)
        task_state.current_progress = progress_val
        bar, percent = _format_sampling_bar(completed_steps, total_steps)
        status_text = f'Sampling step {step + 1}/{total_steps} ({percent}%)'
        task_state.current_status_text = status_text
        now = time.perf_counter()
        step_wall = now - last_step_at
        elapsed_wall = now - sampling_started_at
        average_step_wall = elapsed_wall / float(completed_steps)
        remaining_steps = max(int(total_steps) - completed_steps, 0)
        eta_wall = average_step_wall * float(remaining_steps)
        last_step_at = now
        should_emit_console_log = True
        if disable_pbar and should_emit_console_log:
            timing = f'({step_wall:.2f}s)'
            if debug_mode:
                timing += f' avg={average_step_wall:.2f}s/it eta={eta_wall:.1f}s'
            is_final_step = completed_steps >= int(total_steps)
            console_line = (
                f'[Nex] Sampling: [{bar}] {percent:3d}%  '
                f'Step {step + 1}/{total_steps} {timing}'
            )
            print(
                console_line,
                end='\n' if debug_mode or is_final_step else '\r',
                flush=True,
            )

        preview_image = None
        preview_update_interval = _resolve_preview_update_interval(task_state)
        should_emit_preview_image = (
            y is not None
            and (
                preview_update_interval <= 1
                or completed_steps == int(total_steps)
                or (completed_steps % preview_update_interval) == 0
            )
        )

        if should_emit_preview_image:
            preview_image = y
            if preview_transform is not None and preview_image is not None:
                preview_image = preview_transform(preview_image)

            if (
                preview_image is not None
                and isinstance(preview_image, np.ndarray)
                and preview_stitch_context is not None
            ):
                # Preview transport uses a hard paste only; full-image morphological blending
                # remains reserved for the final stitched output.
                from modules.pipeline.inpaint import InpaintPipeline

                inpaint = InpaintPipeline()
                preview_image = inpaint.paste_back(preview_stitch_context, preview_image)
            elif not isinstance(preview_image, np.ndarray):
                preview_image = None

        should_emit_progress_event = (
            preview_image is not None
            or completed_steps == 1
            or completed_steps == int(total_steps)
            or (completed_steps % _resolve_text_only_progress_update_interval(total_steps)) == 0
        )
        if should_emit_progress_event:
            task_state.yields.append(['preview', (progress_val, status_text, preview_image)])

    return callback


def _build_sdxl_preview_transform(task_state, runtime):
    previewer_holder = {"previewer": None, "latent_format": None, "resolved": False, "device": None}

    def decode_preview(preview_payload):
        try:
            import torch
            from backend.preview import decode_preview_payload, resolve_best_available_previewer
        except Exception:
            return None

        if not isinstance(preview_payload, torch.Tensor):
            return preview_payload if isinstance(preview_payload, np.ndarray) else None

        previewer = previewer_holder["previewer"]
        latent_format = previewer_holder["latent_format"]
        
        preview_device = preview_payload.device
        if not previewer_holder["resolved"] or previewer_holder["device"] != str(preview_device):
            previewer_holder["resolved"] = True
            unet = getattr(runtime, "unet", None)
            vae = getattr(runtime, "vae", None)
            
            load_device = getattr(unet, "load_device", None) if unet else None
            patcher_model = getattr(unet, "model", None) if unet else None
            latent_format = getattr(patcher_model, "latent_format", None) if patcher_model else None
            if latent_format is None:
                latent_format = getattr(getattr(patcher_model, "model", None), "latent_format", None)
            
            if latent_format is None and vae is not None:
                latent_format = getattr(vae, "latent_format", None)
            if load_device is None and vae is not None:
                load_device = getattr(getattr(vae, "patcher", None), "load_device", None)

            previewer_holder["latent_format"] = latent_format
            if latent_format is not None:
                try:
                    from modules.config import path_vae_approx
                except Exception:
                    path_vae_approx = None
                previewer = resolve_best_available_previewer(
                    preview_device or load_device,
                    latent_format,
                    vae_approx_path=path_vae_approx,
                )
            previewer_holder["previewer"] = previewer
            latent_format = previewer_holder["latent_format"]
            previewer_holder["device"] = str(preview_device or load_device)

        if previewer is None:
            return None

        try:
            return decode_preview_payload(previewer, latent_format, preview_payload)
        except Exception:
            return None

    return decode_preview


class _DeferredAssemblyPreviewRuntime:
    def __init__(self, holder):
        self._holder = holder

    @property
    def unet(self):
        assembly = self._holder.get("assembly")
        if assembly is None:
            return None
        return getattr(getattr(assembly, "unet_spine", None), "unet", None)

    @property
    def vae(self):
        assembly = self._holder.get("assembly")
        if assembly is None:
            return None
        return getattr(getattr(assembly, "vae_worker", None), "vae", None)


def _resolve_unified_checkpoint_path(task_state):
    model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
    if model_name == '':
        raise ValueError('Unified SDXL runtime requires a selected base model.')
    return get_file_from_folder_list(model_name, config.paths_checkpoints)


def _resolve_unified_vae_path(task_state):
    vae_name = str(getattr(task_state, 'vae_name', '') or '').strip()
    if vae_name in {'', flags.default_vae, 'Default (model)', 'Default (Same as model)'}:
        return flags.default_vae
    return get_file_from_folder_list(vae_name, config.path_vae)


def _ensure_supported_unified_runtime_request(task_state):
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is None or not bool(getattr(policy, 'enabled', False)):
        raise RuntimeError('Unified SDXL runtime requires an active SDXL execution policy; legacy shared diffusion path is gutted.')

    checkpoint_path = _resolve_unified_checkpoint_path(task_state)
    resolved_taxonomy = config.resolve_model_taxonomy(checkpoint_path)
    if str(checkpoint_path).lower().endswith('.gguf'):
        raise RuntimeError(
            'GGUF model checkpoints are not supported. '
            'Select an SDXL checkpoint instead.'
        )
    if resolved_taxonomy.architecture != model_taxonomy.ARCHITECTURE_SDXL:
        raise RuntimeError('SD 1.5 execution is no longer supported.')
    return checkpoint_path


def _normalize_pathish(value) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(str(value))))


def _is_selected_base_model_candidate(task_state, candidate: str, *, checkpoint_path: str | None = None) -> bool:
    base_model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
    if not candidate:
        return False
    if candidate == base_model_name:
        return True

    if checkpoint_path is None and base_model_name:
        try:
            checkpoint_path = _resolve_unified_checkpoint_path(task_state)
        except Exception:
            checkpoint_path = None

    if not checkpoint_path:
        return False

    if _normalize_pathish(candidate) == _normalize_pathish(checkpoint_path):
        return True

    resolved_checkpoint_candidate = get_file_from_folder_list(candidate, config.paths_checkpoints)
    return (
        os.path.exists(resolved_checkpoint_candidate)
        and _normalize_pathish(resolved_checkpoint_candidate) == _normalize_pathish(checkpoint_path)
    )


def _resolve_single_unified_lora_spec(
    task_state,
    raw_path,
    weight,
    *,
    checkpoint_path: str | None = None,
    strict: bool = False,
):
    candidate = str(raw_path or '').strip()
    if candidate in {'', 'None'}:
        return None

    if _is_selected_base_model_candidate(task_state, candidate, checkpoint_path=checkpoint_path):
        logging.warning(
            '[Nex-LoraResolve] Skipping LoRA candidate %r because it matches the selected base model.',
            candidate,
        )
        return None

    if os.path.exists(candidate):
        return str(candidate), float(weight)

    resolved = get_file_from_folder_list(candidate, config.paths_lora_lookup)
    if resolved and os.path.exists(resolved):
        return str(resolved), float(weight)

    if strict:
        raise FileNotFoundError(
            f'Could not resolve LoRA file {candidate!r} in configured LoRA lookup paths.'
        )

    return str(candidate), float(weight)


def _resolve_unified_sdxl_lora_specs(
    task_state,
    *,
    loras=None,
    base_model_additional_loras=None,
    checkpoint_path: str | None = None,
    strict: bool = False,
):
    resolved_loras = list(getattr(task_state, 'loras_processed', None) or loras or getattr(task_state, 'loras', []) or [])
    if base_model_additional_loras is None:
        base_model_additional_loras = getattr(task_state, 'base_model_additional_loras', []) or []
    resolved_additional_loras = list(base_model_additional_loras or [])
    merged_specs = []
    for path, weight in (resolved_loras + resolved_additional_loras):
        resolved_spec = _resolve_single_unified_lora_spec(
            task_state,
            path,
            weight,
            checkpoint_path=checkpoint_path,
            strict=strict,
        )
        if resolved_spec is not None:
            merged_specs.append(resolved_spec)
    return tuple(merged_specs)


def resolve_unified_sdxl_process_key(task_state, *, loras=None, base_model_additional_loras=None):
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is None or not bool(getattr(policy, 'enabled', False)):
        return None

    return sdxl_runtime_policy.resolve_sdxl_process_key(
        base_model_name=_resolve_unified_checkpoint_path(task_state),
        vae_name=_resolve_unified_vae_path(task_state),
        clip_name=getattr(task_state, 'clip_model_name', None),
        sdxl_policy=policy,
        loras=list(
            _resolve_unified_sdxl_lora_specs(
                task_state,
                loras=loras,
                base_model_additional_loras=base_model_additional_loras,
                strict=False,
            )
        ),
    )


def _build_unified_spatial_kwargs(task_state, image_input_result=None):
    image_input_result = image_input_result or {}
    resolved_spatial_context = getattr(task_state, 'inpaint_context', None)
    plan = require_workflow_plan(task_state)
    def _first(value, fallback):
        return fallback if value is None else value

    if plan.route_id == 'outpaint':
        return {
            'source_pixels': _first(image_input_result.get('outpaint_image'), getattr(task_state, 'outpaint_input_image', None)),
            'source_mask': image_input_result.get('outpaint_mask'),
            'spatial_mode': 'outpaint',
            'resolved_spatial_context': resolved_spatial_context,
            'outpaint_direction': getattr(task_state, 'outpaint_direction', None),
            'outpaint_expansion_size': int(getattr(task_state, 'inpaint_outpaint_expansion_size', 384) or 384),
            'outpaint_pixelate': bool(getattr(task_state, 'inpaint_pixelate_primer', True)),
        }
    if plan.route_id == 'inpaint':
        source_mask = getattr(task_state, 'context_mask', None)
        if source_mask is None:
            source_mask = image_input_result.get('inpaint_mask')
        return {
            'source_pixels': _first(image_input_result.get('inpaint_image'), getattr(task_state, 'inpaint_input_image', None)),
            'source_mask': source_mask,
            'spatial_mode': 'inpaint',
            'resolved_spatial_context': resolved_spatial_context,
        }
    return {}


def _run_unified_sdxl_task(
    task_state,
    task_dict,
    current_task_id,
    total_count,
    all_steps,
    preparation_steps,
    denoising_strength,
    final_scheduler_name,
    *,
    loras,
    base_model_additional_loras=None,
    controlnet_paths=None,
    contextual_assets=None,
    image_input_result=None,
    progressbar_callback=None,
):
    from backend.sdxl_unified_runtime import UnifiedSDXLRuntime, UnifiedSDXLRuntimeConfig
    from backend.sdxl_streaming_runtime import SDXLStreamingRuntime
    from backend.staging_manager import ExecutionClass, SDXL_RESIDENT_EXECUTION_CLASSES
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    workflow_plan = require_workflow_plan(task_state)
    active_structural_types = {
        item.control_type for item in workflow_plan.controlnet_overlay.structural_descriptors
    }
    active_contextual_types = {
        item.control_type for item in workflow_plan.controlnet_overlay.contextual_descriptors
    }
    prepared_structural = getattr(task_state, 'prepared_structural_cn_tasks', {}) or {}
    prepared_contextual = getattr(task_state, 'prepared_contextual_cn_tasks', {}) or {}
    planned_controlnet_paths = {
        cn_type: path for cn_type, path in (controlnet_paths or {}).items()
        if cn_type in active_structural_types
    }
    planned_contextual_assets = dict(contextual_assets or {})
    planned_contextual_assets['contextual_model_paths'] = {
        cn_type: path
        for cn_type, path in (planned_contextual_assets.get('contextual_model_paths') or {}).items()
        if cn_type in active_contextual_types
    }
    if flags.cn_ip not in active_contextual_types:
        planned_contextual_assets['clip_vision_path'] = None
        planned_contextual_assets['ip_negative_path'] = None
    if flags.cn_pulid not in active_contextual_types:
        planned_contextual_assets['eva_clip_path'] = None
        planned_contextual_assets['insightface_model_names'] = []

    checkpoint_path = _ensure_supported_unified_runtime_request(task_state)
    stream_budget = float(getattr(policy, 'stream_budget_mb', 256.0))

    merged_loras = _resolve_unified_sdxl_lora_specs(
        task_state,
        loras=loras,
        base_model_additional_loras=base_model_additional_loras,
        checkpoint_path=checkpoint_path,
        strict=True,
    )

    quality = {
        "sharpness": float(getattr(task_state, 'sharpness', 2.0)),
        "adaptive_cfg": float(getattr(task_state, 'adaptive_cfg', 7.0)),
        "adm_scaler_positive": float(getattr(task_state, 'adm_scaler_positive', 1.5)),
        "adm_scaler_negative": float(getattr(task_state, 'adm_scaler_negative', 0.8)),
        "adm_scaler_end": float(getattr(task_state, 'adm_scaler_end', 0.3)),
        "controlnet_softness": float(getattr(task_state, 'controlnet_softness', 0.25)),
    }
    config_kwargs = dict(
        model_variant='sdxl',
        execution_class=(
            getattr(policy, 'execution_class', None)
            or getattr(task_state, 'sdxl_execution_family', None)
            or getattr(policy, 'execution_family', None)
            or 'standard_sdxl'
        ),
        streamlike_budget_mb=stream_budget,
        quality=quality,
        checkpoint_path=checkpoint_path,
        vae_path=_resolve_unified_vae_path(task_state),
        prompt=str(task_dict.get('task_prompt', task_state.prompt) or ''),
        negative_prompt=str(task_dict.get('task_negative_prompt', task_state.negative_prompt) or ''),
        positive_texts=tuple(str(item) for item in (task_dict.get('positive') or [task_dict.get('task_prompt', task_state.prompt)])),
        negative_texts=tuple(str(item) for item in (task_dict.get('negative') or [task_dict.get('task_negative_prompt', task_state.negative_prompt)])),
        positive_top_k=int(task_dict.get('positive_top_k', 1) or 1),
        negative_top_k=int(task_dict.get('negative_top_k', 1) or 1),
        width=int(task_state.width),
        height=int(task_state.height),
        steps=int(task_state.steps),
        cfg=float(task_state.cfg_scale),
        sampler=str(task_state.sampler_name),
        scheduler=str(final_scheduler_name),
        seed=int(task_dict['task_seed']),
        clip_layer=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
        batch_size=1,
        lora_specs=merged_loras,
        structural_tasks={
            cn_type: tuple(tuple(task) for task in list(tasks))
            for cn_type, tasks in prepared_structural.items()
            if cn_type in active_structural_types
            if tasks
        },
        controlnet_paths=planned_controlnet_paths,
        controlnet_quality=quality,
        contextual_tasks={
            cn_type: tuple(tuple(task) for task in list(tasks))
            for cn_type, tasks in prepared_contextual.items()
            if cn_type in active_contextual_types
            if tasks
        },
        contextual_assets=planned_contextual_assets,
        runtime_policy=policy,
        initial_latent=getattr(task_state, 'initial_latent', None),
        disable_initial_latent=bool(getattr(task_state, 'inpaint_disable_initial_latent', False)),
        denoise_strength=float(denoising_strength) if denoising_strength is not None else None,
        original_scheduler_name=str(task_state.scheduler_name),
    )
    config_kwargs.update(_build_unified_spatial_kwargs(task_state, image_input_result=image_input_result))

    exec_class = config_kwargs.get("execution_class")
    if isinstance(exec_class, str):
        normalized = exec_class.rsplit(".", 1)[-1]
        try:
            exec_class = ExecutionClass[normalized]
        except KeyError:
            pass

    is_resident = exec_class in SDXL_RESIDENT_EXECUTION_CLASSES

    if is_resident:
        runtime = UnifiedSDXLRuntime(UnifiedSDXLRuntimeConfig(**config_kwargs))
    else:
        runtime = SDXLStreamingRuntime(UnifiedSDXLRuntimeConfig(**config_kwargs))
    try:
        if progressbar_callback and image_input_result and any(
            goal in task_state.goals for goal in ('inpaint', 'outpaint', 'remove')
        ):
            progressbar_callback(
                task_state,
                task_state.current_progress,
                'Encoding source image ...',
            )
        prepared_inputs, _ = runtime.prepare_inputs()
        from backend import process_transition

        policy = getattr(task_state, 'sdxl_execution_policy', None)
        execution_mode = getattr(policy, 'execution_mode', None)
        process_transition.publish_sdxl_runtime(
            task_state,
            workflow_plan=workflow_plan,
            loras=loras,
            base_model_additional_loras=base_model_additional_loras,
            runtime_posture="legacy",
            route_owner=(
                getattr(task_state, 'runtime_route_id', None)
                or getattr(task_state, 'runtime_route_family', None)
                or workflow_plan.route_id
            ),
            safe_to_retain=(execution_mode == 'resident'),
        )

        preview_transform = None
        if not getattr(task_state, 'disable_preview', False):
            preview_transform = _build_sdxl_preview_transform(task_state, runtime)

        callback = get_sampling_callback(
            task_state,
            progressbar_callback,
            current_task_id,
            total_count,
            preparation_steps,
            all_steps,
            preview_transform=preview_transform,
            disable_pbar=True,
            preview_stitch_context=_resolve_preview_stitch_context(task_state, workflow_plan),
        )

        if progressbar_callback:
            progressbar_callback(
                task_state,
                task_state.current_progress,
                'Starting inference ...',
            )
        denoise_result = runtime.denoise_prepared_inputs(
            prepared_inputs,
            callback=callback,
            disable_pbar=False,
        )
        decoded_images, _, _ = runtime.decode_latent(denoise_result.samples, tiled=bool(getattr(task_state, 'tiled', False)))
        return core.pytorch_to_numpy(decoded_images)
    finally:
        runtime.close()


def _prepare_gpu_text_legacy_bypass_transition(
    task_state,
    *,
    loras,
    base_model_additional_loras=None,
):
    """Release assembly-owned GPU residents before legacy SDXL admission."""
    from backend import process_transition
    from backend.sdxl_assembly import runtime_state

    current_key = process_transition.get_active_process_key()
    active_gpu_text = runtime_state.get_active_gpu_text_key()
    if (
        active_gpu_text is None
        and (
            current_key is None
            or current_key.family != process_transition.PROCESS_FAMILY_SDXL
            or str(current_key.residency_class or '').lower() != 'resident_unet_gpu_text'
        )
    ):
        return None

    workflow_plan = require_workflow_plan(task_state)
    legacy_key = process_transition.resolve_sdxl_process_key(
        task_state,
        workflow_plan=workflow_plan,
        loras=loras,
        base_model_additional_loras=base_model_additional_loras,
        runtime_posture="legacy",
        allow_legacy_adapter=False,
    )
    if legacy_key is None:
        raise RuntimeError(
            'Refusing legacy SDXL bypass while GPU-text assembly residents are active: '
            'the legacy process identity could not be resolved for deterministic release.'
        )

    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange, release_for_changes

    release_for_changes(
        [LifecycleChange.SPINE_POSTURE_CHANGE],
        reason="gpu_text_legacy_bypass",
    )

    remaining_owners = []
    if runtime_state.get_active_sdxl_resident_spine_key() is not None:
        remaining_owners.append('resident_unet')
    if runtime_state.get_active_gpu_text_key() is not None:
        remaining_owners.append('gpu_text')
    if remaining_owners:
        raise RuntimeError(
            'Refusing legacy SDXL bypass because assembly-owned GPU residents '
            f'remain active after transition: {", ".join(remaining_owners)}.'
        )
    process_transition.set_active_process_key(legacy_key)
    decision = process_transition.ProcessTransitionDecision(
        action="reset",
        reason="gpu_text_legacy_bypass",
        reset_required=True,
        current_key=current_key,
        requested_key=legacy_key,
    )
    return decision


def process_task(task_state, task_dict, current_task_id, total_count, all_steps,
                 preparation_steps, denoising_strength, final_scheduler_name, loras,
                 controlnet_paths=None,
                 progressbar_callback=None, yield_result_callback=None,
                 route_family=None, contextual_assets=None,
                 base_model_additional_loras=None, image_input_result=None):
    """
    Executes a single generation task (one image) using the unified SDXL runtime.
    """
    if task_state.last_stop is not False:
        resources.interrupt_current_processing()

    from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
    from backend.sdxl_assembly.progress import log_telemetry

    controlnet_paths = controlnet_paths or {}
    workflow_plan = require_workflow_plan(task_state)
    
    assembly_eligible, assembly_reason = is_eligible_for_sdxl_assembly(
        task_state=task_state,
        loras=loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
    )

    if assembly_eligible:
        logging.getLogger(__name__).info(
            "[SDXL Route] Routing request %d/%d to SDXL Assembly lane.",
            current_task_id + 1, total_count
        )
        log_telemetry("assembly_route_begin", f"task_id={current_task_id}")

        preview_transform = None
        preview_runtime_holder = None
        if not getattr(task_state, 'disable_preview', False):
            preview_runtime_holder = {}
            preview_transform = _build_sdxl_preview_transform(
                task_state,
                _DeferredAssemblyPreviewRuntime(preview_runtime_holder),
            )

        callback = get_sampling_callback(
            task_state,
            progressbar_callback,
            current_task_id,
            total_count,
            preparation_steps,
            all_steps,
            preview_transform=preview_transform,
            disable_pbar=True,
            preview_stitch_context=_resolve_preview_stitch_context(task_state, workflow_plan),
        )
        if getattr(task_state, 'disable_preview', False):
            setattr(callback, "_sdxl_forward_text_only", True)

        try:
            if task_state.last_stop is not False:
                resources.interrupt_current_processing()

            img = run_sdxl_assembly_task(
                task_state,
                task_dict,
                current_task_id,
                total_count,
                all_steps,
                preparation_steps,
                denoising_strength,
                final_scheduler_name,
                loras=loras,
                controlnet_paths=controlnet_paths,
                contextual_assets=contextual_assets,
                base_model_additional_loras=base_model_additional_loras,
                image_input_result=image_input_result,
                progressbar_callback=callback,
                status_callback=progressbar_callback,
                preview_runtime_holder=preview_runtime_holder,
            )
            imgs = [img]
        except resources.InterruptProcessingException:
            raise
        except Exception as e:
            log_telemetry("assembly_route_failure", f"task_id={current_task_id} error={e}")
            raise
    else:
        logging.getLogger(__name__).info(
            "[SDXL Route] Routing request %d/%d to Legacy Unified SDXL Runtime path. Reason: %s",
            current_task_id + 1, total_count, assembly_reason or "N/A"
        )
        log_telemetry("assembly_route_legacy_bypass", f"reason={assembly_reason}")

        transition_decision = _prepare_gpu_text_legacy_bypass_transition(
            task_state,
            loras=loras,
            base_model_additional_loras=base_model_additional_loras,
        )
        if transition_decision is not None:
            log_telemetry(
                "assembly_route_legacy_transition",
                f"action={transition_decision.action} reason={transition_decision.reason}",
            )

        _ensure_supported_unified_runtime_request(task_state)
        imgs = _run_unified_sdxl_task(
            task_state,
            task_dict,
            current_task_id,
            total_count,
            all_steps,
            preparation_steps,
            denoising_strength,
            final_scheduler_name,
            loras=loras,
            base_model_additional_loras=base_model_additional_loras,
            controlnet_paths=controlnet_paths,
            contextual_assets=contextual_assets,
            image_input_result=image_input_result,
            progressbar_callback=progressbar_callback,
        )

    current_progress = int(preparation_steps + (100 - preparation_steps) / float(all_steps) * task_state.steps)

    if progressbar_callback:
        progressbar_callback(task_state, current_progress, f'Saving image {current_task_id + 1}/{total_count} to system ...')

    try:
        img_paths = save_and_log(
            task_state, task_state.height, task_state.width, imgs,
            task_dict, task_state.use_expansion, loras,
            workflow=resolve_workflow_identity(task_state, task_dict),
        )
        if assembly_eligible:
            log_telemetry("assembly_route_complete", f"task_id={current_task_id}")
    except resources.InterruptProcessingException:
        raise
    except Exception as e:
        if assembly_eligible:
            log_telemetry("assembly_route_failure", f"task_id={current_task_id} phase=save_log error={e}")
        raise

    if yield_result_callback:
        show_results = not task_state.disable_intermediate_results
        yield_result_callback(task_state, img_paths, current_progress, do_not_show_finished_images=not show_results)

    return imgs, img_paths, current_progress
