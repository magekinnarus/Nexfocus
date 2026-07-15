import psutil
import logging
import contextlib
import torch
import sys
import platform
import gc
import ctypes
import threading
from typing import Any, Tuple
import backend.memory_governor as memory_governor
from backend import environment_profile as environment_profiles

# Re-export all hardware capabilities and comfy weight governor elements
from backend.hardware import *
from backend.legacy_governor import *
from backend.hardware import _residency_plan_for_phase, _emit_residency_log

# Provider callbacks for ControlNet caching and residency management
_CONTROLNET_RESIDENCY_HANDLER = None
_LOADED_CONTROLNETS_PROVIDER = None
_REFRESH_CONTROLNETS_CALLBACK = None

def register_controlnet_residency_handler(handler):
    global _CONTROLNET_RESIDENCY_HANDLER
    _CONTROLNET_RESIDENCY_HANDLER = handler

def register_loaded_controlnets_provider(provider):
    global _LOADED_CONTROLNETS_PROVIDER
    _LOADED_CONTROLNETS_PROVIDER = provider

def register_refresh_controlnets_callback(callback):
    global _REFRESH_CONTROLNETS_CALLBACK
    _REFRESH_CONTROLNETS_CALLBACK = callback

def query_loaded_controlnet(model_path):
    if _LOADED_CONTROLNETS_PROVIDER is not None:
        try:
            return _LOADED_CONTROLNETS_PROVIDER(model_path)
        except Exception:
            pass
    try:
        from backend import controlnet_registry
        return controlnet_registry._LOADED_CONTROLNETS.get(model_path)
    except Exception:
        pass
    return None

def trigger_refresh_controlnets(model_paths):
    if _REFRESH_CONTROLNETS_CALLBACK is not None:
        try:
            _REFRESH_CONTROLNETS_CALLBACK(model_paths)
            return
        except Exception:
            pass
    try:
        from backend import controlnet_registry
        controlnet_registry.refresh_controlnets(model_paths)
    except Exception:
        pass

def _eviction_mode_for_resource(plan, resource_id, *, aggressive=False):
    residency_mode = plan.mode_for(resource_id)
    if residency_mode is None or residency_mode == 'pinned':
        return None
    if aggressive:
        return 'destroy'
    if residency_mode == 'warm':
        return None
    profile_name = plan.notes.get('profile')
    if profile_name in (environment_profiles.PROFILE_COLAB_FREE, environment_profiles.PROFILE_LOCAL_LOW_VRAM):
        return 'destroy'
    return 'offload'

def _has_residency_effect(summary):
    if not isinstance(summary, dict):
        return True
    count_keys = (
        'count',
        'contextual_models',
        'clip_vision_models',
        'insightface_apps',
        'eva_clip_models',
        'face_parsers',
    )
    for key in count_keys:
        try:
            if int(summary.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return True
    return not any(key in summary for key in count_keys)

def _apply_support_residency(plan, *, aggressive=False, notes=None):
    actions = {}

    controlnet_action = _eviction_mode_for_resource(plan, 'controlnet', aggressive=aggressive)
    if controlnet_action is not None:
        try:
            if _CONTROLNET_RESIDENCY_HANDLER is not None:
                action = _CONTROLNET_RESIDENCY_HANDLER(controlnet_action)
                if _has_residency_effect(action):
                    actions['controlnet'] = action
            else:
                from backend import controlnet_registry
                action = controlnet_registry.apply_controlnet_residency(controlnet_action)
                if _has_residency_effect(action):
                    actions['controlnet'] = action
        except Exception:
            logging.debug('ControlNet residency cleanup failed.', exc_info=True)

    preprocessor_action = _eviction_mode_for_resource(plan, 'structural_preprocessors', aggressive=aggressive)
    if preprocessor_action is not None:
        try:
            from backend.preprocessors import runtime as preprocessor_runtime
            action = preprocessor_runtime.apply_residency_policy(preprocessor_action)
            if _has_residency_effect(action):
                actions['structural_preprocessors'] = action
        except Exception:
            logging.debug('Structural preprocessor residency cleanup failed.', exc_info=True)

    contextual_action = _eviction_mode_for_resource(plan, 'contextual_adapters', aggressive=aggressive)
    clip_vision_action = _eviction_mode_for_resource(plan, 'clip_vision', aggressive=aggressive)
    insightface_action = _eviction_mode_for_resource(plan, 'insightface', aggressive=aggressive)
    if any(action is not None for action in (contextual_action, clip_vision_action, insightface_action)):
        try:
            import backend.ip_adapter as ip_adapter
            action = ip_adapter.apply_contextual_residency(
                contextual_action or 'offload',
                clip_vision_action=clip_vision_action,
                insightface_action=insightface_action,
            )
            if _has_residency_effect(action):
                actions['contextual_adapters'] = action
        except Exception:
            logging.debug('Contextual adapter residency cleanup failed.', exc_info=True)

    pulid_action = _eviction_mode_for_resource(plan, 'pulid_support', aggressive=aggressive)
    if pulid_action is not None:
        try:
            import backend.pulid_runtime as pulid_runtime
            action = pulid_runtime.apply_contextual_residency(pulid_action)
            if _has_residency_effect(action):
                actions['pulid_support'] = action
        except Exception:
            logging.debug('PuLID residency cleanup failed.', exc_info=True)

    if actions:
        action_notes = dict(notes or {})
        action_notes['support_actions'] = actions
        _emit_residency_log('cleanup', plan=plan, notes=action_notes)

    return actions

def _try_windows_empty_working_set():
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
    except Exception:
        logging.debug('Windows working-set trimming is unavailable.', exc_info=True)
        return False

    try:
        get_current_process = getattr(kernel32, 'GetCurrentProcess', None)
        empty_working_set = getattr(psapi, 'EmptyWorkingSet', None)
        if get_current_process is None or empty_working_set is None:
            return False

        get_current_process.restype = ctypes.c_void_p
        process_handle = get_current_process()
        if not process_handle:
            return False

        empty_working_set.argtypes = [ctypes.c_void_p]
        empty_working_set.restype = ctypes.c_int
        if int(empty_working_set(process_handle)) == 0:
            logging.debug(
                'EmptyWorkingSet call failed last_error=%s.',
                ctypes.get_last_error(),
            )
            return False
        return True
    except Exception:
        logging.debug('EmptyWorkingSet call failed.', exc_info=True)
        return False


def _try_malloc_trim():
    system_name = platform.system()
    if system_name == 'Linux':
        for library_name in ('libc.so.6', 'libc.so'):
            try:
                libc = ctypes.CDLL(library_name)
            except OSError:
                continue

            trim = getattr(libc, 'malloc_trim', None)
            if trim is None:
                continue

            try:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                return bool(trim(0))
            except Exception:
                logging.debug('malloc_trim call failed.', exc_info=True)
                return False
        return False

    if system_name == 'Windows':
        return _try_windows_empty_working_set()

    return False


def _process_rss_mb():
    try:
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def cleanup_memory(reason, *, unload_models=False, force_cache=False, gc_collect=True, trim_host=None, notes=None, target_phase=None, task=None):
    cleanup_notes = dict(notes or {})
    cleanup_notes['reason'] = reason
    cleanup_phase = normalize_memory_phase(target_phase) if target_phase is not None else current_memory_phase()
    cleanup_notes['target_phase'] = cleanup_phase
    before = capture_memory_snapshot(notes={**cleanup_notes, 'stage': 'before_cleanup'})
    process_rss_before_mb = _process_rss_mb()
    residency_plan = _residency_plan_for_phase(target_phase=cleanup_phase, task=task)

    if unload_models:
        unload_all_models()

    support_actions = _apply_support_residency(
        residency_plan,
        aggressive=bool(unload_models or force_cache),
        notes={'reason': reason, 'target_phase': cleanup_phase},
    )

    if gc_collect:
        gc.collect()

    if trim_host is None:
        trim_host = memory_governor.should_trim_host_memory(snapshot=before, aggressive=bool(unload_models and force_cache))

    # If the caller explicitly wants host trimming, flush allocator caches first
    # so the trim attempt has reclaimable pages to work with.
    soft_empty_cache(force=force_cache or unload_models or bool(trim_host) or bool(support_actions))
    trimmed = _try_malloc_trim() if trim_host else False
    process_rss_after_mb = _process_rss_mb()

    after = capture_memory_snapshot(notes={
        **cleanup_notes,
        'stage': 'after_cleanup',
        'trimmed': trimmed,
        'unload_models': bool(unload_models),
        'support_actions': support_actions,
    })

    logging.info(
        '[Nex-Memory] cleanup reason=%s unload_models=%s force_cache=%s target_phase=%s trimmed=%s '
        'ram_before=%sMB ram_after=%sMB vram_before=%sMB vram_after=%sMB '
        'proc_rss_before=%sMB proc_rss_after=%sMB',
        reason,
        unload_models,
        force_cache,
        cleanup_phase,
        trimmed,
        'n/a' if before.free_ram_mb is None else f'{before.free_ram_mb:.1f}',
        'n/a' if after.free_ram_mb is None else f'{after.free_ram_mb:.1f}',
        'n/a' if before.free_vram_mb is None else f'{before.free_vram_mb:.1f}',
        'n/a' if after.free_vram_mb is None else f'{after.free_vram_mb:.1f}',
        'n/a' if process_rss_before_mb is None else f'{process_rss_before_mb:.1f}',
        'n/a' if process_rss_after_mb is None else f'{process_rss_after_mb:.1f}',
    )
    return after

def prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
    affordance = memory_governor.can_afford(
        minimum_free_ram_mb=memory_governor.governor.policy.checkpoint_switch_ram_headroom_mb,
        phase=memory_governor.MemoryPhase.MODEL_REFRESH,
        notes=notes,
    )
    aggressive = memory_governor.governor.policy.aggressive_checkpoint_switch_reclaim or not affordance.allowed
    # A checkpoint/family switch is an explicit ownership boundary.  Once the
    # departing runtime has released its objects, return reclaimable glibc
    # arenas even when the host still has generous free-RAM headroom.  The
    # ordinary pressure threshold is appropriate for phase-local cleanup, but
    # it otherwise leaves large SDXL preprocessor/PuLID allocations charged to
    # the process across SDXL -> Flux -> SDXL transitions.
    full_release_trim = memory_governor.should_trim_host_memory(aggressive=True)
    trim_host = bool(aggressive or full_release_trim)

    logging.info(
        '[Nex-Memory] checkpoint_switch current=%s next=%s allowed=%s aggressive=%s trim_host=%s detail=%s',
        current_model,
        next_model,
        affordance.allowed,
        aggressive,
        trim_host,
        affordance.reason,
    )

    if release_callback is not None:
        release_callback()

    return cleanup_memory(
        'checkpoint_switch',
        unload_models=True,
        force_cache=True,
        gc_collect=True,
        trim_host=trim_host,
        target_phase=MemoryPhase.MODEL_REFRESH,
        notes={
            'current_model': current_model,
            'next_model': next_model,
            'affordance_allowed': affordance.allowed,
            'affordance_reason': affordance.reason,
            **(notes or {}),
        },
    )

MemoryPhase = memory_governor.MemoryPhase

def normalize_memory_phase(phase):
    return memory_governor.normalize_phase(phase)

def memory_phase_scope(phase, task=None, notes=None, end_notes=None):
    return memory_governor.phase_scope(phase, task=task, notes=notes, end_notes=end_notes)

def begin_memory_phase(phase, task=None, notes=None):
    return memory_governor.begin_phase(phase, task=task, notes=notes)

def end_memory_phase(phase=None, notes=None):
    return memory_governor.end_phase(phase, notes=notes)

def capture_memory_snapshot(notes=None, task=None):
    return memory_governor.capture_snapshot(notes=notes, task=task)

def current_memory_phase():
    return memory_governor.current_phase()

def active_memory_environment_profile():
    return memory_governor.environment_profile()

def memory_policy_summary():
    return memory_governor.policy_summary()

def memory_can_afford(**kwargs):
    return memory_governor.can_afford(**kwargs)

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    stream_context = contextlib.nullcontext()
    if stream is not None:
        stream_device = getattr(getattr(stream, "device", None), "type", None)
        if stream_device == "cuda":
            stream_context = torch.cuda.stream(stream)
        elif stream_device == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "stream"):
            stream_context = torch.xpu.stream(stream)

    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            with stream_context:
                return weight.to(dtype=dtype, copy=copy)
        return weight.to(dtype=dtype, copy=copy)

    if stream is not None:
        with stream_context:
            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()
interrupt_processing = False

def interrupt_current_processing(value=True):
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

def prepare_models_for_stage(
    models,
    *,
    stage_name=None,
    target_phase=None,
    memory_required=0,
    force_patch_weights=False,
    minimum_memory_required=None,
    force_full_load=False,
    force_high_vram=False,
):
    if models is None:
        return 0
    if not isinstance(models, (list, tuple, set)):
        models = [models]
    models = [model for model in models if model is not None]
    if len(models) == 0:
        return 0

    phase_name = normalize_memory_phase(target_phase) if target_phase is not None else current_memory_phase()
    stage_label = stage_name or "stage"
    logging.info(
        "[Nex-Memory] prepare_models_for_stage stage=%s phase=%s models=%d",
        stage_label,
        phase_name,
        len(models),
    )
    try:
        from backend import process_transition
        process_transition.log_stage_telemetry(stage_label, target_phase)
    except Exception:
        pass
    return load_models_gpu(
        models,
        memory_required=memory_required,
        force_patch_weights=force_patch_weights,
        minimum_memory_required=minimum_memory_required,
        force_full_load=force_full_load,
        force_high_vram=force_high_vram,
        target_phase=target_phase,
    )

def load_model_gpu(model):
    return load_models_gpu([model])

def teardown_runtime_family(family: str, reason: str = None) -> None:
    logging.info(f"[Nex-Residency] Explicit teardown requested for runtime family: {family} (Reason: {reason})")
    
    # 1. Clear the active process registry if it matches the family
    try:
        from backend import process_transition
        active_key = process_transition.get_active_process_key()
        if active_key is not None and active_key.family == family:
            process_transition.clear_active_process_key()
    except Exception:
        logging.debug(f"Failed to clear process registry for family {family}", exc_info=True)
        
    # 2. Clear components cache for SDXL if family is SDXL
    if family == "sdxl":
        try:
            from backend import sdxl_unified_runtime
            sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache(teardown=True)
        except Exception:
            logging.debug("Failed to clear SDXL component cache during teardown", exc_info=True)
    elif family == "flux_fill":
        try:
            from backend.flux_fill_v3 import (
                release_active_flux_resident_spine,
                release_flux_latent_artifacts,
            )

            released_spine = release_active_flux_resident_spine(reason=reason or "explicit_teardown")
            released_artifacts = release_flux_latent_artifacts()
            logging.debug(
                "Flux Fill teardown released spine=%s artifacts=%s",
                released_spine,
                released_artifacts,
            )
        except Exception:
            logging.debug("Failed to release Flux Fill runtime during teardown", exc_info=True)

    # 3. Offload all weights from GPU
    unload_all_models()
    soft_empty_cache(force=True)

def teardown_active_runtime(reason: str = None) -> None:
    try:
        from backend import process_transition
        active_key = process_transition.get_active_process_key()
        if active_key is not None:
            teardown_runtime_family(active_key.family, reason=reason)
    except Exception:
        logging.debug("Failed to resolve active runtime family for teardown", exc_info=True)
