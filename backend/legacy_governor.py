import sys
import logging
import weakref
import gc
import time
import torch
import backend.memory_governor as memory_governor
from backend import environment_profile as environment_profiles
from backend.hardware import (
    config,
    get_free_memory,
    get_torch_device,
    is_intel_xpu,
    ipex,
    VRAMState,
    minimum_inference_memory,
    extra_reserved_memory,
    _memory_mb,
    _classify_model_role,
    _describe_model_for_logs,
    _residency_plan_for_phase,
    MIN_WEIGHT_MEMORY_RATIO,
    is_device_cpu,
    _emit_residency_log,
    soft_empty_cache,
)

current_loaded_models = []
_last_soft_empty_cache = 0.0

class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if hasattr(model, "parent") and model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
        real_model = self.model.model

        if is_intel_xpu() and not config.disable_ipex_optimize and ipex is not None and real_model is not None:
            with torch.no_grad():
                real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if hasattr(self.model, "can_runtime_release") and self.model.can_runtime_release():
                self.model.detach(unpatch_weights)
                if self.model_finalizer:
                    self.model_finalizer.detach()
                    self.model_finalizer = None
                self.real_model = None
                return True
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        if self.model_finalizer:
            self.model_finalizer.detach()
            self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model and self.real_model() is not None and self.model is None

def _residency_profile_name():
    profile = memory_governor.environment_profile()
    return getattr(profile, 'name', environment_profiles.PROFILE_CUSTOM)

def _warn_legacy_vram_mode_if_needed(current_vram_state):
    if current_vram_state not in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        return

    warned = getattr(_warn_legacy_vram_mode_if_needed, '_warned', set())
    if current_vram_state.name in warned:
        return

    logging.warning(
        '[Nex-Memory] Legacy VRAM flag behavior (%s) is deprecated; residency policy now follows profile=%s phase=%s with compatibility fallback.',
        current_vram_state.name,
        _residency_profile_name(),
        memory_governor.current_phase(),
    )
    warned = set(warned)
    warned.add(current_vram_state.name)
    _warn_legacy_vram_mode_if_needed._warned = warned

def describe_model_patcher(model_patcher):
    return _describe_model_for_logs(model_patcher)

def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    free_mem = get_free_memory(device)
    if memory_required >= 1e29:
        logging.info(
            "[Nex-Memory] Requesting full reclaim. Free: %.1f MB",
            free_mem / (1024**2),
        )
    else:
        logging.info(
            "[Nex-Memory] Requesting %.1f MB. Free: %.1f MB",
            memory_required / (1024**2),
            free_mem / (1024**2),
        )

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not config.disable_smart_memory:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        
        if current_loaded_models[i].model_unload(memory_to_free):
            m_size = current_loaded_models[i].model_memory()
            logging.info(f"[Nex-Memory] Offloading model to CPU to free {m_size / (1024**2):.1f} MB")
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False, force_high_vram=False, target_phase=None):
    cleanup_models_gc()
    
    current_vram_state = VRAMState.HIGH_VRAM if force_high_vram else VRAMState(config.lowvram or VRAMState.NORMAL_VRAM.value) # Fallback to normalize
    # Realize VRAM state using globals
    from backend import hardware
    current_vram_state = VRAMState.HIGH_VRAM if force_high_vram else hardware.vram_state
    _warn_legacy_vram_mode_if_needed(current_vram_state)
    residency_plan = _residency_plan_for_phase(target_phase=target_phase)

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models = set(models)
    logging.info(f"[Nex-Memory] load_models_gpu: {len(models)} models requested")
    models_to_load = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models.pop(loaded_model_index)
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            models_to_load.append(loaded_model)
    
    load_device = get_torch_device()

    def _needs_patching(m_wrapper):
        patcher = m_wrapper.model
        model_obj = getattr(patcher, "model", None)
        if model_obj is None:
            return False
        current_uuid = getattr(model_obj, "current_weight_patches_uuid", None)
        return patcher.patches_uuid != current_uuid

    def _log_needs_patching(m_wrapper):
        res = _needs_patching(m_wrapper)
        if res:
            logging.info(f"[Nex-Memory] Model {m_wrapper.model.__class__.__name__} needs patch reconciliation (UUID mismatch)")
        return res

    models_to_load = [m for m in models_to_load if m.model.current_loaded_device() != load_device or force_patch_weights or _log_needs_patching(m)]
    
    if len(models_to_load) == 0:
        return 0.0

    start_time = time.time()
    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            old_p = current_loaded_models.pop(i).model
            model_obj = getattr(old_p, "model", None)
            is_current = model_obj is not None and old_p.patches_uuid == getattr(model_obj, "current_weight_patches_uuid", None)
            old_p.detach(unpatch_all=is_current)

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            if current_vram_state != VRAMState.HIGH_VRAM:
                free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = current_vram_state

        model_role, _, _ = _classify_model_role(model)
        residency_mode = residency_plan.mode_for(model_role) or 'evictable'
        profile_name = residency_plan.notes.get('profile')
        pinned_full_load = (
            residency_mode == 'pinned'
            and not force_high_vram
            and vram_set_state in (VRAMState.NORMAL_VRAM, VRAMState.HIGH_VRAM)
            and profile_name not in (environment_profiles.PROFILE_LOCAL_LOW_VRAM,)
        )
        effective_force_full_load = force_full_load or pinned_full_load

        lowvram_model_memory = 0
        loaded_memory = loaded_model.model_loaded_memory()
        current_free_mem = None
        if vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not effective_force_full_load:
            current_free_mem = get_free_memory(torch_dev) + loaded_memory
            lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            if hardware.total_vram < 4096:
                lowvram_model_memory = max(lowvram_model_memory, 256 * 1024 * 1024)
            lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        model_label = _describe_model_for_logs(model)
        target_memory = loaded_model.model_memory_required(torch_dev)
        load_mode = 'partial' if lowvram_model_memory > 0 and lowvram_model_memory < target_memory else 'full'
        _emit_residency_log(
            'load_plan',
            plan=residency_plan,
            role=model_role,
            item=model_label,
            action=load_mode,
            notes={'full_load': effective_force_full_load, 'legacy_vram': vram_set_state.name},
        )
        free_mem_text = 'n/a' if current_free_mem is None else f"{_memory_mb(current_free_mem):.1f}"
        perf_message = (
            f"[Nex-Perf] load_models_gpu item={model_label} device={torch_dev} mode={load_mode} "
            f"vram_state={vram_set_state.name} target={_memory_mb(target_memory):.1f}MB "
            f"loaded={_memory_mb(loaded_memory):.1f}MB budget={_memory_mb(lowvram_model_memory):.1f}MB "
            f"free={_memory_mb(extra_mem):.1f}MB min_req={_memory_mb(minimum_memory_required):.1f}MB "
            f"current_free={free_mem_text}MB"
        )
        print(perf_message)
        logging.info(perf_message)

        model_load_start = time.perf_counter()
        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        model_load_duration = time.perf_counter() - model_load_start
        current_loaded_models.insert(0, loaded_model)

        lowvram_patch_counter = model.lowvram_patch_counter() if hasattr(model, 'lowvram_patch_counter') else 0
        perf_message = (
            f"[Nex-Perf] load_models_gpu complete item={model_label} "
            f"loaded_now={_memory_mb(loaded_model.model_loaded_memory()):.1f}MB "
            f"total={_memory_mb(loaded_model.model_memory()):.1f}MB "
            f"lowvram_patches={lowvram_patch_counter} duration={model_load_duration:.3f}s"
        )
        print(perf_message)
        logging.info(perf_message)

    load_time = time.time() - start_time
    logging.info(f"Nex Model Loading Time: {load_time:.2f} seconds")
    return load_time

def cleanup_models_gc():
    do_gc = False
    for cur in current_loaded_models:
        if cur.is_dead():
            do_gc = True
            break
    if do_gc:
        gc.collect()
        soft_empty_cache()
    cleanup_models()

def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].model is None or current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete
    for i in to_delete:
        del current_loaded_models[i]

def loaded_model_state():
    cleanup_models_gc()
    state = []

    for loaded_model in current_loaded_models:
        model_patcher = loaded_model.model
        if model_patcher is None:
            continue

        model_obj = getattr(model_patcher, "model", None)
        role, patcher_name, model_name = _classify_model_role(model_patcher)

        try:
            current_device = model_patcher.current_loaded_device()
        except Exception:
            current_device = getattr(model_obj, "device", None)

        try:
            loaded_memory_mb = _memory_mb(loaded_model.model_loaded_memory())
        except Exception:
            loaded_memory_mb = None

        try:
            total_memory_mb = _memory_mb(loaded_model.model_memory())
        except Exception:
            total_memory_mb = None

        lowvram_patch_counter = None
        if hasattr(model_patcher, "lowvram_patch_counter"):
            try:
                lowvram_patch_counter = model_patcher.lowvram_patch_counter()
            except Exception:
                lowvram_patch_counter = None

        state.append({
            "label": _describe_model_for_logs(model_patcher),
            "role": role,
            "patcher_class": patcher_name,
            "model_class": model_name,
            "load_device": str(getattr(model_patcher, "load_device", None)),
            "offload_device": str(getattr(model_patcher, "offload_device", None)),
            "current_loaded_device": str(current_device),
            "currently_used": bool(getattr(loaded_model, "currently_used", False)),
            "loaded_memory_mb": loaded_memory_mb,
            "total_memory_mb": total_memory_mb,
            "model_lowvram": bool(getattr(model_obj, "model_lowvram", False)),
            "lowvram_patch_counter": lowvram_patch_counter,
            "current_weight_patches_uuid": str(getattr(model_obj, "current_weight_patches_uuid", None)),
        })

    return state

def unload_all_models():
    free_memory(1e30, get_torch_device())

def eject_model(model_patcher):
    detached = False
    try:
        if model_patcher is None:
            return False
        model_patcher.detach()
        detached = True
        for i in range(len(current_loaded_models) - 1, -1, -1):
            if current_loaded_models[i].model is model_patcher:
                current_loaded_models.pop(i)
        return True
    except Exception:
        logging.warning(
            "[Nex-Memory] Failed to eject model patcher %s cleanly; leaving live wrapper registered.",
            type(model_patcher).__name__ if model_patcher is not None else "<none>",
            exc_info=True,
        )
        return False
    finally:
        try:
            cleanup_models()
        except Exception:
            logging.warning("[Nex-Memory] cleanup_models failed during eject_model.", exc_info=True)
        try:
            soft_empty_cache()
        except Exception:
            logging.warning(
                "[Nex-Memory] soft_empty_cache failed during eject_model cleanup (detached=%s).",
                detached,
                exc_info=True,
            )

def load_model_gpu(model):
    return load_models_gpu([model])
