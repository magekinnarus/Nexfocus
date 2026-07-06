"""
Nex Retained Compatibility Remainder / Legacy Support Module.
This module is kept solely for backwards compatibility and to support legacy tests/tooling.
It is no longer the production owner of the pipeline execution processes.
"""
import modules.core as core
import torch
import modules.config
import modules.flags
import modules.model_taxonomy
from backend import conditioning, process_transition, resources, schedulers, lora
from backend import sdxl_runtime_policy
import extras.vae_interpose as vae_interpose


from ldm_patched.modules.model_base import SDXL
from modules.util import get_file_from_folder_list, get_enabled_loras


model_base = core.StableDiffusionModel()
final_unet = None
final_clip = None
final_vae = None

class ControlNetDict(dict):
    @property
    def _registry(self):
        from backend import controlnet_registry
        return controlnet_registry._LOADED_CONTROLNETS

    def __getitem__(self, key):
        return self._registry[key]

    def __setitem__(self, key, value):
        self._registry[key] = value

    def __delitem__(self, key):
        del self._registry[key]

    def __contains__(self, key):
        return key in self._registry

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    def get(self, key, default=None):
        return self._registry.get(key, default)

    def pop(self, key, default=None):
        return self._registry.pop(key, default)

    def clear(self):
        self._registry.clear()

    def update(self, *args, **kwargs):
        self._registry.update(*args, **kwargs)

    def setdefault(self, key, default=None):
        return self._registry.setdefault(key, default)

    def popitem(self):
        return self._registry.popitem()

    def copy(self):
        return self._registry.copy()

loaded_ControlNets = ControlNetDict()


def _should_skip_eager_pipeline_preload() -> bool:
    # Nex Universal Clean Slate Policy: Never load models at startup.
    # Models are slotted in UI settings but remain unloaded until the first task.
    return True


def _offload_controlnet(model):
    if model is None:
        return
    patcher = getattr(model, 'control_model_wrapped', None)
    if patcher is not None:
        try:
            patcher.detach()
        except Exception:
            pass


def _destroy_controlnet(model):
    if model is None:
        return
    _offload_controlnet(model)
    try:
        model.cleanup()
    except Exception:
        pass


def apply_controlnet_residency(mode='offload'):
    actions = {'mode': mode, 'count': len(loaded_ControlNets)}
    if mode == 'destroy':
        stale = list(loaded_ControlNets.values())
        loaded_ControlNets.clear()
        for model in stale:
            _destroy_controlnet(model)
    else:
        for model in loaded_ControlNets.values():
            _offload_controlnet(model)
    return actions


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    cache = {}
    requested_paths = {p for p in model_paths if p is not None}
    stale_paths = [p for p in loaded_ControlNets.keys() if p not in requested_paths]

    for stale_path in stale_paths:
        _destroy_controlnet(loaded_ControlNets.pop(stale_path, None))

    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets.clear()
    loaded_ControlNets.update(cache)
    return


# W02.5 Purge: ControlNet residency, cache provider, and refresh callbacks are no longer registered at import time.
# Production queries/refreshes delegate directly to the backend.controlnet_registry.



@torch.inference_mode()
def assert_model_integrity():
    if model_base.unet_with_lora is None:
        return True

    from ldm_patched.modules.model_base import BaseModel, SDXL
    if not isinstance(model_base.unet_with_lora.model, BaseModel):
        raise NotImplementedError('Unknown model type loaded.')

    if not isinstance(model_base.unet_with_lora.model, SDXL):
        print('[Nex Warning] Non-SDXL base model loaded. Some features may not work.')

    return True


def _is_sdxl_model_base() -> bool:
    return getattr(model_base, 'architecture', None) == modules.model_taxonomy.ARCHITECTURE_SDXL


def _policy_signature(policy) -> tuple:
    if policy is None:
        return ()
    return (
        getattr(policy, 'execution_family', None),
        getattr(policy, 'residency_class', None),
        getattr(policy, 'clip_residency_mode', None),
        getattr(policy, 'vae_encode_mode', None),
        bool(getattr(policy, 'keep_clip_loaded', False)),
    )


def _sdxl_process_class(policy) -> str:
    return sdxl_runtime_policy._sdxl_process_class(policy)


def _sdxl_route_family(policy, base_model_name=None) -> str:
    return sdxl_runtime_policy._sdxl_route_family(policy, base_model_name)


def _sdxl_process_key(
    *,
    base_model_name,
    vae_name=None,
    clip_name=None,
    sdxl_policy=None,
    loras=None,
):
    return sdxl_runtime_policy.resolve_sdxl_process_key(
        base_model_name=base_model_name,
        vae_name=vae_name,
        clip_name=clip_name,
        sdxl_policy=sdxl_policy,
        loras=loras,
    )


def _apply_sdxl_policy_to_model_base(policy) -> None:
    setattr(model_base, 'sdxl_execution_policy', policy)
    setattr(model_base, 'sdxl_execution_family', getattr(policy, 'execution_family', None))
    setattr(model_base, 'sdxl_residency_class', getattr(policy, 'residency_class', None))
    setattr(model_base, 'sdxl_clip_residency_mode', getattr(policy, 'clip_residency_mode', None))
    setattr(model_base, 'sdxl_vae_encode_mode', getattr(policy, 'vae_encode_mode', None))
    setattr(model_base, 'sdxl_keep_clip_loaded', bool(getattr(policy, 'keep_clip_loaded', False)))
    clip_load_device = resources.get_torch_device() if bool(getattr(policy, 'prefer_clip_gpu', False)) else torch.device('cpu')
    clip_offload_device = torch.device('cpu')
    for component in (
        getattr(model_base, 'clip', None),
        getattr(model_base, 'clip_with_lora', None),
    ):
        if component is not None:
            setattr(component, 'runtime_policy', policy)
            patcher = getattr(component, 'patcher', None)
            if patcher is not None:
                patcher.load_device = clip_load_device
                patcher.offload_device = clip_offload_device
    vae = getattr(model_base, 'vae', None)
    if vae is not None:
        setattr(vae, 'runtime_policy', policy)




@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, vae_name=None, clip_name=None, sdxl_policy=None):
    global model_base

    if name == 'None':
        print('Skipping base model load (name is None)')
        return

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    current_clip_name = getattr(model_base, 'clip_filename', None)
    if model_base.filename == filename and model_base.vae_filename == vae_filename and current_clip_name == clip_name:
        _apply_sdxl_policy_to_model_base(sdxl_policy)
        return

    previous_model_filename = getattr(model_base, 'filename', None)
    if previous_model_filename is not None:
        def release_previous_model_state():
            global model_base
            previous_model = model_base
            
            # 1. Clear Conditioning Caches (Fixes VRAM leak/stale states)
            for component in [getattr(previous_model, 'clip', None), getattr(previous_model, 'clip_with_lora', None)]:
                if component is not None:
                    if hasattr(component, 'fcs_cond_cache'):
                        component.fcs_cond_cache.clear()
            
            # 2. GGUF Specific Destruction (Fixes UNet destruction bug)
            for unet in [getattr(previous_model, 'unet', None), getattr(previous_model, 'unet_with_lora', None)]:
                if unet is not None:
                    if hasattr(unet, 'unpatch_model'):
                        unet.unpatch_model(unpatch_weights=True)

            try:
                from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange
                changes = []
                current_vae_name = getattr(previous_model, 'vae_filename', None)

                if previous_model_filename != filename:
                    changes.append(LifecycleChange.CHECKPOINT_CHANGE)
                if current_clip_name != clip_name:
                    changes.append(LifecycleChange.MODEL_CHANGE)
                if current_vae_name != vae_filename:
                    changes.append(LifecycleChange.SPATIAL_VAE_CHANGE)
                if not changes:
                    changes.append(LifecycleChange.MODEL_CHANGE)

                release_for_changes(changes, reason="checkpoint_switch")
            except Exception:
                pass
            
            model_base = core.StableDiffusionModel()
            del previous_model

        resources.prepare_for_checkpoint_switch(
            current_model=previous_model_filename,
            next_model=filename,
            release_callback=release_previous_model_state,
            notes={
                'current_vae': getattr(model_base, 'vae_filename', None),
                'next_vae': vae_filename,
                'current_clip': current_clip_name,
                'next_clip': clip_name,
            },
        )

    model_base = core.load_model(
        filename,
        vae_filename,
        clip_name,
        sdxl_policy=sdxl_policy,
    )
    model_base.clip_filename = clip_name
    _apply_sdxl_policy_to_model_base(sdxl_policy)

    print(f'Base model loaded: {model_base.filename}')
    if model_base.vae_filename:
        print(f'VAE loaded: {model_base.vae_filename}')
    if clip_name:
        print(f'Force CLIP: {clip_name}')
    return




@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None):
    global model_base

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    model_base.refresh_loras(loras + base_model_additional_loras)

    return


def _resolve_sdxl_policy(policy=None):
    resolved = policy
    if resolved is None:
        resolved = getattr(model_base, 'sdxl_execution_policy', None)
    return resolved


def _resolve_sdxl_execution_family(execution_family=None, policy=None):
    if execution_family is not None:
        return execution_family
    resolved_policy = _resolve_sdxl_policy(policy)
    return getattr(resolved_policy, 'execution_family', None)


def _resolve_sdxl_clip_residency_mode(clip_residency_mode=None, policy=None):
    if clip_residency_mode is not None:
        return clip_residency_mode
    resolved_policy = _resolve_sdxl_policy(policy)
    return getattr(resolved_policy, 'clip_residency_mode', None)


def _resolve_sdxl_residency_class(residency_class=None):
    resolved = residency_class
    if resolved is None:
        resolved = getattr(model_base, 'sdxl_residency_class', None)
    if resolved is None:
        resolved = getattr(model_base, 'residency_class', None)
    return resources.normalize_sdxl_residency_class(resolved)


def _build_sdxl_text_conditioning_fingerprint(clip, text, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
    text_encoder_identity = (
        type(getattr(clip, 'model', clip)).__name__,
        getattr(clip, 'layer_idx', None),
    )
    return conditioning.build_sdxl_text_conditioning_fingerprint(
        prompt=text,
        negative_prompt='',
        model_identity=getattr(model_base, 'filename', None),
        text_encoder_identity=text_encoder_identity,
        clip_patch_uuid=resources.model_reconciliation_signature(clip.patcher),
        clip_layer_idx=getattr(clip, 'layer_idx', None),
        lora_artifacts_state=getattr(model_base, 'lora_artifact_registry', ()),
        route_family_reconciliation_signature=route_family or getattr(model_base, 'compatibility_family', None),
        residency_class=_resolve_sdxl_residency_class(residency_class),
        route_family=route_family,
        execution_family=_resolve_sdxl_execution_family(execution_family),
        clip_residency_mode=_resolve_sdxl_clip_residency_mode(clip_residency_mode),
    )


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
    if _is_sdxl_model_base():
        cache_key = _build_sdxl_text_conditioning_fingerprint(
            clip,
            text,
            route_family=route_family,
            residency_class=residency_class,
            execution_family=execution_family,
            clip_residency_mode=clip_residency_mode,
        ).digest()
    else:
        cache_key = (text, clip.layer_idx, resources.model_reconciliation_signature(clip.patcher))
    cached = clip.fcs_cond_cache.get(cache_key, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[cache_key] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
    global final_clip

    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(
            final_clip,
            text,
            route_family=route_family,
            residency_class=residency_class,
            execution_family=execution_family,
            clip_residency_mode=clip_residency_mode,
        )
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def set_clip_skip(clip_skip: int):
    global final_clip

    if final_clip is None:
        return

    final_clip.clip_layer(-abs(clip_skip))
    return

@torch.inference_mode()
def clear_all_caches():
    if final_clip is not None:
        final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        if _should_skip_eager_pipeline_preload():
            return
    assert_model_integrity()
    
    if final_clip is None:
        return

    resources.prepare_models_for_stage(
        [final_clip.patcher],
        stage_name="text_encode",
        target_phase=resources.MemoryPhase.PROMPT_ENCODE,
        force_full_load=True,
    )
    return


def release_sdxl_runtime_state(
    *,
    current_process_key=None,
    next_process_key=None,
    reason=None,
    hard_reset=False,
    current_model_name=None,
    next_model_name=None,
    current_vae_name=None,
    next_vae_name=None,
):
    global final_unet, final_clip, final_vae, refresh_state

    def _reset_refresh_state():
        global refresh_state
        refresh_state = {
            'base_model_name': None,
            'loras': None,
            'base_model_additional_loras': None,
            'vae_name': None,
            'clip_name': None,
            'sdxl_policy': None,
            'sdxl_process_class': None,
            'sdxl_process_key': None,
        }

    def _release_cached_sdxl_state():
        global final_unet, final_clip, final_vae
        try:
            from backend import sdxl_unified_runtime, conditioning
            teardown = (next_process_key is None or next_process_key.family != process_transition.PROCESS_FAMILY_SDXL)
            sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache(teardown=teardown)
            conditioning.clear_prompt_conditioning_cache()
        except Exception:
            pass

        try:
            from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange
            changes = process_transition.classify_sdxl_process_key_changes(
                current_process_key,
                next_process_key,
            )
            if current_vae_name != next_vae_name and LifecycleChange.SPATIAL_VAE_CHANGE not in changes:
                changes.append(LifecycleChange.SPATIAL_VAE_CHANGE)
            if not changes:
                changes.append(LifecycleChange.MODEL_CHANGE)
            release_for_changes(changes, reason=reason or "sdxl_process_transition")
        except Exception:
            pass

        clear_all_caches()
        final_unet = None
        final_clip = None
        final_vae = None

    if hard_reset and (current_process_key is not None or next_process_key is not None):
        resources.prepare_for_checkpoint_switch(
            current_model=current_model_name,
            next_model=next_model_name,
            release_callback=_release_cached_sdxl_state,
            notes={
                'reason': reason or 'sdxl_process_transition',
                'current_process_key': process_transition.describe_process_key(current_process_key),
                'next_process_key': process_transition.describe_process_key(next_process_key),
            },
        )
    else:
        _release_cached_sdxl_state()

    _reset_refresh_state()
    return {
        'released': bool(final_unet is None and final_clip is None and final_vae is None),
        'reason': reason,
        'hard_reset': bool(hard_reset),
        'current_process_key': current_process_key,
        'next_process_key': next_process_key,
    }


refresh_state = {
    'base_model_name': None,
    'loras': None,
    'base_model_additional_loras': None,
    'vae_name': None,
    'clip_name': None,
    'sdxl_policy': None,
    'sdxl_process_class': None,
    'sdxl_process_key': None,
}


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(base_model_name, loras,
                       base_model_additional_loras=None, vae_name=None, clip_name=None, sdxl_policy=None):
    global final_unet, final_clip, final_vae, refresh_state

    # Sort loras to ensure consistent comparison
    loras = sorted(loras) if loras is not None else []
    base_model_additional_loras = sorted(base_model_additional_loras) if base_model_additional_loras is not None else []

    current_state = {
        'base_model_name': base_model_name,
        'loras': loras,
        'base_model_additional_loras': base_model_additional_loras,
        'vae_name': vae_name,
        'clip_name': clip_name,
        'sdxl_policy': _policy_signature(sdxl_policy),
        'sdxl_process_class': _sdxl_process_class(sdxl_policy),
        'sdxl_process_key': _sdxl_process_key(
            base_model_name=base_model_name,
            vae_name=vae_name,
            clip_name=clip_name,
            sdxl_policy=sdxl_policy,
            loras=list(loras) + list(base_model_additional_loras),
        ),
    }

    if refresh_state == current_state and final_unet is not None:
        _apply_sdxl_policy_to_model_base(sdxl_policy)
        process_transition.set_active_runtime(
            family=process_transition.PROCESS_FAMILY_SDXL,
            key=current_state['sdxl_process_key'],
            route_owner='sdxl',
            safe_to_retain=False,
        )
        return

    print(f'[Nex-Pipeline] Reconciling model state (LoRAs: {len(loras)} slots, Additional: {len(base_model_additional_loras)} slots)')

    process_key_changed = refresh_state.get('sdxl_process_key') != current_state.get('sdxl_process_key')
    release_sdxl_runtime_state(
        current_process_key=refresh_state.get('sdxl_process_key'),
        next_process_key=current_state.get('sdxl_process_key'),
        current_model_name=refresh_state.get('base_model_name') or getattr(model_base, 'filename', None),
        next_model_name=base_model_name,
        current_vae_name=refresh_state.get('vae_name') or getattr(model_base, 'vae_filename', None),
        next_vae_name=vae_name,
        reason='refresh_everything_transition',
        hard_reset=bool(process_key_changed),
    )

    refresh_base_model(base_model_name, vae_name, clip_name, sdxl_policy=sdxl_policy)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    prepare_text_encoder(async_call=True)
    clear_all_caches()

    refresh_state = current_state
    process_transition.set_active_runtime(
        family=process_transition.PROCESS_FAMILY_SDXL,
        key=refresh_state['sdxl_process_key'],
        route_owner='sdxl',
        safe_to_retain=False,
    )
    return


# Startup preloading is skipped to enforce the Nex Universal Clean Slate Policy.
print('[Startup] Skipping eager default SDXL preload for the active memory/profile route policy.')




@torch.no_grad()
@torch.inference_mode()
def get_candidate_vae(steps, denoise=1.0):
    return final_vae, None
