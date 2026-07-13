from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL, normalize_objr_engine
from modules.route_intent import resolve_route_intent


PROCESS_FAMILY_SDXL = "sdxl"
PROCESS_FAMILY_FLUX_FILL = "flux_fill"

PROCESS_CLASS_STANDARD_SDXL = "standard_sdxl"
PROCESS_CLASS_SDXL_GGUF_STAGED = "sdxl_gguf_staged"
PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING = "sdxl_gguf_true_streaming"
PROCESS_CLASS_FLUX_FILL = "flux_fill"

_TOKEN_ALIASES = {
    "sdxl": PROCESS_FAMILY_SDXL,
    "flux": PROCESS_FAMILY_FLUX_FILL,
    "flux_fill": PROCESS_FAMILY_FLUX_FILL,
    "flux-fill": PROCESS_FAMILY_FLUX_FILL,
    "standard sdxl": PROCESS_CLASS_STANDARD_SDXL,
    "sdxl standard": PROCESS_CLASS_STANDARD_SDXL,
    "full_resident": PROCESS_CLASS_STANDARD_SDXL,
    "unified_streaming": PROCESS_CLASS_STANDARD_SDXL,
    "unified streaming": PROCESS_CLASS_STANDARD_SDXL,
    "full resident": PROCESS_CLASS_STANDARD_SDXL,
    "full": PROCESS_CLASS_STANDARD_SDXL,
    "gguf staged": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "sdxl gguf staged": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf_staged_residency": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf staged residency": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf true streaming": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "sdxl gguf true streaming": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "benchmark only": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "benchmark-only": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
}


def _normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _TOKEN_ALIASES.get(token, token)


def normalize_process_family(value: Any) -> str:
    return _normalize_token(value)


def resolve_process_class(
    value: Any = None,
    *,
    family: Any = None,
    execution_family: Any = None,
    residency_class: Any = None,
) -> str:
    if value is not None:
        return normalize_process_class(value, family=family)
    if execution_family is not None:
        return normalize_process_class(execution_family, family=family)
    if residency_class is not None:
        return normalize_process_class(residency_class, family=family)
    normalized_family = normalize_process_family(family)
    if normalized_family == PROCESS_FAMILY_FLUX_FILL:
        return PROCESS_CLASS_FLUX_FILL
    return PROCESS_CLASS_STANDARD_SDXL if normalized_family == PROCESS_FAMILY_SDXL else _normalize_token(normalized_family)


def normalize_process_class(value: Any, *, family: Any = None) -> str:
    token = _normalize_token(value)
    normalized_family = normalize_process_family(family)

    if normalized_family == PROCESS_FAMILY_FLUX_FILL:
        if token in {PROCESS_CLASS_FLUX_FILL, "flux", "flux_fill"}:
            return PROCESS_CLASS_FLUX_FILL
        return token

    if normalized_family == PROCESS_FAMILY_SDXL:
        if token in {
            PROCESS_CLASS_STANDARD_SDXL,
            "full_resident",
            "unified_streaming",
            "full",
            "standard",
            "standard_sdxl",
        }:
            return PROCESS_CLASS_STANDARD_SDXL
        if token in {
            PROCESS_CLASS_SDXL_GGUF_STAGED,
            "gguf_staged_residency",
            "staged",
            "gguf_staged",
        }:
            return PROCESS_CLASS_SDXL_GGUF_STAGED
        if token in {
            PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
            "gguf_true_streaming",
            "true_streaming",
        }:
            return PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING
    return token


@dataclass(frozen=True)
class ProcessKey:
    family: str
    process_class: str
    authoritative_identity: Any
    execution_family: Optional[str] = None
    residency_class: Optional[str] = None
    route_family: Optional[str] = None

    def normalized(self) -> "ProcessKey":
        return ProcessKey(
            family=normalize_process_family(self.family),
            process_class=normalize_process_class(self.process_class, family=self.family),
            authoritative_identity=self.authoritative_identity,
            execution_family=self.execution_family if self.execution_family is None else str(self.execution_family),
            residency_class=self.residency_class if self.residency_class is None else str(self.residency_class),
            route_family=self.route_family if self.route_family is None else str(self.route_family),
        )


@dataclass(frozen=True)
class ProcessTransitionDecision:
    action: str
    reason: str
    reset_required: bool
    current_key: ProcessKey | None
    requested_key: ProcessKey

    @property
    def reuse_allowed(self) -> bool:
        return not self.reset_required


class SharedProcessRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._active_key: ProcessKey | None = None
        self._active_family: str | None = None
        self._active_route_owner: str | None = None
        self._safe_to_retain: bool = False

    def get_active_key(self) -> ProcessKey | None:
        with self._lock:
            return self._active_key

    def set_active_key(self, key: ProcessKey | None) -> ProcessKey | None:
        normalized = key.normalized() if key is not None else None
        with self._lock:
            self._active_key = normalized
            if normalized is not None:
                self._active_family = normalized.family
                self._active_route_owner = None
                self._safe_to_retain = False
            else:
                self._active_family = None
                self._active_route_owner = None
                self._safe_to_retain = False
            return self._active_key

    def clear_active_key(self) -> None:
        with self._lock:
            self._active_key = None
            self._active_family = None
            self._active_route_owner = None
            self._safe_to_retain = False

    def get_active_family(self) -> str | None:
        with self._lock:
            return self._active_family

    def set_active_family(self, family: str | None) -> None:
        with self._lock:
            self._active_family = family

    def get_active_route_owner(self) -> str | None:
        with self._lock:
            return self._active_route_owner

    def set_active_route_owner(self, route_owner: str | None) -> None:
        with self._lock:
            self._active_route_owner = route_owner

    def is_safe_to_retain(self) -> bool:
        with self._lock:
            return self._safe_to_retain

    def set_safe_to_retain(self, safe: bool) -> None:
        with self._lock:
            self._safe_to_retain = safe

    def set_active_runtime(self, family: str | None, key: ProcessKey | None, route_owner: str | None, safe_to_retain: bool = False) -> None:
        normalized = key.normalized() if key is not None else None
        with self._lock:
            self._active_family = family
            self._active_key = normalized
            self._active_route_owner = route_owner
            self._safe_to_retain = safe_to_retain

    def clear_active_runtime(self) -> None:
        with self._lock:
            self._active_key = None
            self._active_family = None
            self._active_route_owner = None
            self._safe_to_retain = False

    def evaluate_transition(self, requested_key: ProcessKey) -> ProcessTransitionDecision:
        requested = requested_key.normalized()
        current = self.get_active_key()
        if current is None:
            return ProcessTransitionDecision(
                action="start",
                reason="no_active_process",
                reset_required=False,
                current_key=None,
                requested_key=requested,
            )

        if current == requested:
            return ProcessTransitionDecision(
                action="reuse",
                reason="same_process_identity",
                reset_required=False,
                current_key=current,
                requested_key=requested,
            )

        if current.family != requested.family:
            reason = "family_change"
        elif current.process_class != requested.process_class:
            reason = "process_class_change"
        elif current.authoritative_identity != requested.authoritative_identity:
            is_same_base_components = False
            if current.family == PROCESS_FAMILY_SDXL and requested.family == PROCESS_FAMILY_SDXL:
                curr_id = current.authoritative_identity
                req_id = requested.authoritative_identity
                if isinstance(curr_id, tuple) and isinstance(req_id, tuple):
                    curr_base_len = 1 if getattr(current, 'route_family', None) == 'gguf' else 2
                    req_base_len = 1 if getattr(requested, 'route_family', None) == 'gguf' else 2
                    if curr_base_len == req_base_len and len(curr_id) >= curr_base_len and len(req_id) >= req_base_len:
                        if curr_id[:curr_base_len] == req_id[:req_base_len]:
                            is_same_base_components = True
            
            if is_same_base_components:
                return ProcessTransitionDecision(
                    action="reuse",
                    reason="lora_stack_change",
                    reset_required=False,
                    current_key=current,
                    requested_key=requested,
                )
            reason = "identity_change"
        else:
            reason = "same_process_identity"
            return ProcessTransitionDecision(
                action="reuse",
                reason=reason,
                reset_required=False,
                current_key=current,
                requested_key=requested,
            )

        return ProcessTransitionDecision(
            action="reset",
            reason=reason,
            reset_required=True,
            current_key=current,
            requested_key=requested,
        )


_DEFAULT_REGISTRY = SharedProcessRegistry()


def get_active_process_key() -> ProcessKey | None:
    return _DEFAULT_REGISTRY.get_active_key()


def set_active_process_key(key: ProcessKey | None) -> ProcessKey | None:
    return _DEFAULT_REGISTRY.set_active_key(key)


def clear_active_process_key() -> None:
    _DEFAULT_REGISTRY.clear_active_key()


def get_active_family() -> str | None:
    return _DEFAULT_REGISTRY.get_active_family()


def set_active_family(family: str | None) -> None:
    _DEFAULT_REGISTRY.set_active_family(family)


def get_active_route_owner() -> str | None:
    return _DEFAULT_REGISTRY.get_active_route_owner()


def set_active_route_owner(route_owner: str | None) -> None:
    _DEFAULT_REGISTRY.set_active_route_owner(route_owner)


def is_safe_to_retain() -> bool:
    return _DEFAULT_REGISTRY.is_safe_to_retain()


def set_safe_to_retain(safe: bool) -> None:
    _DEFAULT_REGISTRY.set_safe_to_retain(safe)


def set_active_runtime(family: str | None, key: ProcessKey | None, route_owner: str | None, safe_to_retain: bool = False) -> None:
    _DEFAULT_REGISTRY.set_active_runtime(family, key, route_owner, safe_to_retain)


def clear_active_runtime() -> None:
    _DEFAULT_REGISTRY.clear_active_runtime()


def evaluate_process_transition(requested_key: ProcessKey) -> ProcessTransitionDecision:
    return _DEFAULT_REGISTRY.evaluate_transition(requested_key)


def build_process_key(
    *,
    family: Any,
    process_class: Any = None,
    authoritative_identity: Any,
    execution_family: Any = None,
    residency_class: Any = None,
    route_family: Any = None,
) -> ProcessKey:
    resolved_family = normalize_process_family(family)
    resolved_process_class = resolve_process_class(
        process_class,
        family=resolved_family,
        execution_family=execution_family,
        residency_class=residency_class,
    )
    return ProcessKey(
        family=resolved_family,
        process_class=resolved_process_class,
        authoritative_identity=authoritative_identity,
        execution_family=None if execution_family is None else str(execution_family),
        residency_class=None if residency_class is None else str(residency_class),
        route_family=None if route_family is None else str(route_family),
    )


def describe_process_key(key: ProcessKey | None) -> str:
    if key is None:
        return "<none>"
    return (
        f"family={key.family} "
        f"class={key.process_class} "
        f"identity={key.authoritative_identity!r}"
    )


def _sdxl_identity_components(key: ProcessKey | None) -> tuple[Any | None, Any | None, tuple[Any, ...]]:
    if key is None:
        return None, None, ()

    raw_identity = getattr(key, "authoritative_identity", None)
    if isinstance(raw_identity, tuple):
        identity = raw_identity
    elif raw_identity is None:
        identity = ()
    elif isinstance(raw_identity, list):
        identity = tuple(raw_identity)
    else:
        identity = (raw_identity,)

    checkpoint_identity = identity[0] if len(identity) > 0 else None
    is_gguf_route = str(getattr(key, "route_family", "") or "").lower() == "gguf"
    clip_identity = None if is_gguf_route else (identity[1] if len(identity) > 1 else None)
    lora_offset = 1 if is_gguf_route else 2
    lora_identity = tuple(identity[lora_offset:]) if len(identity) > lora_offset else ()
    return checkpoint_identity, clip_identity, lora_identity


def _resolve_process_checkpoint_label(key: ProcessKey | None) -> Any | None:
    if key is None:
        return None

    identity = getattr(key, "authoritative_identity", None)
    if key.family == PROCESS_FAMILY_SDXL:
        if isinstance(identity, (tuple, list)) and len(identity) > 0:
            return identity[0]
        return identity

    if key.family == PROCESS_FAMILY_FLUX_FILL:
        if isinstance(identity, (tuple, list)):
            for item in identity:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) >= 2
                    and str(item[0]).strip().lower() == "unet_path"
                ):
                    return item[1]
        return PROCESS_FAMILY_FLUX_FILL

    return identity


def classify_sdxl_process_key_changes(
    current_key: ProcessKey | None,
    requested_key: ProcessKey | None,
) -> list[Any]:
    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange

    changes: list[LifecycleChange] = []

    def add(change: LifecycleChange) -> None:
        if change not in changes:
            changes.append(change)

    if requested_key is None or requested_key.family != PROCESS_FAMILY_SDXL:
        add(LifecycleChange.FAMILY_CHANGE)
        return changes

    if current_key is None:
        add(LifecycleChange.CHECKPOINT_CHANGE)
        return changes

    if current_key.process_class != requested_key.process_class:
        add(LifecycleChange.SPINE_POSTURE_CHANGE)

    if current_key.authoritative_identity != requested_key.authoritative_identity:
        current_checkpoint, current_clip, current_loras = _sdxl_identity_components(current_key)
        requested_checkpoint, requested_clip, requested_loras = _sdxl_identity_components(requested_key)

        if current_checkpoint != requested_checkpoint:
            add(LifecycleChange.CHECKPOINT_CHANGE)
        if current_clip != requested_clip:
            add(LifecycleChange.MODEL_CHANGE)
        if current_loras != requested_loras:
            add(LifecycleChange.LORA_STACK_CHANGE)

        if not changes:
            add(LifecycleChange.MODEL_CHANGE)

    return changes


def log_stage_telemetry(
    stage_name: str,
    target_phase: str | None = None,
    *,
    prefetch_count: int | None = None,
    posture_override: str | None = None,
) -> None:
    import torch
    import logging
    if posture_override is not None:
        posture = str(posture_override)
    else:
        active_key = get_active_process_key()
        posture = "Unknown"
        if active_key is not None:
            p_class = str(active_key.process_class or "").lower()
            res_class = str(active_key.residency_class or "").lower()
            if "streaming" in p_class or "streaming" in res_class:
                posture = "Streaming"
            else:
                posture = "Resident"

    free_vram_bytes = 0.0
    cached_vram_bytes = 0.0
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free_vram_bytes = float(torch.cuda.mem_get_info(device)[0])
            cached_vram_bytes = float(torch.cuda.memory_reserved(device))
    except Exception:
        pass

    if prefetch_count is None:
        prefetch_count = 0

    logging.info(
        f"[Nex-Telemetry] Stage Switch: stage={stage_name} | "
        f"posture={posture} | "
        f"free_vram={free_vram_bytes / (1024*1024):.1f}MB | "
        f"cached_vram={cached_vram_bytes / (1024*1024):.1f}MB | "
        f"prefetch_queue={prefetch_count}"
    )


def resolve_preflight_additional_loras(task_state) -> list:
    additional_loras = []
    route_intent = resolve_route_intent(task_state)

    # 1. Inpaint / Outpaint patch LoRA
    try:
        from modules import flags, config
        is_outpaint = route_intent.wants_outpaint
        is_inpaint = route_intent.wants_inpaint
        use_flux_fill_inpaint = route_intent.wants_flux_inpaint

        if (is_outpaint or is_inpaint) and not use_flux_fill_inpaint:
            engine = getattr(task_state, 'outpaint_engine', 'None') if is_outpaint else getattr(task_state, 'inpaint_engine', 'None')
            engine = flags.normalize_inpaint_engine_version(engine, default=flags.INPAINT_ENGINE_NONE)
            if engine != flags.INPAINT_ENGINE_NONE:
                inpaint_patch_model_path = config.downloading_inpaint_models(engine)
                additional_loras.append((inpaint_patch_model_path, 1.0))
    except Exception:
        pass

    return additional_loras


def resolve_sdxl_process_key(task_state) -> ProcessKey | None:
    from modules.pipeline.inference import resolve_unified_sdxl_process_key

    return resolve_unified_sdxl_process_key(
        task_state,
        loras=getattr(task_state, 'loras', []) or [],
        base_model_additional_loras=getattr(task_state, 'base_model_additional_loras', []) or [],
    )


def resolve_flux_fill_process_key(
    task_state,
    *,
    route_family: str | None = None,
    selected_engine: str | None = None,
) -> ProcessKey | None:
    from backend.flux_fill_v3 import resolve_flux_fill_process_key as resolve_greenfield
    return resolve_greenfield(task_state, route_family=route_family, selected_engine=selected_engine)


def _is_auxiliary_only_route(route, task_state) -> bool:
    route_id = str(getattr(route, 'route_id', '') or '').strip().lower()
    selected_engine = normalize_objr_engine(getattr(task_state, 'objr_engine', None))

    if route_id == 'upscale':
        return True
    if route_id == 'removal' and selected_engine != OBJR_ENGINE_FLUX_FILL:
        return True
    return False


def resolve_requested_process_key(task_state, route) -> ProcessKey | None:
    selected_engine = normalize_objr_engine(getattr(task_state, 'objr_engine', None))
    route_id = str(getattr(route, 'route_id', '') or '').strip().lower()
    expects_flux_process = (
        route.family == 'flux_fill'
        or (route_id in {'removal', 'flux_removal'} and selected_engine == OBJR_ENGINE_FLUX_FILL)
    )
    if expects_flux_process:
        return resolve_flux_fill_process_key(
            task_state,
            route_family=route.family,
            selected_engine=selected_engine,
        )
    if _is_auxiliary_only_route(route, task_state):
        # Auxiliary-only routes do not own a major-family process. Preserve an
        # already-active SDXL/Flux family if one exists, otherwise publish no
        # process identity at all.
        return get_active_process_key()
    if getattr(task_state.sdxl_execution_policy, 'enabled', False):
        return resolve_sdxl_process_key(task_state)
    return None


def release_process_boundary(current_key: ProcessKey | None, requested_key: ProcessKey | None) -> Any:
    if current_key is None:
        return None

    if current_key.family == PROCESS_FAMILY_FLUX_FILL:
        import backend.resources as resources
        from backend.flux_fill_v3 import (
            release_active_flux_resident_spine,
            release_flux_latent_artifacts,
        )

        release_state = {
            'released_spine': False,
            'released_artifacts': False,
        }

        def _release_callback() -> None:
            release_state['released_spine'] = bool(
                release_active_flux_resident_spine(reason='route_transition')
            )
            release_state['released_artifacts'] = bool(release_flux_latent_artifacts())

        resources.prepare_for_checkpoint_switch(
            current_model=_resolve_process_checkpoint_label(current_key),
            next_model=_resolve_process_checkpoint_label(requested_key),
            release_callback=_release_callback,
            notes={
                'reason': 'route_transition',
                'current_process_key': describe_process_key(current_key),
                'next_process_key': describe_process_key(requested_key),
            },
        )
        return {
            'released': release_state['released_spine'] or release_state['released_artifacts'],
            'reason': 'greenfield_flux_route_transition',
            'hard_reset': False,
            'current_process_key': current_key,
            'next_process_key': requested_key,
        }

    if requested_key is None:
        return None

    if current_key.family == PROCESS_FAMILY_SDXL:
        import backend.resources as resources
        from backend import sdxl_unified_runtime

        current_model_name = _resolve_process_checkpoint_label(current_key)
        next_model_name = _resolve_process_checkpoint_label(requested_key)

        def _release_callback():
            teardown = (requested_key is None or requested_key.family != PROCESS_FAMILY_SDXL)
            try:
                sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache(teardown=teardown)
            except TypeError as exc:
                if "unexpected keyword argument 'teardown'" not in str(exc):
                    raise
                sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache()
            try:
                from backend import conditioning
                conditioning.clear_prompt_conditioning_cache()
            except Exception:
                pass
            try:
                from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange
                changes = classify_sdxl_process_key_changes(current_key, requested_key)
                if not changes:
                    changes.append(LifecycleChange.MODEL_CHANGE)
                release_for_changes(changes, reason='route_transition')
            except Exception:
                pass

        resources.prepare_for_checkpoint_switch(
            current_model=current_model_name,
            next_model=next_model_name,
            release_callback=_release_callback,
            notes={
                'reason': 'route_transition',
                'current_process_key': describe_process_key(current_key),
                'next_process_key': describe_process_key(requested_key),
            },
        )

        released = False
        import sys
        if 'modules.default_pipeline' in sys.modules:
            try:
                import modules.default_pipeline as default_pipeline
                default_pipeline.release_sdxl_runtime_state(
                    current_process_key=current_key,
                    next_process_key=requested_key,
                    current_model_name=current_model_name,
                    next_model_name=next_model_name,
                    current_vae_name=None,
                    next_vae_name=None,
                    reason='route_transition',
                    hard_reset=False,
                )
                released = True
            except Exception:
                pass
        else:
            released = True

        return {
            'released': released,
            'reason': 'route_transition',
            'hard_reset': False,
            'current_process_key': current_key,
            'next_process_key': requested_key,
        }

    return None


def apply_process_transition_gate(requested_key: ProcessKey | None) -> ProcessTransitionDecision | None:
    current_key = get_active_process_key()
    if requested_key is None:
        if current_key is not None and current_key.family == PROCESS_FAMILY_FLUX_FILL:
            release_process_boundary(current_key, None)
            clear_active_runtime()
        return None

    decision = evaluate_process_transition(requested_key)
    if decision.reset_required:
        release_process_boundary(current_key, requested_key)
        clear_active_runtime()
    return decision


def sync_route_process_activation(route, task_state, requested_process_key: ProcessKey | None) -> Any:
    if _is_auxiliary_only_route(route, task_state):
        # Auxiliary-only routes borrow the currently active major-family
        # posture if one exists, but they never replace the registry with their
        # own route-owned identity.
        return None

    if route.family == "flux_fill":
        from backend.flux_fill_v3 import sync_flux_fill_process_activation
        return sync_flux_fill_process_activation(route, task_state, requested_process_key)

    elif route.family == "sdxl" or getattr(task_state.sdxl_execution_policy, "enabled", False):
        if requested_process_key is not None and requested_process_key.family == PROCESS_FAMILY_SDXL:
            policy = getattr(task_state, 'sdxl_execution_policy', None)
            execution_mode = getattr(policy, 'execution_mode', None)
            safe_to_retain = (execution_mode == 'resident')

            set_active_runtime(
                family=PROCESS_FAMILY_SDXL,
                key=requested_process_key,
                route_owner=route.route_id,
                safe_to_retain=safe_to_retain
            )
        else:
            clear_active_runtime()
        return None

    else:
        clear_active_runtime()
        return None


def reconcile_runtime_state(route, task_state) -> ProcessTransitionDecision | None:
    task_state.base_model_additional_loras = resolve_preflight_additional_loras(task_state)
    requested_process_key = resolve_requested_process_key(task_state, route)
    decision = apply_process_transition_gate(requested_process_key)
    sync_route_process_activation(route, task_state, requested_process_key)
    return decision
