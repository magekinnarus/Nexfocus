"""
Central memory governance scaffold inspired by ComfyUI Dynamic VRAM.

The goal here is not to reproduce ComfyUI's allocator internals. Instead, we
provide a single place where phase transitions, cache policy, environment-aware
thresholds, and lightweight telemetry can live so the rest of the app stops
making ad hoc memory decisions.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import logging
import platform
import threading
import time
from typing import Any, Deque, Dict, Optional

from backend import environment_profile as environment_profiles
from backend import sdxl_runtime_policy
from modules.route_intent import resolve_route_intent

import psutil


class MemoryPhase(str, Enum):
    IDLE = 'idle'
    TASK = 'task'
    ROUTE_SELECT = 'route_select'
    MODEL_REFRESH = 'model_refresh'
    PROMPT_ENCODE = 'prompt_encode'
    IMAGE_INPUT_PREPARE = 'image_input_prepare'
    INPAINT_PREPARE = 'inpaint_prepare'
    OUTPAINT_PREPARE = 'outpaint_prepare'
    VAE_ENCODE = 'vae_encode'
    STRUCTURAL_PREPROCESS = 'structural_preprocess'
    CONTEXTUAL_PREPROCESS = 'contextual_preprocess'
    CONTROL_APPLY = 'control_apply'
    REMOVAL = 'removal'
    DIFFUSION = 'diffusion'
    DECODE = 'decode'
    STITCH = 'stitch'
    UPSCALE = 'upscale'
    TILED_REFINE = 'tiled_refine'
    FINALIZE = 'finalize'


PHASE_ALIASES = {
    'prepare': MemoryPhase.MODEL_REFRESH.value,
    'image_input': MemoryPhase.IMAGE_INPUT_PREPARE.value,
    'control': MemoryPhase.CONTROL_APPLY.value,
    'postprocess': MemoryPhase.FINALIZE.value,
}


RESIDENCY_RESOURCE_GROUPS = {
    'core_checkpoint': ('unet', 'clip', 'vae'),
    'support_caches': ('controlnet', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support'),
    'route_artifacts': ('prompt_conditions', 'route_state'),
}


def _ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
    return tuple(ordered)


def _plan(*, pinned: tuple[str, ...] = (), warm: tuple[str, ...] = (), evictable: tuple[str, ...] = ()) -> Dict[str, tuple[str, ...]]:
    return {
        'pinned': _ordered_unique(pinned),
        'warm': _ordered_unique(warm),
        'evictable': _ordered_unique(evictable),
    }


def _task_expects_controlnet(task) -> bool:
    if task is None:
        return True
    try:
        return resolve_route_intent(task).expects_controlnet
    except Exception:
        return True
    return False


def _move_resource_to_evictable(resource_id: str, pinned: list[str], warm: list[str], evictable: list[str]) -> None:
    if resource_id in pinned:
        return
    if resource_id in warm:
        warm.remove(resource_id)
    if resource_id not in evictable:
        evictable.append(resource_id)


def _task_uses_dedicated_gguf_runtime(task) -> bool:
    if task is None:
        return False

    try:
        policy = getattr(task, 'sdxl_execution_policy', None)
        runtime_family = str(
            getattr(policy, 'runtime_family', None)
            or getattr(task, 'sdxl_runtime_family', None)
            or ''
        ).strip().lower()

        if sdxl_runtime_policy.policy_marks_legacy_sdxl_gguf(policy):
            return False

        if runtime_family == "gguf_sdxl":
            return True

        execution_family = str(
            getattr(task, 'sdxl_execution_family', None)
            or getattr(policy, 'execution_family', None)
            or ''
        ).strip().lower()
        residency_class = str(
            getattr(task, 'sdxl_residency_class', None)
            or getattr(policy, 'residency_class', None)
            or ''
        ).strip().lower()

        return (
            execution_family == sdxl_runtime_policy.EXECUTION_FAMILY_GGUF_STAGED
            or residency_class == sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_GGUF_STAGED
            or residency_class == sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING
        )
    except Exception:
        return False


GGUF_RESIDENCY_PLANS = {
    MemoryPhase.TASK.value: _plan(
        warm=('unet',),
        evictable=('clip', 'vae') + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',),
    ),
    MemoryPhase.ROUTE_SELECT.value: _plan(
        warm=('unet',),
        evictable=('clip', 'vae') + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',),
    ),
    MemoryPhase.MODEL_REFRESH.value: _plan(
        pinned=('unet', 'clip'),
        evictable=('vae',) + RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts'],
    ),
    MemoryPhase.PROMPT_ENCODE.value: _plan(
        pinned=('clip',),
        warm=('unet',),
        evictable=('vae',) + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',),
    ),
    MemoryPhase.IMAGE_INPUT_PREPARE.value: _plan(
        pinned=('vae',),
        warm=('unet',),
        evictable=('clip', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.INPAINT_PREPARE.value: _plan(
        pinned=('vae',),
        warm=('unet',),
        evictable=('clip', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.OUTPAINT_PREPARE.value: _plan(
        pinned=('vae',),
        warm=('unet',),
        evictable=('clip', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.VAE_ENCODE.value: _plan(
        pinned=('vae',),
        warm=('unet',),
        evictable=('clip', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.DIFFUSION.value: _plan(
        pinned=('unet',),
        evictable=('clip', 'vae', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.DECODE.value: _plan(
        pinned=('vae',),
        warm=('unet',),
        evictable=('clip', 'controlnet') + RESIDENCY_RESOURCE_GROUPS['support_caches'],
    ),
    MemoryPhase.STITCH.value: _plan(
        evictable=('clip', 'vae') + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',),
    ),
    MemoryPhase.FINALIZE.value: _plan(
        warm=('unet',),
        evictable=('clip', 'vae') + RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts'],
    ),
}

BASE_RESIDENCY_PLANS = {
    MemoryPhase.IDLE.value: _plan(evictable=RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts']),
    MemoryPhase.TASK.value: _plan(warm=('unet', 'vae'), evictable=('clip',) + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',)),
    MemoryPhase.ROUTE_SELECT.value: _plan(warm=('unet', 'vae'), evictable=('clip',) + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',)),
    MemoryPhase.MODEL_REFRESH.value: _plan(pinned=('unet', 'clip', 'vae'), evictable=RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts']),
    MemoryPhase.PROMPT_ENCODE.value: _plan(pinned=('clip',), warm=('unet', 'vae'), evictable=RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',)),
    MemoryPhase.IMAGE_INPUT_PREPARE.value: _plan(pinned=('vae',), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.INPAINT_PREPARE.value: _plan(pinned=('vae',), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.OUTPAINT_PREPARE.value: _plan(pinned=('vae',), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.VAE_ENCODE.value: _plan(pinned=('vae',), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.STRUCTURAL_PREPROCESS.value: _plan(pinned=('controlnet', 'structural_preprocessors'), warm=('unet', 'vae'), evictable=('clip', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.CONTEXTUAL_PREPROCESS.value: _plan(pinned=('contextual_adapters', 'clip_vision', 'insightface', 'pulid_support'), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors')),
    MemoryPhase.CONTROL_APPLY.value: _plan(pinned=('unet', 'controlnet'), warm=('contextual_adapters', 'vae'), evictable=('clip', 'structural_preprocessors', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.REMOVAL.value: _plan(pinned=('removal_models',), evictable=RESIDENCY_RESOURCE_GROUPS['core_checkpoint'] + RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts']),
    MemoryPhase.DIFFUSION.value: _plan(pinned=('unet',), warm=('vae', 'controlnet', 'contextual_adapters'), evictable=('clip', 'structural_preprocessors', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.DECODE.value: _plan(pinned=('vae',), warm=('unet', 'controlnet'), evictable=('clip', 'structural_preprocessors', 'contextual_adapters', 'clip_vision', 'insightface', 'pulid_support')),
    MemoryPhase.STITCH.value: _plan(warm=('vae',), evictable=('clip',) + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',)),
    MemoryPhase.UPSCALE.value: _plan(pinned=('upscaler_model',), warm=('prompt_conditions',), evictable=RESIDENCY_RESOURCE_GROUPS['core_checkpoint'] + RESIDENCY_RESOURCE_GROUPS['support_caches'] + ('route_state',)),
    MemoryPhase.TILED_REFINE.value: _plan(pinned=('unet', 'vae'), warm=('prompt_conditions',), evictable=('clip',) + RESIDENCY_RESOURCE_GROUPS['support_caches']),
    MemoryPhase.FINALIZE.value: _plan(evictable=RESIDENCY_RESOURCE_GROUPS['support_caches'] + RESIDENCY_RESOURCE_GROUPS['route_artifacts']),
}


PROFILE_RESIDENCY_OVERRIDES = {
    environment_profiles.PROFILE_COLAB_FREE: {
        'extra_evictable': ('contextual_adapters', 'clip_vision', 'insightface', 'pulid_support'),
    },
    environment_profiles.PROFILE_LOCAL_LOW_VRAM: {
        'extra_evictable': ('contextual_adapters', 'clip_vision', 'insightface', 'pulid_support'),
    },
    environment_profiles.PROFILE_COLAB_PRO: {
        MemoryPhase.DIFFUSION.value: {
            'warm': ('clip_vision', 'insightface', 'pulid_support'),
        },
        MemoryPhase.DECODE.value: {
            'warm': ('controlnet', 'contextual_adapters'),
        },
    },
    environment_profiles.PROFILE_LOCAL_NORMAL: {
        MemoryPhase.DIFFUSION.value: {
            'warm': ('controlnet', 'contextual_adapters'),
        },
    },
}


@dataclass
class MemorySnapshot:
    timestamp: float
    phase: str
    total_vram_mb: Optional[float]
    free_vram_mb: Optional[float]
    total_ram_mb: Optional[float]
    free_ram_mb: Optional[float]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResidencyPlan:
    pinned: tuple[str, ...] = ()
    warm: tuple[str, ...] = ()
    evictable: tuple[str, ...] = ()
    notes: Dict[str, Any] = field(default_factory=dict)

    def mode_for(self, resource_id: str) -> str | None:
        if resource_id in self.pinned:
            return 'pinned'
        if resource_id in self.warm:
            return 'warm'
        if resource_id in self.evictable:
            return 'evictable'
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            'pinned': list(self.pinned),
            'warm': list(self.warm),
            'evictable': list(self.evictable),
            'notes': dict(self.notes),
        }


@dataclass
class MemoryPolicy:
    low_vram_threshold_mb: float = 8192.0
    medium_vram_threshold_mb: float = 16384.0
    low_vram_cache_cooldown_s: float = 0.5
    medium_vram_cache_cooldown_s: float = 1.5
    high_vram_cache_cooldown_s: float = 4.0
    low_ram_headroom_mb: float = 2048.0
    critical_ram_headroom_mb: float = 1024.0
    checkpoint_switch_ram_headroom_mb: float = 4096.0
    minimum_cache_cooldown_s: float = 0.25
    linux_malloc_trim_enabled: bool = False
    linux_malloc_trim_trigger_mb: float = 2048.0
    aggressive_checkpoint_switch_reclaim: bool = False


@dataclass
class MemoryAffordance:
    allowed: bool
    phase: str
    required_ram_mb: float
    required_vram_mb: float
    minimum_free_ram_mb: float
    minimum_free_vram_mb: float
    free_ram_mb: Optional[float]
    free_vram_mb: Optional[float]
    free_ram_after_mb: Optional[float]
    free_vram_after_mb: Optional[float]
    reason: str
    notes: Dict[str, Any] = field(default_factory=dict)


class MemoryGovernor:
    def __init__(self, policy: MemoryPolicy | None = None):
        self._lock = threading.RLock()
        self._phase = MemoryPhase.IDLE.value
        self._phase_started_at = time.time()
        self._last_cache_flush = 0.0
        self._history: Deque[MemorySnapshot] = deque(maxlen=64)
        self._phase_stack: list[tuple[str, float, Any | None]] = []
        self._base_policy = policy or MemoryPolicy()
        self._current_task = None
        self.policy = MemoryPolicy(**vars(self._base_policy))
        self._environment_profile = None

    def configure_environment(self, profile=None, policy: MemoryPolicy | None = None):
        with self._lock:
            if profile is not None:
                self._environment_profile = profile
                merged = dict(vars(self._base_policy))
                merged.update(getattr(profile, 'policy_overrides', {}) or {})
                if policy is not None:
                    merged.update(vars(policy))
                self.policy = MemoryPolicy(**merged)
            elif policy is not None:
                self.policy = MemoryPolicy(**vars(policy))

    def environment_profile(self):
        return self._environment_profile

    def profile_name(self):
        profile = self.environment_profile()
        return getattr(profile, 'name', 'unconfigured')

    def policy_summary(self):
        return {
            'profile': self.profile_name(),
            'low_ram_headroom_mb': self.policy.low_ram_headroom_mb,
            'critical_ram_headroom_mb': self.policy.critical_ram_headroom_mb,
            'checkpoint_switch_ram_headroom_mb': self.policy.checkpoint_switch_ram_headroom_mb,
            'low_vram_threshold_mb': self.policy.low_vram_threshold_mb,
            'medium_vram_threshold_mb': self.policy.medium_vram_threshold_mb,
            'linux_malloc_trim_enabled': self.policy.linux_malloc_trim_enabled,
            'linux_malloc_trim_trigger_mb': self.policy.linux_malloc_trim_trigger_mb,
            'aggressive_checkpoint_switch_reclaim': self.policy.aggressive_checkpoint_switch_reclaim,
        }

    def begin_phase(self, phase: str | MemoryPhase, task=None, notes: Dict[str, Any] | None = None):
        phase_name = self._normalize_phase(phase)
        with self._lock:
            started_at = time.time()
            self._phase_stack.append((phase_name, started_at, task))
            self._phase = phase_name
            self._phase_started_at = started_at
            self._current_task = task
            snapshot = self.capture_snapshot(notes=notes, task=task)
            self._history.append(snapshot)
            return snapshot

    def end_phase(self, phase: str | MemoryPhase | None = None, notes: Dict[str, Any] | None = None):
        with self._lock:
            if phase is None:
                if self._phase_stack:
                    self._phase_stack.pop()
            else:
                phase_name = self._normalize_phase(phase)
                for index in range(len(self._phase_stack) - 1, -1, -1):
                    if self._phase_stack[index][0] == phase_name:
                        del self._phase_stack[index]
                        break

            if self._phase_stack:
                self._phase, self._phase_started_at, self._current_task = self._phase_stack[-1]
            else:
                self._phase = MemoryPhase.IDLE.value
                self._phase_started_at = time.time()
                self._current_task = None
            snapshot = self.capture_snapshot(notes=notes)
            self._history.append(snapshot)
            return snapshot

    @contextmanager
    def phase_scope(
        self,
        phase: str | MemoryPhase,
        task=None,
        notes: Dict[str, Any] | None = None,
        end_notes: Dict[str, Any] | None = None,
    ):
        phase_name = self._normalize_phase(phase)
        self.begin_phase(phase_name, task=task, notes=notes)
        try:
            yield phase_name
        finally:
            self.end_phase(phase_name, notes=end_notes)

    def capture_snapshot(self, notes: Dict[str, Any] | None = None, task=None):
        total_vram_mb = None
        free_vram_mb = None
        total_ram_mb = float(psutil.virtual_memory().total) / (1024 * 1024)
        free_ram_mb = float(psutil.virtual_memory().available) / (1024 * 1024)

        try:
            from backend import resources as resource_state

            total_vram_mb = getattr(resource_state, 'total_vram', None)
            if total_vram_mb is not None:
                total_vram_mb = float(total_vram_mb)

            try:
                device = resource_state.get_torch_device()
                free_vram_mb = float(resource_state.get_free_memory(device)) / (1024 * 1024)
            except Exception:
                free_vram_mb = None
        except Exception:
            logging.debug('MemoryGovernor could not import backend.resources for snapshot capture.', exc_info=True)

        payload = dict(notes or {})
        payload.setdefault('profile', self.profile_name())
        if task is not None:
            payload.setdefault('task_type', task.__class__.__name__)

        return MemorySnapshot(
            timestamp=time.time(),
            phase=self.current_phase(),
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            total_ram_mb=total_ram_mb,
            free_ram_mb=free_ram_mb,
            notes=payload,
        )

    def plan_for_task(self, task=None, phase: str | MemoryPhase | None = None):
        phase_name = self._normalize_phase(phase) if phase is not None else self.current_phase()
        if task is None:
            task = self._current_task
        notes = {'phase': phase_name}
        if task is not None:
            notes['task_type'] = task.__class__.__name__
        profile_name = self.profile_name()
        notes['profile'] = profile_name

        # This governor only decides phase-local warmth and eviction for compatibility surfaces.
        # It must not re-plan model family, runtime posture, or Flux fallback policy.
        uses_dedicated_gguf_route = _task_uses_dedicated_gguf_runtime(task)
        plan_table = GGUF_RESIDENCY_PLANS if uses_dedicated_gguf_route else BASE_RESIDENCY_PLANS
        base_plan = plan_table.get(phase_name, BASE_RESIDENCY_PLANS[MemoryPhase.IDLE.value])
        pinned = list(base_plan['pinned'])
        warm = [item for item in base_plan['warm'] if item not in pinned]
        evictable = [item for item in base_plan['evictable'] if item not in pinned and item not in warm]

        overrides = PROFILE_RESIDENCY_OVERRIDES.get(profile_name, {})
        extra_evictable = tuple(overrides.get('extra_evictable', ()))
        for resource_id in extra_evictable:
            if resource_id in warm:
                warm.remove(resource_id)
            if resource_id not in pinned and resource_id not in evictable:
                evictable.append(resource_id)

        phase_override = overrides.get(phase_name, {})
        for resource_id in phase_override.get('pinned', ()):
            if resource_id not in pinned:
                pinned.append(resource_id)
            if resource_id in warm:
                warm.remove(resource_id)
            if resource_id in evictable:
                evictable.remove(resource_id)
        for resource_id in phase_override.get('warm', ()):
            if resource_id in pinned:
                continue
            if resource_id not in warm:
                warm.append(resource_id)
            if resource_id in evictable:
                evictable.remove(resource_id)
        for resource_id in phase_override.get('evictable', ()):
            if resource_id in pinned:
                continue
            if resource_id in warm:
                warm.remove(resource_id)
            if resource_id not in evictable:
                evictable.append(resource_id)

        if phase_name in {MemoryPhase.DIFFUSION.value, MemoryPhase.DECODE.value} and not _task_expects_controlnet(task):
            _move_resource_to_evictable('controlnet', pinned, warm, evictable)
            notes['controlnet_expected'] = False
        notes['source'] = 'gguf_phase_residency' if uses_dedicated_gguf_route else 'profile_phase_residency'
        if uses_dedicated_gguf_route:
            notes['route_family'] = 'gguf'
        return ResidencyPlan(
            pinned=tuple(pinned),
            warm=tuple(warm),
            evictable=tuple(evictable),
            notes=notes,
        )

    def can_afford(
        self,
        *,
        required_ram_mb: float = 0.0,
        required_vram_mb: float = 0.0,
        minimum_free_ram_mb: float | None = None,
        minimum_free_vram_mb: float = 0.0,
        phase: str | MemoryPhase | None = None,
        notes: Dict[str, Any] | None = None,
    ):
        phase_name = self._normalize_phase(phase) if phase is not None else self.current_phase()
        snapshot = self.capture_snapshot(notes=notes)
        ram_floor = self.policy.low_ram_headroom_mb if minimum_free_ram_mb is None else float(minimum_free_ram_mb)
        vram_floor = float(minimum_free_vram_mb)

        free_ram_after = None if snapshot.free_ram_mb is None else float(snapshot.free_ram_mb) - float(required_ram_mb)
        free_vram_after = None if snapshot.free_vram_mb is None else float(snapshot.free_vram_mb) - float(required_vram_mb)

        ram_ok = free_ram_after is None or free_ram_after >= ram_floor
        vram_ok = free_vram_after is None or free_vram_after >= vram_floor
        allowed = ram_ok and vram_ok

        reason_parts = []
        if not ram_ok:
            reason_parts.append(
                f"ram_after={free_ram_after:.1f}MB below floor={ram_floor:.1f}MB"
            )
        if not vram_ok:
            reason_parts.append(
                f"vram_after={free_vram_after:.1f}MB below floor={vram_floor:.1f}MB"
            )
        if not reason_parts:
            reason_parts.append('headroom_ok')

        return MemoryAffordance(
            allowed=allowed,
            phase=phase_name,
            required_ram_mb=float(required_ram_mb),
            required_vram_mb=float(required_vram_mb),
            minimum_free_ram_mb=ram_floor,
            minimum_free_vram_mb=vram_floor,
            free_ram_mb=snapshot.free_ram_mb,
            free_vram_mb=snapshot.free_vram_mb,
            free_ram_after_mb=free_ram_after,
            free_vram_after_mb=free_vram_after,
            reason='; '.join(reason_parts),
            notes=dict(notes or {}),
        )

    def needs_host_cleanup(self, *, required_ram_mb: float = 0.0, minimum_free_ram_mb: float | None = None, aggressive: bool = False):
        affordance = self.can_afford(
            required_ram_mb=required_ram_mb,
            minimum_free_ram_mb=minimum_free_ram_mb,
        )
        if aggressive:
            return True
        if affordance.free_ram_after_mb is not None and affordance.free_ram_after_mb < self.policy.critical_ram_headroom_mb:
            return True
        return not affordance.allowed

    def should_trim_host_memory(self, snapshot: MemorySnapshot | None = None, *, aggressive: bool = False):
        if platform.system() != 'Linux':
            return False
        if aggressive:
            return True
        if not self.policy.linux_malloc_trim_enabled:
            return False
        snapshot = snapshot or self.capture_snapshot()
        return snapshot.free_ram_mb is not None and snapshot.free_ram_mb < self.policy.linux_malloc_trim_trigger_mb

    def should_flush_cache(self, force: bool = False):
        if force:
            return True

        snapshot = self.capture_snapshot()
        total_vram = snapshot.total_vram_mb
        if total_vram is None:
            cooldown = self.policy.minimum_cache_cooldown_s
        elif total_vram < self.policy.low_vram_threshold_mb:
            cooldown = self.policy.low_vram_cache_cooldown_s
        elif total_vram < self.policy.medium_vram_threshold_mb:
            cooldown = self.policy.medium_vram_cache_cooldown_s
        else:
            cooldown = self.policy.high_vram_cache_cooldown_s

        if snapshot.phase in {MemoryPhase.DIFFUSION.value, MemoryPhase.DECODE.value}:
            cooldown *= 1.25
        elif snapshot.phase in {MemoryPhase.REMOVAL.value, MemoryPhase.UPSCALE.value}:
            cooldown *= 0.75

        if snapshot.free_ram_mb is not None and snapshot.free_ram_mb < self.policy.low_ram_headroom_mb:
            cooldown = max(self.policy.minimum_cache_cooldown_s, cooldown * 0.5)

        vram_pressure = False
        if snapshot.free_vram_mb is not None and snapshot.total_vram_mb is not None and snapshot.total_vram_mb > 0.0:
            vram_pressure = (snapshot.free_vram_mb / snapshot.total_vram_mb) < 0.12

        ram_pressure = False
        if snapshot.free_ram_mb is not None:
            ram_pressure = snapshot.free_ram_mb < self.policy.low_ram_headroom_mb

        under_pressure = vram_pressure or ram_pressure

        elapsed = time.time() - self._last_cache_flush
        return (elapsed >= max(self.policy.minimum_cache_cooldown_s, cooldown)) and under_pressure

    def note_cache_flush(self):
        with self._lock:
            self._last_cache_flush = time.time()

    def current_phase(self):
        return self._phase

    def phase_age(self):
        return time.time() - self._phase_started_at

    def history(self):
        return tuple(self._history)

    @staticmethod
    def _normalize_phase(phase: str | MemoryPhase):
        if isinstance(phase, MemoryPhase):
            return phase.value
        phase_name = str(phase).strip().lower()
        return PHASE_ALIASES.get(phase_name, phase_name)


governor = MemoryGovernor()


def configure_environment(profile=None, policy: MemoryPolicy | None = None):
    governor.configure_environment(profile=profile, policy=policy)


def environment_profile():
    return governor.environment_profile()


def policy_summary():
    return governor.policy_summary()


def begin_phase(phase: str | MemoryPhase, task=None, notes: Dict[str, Any] | None = None):
    return governor.begin_phase(phase, task=task, notes=notes)


def end_phase(phase: str | MemoryPhase | None = None, notes: Dict[str, Any] | None = None):
    return governor.end_phase(phase, notes=notes)


def phase_scope(
    phase: str | MemoryPhase,
    task=None,
    notes: Dict[str, Any] | None = None,
    end_notes: Dict[str, Any] | None = None,
):
    return governor.phase_scope(phase, task=task, notes=notes, end_notes=end_notes)


def capture_snapshot(notes: Dict[str, Any] | None = None, task=None):
    return governor.capture_snapshot(notes=notes, task=task)


def can_afford(**kwargs):
    return governor.can_afford(**kwargs)


def needs_host_cleanup(**kwargs):
    return governor.needs_host_cleanup(**kwargs)


def should_trim_host_memory(snapshot: MemorySnapshot | None = None, *, aggressive: bool = False):
    return governor.should_trim_host_memory(snapshot=snapshot, aggressive=aggressive)


def should_flush_cache(force: bool = False):
    return governor.should_flush_cache(force=force)


def note_cache_flush():
    governor.note_cache_flush()


def plan_for_task(task=None, phase: str | MemoryPhase | None = None):
    return governor.plan_for_task(task=task, phase=phase)


def current_phase():
    return governor.current_phase()


def normalize_phase(phase: str | MemoryPhase):
    return governor._normalize_phase(phase)
