import os
import sys
from types import SimpleNamespace

import pytest

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import environment_profile, memory_governor, resources


@pytest.fixture
def restore_profile():
    original_profile = memory_governor.environment_profile()
    original_policy = memory_governor.governor.policy
    yield
    memory_governor.configure_environment(original_profile, original_policy)


def test_residency_plan_differs_by_profile_and_phase(restore_profile):
    free_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_COLAB_FREE,
        total_ram_mb=16384,
        total_vram_mb=15360,
        is_colab=True,
    )
    memory_governor.configure_environment(free_profile)
    free_plan = memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.DIFFUSION)

    pro_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_COLAB_PRO,
        total_ram_mb=53248,
        total_vram_mb=23000,
        is_colab=True,
    )
    memory_governor.configure_environment(pro_profile)
    pro_plan = memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.DIFFUSION)

    assert free_plan.mode_for('unet') == 'pinned'
    assert free_plan.mode_for('controlnet') == 'warm'
    assert free_plan.mode_for('clip_vision') == 'evictable'
    decode_plan = memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.DECODE)
    assert decode_plan.mode_for('controlnet') == 'warm'
    assert pro_plan.mode_for('unet') == 'pinned'
    assert pro_plan.mode_for('clip_vision') == 'warm'
    assert memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.PROMPT_ENCODE).mode_for('clip') == 'pinned'


def test_txt2img_does_not_keep_controlnet_warm_on_constrained_profiles(restore_profile):
    low_vram_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    memory_governor.configure_environment(low_vram_profile)

    txt2img_task = SimpleNamespace(current_tab='txt2img', cn_tasks={})
    diffusion_plan = memory_governor.plan_for_task(task=txt2img_task, phase=memory_governor.MemoryPhase.DIFFUSION)
    decode_plan = memory_governor.plan_for_task(task=txt2img_task, phase=memory_governor.MemoryPhase.DECODE)

    assert diffusion_plan.mode_for('controlnet') == 'evictable'
    assert decode_plan.mode_for('controlnet') == 'evictable'

    controlnet_task = SimpleNamespace(current_tab='ip', input_image_checkbox=True, cn_tasks={'canny': [[1, 2, 3]]})
    controlnet_plan = memory_governor.plan_for_task(task=controlnet_task, phase=memory_governor.MemoryPhase.DECODE)

    assert controlnet_plan.mode_for('controlnet') == 'warm'


def test_stale_inpaint_mix_checkbox_does_not_keep_controlnet_warm_without_live_ip_route(restore_profile):
    low_vram_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    memory_governor.configure_environment(low_vram_profile)

    stale_task = SimpleNamespace(
        current_tab='txt2img',
        input_image_checkbox=True,
        mixing_image_prompt_and_inpaint=True,
        cn_tasks={},
    )
    diffusion_plan = memory_governor.plan_for_task(task=stale_task, phase=memory_governor.MemoryPhase.DIFFUSION)

    assert diffusion_plan.mode_for('controlnet') == 'evictable'

def test_cleanup_memory_ignores_zero_count_support_actions(monkeypatch, restore_profile):
    low_vram_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    memory_governor.configure_environment(low_vram_profile)

    cache_calls = []

    def snapshot(*args, **kwargs):
        return SimpleNamespace(free_ram_mb=8192.0, free_vram_mb=2048.0)

    monkeypatch.setattr(resources, 'capture_memory_snapshot', snapshot)
    monkeypatch.setattr(resources, 'soft_empty_cache', lambda force=False: cache_calls.append(force))
    monkeypatch.setattr(resources, '_try_malloc_trim', lambda: False)
    monkeypatch.setattr(resources.gc, 'collect', lambda: None)

    from backend import controlnet_registry
    from backend.preprocessors import runtime as preprocessor_runtime
    import backend.ip_adapter as ip_adapter
    import backend.pulid_runtime as pulid_runtime

    monkeypatch.setattr(controlnet_registry, 'apply_controlnet_residency', lambda mode: {'mode': mode, 'count': 0})
    monkeypatch.setattr(preprocessor_runtime, 'apply_residency_policy', lambda mode: {'mode': mode, 'count': 0})
    monkeypatch.setattr(
        ip_adapter,
        'apply_contextual_residency',
        lambda mode, clip_vision_action=None, insightface_action=None: {
            'mode': mode,
            'clip_vision_action': clip_vision_action,
            'insightface_action': insightface_action,
            'contextual_models': 0,
            'clip_vision_models': 0,
            'insightface_apps': 0,
        },
    )
    monkeypatch.setattr(pulid_runtime, 'apply_contextual_residency', lambda mode: {'mode': mode, 'eva_clip_models': 0, 'face_parsers': 0})

    txt2img_task = SimpleNamespace(current_tab='txt2img', cn_tasks={})
    resources.cleanup_memory(
        'unit_test_noop_txt2img',
        gc_collect=False,
        target_phase=resources.MemoryPhase.DIFFUSION,
        task=txt2img_task,
    )

    assert cache_calls == [False]

def test_cleanup_memory_dispatches_residency_handlers_by_target_phase(monkeypatch, restore_profile):
    low_vram_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    memory_governor.configure_environment(low_vram_profile)

    calls = []

    def snapshot(*args, **kwargs):
        return SimpleNamespace(free_ram_mb=8192.0, free_vram_mb=2048.0)

    monkeypatch.setattr(resources, 'capture_memory_snapshot', snapshot)
    monkeypatch.setattr(resources, 'soft_empty_cache', lambda force=False: calls.append(('soft_empty_cache', force)))
    monkeypatch.setattr(resources, '_try_malloc_trim', lambda: False)
    monkeypatch.setattr(resources.gc, 'collect', lambda: None)

    from backend import controlnet_registry
    from backend.preprocessors import runtime as preprocessor_runtime
    import backend.ip_adapter as ip_adapter
    import backend.pulid_runtime as pulid_runtime

    monkeypatch.setattr(
        controlnet_registry,
        'apply_controlnet_residency',
        lambda mode: calls.append(('controlnet', mode)) or {'mode': mode},
    )
    monkeypatch.setattr(
        preprocessor_runtime,
        'apply_residency_policy',
        lambda mode: calls.append(('preprocessors', mode)) or {'mode': mode},
    )
    monkeypatch.setattr(
        ip_adapter,
        'apply_contextual_residency',
        lambda mode, clip_vision_action=None, insightface_action=None: calls.append(
            ('contextual', mode, clip_vision_action, insightface_action)
        ) or {'mode': mode, 'clip_vision_action': clip_vision_action, 'insightface_action': insightface_action},
    )
    monkeypatch.setattr(
        pulid_runtime,
        'apply_contextual_residency',
        lambda mode: calls.append(('pulid', mode)) or {'mode': mode},
    )

    resources.cleanup_memory(
        'unit_test_finalize',
        gc_collect=False,
        target_phase=resources.MemoryPhase.FINALIZE,
        notes={'test': True},
    )

    assert ('controlnet', 'destroy') in calls
    assert ('preprocessors', 'destroy') in calls
    assert ('pulid', 'destroy') in calls
    assert ('contextual', 'destroy', 'destroy', 'destroy') in calls
    assert any(name == 'soft_empty_cache' for name, *_ in calls)


def test_affordable_checkpoint_switch_trims_supported_host_after_release(monkeypatch):
    calls = []
    trim_requests = []

    monkeypatch.setattr(
        resources.memory_governor,
        'can_afford',
        lambda **kwargs: SimpleNamespace(allowed=True, reason='headroom_ok'),
    )
    monkeypatch.setattr(
        resources.memory_governor,
        'should_trim_host_memory',
        lambda *, aggressive=False, **kwargs: trim_requests.append(aggressive) or aggressive,
    )
    monkeypatch.setattr(
        resources,
        'cleanup_memory',
        lambda reason, **kwargs: calls.append((reason, kwargs)) or {'done': True},
    )
    monkeypatch.setattr(
        resources.memory_governor.governor.policy,
        'aggressive_checkpoint_switch_reclaim',
        False,
    )

    result = resources.prepare_for_checkpoint_switch(
        current_model='sdxl.safetensors',
        next_model='flux.safetensors',
        release_callback=lambda: calls.append(('release', {})),
    )

    assert result == {'done': True}
    assert trim_requests == [True]
    assert calls[0] == ('release', {})
    assert calls[1][0] == 'checkpoint_switch'
    assert calls[1][1]['trim_host'] is True


def test_aggressive_linux_trim_bypasses_optional_background_trim_policy(monkeypatch):
    monkeypatch.setattr(memory_governor.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(
        memory_governor.governor.policy,
        'linux_malloc_trim_enabled',
        False,
    )

    assert memory_governor.should_trim_host_memory(aggressive=True) is True
    assert memory_governor.should_trim_host_memory(aggressive=False) is False


def test_linux_malloc_trim_uses_process_handle_fallback_truthfully(monkeypatch):
    calls = []

    class FakeTrim:
        argtypes = None
        restype = None

        def __call__(self, pad):
            calls.append(pad)
            return 1

    class FakeLibc:
        malloc_trim = FakeTrim()

    def fake_cdll(name):
        if name is not None:
            raise OSError(name)
        return FakeLibc()

    monkeypatch.setattr(resources.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(resources.ctypes, 'CDLL', fake_cdll)

    assert resources._try_malloc_trim() is True
    assert calls == [0]
