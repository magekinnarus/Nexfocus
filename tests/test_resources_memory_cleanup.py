import os
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.memory_governor import MemoryAffordance
from backend import resources


def test_try_malloc_trim_uses_empty_working_set_on_windows(monkeypatch):
    calls = []

    def fake_get_current_process():
        return 12345

    def fake_empty_working_set(handle):
        calls.append(handle)
        return 1

    class FakeDll:
        def __init__(self, library_name):
            if library_name == 'kernel32':
                self.GetCurrentProcess = fake_get_current_process
            elif library_name == 'psapi':
                self.EmptyWorkingSet = fake_empty_working_set

    monkeypatch.setattr(resources.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(resources.ctypes, 'WinDLL', lambda name, use_last_error=True: FakeDll(name), raising=False)
    monkeypatch.setattr(resources.ctypes, 'get_last_error', lambda: 0, raising=False)

    assert resources._try_malloc_trim() is True
    assert calls == [12345]


def test_cleanup_memory_forces_cache_flush_when_host_trim_is_requested(monkeypatch):
    soft_empty_cache_calls = []

    snapshot = SimpleNamespace(free_ram_mb=1024.0, free_vram_mb=2048.0)

    monkeypatch.setattr(resources, 'capture_memory_snapshot', lambda notes=None: snapshot)
    monkeypatch.setattr(resources, 'current_memory_phase', lambda: 'diffusion')
    monkeypatch.setattr(resources, '_residency_plan_for_phase', lambda **kwargs: SimpleNamespace(notes={}))
    monkeypatch.setattr(resources, '_apply_support_residency', lambda *args, **kwargs: {})
    monkeypatch.setattr(resources.gc, 'collect', lambda: 0)
    monkeypatch.setattr(resources.memory_governor, 'should_trim_host_memory', lambda **kwargs: True)
    monkeypatch.setattr(resources, 'soft_empty_cache', lambda force=False: soft_empty_cache_calls.append(force))
    monkeypatch.setattr(resources, '_try_malloc_trim', lambda: True)

    resources.cleanup_memory('unit_test_cleanup')

    assert soft_empty_cache_calls == [True]


def test_prepare_for_checkpoint_switch_releases_then_cleans(monkeypatch):
    calls = []

    monkeypatch.setattr(
        resources.memory_governor,
        'can_afford',
        lambda **kwargs: MemoryAffordance(
            allowed=False,
            phase='model_refresh',
            required_ram_mb=0.0,
            required_vram_mb=0.0,
            minimum_free_ram_mb=4096.0,
            minimum_free_vram_mb=0.0,
            free_ram_mb=1024.0,
            free_vram_mb=2048.0,
            free_ram_after_mb=1024.0,
            free_vram_after_mb=2048.0,
            reason='ram_after=1024.0MB below floor=4096.0MB',
        ),
    )
    monkeypatch.setattr(resources, 'cleanup_memory', lambda reason, **kwargs: calls.append(('cleanup', reason, kwargs)) or {'done': True})

    original_flag = resources.memory_governor.governor.policy.aggressive_checkpoint_switch_reclaim
    resources.memory_governor.governor.policy.aggressive_checkpoint_switch_reclaim = False
    try:
        result = resources.prepare_for_checkpoint_switch(
            current_model='old.safetensors',
            next_model='new.safetensors',
            release_callback=lambda: calls.append(('release',)),
        )
    finally:
        resources.memory_governor.governor.policy.aggressive_checkpoint_switch_reclaim = original_flag

    assert result == {'done': True}
    assert calls[0] == ('release',)
    assert calls[1][0] == 'cleanup'
    assert calls[1][1] == 'checkpoint_switch'
    assert calls[1][2]['unload_models'] is True
    assert calls[1][2]['force_cache'] is True
    assert calls[1][2]['trim_host'] is True


def test_teardown_runtime_family_releases_flux_runtime_before_cache_flush(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "backend.flux_fill_v3.release_active_flux_resident_spine",
        lambda **kwargs: calls.append(("spine", kwargs)) or True,
    )
    monkeypatch.setattr(
        "backend.flux_fill_v3.release_flux_latent_artifacts",
        lambda: calls.append(("artifacts", None)) or True,
    )
    monkeypatch.setattr(resources, "unload_all_models", lambda: calls.append(("unload", None)))
    monkeypatch.setattr(
        resources,
        "soft_empty_cache",
        lambda force=False: calls.append(("soft_empty_cache", force)),
    )

    resources.teardown_runtime_family("flux_fill", reason="test_teardown")

    assert calls == [
        ("spine", {"reason": "test_teardown"}),
        ("artifacts", None),
        ("unload", None),
        ("soft_empty_cache", True),
    ]


def test_cleanup_models_prunes_garbage_collected_patchers():
    from backend import legacy_governor
    import weakref

    class DummyPatcher:
        def __init__(self):
            self.detached = False
            self.load_device = "cpu"
        def detach(self):
            self.detached = True

    class DummyRealModel:
        pass

    patcher = DummyPatcher()
    real_model = DummyRealModel()

    loaded = legacy_governor.LoadedModel(patcher)
    loaded.real_model = weakref.ref(real_model)

    legacy_governor.current_loaded_models.append(loaded)

    # Initially, it is not dead and should remain in the list
    legacy_governor.cleanup_models()
    assert loaded in legacy_governor.current_loaded_models

    # Simulate garbage collection of the patcher wrapper (LoadedModel.model returns None)
    del patcher
    import gc
    gc.collect()

    # It is now dead/stale, and cleanup_models should prune it even if real_model is still alive
    legacy_governor.cleanup_models()
    assert loaded not in legacy_governor.current_loaded_models


def test_eject_model_removes_from_current_loaded_models():
    from backend import legacy_governor

    class DummyPatcher:
        def __init__(self):
            self.detached = False
            self.load_device = "cpu"
        def detach(self):
            self.detached = True

    patcher = DummyPatcher()
    loaded = legacy_governor.LoadedModel(patcher)
    legacy_governor.current_loaded_models.append(loaded)

    assert loaded in legacy_governor.current_loaded_models

    result = legacy_governor.eject_model(patcher)

    assert result is True
    assert patcher.detached is True
    assert loaded not in legacy_governor.current_loaded_models


def test_cleanup_models_gc_prunes_garbage_collected_patchers():
    from backend import legacy_governor
    import weakref

    class DummyPatcher:
        def __init__(self):
            self.load_device = "cpu"
        def detach(self):
            pass

    class DummyRealModel:
        pass

    patcher = DummyPatcher()
    real_model = DummyRealModel()

    loaded = legacy_governor.LoadedModel(patcher)
    loaded.real_model = weakref.ref(real_model)
    legacy_governor.current_loaded_models.append(loaded)

    del patcher
    import gc
    gc.collect()

    legacy_governor.cleanup_models_gc()

    assert loaded not in legacy_governor.current_loaded_models


def test_eject_model_detach_failure_keeps_live_wrapper_and_prunes_stale_ones(monkeypatch):
    from backend import legacy_governor
    import weakref

    soft_empty_cache_calls = []
    monkeypatch.setattr(legacy_governor, "soft_empty_cache", lambda: soft_empty_cache_calls.append(True))

    class FailingPatcher:
        def __init__(self):
            self.load_device = "cpu"
            self.detach_calls = 0
        def detach(self):
            self.detach_calls += 1
            raise RuntimeError("detach failed")

    class DeadPatcher:
        def __init__(self):
            self.load_device = "cpu"
        def detach(self):
            pass

    class DummyRealModel:
        pass

    failing_patcher = FailingPatcher()
    live_real_model = DummyRealModel()
    live_loaded = legacy_governor.LoadedModel(failing_patcher)
    live_loaded.real_model = weakref.ref(live_real_model)

    dead_patcher = DeadPatcher()
    dead_real_model = DummyRealModel()
    dead_loaded = legacy_governor.LoadedModel(dead_patcher)
    dead_loaded.real_model = weakref.ref(dead_real_model)

    legacy_governor.current_loaded_models.append(live_loaded)
    legacy_governor.current_loaded_models.append(dead_loaded)

    del dead_patcher
    import gc
    gc.collect()

    result = legacy_governor.eject_model(failing_patcher)

    assert result is False
    assert failing_patcher.detach_calls == 1
    assert live_loaded in legacy_governor.current_loaded_models
    assert dead_loaded not in legacy_governor.current_loaded_models
    assert soft_empty_cache_calls == [True]

