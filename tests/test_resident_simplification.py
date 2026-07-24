import sys
from types import SimpleNamespace

# Pre-mock args_manager to avoid argparse conflicts during test execution
fake_args = SimpleNamespace(
    colab=False,
    preset="",
    output_path="",
    temp_path="",
    skip_model_load=True,
    disable_metadata=True,
)
sys.modules["args_manager"] = SimpleNamespace(
    args=fake_args,
    args_parser=SimpleNamespace(args=fake_args, parser=SimpleNamespace()),
)

import time
import traceback
import pytest
import torch

import backend.sdxl_unified_runtime as runtime_mod
import backend.resources as resources
import backend.memory_governor as memory_governor
import modules.pipeline.preprocessing as preprocessing
from backend.staging_manager import ResidencyMode, ExecutionClass, PlacementPlanner
from backend.sdxl_runtime_policy import resolve_sdxl_execution_policy
from tests.test_sdxl_unified_runtime import FakePatcher, FakeClip, FakeVAE


# Tests for consolidated resident policies


def test_sdxl_clip_unconditionally_cpu(monkeypatch):
    # Enforce residency checks across different inference profiles
    from backend import environment_profile as environment_profiles
    from backend.staging_manager import InferenceCostProfile, ResourceLedger, ComponentPlacement

    ledger = ResourceLedger()
    profile = InferenceCostProfile(
        family="sdxl",
        variant="sdxl-base",
        weights_mb=1000.0,
        overhead_mb=100.0,
        vae_mb=200.0,
        clip_mb=500.0,
    )
    
    planner = PlacementPlanner()
    
    # Test for standard tier
    placement = planner._sdxl_clip(
        family="sdxl",
        variant="sdxl-base",
        profile=profile,
        greedy=False,
        ledger=ledger,
        available_gpu_mb=8000.0,
    )
    assert placement.residency_mode == ResidencyMode.CPU_ONLY.value
    assert placement.load_device == "cpu"
    assert placement.compute_device == "cpu"

    # Test for greedy tier (should still be CPU_ONLY under W14c contract)
    placement_greedy = planner._sdxl_clip(
        family="sdxl",
        variant="sdxl-base",
        profile=profile,
        greedy=True,
        ledger=ledger,
        available_gpu_mb=16000.0,
    )
    assert placement_greedy.residency_mode == ResidencyMode.CPU_ONLY.value


def test_colab_free_pinned_full_load():
    from backend import environment_profile as environment_profiles
    from backend.resources import VRAMState
    
    class FakeResidencyPlan:
        def __init__(self, notes):
            self.notes = notes
        def mode_for(self, role):
            return "pinned"

    residency_plan = FakeResidencyPlan(notes={"profile": environment_profiles.PROFILE_COLAB_FREE})
    
    # Test pinned_full_load evaluation for PROFILE_COLAB_FREE.
    # It should evaluate to True since we removed PROFILE_COLAB_FREE from the exclusion list.
    residency_mode = "pinned"
    force_high_vram = False
    vram_set_state = VRAMState.NORMAL_VRAM
    profile_name = environment_profiles.PROFILE_COLAB_FREE
    
    pinned_full_load = (
        residency_mode == 'pinned'
        and not force_high_vram
        and vram_set_state in (VRAMState.NORMAL_VRAM, VRAMState.HIGH_VRAM)
        and profile_name not in (environment_profiles.PROFILE_LOCAL_LOW_VRAM,)
    )
    assert pinned_full_load is True


def test_memory_governor_cache_flushing_pressure():
    from backend.memory_governor import MemoryGovernor, MemorySnapshot, MemoryPolicy
    
    class FakeGovernor(MemoryGovernor):
        def __init__(self, policy):
            self.policy = policy
            self._last_cache_flush = time.time() - 100.0
            self._lock = time # fake lock placeholder
            
        def capture_snapshot(self):
            return self._snapshot
            
    policy = MemoryPolicy(
        low_vram_threshold_mb=8000.0,
        low_vram_cache_cooldown_s=1.0,
        minimum_cache_cooldown_s=0.5,
        low_ram_headroom_mb=2048.0,
    )
    gov = FakeGovernor(policy)
    
    # 1. No memory pressure -> should NOT flush cache
    gov._snapshot = MemorySnapshot(
        timestamp=time.time(),
        phase="diffusion",
        total_vram_mb=8192.0,
        free_vram_mb=4000.0, # plenty of VRAM (ratio: 0.48 > 0.12)
        total_ram_mb=16384.0,
        free_ram_mb=8000.0, # plenty of RAM
    )
    assert gov.should_flush_cache() is False

    # 2. VRAM pressure -> should flush cache
    gov._snapshot = MemorySnapshot(
        timestamp=time.time(),
        phase="diffusion",
        total_vram_mb=8192.0,
        free_vram_mb=500.0, # low VRAM (ratio: 0.06 < 0.12)
        total_ram_mb=16384.0,
        free_ram_mb=8000.0,
    )
    assert gov.should_flush_cache() is True

    # 3. RAM pressure -> should flush cache
    gov._snapshot = MemorySnapshot(
        timestamp=time.time(),
        phase="diffusion",
        total_vram_mb=8192.0,
        free_vram_mb=4000.0,
        total_ram_mb=16384.0,
        free_ram_mb=1000.0, # low RAM (1000MB < low_ram_headroom_mb)
    )
    assert gov.should_flush_cache() is True

    # 4. Unknown VRAM totals without RAM pressure -> should NOT flush cache
    gov._snapshot = MemorySnapshot(
        timestamp=time.time(),
        phase="diffusion",
        total_vram_mb=None,
        free_vram_mb=None,
        total_ram_mb=16384.0,
        free_ram_mb=8000.0,
    )
    assert gov.should_flush_cache() is False


def test_resident_close_does_not_detach(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    
    from backend.sdxl_runtime_policy import SDXLExecutionPolicy
    
    resident_policy = SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
        allow_cpu_shadow=True,
    )
    
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=1024,
        height=1024,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        runtime_policy=resident_policy,
    )
    
    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    runtime.load_components()
    
    unet_ref = runtime.unet
    assert len(unet_ref.detach_calls) == 0
    
    # Close should NOT detach the components under resident mode
    runtime.close()
    assert len(unet_ref.detach_calls) == 0


def test_resident_lora_stack_reuses_same_shell_and_restores_clean_state(monkeypatch):
    compile_calls: list[tuple[str, int]] = []
    runtime_mod.clear_unified_sdxl_runtime_component_cache()

    def fake_load_checkpoint(*args, **kwargs):
        unet = FakePatcher("unet")
        clip = FakeClip()

        def fake_unet_reload(model, device):
            if hasattr(model, "_patched_marker"):
                delattr(model, "_patched_marker")
            model.device = device

        def fake_clip_reload(model, device):
            if hasattr(model, "_patched_marker"):
                delattr(model, "_patched_marker")
            model.device = device

        unet.runtime_reload = fake_unet_reload
        clip.patcher.runtime_reload = fake_clip_reload
        return unet, clip, FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        compile_calls.append((getattr(patcher, "name", "?"), patch_count))
        if patch_count > 0:
            patcher.model._patched_marker = True
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model: {"clip_key": "clip_key"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model: {"unet_key": "unet_key"})
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    import backend.sdxl_resident_runtime as resident_runtime_mod
    monkeypatch.setattr(resident_runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(resident_runtime_mod.GpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(resident_runtime_mod, "SafeOpenHeaderOnly", lambda path: {"path": path})

    from backend.sdxl_runtime_policy import SDXLExecutionPolicy

    resident_policy = SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
        allow_cpu_shadow=True,
    )

    common_kwargs = dict(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=1024,
        height=1024,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        runtime_policy=resident_policy,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(("lora-a.safetensors", 1.0),),
        )
    )
    prepared_a, _ = runtime_a.prepare_inputs()
    assert prepared_a.metrics["compiled_unet_cache_hit"] == 0.0
    assert getattr(runtime_a.unet.model, "_patched_marker", False) is True
    assert getattr(runtime_a.clip.patcher.model, "_patched_marker", False) is False
    runtime_a.close()

    runtime_b = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(("lora-a.safetensors", 1.0),),
        )
    )
    prepared_b, _ = runtime_b.prepare_inputs()
    assert prepared_b.metrics["compiled_unet_cache_hit"] == 1.0
    assert getattr(runtime_b.unet.model, "_patched_marker", False) is True
    assert getattr(runtime_b.clip.patcher.model, "_patched_marker", False) is False
    runtime_b.close()

    runtime_c = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(),
        )
    )
    prepared_c, _ = runtime_c.prepare_inputs()
    assert prepared_c.metrics["compiled_unet_cache_hit"] == 0.0
    assert getattr(runtime_c.unet.model, "_patched_marker", False) is False
    assert getattr(runtime_c.clip.patcher.model, "_patched_marker", False) is False
    runtime_c.close()

    assert sum(1 for name, patch_count in compile_calls if name == "clip" and patch_count > 0) == 1
    assert sum(1 for name, patch_count in compile_calls if name == "unet" and patch_count > 0) == 1
    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_prompt_cache_seed_update(monkeypatch):
    # Mocking necessary methods in preprocessing
    original_tasks = [
        {"task_seed": 100, "task_prompt": "hello"},
        {"task_seed": 101, "task_prompt": "hello"},
    ]
    
    # Mock the cache to return a copy of our original tasks
    monkeypatch.setattr(preprocessing, "_load_prompt_tasks_from_cache", lambda fp: [dict(t) for t in original_tasks])
    monkeypatch.setattr(preprocessing, "_build_prompt_task_fingerprint", lambda *args, **kwargs: SimpleNamespace(digest=lambda: b"fake"))
    
    task_state = SimpleNamespace(
        prompt="hello",
        negative_prompt="",
        image_number=2,
        disable_seed_increment=False,
        use_expansion=False,
        use_style=False,
        style_selections=[],
        seed=555,
        current_progress=0,
        goals=[],
        loras=[],
    )
    
    # Requesting with seed=555
    tasks = preprocessing.process_prompt(task_state, [])
    
    # Assert tasks seeds are updated to the new seed
    assert tasks[0]["task_seed"] == 555
    assert tasks[1]["task_seed"] == 556
    
    # Check that original_tasks was not mutated
    assert original_tasks[0]["task_seed"] == 100


def test_spatial_latent_cache_image_mode(monkeypatch):
    # Verify spatial caching handles None mask and caches the latent representation
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()
        
    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=512,
        height=512,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=torch.zeros((1, 512, 512, 3), dtype=torch.float32),
    )
    
    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    
    # Let's count VAE encode calls
    vae_calls = {"calls": 0}
    runtime.load_components()
    # Replace VAE with CountingVAE
    runtime.vae = CountingVAE(vae_calls)
    
    # First run (cache miss)
    art1, payload1, metrics1 = runtime._prepare_spatial_conditioning_artifacts()
    assert vae_calls["calls"] == 1
    assert metrics1["spatial_cache_hit"] == 0.0
    assert art1.mask_fingerprint is None
    assert art1.denoise_mask_fingerprint is None
    
    # Second run (cache hit)
    art2, payload2, metrics2 = runtime._prepare_spatial_conditioning_artifacts()
    assert vae_calls["calls"] == 1  # Should not increase
    assert metrics2["spatial_cache_hit"] == 1.0
    assert art2.artifact_fingerprint == art1.artifact_fingerprint
    
    runtime.close()


class CountingVAE(FakeVAE):
    def __init__(self, counter: dict[str, int]) -> None:
        super().__init__()
        self._counter = counter

    def encode(self, pixels):
        self._counter["calls"] = int(self._counter.get("calls", 0)) + 1
        return super().encode(pixels)

    def clone(self):
        cloned = CountingVAE(self._counter)
        cloned.patcher = self.patcher.clone()
        return cloned
