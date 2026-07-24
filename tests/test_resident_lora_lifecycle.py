import sys
import os
import time
import copy
from types import SimpleNamespace
from typing import Any

# Pre-mock args_manager
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

import pytest
import torch

from backend import resources, sdxl_runtime_policy
from backend.staging_manager import ExecutionClass
from backend.sdxl_unified_runtime import UnifiedSDXLRuntimeConfig
import backend.sdxl_resident_runtime as resident_mod
from tests.test_sdxl_unified_runtime import FakePatcher, FakeClip, FakeVAE, CountingVAE


@pytest.fixture(autouse=True)
def _clear_cache_and_active_key():
    from backend import process_transition
    process_transition.clear_active_process_key()
    from backend.sdxl_unified_runtime import clear_unified_sdxl_runtime_component_cache
    clear_unified_sdxl_runtime_component_cache()
    yield
    process_transition.clear_active_process_key()
    clear_unified_sdxl_runtime_component_cache()


def test_resident_unet_lifecycle(monkeypatch):
    # Setup compile tracking and load mocks
    gpu_compile_calls = []
    cpu_compile_calls = []
    unet_reload_calls = []
    clip_reload_calls = []

    def fake_load_checkpoint(*args, **kwargs):
        unet = FakePatcher("unet")
        clip = FakeClip()
        vae = FakeVAE()

        def fake_unet_reload(model, device):
            unet_reload_calls.append(device)
            model.device = device

        def fake_clip_reload(model, device):
            clip_reload_calls.append(device)
            model.device = device

        unet.runtime_reload = fake_unet_reload
        clip.patcher.runtime_reload = fake_clip_reload
        return unet, clip, vae

    def fake_load_lora(header, to_load, log_missing=False):
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_gpu_compile(patcher, clean_source, target_device, intermediate_dtype):
        gpu_compile_calls.append((patcher.name, len(patcher.patches)))
        patcher.patches = {}
        return {
            "status": "compiled",
            "patch_count": 1,
            "materialized_patch_keys": 1,
            "host_pinned_bytes": 0,
        }

    def fake_cpu_compile(patcher, *args, **kwargs):
        cpu_compile_calls.append((patcher.name, len(patcher.patches)))
        patcher.patches = {}
        return {
            "status": "compiled",
            "patch_count": 1,
            "materialized_patch_keys": 1,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(resident_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resident_mod.resources, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(resident_mod.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)
    monkeypatch.setattr(resident_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(resident_mod.backend_lora, "model_lora_keys_clip", lambda model: {"clip_key": "clip_key"})
    monkeypatch.setattr(resident_mod.backend_lora, "model_lora_keys_unet", lambda model: {"unet_key": "unet_key"})
    monkeypatch.setattr(resident_mod.GpuArtifactCompiler, "compile_patcher", fake_gpu_compile)
    monkeypatch.setattr(resident_mod.CpuArtifactCompiler, "compile_patcher", fake_cpu_compile)
    monkeypatch.setattr(resident_mod, "SafeOpenHeaderOnly", lambda path: {"path": path})

    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
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
        prompt="a blue bird",
        negative_prompt="bad",
        width=1024,
        height=1024,
        steps=5,
        cfg=5.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        runtime_policy=policy,
    )

    # 1. First apply of LoRA
    runtime_1 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(("lora-1.safetensors", 1.0),),
        )
    )
    prepared_1, _ = runtime_1.prepare_inputs()
    assert prepared_1.metrics["compiled_unet_cache_hit"] == 0.0
    assert len(gpu_compile_calls) == 1
    assert gpu_compile_calls[0] == ("unet", 1)
    assert len(cpu_compile_calls) == 1
    assert cpu_compile_calls[0] == ("clip", 1)
    assert getattr(runtime_1.unet.model, "_nex_clean_unet_source", None) is None
    assert getattr(runtime_1.clip.patcher.model, "_nex_clean_clip_source", None) is None
    assert len(unet_reload_calls) == 0
    assert len(clip_reload_calls) == 1
    runtime_1.close()

    # 2. Warm same-stack reuse
    runtime_2 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(("lora-1.safetensors", 1.0),),
        )
    )
    prepared_2, _ = runtime_2.prepare_inputs()
    assert prepared_2.metrics["compiled_unet_cache_hit"] == 1.0
    # No new compilation calls
    assert len(gpu_compile_calls) == 1
    assert len(cpu_compile_calls) == 1
    assert len(unet_reload_calls) == 0
    assert len(clip_reload_calls) == 1
    runtime_2.close()

    # 3. Stack change
    runtime_3 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(("lora-2.safetensors", 1.0),),
        )
    )
    prepared_3, _ = runtime_3.prepare_inputs()
    assert prepared_3.metrics["compiled_unet_cache_hit"] == 0.0
    assert len(gpu_compile_calls) == 2
    assert gpu_compile_calls[1] == ("unet", 1)
    # CLIP compiled again for the new stack
    assert len(cpu_compile_calls) == 2
    assert cpu_compile_calls[1] == ("clip", 1)
    assert len(unet_reload_calls) == 1
    assert len(clip_reload_calls) == 2
    runtime_3.close()

    # 4. LoRA removal back to empty stack
    runtime_4 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            lora_specs=(),
        )
    )
    prepared_4, _ = runtime_4.prepare_inputs()
    assert prepared_4.metrics["compiled_unet_cache_hit"] == 0.0
    # Empty stack does not compile UNet
    assert len(gpu_compile_calls) == 2
    assert len(cpu_compile_calls) == 2
    assert getattr(runtime_4.unet.model, "_nex_clean_unet_source", None) is None
    assert len(unet_reload_calls) == 2
    assert len(clip_reload_calls) == 2
    runtime_4.close()


def test_vae_seams_and_residency(monkeypatch):
    vae_calls = {"calls": 0}

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), CountingVAE(vae_calls)

    monkeypatch.setattr(resident_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resident_mod.resources, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(resident_mod.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
        vae_encode_mode="gpu_resident",
    )

    config = UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a blue bird",
        negative_prompt="bad",
        width=512,
        height=512,
        steps=5,
        cfg=5.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        runtime_policy=policy,
    )

    runtime = resident_mod.ResidentSDXLRuntime(config)
    runtime.load_components()

    # Test encode_spatial_pixels seam
    pixels = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    latent = runtime.encode_spatial_pixels(pixels)
    assert latent.shape == (1, 4, 64, 64)
    assert vae_calls["calls"] == 1

    # Test decode_spatial_latents seam
    decoded = runtime.decode_spatial_latents(latent)
    assert decoded.shape == (1, 512, 512, 3)

    # Close should clean up memory but preserve model loaders
    runtime.close()


def test_support_model_teardown_discipline(monkeypatch):
    cleanup_calls = []

    def fake_cleanup_memory(reason, *args, **kwargs):
        cleanup_calls.append((reason, kwargs))
        return None

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    monkeypatch.setattr(resident_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resident_mod.resources, "cleanup_memory", fake_cleanup_memory)

    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
    )

    config = UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a bird",
        negative_prompt="bad",
        width=512,
        height=512,
        steps=5,
        cfg=5.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        runtime_policy=policy,
    )

    runtime = resident_mod.ResidentSDXLRuntime(config)
    
    # 1. During prepare_inputs, preprocessor teardown is triggered
    runtime.prepare_inputs()
    assert any(reason == "preprocessors_teardown" for reason, _kwargs in cleanup_calls)
    preflight_kwargs = next(kwargs for reason, kwargs in cleanup_calls if reason == "preprocessors_teardown")
    assert getattr(preflight_kwargs.get("task"), "current_tab", None) == "txt2img"

    # 2. During close, resident request complete teardown is triggered
    runtime.close()
    assert any(reason == "resident_request_complete" for reason, _kwargs in cleanup_calls)
    close_kwargs = next(kwargs for reason, kwargs in cleanup_calls if reason == "resident_request_complete")
    assert getattr(close_kwargs.get("task"), "current_tab", None) == "txt2img"


def test_memory_task_hint_marks_inpaint_controlnet_routes():
    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
    )
    runtime = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="standard_sdxl",
            checkpoint_path="checkpoint.safetensors",
            prompt="a bird",
            negative_prompt="bad",
            width=512,
            height=512,
            steps=5,
            cfg=5.0,
            sampler="euler",
            scheduler="karras",
            seed=123,
            runtime_policy=policy,
            source_pixels=object(),
            source_mask=object(),
            structural_tasks={"canny": ((object(), 0.5, 1.0),)},
            controlnet_paths={"canny": "controlnet.safetensors"},
        )
    )

    hint = runtime._memory_task_hint()

    assert hint.current_tab == "inpaint"
    assert hint.input_image_checkbox is True
    assert hint.mixing_image_prompt_and_inpaint is True
    assert "canny" in hint.cn_tasks


def test_patcher_load_fast_path(monkeypatch):
    load_calls = []

    def fake_load_checkpoint(*args, **kwargs):
        unet = FakePatcher("unet")
        original_load = unet.load
        
        def mock_load(device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
            load_calls.append(device_to)
            return original_load(device_to, lowvram_model_memory, force_patch_weights, full_load)
            
        unet.load = mock_load
        return unet, FakeClip(), FakeVAE()

    monkeypatch.setattr(resident_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resident_mod.resources, "get_torch_device", lambda: torch.device("cuda"))

    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_mode="resident",
    )

    config = UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="standard_sdxl",
        checkpoint_path="checkpoint.safetensors",
        prompt="a bird",
        negative_prompt="bad",
        width=512,
        height=512,
        steps=5,
        cfg=5.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        runtime_policy=policy,
    )

    from backend import patching
    
    # We patch a model and call load twice
    model = torch.nn.Linear(2, 2)
    model.device = torch.device("cuda")
    patcher = patching.NexModelPatcher(model, torch.device("cuda"), torch.device("cuda"))
    
    # First load
    patcher.load(device_to=torch.device("cuda"), full_load=True)
    # Second load (should early return without calling standard load loop)
    # Standard load implementation writes to model.lowvram_patch_counter or log completely, 
    # but with early return, it does not do anything.
    patcher.load(device_to=torch.device("cuda"), full_load=True)


def test_resident_unet_standard_scheduler_changes_reuse_clean_state(monkeypatch):
    unet_reload_calls = []

    def fake_load_checkpoint(*args, **kwargs):
        unet = FakePatcher("unet")
        clip = FakeClip()
        vae = FakeVAE()

        def fake_unet_reload(model, device):
            unet_reload_calls.append(device)
            model.device = device

        unet.runtime_reload = fake_unet_reload
        return unet, clip, vae

    monkeypatch.setattr(resident_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resident_mod.resources, "get_torch_device", lambda: torch.device("cuda"))
    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
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
        prompt="a blue bird",
        negative_prompt="bad",
        width=1024,
        height=1024,
        steps=5,
        cfg=5.0,
        sampler="euler",
        seed=123,
        runtime_policy=policy,
    )

    # 1. Start with no LoRAs and a standard scheduler.
    runtime_1 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            scheduler="karras",
            lora_specs=(),
        )
    )
    runtime_1.prepare_inputs()
    # The newly loaded UNet is clean.
    assert len(unet_reload_calls) == 0
    runtime_1.close()

    # 2. Change to another standard scheduler; the UNet is reused from cache.
    runtime_2 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            scheduler="Beta",
            lora_specs=(),
        )
    )
    runtime_2.prepare_inputs()
    # Ordinary scheduler changes do not mutate the resident UNet.
    assert len(unet_reload_calls) == 0
    runtime_2.close()

    # 3. Another ordinary scheduler remains a warm hit.
    runtime_3 = resident_mod.ResidentSDXLRuntime(
        UnifiedSDXLRuntimeConfig(
            **common_kwargs,
            scheduler="Beta",
            lora_specs=(),
        )
    )
    runtime_3.prepare_inputs()
    assert len(unet_reload_calls) == 0
    runtime_3.close()
