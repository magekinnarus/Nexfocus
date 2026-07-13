from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import torch

from backend.sdxl_assembly.contracts import (
    ResolvedFileIdentity,
    SDXLAssemblyRequest,
    SDXLLoraSpec,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
)
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.runtime_state import (
    acquire_active_sdxl_resident_spine,
    release_active_sdxl_resident_spine,
    clear_all_caches,
    debug_component_cache_report,
)
from backend.sdxl_assembly.resident_unet import ResidentUnetSpine
from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker
from backend.sdxl_assembly.cpu_lora_worker import CpuLoraWorker
from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange

class FakeModel:
    def __init__(self) -> None:
        self.parameters_dict = {"param1": torch.nn.Parameter(torch.empty(2, 2, device="meta"))}
        self.buffers_dict = {}
        self.current_weight_patches_uuid = None
        self.model_loaded_weight_memory = 0
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.device = torch.device("cpu")
        self.model_sampling = SimpleNamespace(
            sigma_max=1.0,
            noise_scaling=lambda sigma, noise, latent, max_denoise=False: noise,
            inverse_noise_scaling=lambda sigma, samples: samples,
        )

    def state_dict(self):
        return {"param1": self.parameters_dict["param1"]}

    def named_modules(self):
        return [("", self)]

    def named_parameters(self, recurse=True):
        return [("param1", self.parameters_dict["param1"])]

    def named_buffers(self, recurse=True):
        return []

    def requires_grad_(self, val):
        pass

    def eval(self):
        pass

class FakePatcher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = FakeModel()
        self.patches = {}
        self.weight_wrapper_patches = {}
        self.backup = {}
        self.object_patches_backup = {}
        self.runtime_release_to_meta = True
        self.runtime_reload = None
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.current_device = torch.device("cpu")
        self.attach_calls = []
        self.detach_calls = []

    def model_size(self):
        return 1024

    def add_patches(self, patches, weight):
        for k, v in patches.items():
            self.patches[k] = [(weight, v, 1.0, None, lambda x: x)]
        return list(patches.keys())

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        device = torch.device(device_to or "cpu")
        self.current_device = device
        self.attach_calls.append((device, lowvram_model_memory))
        return self.model

    def detach(self):
        self.current_device = self.offload_device
        self.detach_calls.append((self.current_device, True))

    def current_loaded_device(self):
        return self.current_device


def _identity(name: str, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=Path(name),
        sha256=sha,
        size_bytes=1,
        modified_ns=1,
    )


def _request(**overrides) -> SDXLAssemblyRequest:
    kwargs = {
        'request_id': 'req_w12a_test',
        'route_id': 'txt2img_assembly',
        'image_index': 0,
        'image_count': 1,
        'checkpoint': _identity('checkpoint.safetensors', 'checkpoint_sha'),
        'vae': _identity('vae.safetensors', 'vae_sha'),
        'model_variant_key': 'sdxl',
        'prompt': 'prompt',
        'negative_prompt': 'negative',
        'positive_texts': ('prompt',),
        'negative_texts': ('negative',),
        'width': 64,
        'height': 64,
        'steps': 3,
        'cfg': 5.0,
        'sampler': 'euler',
        'scheduler': 'karras',
        'seed': 123,
        'device': 'cpu',
        'unet_posture': UNetPostureKind.RESIDENT,
        'lora_posture': LoraPatchPostureKind.RESIDENT,
        'clip_posture': TextEncoderPostureKind.CPU_PINNED,
        'vae_posture': VAEPostureKind.TRANSIENT,
    }
    kwargs.update(overrides)
    return SDXLAssemblyRequest(**kwargs)


@pytest.fixture(autouse=True)
def _cleanup():
    clear_all_caches()
    yield
    clear_all_caches()


def test_resident_spine_cold_load_and_warm_reuse(monkeypatch):
    load_calls = []
    gpu_compile_calls = []

    def fake_stream_load(ckpt_path, **kwargs):
        load_calls.append((ckpt_path, kwargs))
        unet = FakePatcher("unet")
        def reload_func(model, device):
            # Create actual tensor on device
            model.parameters_dict["param1"] = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        unet.runtime_reload = reload_func
        return unet

    def fake_gpu_compile(patcher, **kwargs):
        gpu_compile_calls.append(patcher)
        patcher.patches = {}
        return {"status": "compiled", "patch_count": len(patcher.patches)}

    monkeypatch.setattr("backend.loader._stream_load_sdxl_unet_from_checkpoint", fake_stream_load)
    monkeypatch.setattr("backend.gpu_compiler.GpuArtifactCompiler.compile_patcher", fake_gpu_compile)

    req1 = _request(lora_specs=())
    # 1. Cold Load
    spine1, reused = acquire_active_sdxl_resident_spine(req1)
    assert not reused
    assert len(load_calls) == 1
    assert load_calls[0][1]["load_device"].type == "cpu"  # request device is cpu for testing
    
    # Start the spine to trigger start code
    spine1.start()

    # Verify debug cache report
    report = debug_component_cache_report()
    assert report["active_resident_spine"]
    assert report["clean_shadow_bytes"] == 0.0

    # 2. Warm Reuse (exact key match)
    spine2, reused = acquire_active_sdxl_resident_spine(req1)
    assert reused
    assert spine2 is spine1
    assert len(load_calls) == 1


def test_resident_director_splits_unet_gpu_lora_from_cpu_clip_lora(monkeypatch):
    def fake_stream_load(ckpt_path, **kwargs):
        unet = FakePatcher("unet")
        unet.runtime_reload = lambda model, device: None
        return unet

    monkeypatch.setattr("backend.loader._stream_load_sdxl_unet_from_checkpoint", fake_stream_load)

    spec = SDXLLoraSpec(
        file_identity=_identity("clip_only.safetensors", "clip_hash"),
        unet_weight=0.0,
        clip_weight=1.0,
    )
    req = _request(lora_specs=(spec,), lora_stack_hash="clip_stack")

    assembly = SDXLAssemblyDirector.select_assembly(req)

    assert isinstance(assembly.lora_worker, GpuLoraWorker)
    assert isinstance(assembly.unet_spine.lora_worker, GpuLoraWorker)
    assert isinstance(assembly.text_encode_worker.lora_worker, CpuLoraWorker)


def test_resident_unet_rejects_non_safetensors_checkpoint():
    from backend.sdxl_assembly.runtime_state import acquire_resident_unet_component

    req = _request(checkpoint=_identity("checkpoint.ckpt", "checkpoint_sha"))

    with pytest.raises(RuntimeError, match="requires a safetensors checkpoint"):
        acquire_resident_unet_component(req)


def test_resident_unet_lora_lifecycle(monkeypatch):
    gpu_compile_calls = []
    reload_calls = []
    
    # We will use lora cache from cpu_lora_worker
    from backend.sdxl_assembly.cpu_lora_worker import _PARSED_LORA_CACHE
    _PARSED_LORA_CACHE.clear()

    def fake_stream_load(ckpt_path, **kwargs):
        unet = FakePatcher("unet")
        def reload_func(model, device):
            reload_calls.append(device)
            model.parameters_dict["param1"] = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        unet.runtime_reload = reload_func
        return unet

    def fake_load_lora(header, key_map, log_missing=False):
        return {"param1": torch.ones(2, 2)}

    def fake_gpu_compile(patcher, **kwargs):
        gpu_compile_calls.append(patcher)
        # Clear patches to simulate compiler behavior
        patcher.patches = {}
        return {"status": "compiled", "patch_count": 1}

    monkeypatch.setattr("backend.loader._stream_load_sdxl_unet_from_checkpoint", fake_stream_load)
    monkeypatch.setattr("backend.gpu_compiler.GpuArtifactCompiler.compile_patcher", fake_gpu_compile)
    monkeypatch.setattr("backend.lora.load_lora", fake_load_lora)
    monkeypatch.setattr("backend.lora.model_lora_keys_unet", lambda m: {"param1": "param1"})
    monkeypatch.setattr("backend.sdxl_assembly.gpu_lora_worker.SafeOpenHeaderOnly", lambda path: {"path": path})

    # 1. Direct GPU load with UNet LoRA specs
    spec1 = SDXLLoraSpec(file_identity=_identity("lora1.safetensors", "hash1"), unet_weight=1.0, clip_weight=0.0)
    req1 = _request(lora_specs=(spec1,), lora_stack_hash="stack1")

    spine1, reused = acquire_active_sdxl_resident_spine(req1)
    spine1.start()
    assert not reused
    assert spine1.lora_worker.unet_patch_count == 1
    assert len(gpu_compile_calls) == 1
    assert len(_PARSED_LORA_CACHE) == 0

    # 2. Warm reuse with same stack
    spine2, reused = acquire_active_sdxl_resident_spine(req1)
    spine2.start()
    assert reused
    assert len(gpu_compile_calls) == 1

    # 3. Stack change (UNet LoRA signature changes) -> In-place reload and GPU prepatch
    spec2 = SDXLLoraSpec(file_identity=_identity("lora2.safetensors", "hash2"), unet_weight=0.8, clip_weight=0.0)
    req2 = _request(lora_specs=(spec2,), lora_stack_hash="stack2")

    spine3, reused = acquire_active_sdxl_resident_spine(req2)
    spine3.start()
    # Key mismatch -> acquire triggers in-place reload/compile and returns spine3 (which is spine1)
    assert not reused
    assert spine3 is spine1
    assert len(reload_calls) == 1  # Verify clean weight reload occurred
    assert len(gpu_compile_calls) == 2  # Verify compiled again

    # 4. CLIP-only LoRA change -> neutrality verification
    # Request has different stack hash and CLIP spec, but UNet spec remains identical to spec2
    spec2_clip = SDXLLoraSpec(file_identity=_identity("lora2.safetensors", "hash2"), unet_weight=0.8, clip_weight=1.0)
    req3 = _request(lora_specs=(spec2_clip,), lora_stack_hash="stack3")

    spine4, reused = acquire_active_sdxl_resident_spine(req3)
    spine4.start()
    # UNet signature is identical, so it is reused warm without reloading or recompiling!
    assert reused
    assert spine4 is spine1
    assert len(reload_calls) == 1
    assert len(gpu_compile_calls) == 2


def test_coordination_invalidation_rules(monkeypatch):
    released_spines = []

    def fake_stream_load(ckpt_path, **kwargs):
        unet = FakePatcher("unet")
        unet.runtime_reload = lambda model, device: None
        return unet

    monkeypatch.setattr("backend.loader._stream_load_sdxl_unet_from_checkpoint", fake_stream_load)
    monkeypatch.setattr("backend.sdxl_assembly.runtime_state.release_active_sdxl_resident_spine", lambda reason=None: released_spines.append(reason))

    # Load resident spine
    req = _request(lora_specs=())
    spine, _ = acquire_active_sdxl_resident_spine(req)
    spine.start()

    # Trigger model/prompt release for LORA_STACK_CHANGE
    # W12a handles lora stack change in-place inside acquire, so lifecycle coordinator should NOT release it.
    release_for_changes([LifecycleChange.LORA_STACK_CHANGE], reason="lora_changed")
    assert len(released_spines) == 0

    # Trigger model/prompt release for CHECKPOINT_CHANGE
    # Coordinator MUST release resident spine for checkpoint change!
    release_for_changes([LifecycleChange.CHECKPOINT_CHANGE], reason="checkpoint_changed")
    assert len(released_spines) == 1
    assert released_spines[0] == "checkpoint_changed"


def test_failure_cleanup_path(monkeypatch):
    loaded_unets = []

    def fake_stream_load(ckpt_path, **kwargs):
        unet = FakePatcher("unet")
        unet.runtime_reload = lambda model, device: None
        loaded_unets.append(unet)
        return unet

    def fake_gpu_compile_fail(patcher, **kwargs):
        # Simulate compilation error
        raise RuntimeError("GPU compile ran out of memory")

    monkeypatch.setattr("backend.loader._stream_load_sdxl_unet_from_checkpoint", fake_stream_load)
    monkeypatch.setattr("backend.gpu_compiler.GpuArtifactCompiler.compile_patcher", fake_gpu_compile_fail)
    monkeypatch.setattr("backend.lora.load_lora", lambda *args, **kwargs: {"param1": torch.ones(2, 2)})
    monkeypatch.setattr("backend.lora.model_lora_keys_unet", lambda m: {"param1": "param1"})
    monkeypatch.setattr("backend.sdxl_assembly.gpu_lora_worker.SafeOpenHeaderOnly", lambda path: {"path": path})

    spec = SDXLLoraSpec(file_identity=_identity("lora.safetensors", "hash"), unet_weight=1.0, clip_weight=0.0)
    req = _request(lora_specs=(spec,))

    # Loading the spine should propagate the compile error and clean up
    with pytest.raises(RuntimeError, match="GPU compile ran out of memory"):
        acquire_active_sdxl_resident_spine(req)

    # Verify resident spine state was cleared completely
    report = debug_component_cache_report()
    assert not report["active_resident_spine"]
    assert loaded_unets
    assert loaded_unets[0].detach_calls
