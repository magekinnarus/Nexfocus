from __future__ import annotations

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

import contextlib
import copy

import numpy as np
import pytest
import torch

import backend.sdxl_unified_runtime as runtime_mod
import backend.sdxl_resident_runtime as resident_mod
import backend.sdxl_streaming_runtime as streaming_mod
from backend import patching as backend_patching, sdxl_runtime_policy
from backend.staging_manager import ExecutionClass
from modules import flags


class FakePatcher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = torch.nn.Linear(4, 4, bias=False)
        self.model.to(dtype=torch.float16)
        self.model.device = torch.device("cpu")
        self.model.model_loaded_weight_memory = 0
        self.model.model_lowvram = False
        self.model.lowvram_patch_counter = 0
        self.model.current_weight_patches_uuid = None
        self.model.model_sampling = SimpleNamespace(
            sigma_max=1.0,
            noise_scaling=lambda sigma, noise, latent, max_denoise: noise,
            inverse_noise_scaling=lambda sigma, samples: samples,
        )
        self.patches: dict[str, list[tuple[float, object, float, None, object]]] = {}
        self.runtime_release_to_meta = True
        self.runtime_reload = None
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.current_device = torch.device("cpu")
        self.attach_calls: list[tuple[str, str, int]] = []
        self.detach_calls: list[tuple[str, bool]] = []
        self.partial_unload_calls: list[tuple[str, str, int]] = []
        import uuid
        self.patches_uuid = uuid.uuid4()

    def add_patches(self, patches, weight):
        loaded_keys = []
        patch_index = len(self.patches)
        for key, payload in patches.items():
            patch_key = f"{key}:{patch_index}"
            self.patches.setdefault(patch_key, []).append((weight, payload, 1.0, None, lambda x: x))
            loaded_keys.append(patch_key)
            patch_index += 1
        return loaded_keys

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        _ = load_weights
        _ = force_patch_weights
        device = torch.device(device_to) if device_to is not None else torch.device("cpu")
        self.attach_calls.append((self.name, str(device), int(lowvram_model_memory)))
        self.current_device = device
        self.model.device = device
        self.model.model_loaded_weight_memory = self.model_size() if int(lowvram_model_memory) == 0 else int(lowvram_model_memory)
        return self.model

    def detach(self, unpatch_all=True):
        self.detach_calls.append((self.name, bool(unpatch_all)))
        self.current_device = torch.device("cpu")
        self.model.device = torch.device("cpu")
        self.model.model_loaded_weight_memory = 0

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.model.current_weight_patches_uuid = None
        self.model.model_loaded_weight_memory = 0

    def model_size(self):
        total = 0
        for tensor in self.model.parameters():
            total += tensor.numel() * tensor.element_size()
        return total

    def loaded_size(self):
        return int(getattr(self.model, "model_loaded_weight_memory", 0))

    def current_loaded_device(self):
        return self.current_device

    def model_patches_to(self, device):
        pass

    def model_dtype(self):
        return torch.float16

    def partially_unload(self, device_to, memory_to_free=0):
        device = torch.device(device_to) if device_to is not None else torch.device("cpu")
        self.partial_unload_calls.append((self.name, str(device), int(memory_to_free)))
        self.current_device = device
        self.model.device = device
        self.model.model_loaded_weight_memory = 0
        return int(memory_to_free)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        device = torch.device(device_to) if device_to is not None else torch.device("cpu")
        self.current_device = device
        self.model.device = device
        self.model.model_loaded_weight_memory = self.model_size() if int(extra_memory) == 0 else int(extra_memory)
        return int(extra_memory)

    def clone(self):
        cloned = FakePatcher(self.name)
        cloned.model = self.model
        cloned.model.device = getattr(self.model, "device", torch.device("cpu"))
        cloned.model.model_loaded_weight_memory = getattr(self.model, "model_loaded_weight_memory", 0)
        cloned.model.model_lowvram = getattr(self.model, "model_lowvram", False)
        cloned.model.lowvram_patch_counter = getattr(self.model, "lowvram_patch_counter", 0)
        cloned.model.current_weight_patches_uuid = getattr(self.model, "current_weight_patches_uuid", None)
        cloned.runtime_release_to_meta = self.runtime_release_to_meta
        cloned.runtime_reload = self.runtime_reload
        cloned.load_device = self.load_device
        cloned.offload_device = self.offload_device
        cloned.current_device = getattr(self, "current_device", torch.device("cpu"))
        cloned.patches_uuid = self.patches_uuid
        return cloned

    def isolated_clone(self):
        cloned = self.clone()
        cloned.model = copy.deepcopy(self.model)
        cloned.model.device = getattr(self.model, "device", torch.device("cpu"))
        return cloned


class FakeClip:
    def __init__(self) -> None:
        self.patcher = FakePatcher("clip")
        self.layer_idx = None
        self.encode_calls = 0

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text):
        return text.split()

    def encode_from_tokens_resident(self, tokens, return_pooled=False):
        self.encode_calls += 1
        scale = float(len(tokens) or 1)
        cond = torch.full((1, 2, 4), scale, dtype=torch.float16)
        pooled = torch.full((1, 4), scale + 0.5, dtype=torch.float16)
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens(self, tokens, return_pooled=False):
        return self.encode_from_tokens_resident(tokens, return_pooled=return_pooled)

    def clone(self):
        cloned = FakeClip()
        cloned.patcher = self.patcher.clone()
        cloned.layer_idx = self.layer_idx
        return cloned


class FakeFirstStageModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def parameters(self):
        return iter([torch.zeros(1, dtype=self.dtype, device=self.device)])

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def encode(self, batch):
        B, C, H, W = batch.shape
        latent_h = max(1, H // 8)
        latent_w = max(1, W // 8)
        return torch.zeros((B, 4, latent_h, latent_w), dtype=self.dtype, device=self.device)

    def decode(self, batch):
        B, C, H, W = batch.shape
        return torch.zeros((B, 3, H * 8, W * 8), dtype=self.dtype, device=self.device)



class FakeLatentFormat:
    def process_in(self, tensor):
        return tensor

    def process_out(self, tensor):
        return tensor


class FakeVAE:
    def __init__(self) -> None:
        self.patcher = FakePatcher("vae")
        self.first_stage_model = FakeFirstStageModel()
        self.latent_format = FakeLatentFormat()

    def encode(self, pixels):
        batch, height, width, _ = pixels.shape
        latent_h = max(1, height // 8)
        latent_w = max(1, width // 8)
        base = pixels.mean(dim=-1, keepdim=False).unsqueeze(1).to(dtype=torch.float32)
        latent = torch.nn.functional.interpolate(base, size=(latent_h, latent_w), mode="nearest")
        return {"samples": latent.repeat(1, 4, 1, 1)}

    def clone(self):
        cloned = FakeVAE()
        cloned.patcher = self.patcher.clone()
        cloned.first_stage_model.device = self.first_stage_model.device
        cloned.first_stage_model.dtype = self.first_stage_model.dtype
        return cloned


class CountingVAE(FakeVAE):
    def __init__(self, counter: dict[str, int]) -> None:
        super().__init__()
        self._counter = counter
        
        orig_first_stage_encode = self.first_stage_model.encode
        def counting_first_stage_encode(batch):
            self._counter["calls"] = int(self._counter.get("calls", 0)) + 1
            return orig_first_stage_encode(batch)
        self.first_stage_model.encode = counting_first_stage_encode

    def encode(self, pixels):
        self._counter["calls"] = int(self._counter.get("calls", 0)) + 1
        return super().encode(pixels)

    def clone(self):
        cloned = CountingVAE(self._counter)
        cloned.patcher = self.patcher.clone()
        cloned.first_stage_model.device = self.first_stage_model.device
        cloned.first_stage_model.dtype = self.first_stage_model.dtype
        return cloned


def test_compile_patcher_routes_gpu_backed_models_to_gpu_compiler(monkeypatch):
    runtime = runtime_mod.UnifiedSDXLRuntime.__new__(runtime_mod.UnifiedSDXLRuntime)

    class FakeTensor:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)
            self.dtype = torch.float16

    class FakeModel:
        def parameters(self):
            return [FakeTensor("cuda")]

        def buffers(self):
            return []

    patcher = SimpleNamespace(model=FakeModel())

    with pytest.raises(AssertionError) as excinfo:
        runtime_mod.UnifiedSDXLRuntime._compile_patcher(runtime, patcher, pin_model_host=False)
    assert "only supports CPU-backed models" in str(excinfo.value)


def test_compile_patcher_routes_cpu_backed_models_to_cpu_compiler(monkeypatch):
    runtime = runtime_mod.UnifiedSDXLRuntime.__new__(runtime_mod.UnifiedSDXLRuntime)
    patcher = SimpleNamespace(model=torch.nn.Linear(4, 4, bias=False))
    cpu_calls: list[bool] = []

    def fake_cpu_compile(patcher_arg, *, pin_unet_host):
        _ = patcher_arg
        cpu_calls.append(bool(pin_unet_host))
        return {
            "status": "compiled",
            "materialized_patch_keys": 3,
            "host_pinned_bytes": 128,
        }

    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_cpu_compile))
    monkeypatch.setattr(
        runtime_mod.GpuArtifactCompiler,
        "compile_patcher",
        staticmethod(lambda *args, **kwargs: pytest.fail("GPU compiler should not handle CPU-backed compile")),
    )

    result = runtime_mod.UnifiedSDXLRuntime._compile_patcher(runtime, patcher, pin_model_host=True)

    assert cpu_calls == [True]
    assert result["status"] == "compiled"
    assert result["patch_count"] == 3.0
    assert result["host_pinned_bytes"] == 128.0


def test_unified_runtime_dispatch_initializes_streaming_runtime_once(monkeypatch):
    init_calls: list[str] = []
    original_init = streaming_mod.SDXLStreamingRuntime.__init__

    def tracking_init(self, config):
        init_calls.append(type(self).__name__)
        original_init(self, config)

    monkeypatch.setattr(streaming_mod.SDXLStreamingRuntime, "__init__", tracking_init)

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class=ExecutionClass.SDXL_STREAMING_T1,
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
            clip_layer=-2,
            batch_size=1,
        )
    )

    assert type(runtime) is streaming_mod.SDXLStreamingRuntime
    assert init_calls == ["SDXLStreamingRuntime"]


def test_unified_runtime_dispatch_initializes_resident_runtime_once(monkeypatch):
    init_calls: list[str] = []
    original_init = resident_mod.ResidentSDXLRuntime.__init__

    def tracking_init(self, config):
        init_calls.append(type(self).__name__)
        original_init(self, config)

    monkeypatch.setattr(resident_mod.ResidentSDXLRuntime, "__init__", tracking_init)

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class=ExecutionClass.SDXL_RESIDENT_T2,
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
            clip_layer=-2,
            batch_size=1,
        )
    )

    assert type(runtime) is resident_mod.ResidentSDXLRuntime
    assert init_calls == ["ResidentSDXLRuntime"]


def test_load_components_reuses_shared_base_shell_across_runtime_instances(monkeypatch):
    load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_checkpoint(*args, **kwargs):
        load_calls.append((args, kwargs))
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        clip_layer=-2,
        batch_size=1,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(config)
    cold_wall = runtime_a.load_components()
    unet_a = runtime_a.unet
    clip_a = runtime_a.clip
    vae_a = runtime_a.vae
    runtime_a_cache_hit = runtime_a._base_component_cache_hit
    runtime_a.close()

    runtime_b = runtime_mod.UnifiedSDXLRuntime(config)
    warm_wall = runtime_b.load_components()
    unet_b = runtime_b.unet
    clip_b = runtime_b.clip
    vae_b = runtime_b.vae
    runtime_b_cache_hit = runtime_b._base_component_cache_hit
    runtime_b.close()

    assert len(load_calls) == 1
    assert cold_wall >= 0.0
    assert warm_wall == 0.0
    assert runtime_a_cache_hit is False
    assert runtime_b_cache_hit is True
    assert unet_a is not unet_b
    assert clip_a is not clip_b
    assert vae_a is not vae_b


def test_load_components_reuses_shared_default_vae_across_checkpoint_switches(monkeypatch):
    load_calls: list[dict[str, object]] = []

    def fake_load_checkpoint(*args, **kwargs):
        _ = args
        vae_source = kwargs.get("vae_source")
        returned_vae = FakeVAE()
        load_calls.append({"vae_source": vae_source, "returned_vae": returned_vae})
        return FakePatcher("unet"), FakeClip(), returned_vae

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)

    config_a = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint-a.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=1024,
        height=1024,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        vae_path=flags.default_vae,
    )
    config_b = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint-b.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=1024,
        height=1024,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=123,
        vae_path=flags.default_vae,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(config_a)
    runtime_b = runtime_mod.UnifiedSDXLRuntime(config_b)

    runtime_a.load_components()
    runtime_b.load_components()

    assert len(load_calls) == 2
    assert load_calls[0]["vae_source"] is not None
    assert load_calls[1]["vae_source"] is load_calls[0]["vae_source"]

    runtime_a.close()
    runtime_b.close()


def test_denoise_consumes_prepared_artifacts_and_transitions_execution_state(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    decode_calls: list[tuple[str, tuple[int, ...], bool]] = []
    sampler_calls: list[dict[str, object]] = []

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = sampler
            _ = scheduler
            _ = denoise
            _ = model_options
            self.sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    def fake_prepare_direct_conds(self, *, execution_unet, noise, positive, negative, latent_image, denoise_mask, device):
        _ = self
        _ = execution_unet
        _ = noise
        _ = latent_image
        _ = denoise_mask
        _ = device
        return {"positive": positive, "negative": negative}, 0.125

    def fake_calc_fullframe_cond_batch(self, execution_unet, conds, x_in, timestep):
        _ = self
        _ = execution_unet
        _ = timestep
        positive_cond = conds[0][0][0]
        negative_cond = conds[1][0][0]
        cond_pred = x_in + positive_cond.mean().to(x_in.dtype)
        uncond_pred = x_in + negative_cond.mean().to(x_in.dtype)
        return [cond_pred, uncond_pred]

    def fake_sample_euler(model_fn, noise, sigmas, extra_args, callback, disable):
        _ = extra_args
        _ = callback
        sampler_calls.append(
            {
                "disable": disable,
                "sigmas": tuple(float(value) for value in sigmas.detach().cpu().tolist()),
                "output_mean": float(model_fn(noise, sigmas[:1]).mean().item()),
            }
        )
        return model_fn(noise, sigmas[:1])

    def fake_decode_preloaded_vae(vae, latent, tiled=False, tile_size=64):
        _ = tile_size
        decode_calls.append((getattr(vae.patcher, "name", "?"), tuple(int(dim) for dim in latent.shape), bool(tiled)))
        return latent.detach().cpu().movedim(1, -1)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model, key_map=None: {"clip_target": "clip.weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model, key_map=None: {"unet_target": "unet.weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(runtime_mod.sampling, "KSampler", FakeKSampler)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_prepare_direct_conds", fake_prepare_direct_conds)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_calc_fullframe_cond_batch", fake_calc_fullframe_cond_batch)
    monkeypatch.setattr(runtime_mod.k_diffusion, "sample_euler", fake_sample_euler)
    monkeypatch.setattr(
        runtime_mod.sampling,
        "sample_prepared_sdxl",
        lambda *args, **kwargs: pytest.fail("generic prepared sampler path should not run"),
    )
    monkeypatch.setattr(runtime_mod.precision, "autocast_context", lambda device, enabled=True: contextlib.nullcontext())
    monkeypatch.setattr(runtime_mod.decode, "decode_preloaded_vae", fake_decode_preloaded_vae)

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        clip_layer=-2,
        batch_size=1,
        lora_specs=(("lora-a.safetensors", 1.0),),
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()
    clip_encode_calls = runtime.clip.encode_calls

    denoise_result = runtime.denoise_prepared_inputs(prepared)

    assert runtime.clip.encode_calls == clip_encode_calls
    assert prepared.gpu_attached_execution_state is not None
    assert prepared.gpu_attached_execution_state.active_phase == "diffusion"
    assert denoise_result.execution_state is prepared.gpu_attached_execution_state
    assert denoise_result.execution_state.active_phase == "diffusion"
    assert any(component.startswith("compiled_unet:") for component in denoise_result.execution_state.attached_component_ids)
    assert any(component.startswith("conditioning:") for component in denoise_result.execution_state.attached_component_ids)
    assert runtime.unet.attach_calls[0][0] == "unet"
    assert runtime.unet.attach_calls[0][1] == "cpu"
    assert denoise_result.samples.device.type == "cpu"
    assert denoise_result.metrics["prepared_conditioning_reused"] == 1.0
    assert denoise_result.metrics["prepared_unet_reused"] == 1.0
    assert denoise_result.metrics["cond_prepare_explicit"] == pytest.approx(0.125)
    assert denoise_result.metrics["denoise_wall"] >= 0.0
    assert sampler_calls[0]["disable"] is True
    assert sampler_calls[0]["sigmas"] == (1.0, 0.0)
    assert torch.count_nonzero(denoise_result.samples) > 0

    resident_unet = runtime.unet
    images, vae_attach, vae_decode = runtime.decode_latent(denoise_result.samples)

    assert resident_unet.detach_calls == []
    assert runtime.vae.patcher.attach_calls[0][0] == "vae"
    assert decode_calls == [("vae", tuple(int(dim) for dim in denoise_result.samples.shape), False)]
    assert images.shape[-1] == 4
    assert vae_attach >= 0.0
    assert vae_decode >= 0.0
    assert runtime.execution_state is not None
    assert runtime.execution_state.active_phase == "finalize"

    runtime.close()
    assert resident_unet.detach_calls[0][0] == "unet"
    assert runtime.execution_state is None


def test_load_components_keeps_streaming_unet_cpu_authoritative(monkeypatch):
    captured = {}

    def fake_load_checkpoint(*args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    streaming_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_STREAMING_T1,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
    )

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)

    runtime = streaming_mod.SDXLStreamingRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
            runtime_policy=streaming_policy,
        )
    )

    runtime.load_components()

    assert captured["kwargs"]["load_device"] == torch.device("cpu")
    assert captured["kwargs"]["offload_device"] == torch.device("cpu")
    assert runtime.unet.runtime_release_to_meta is False

    runtime.close()


def test_load_components_uses_gpu_unet_and_cpu_shared_vae_when_policy_execution_class_is_resident(monkeypatch):
    captured = {}

    def fake_load_checkpoint(*args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    resident_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
    )

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cuda"))

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
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
    )

    runtime.load_components()

    assert runtime.policy is resident_policy
    assert captured["kwargs"]["load_device"] == torch.device("cuda")
    assert captured["kwargs"]["offload_device"] == torch.device("cuda")
    assert captured["kwargs"]["clip_load_device"] == torch.device("cpu")
    assert captured["kwargs"]["clip_offload_device"] == torch.device("cpu")
    assert captured["kwargs"]["vae_load_device"] == torch.device("cpu")
    assert captured["kwargs"]["vae_offload_device"] == torch.device("cpu")
    assert runtime.unet.runtime_release_to_meta is True


def test_clear_unified_component_cache_also_clears_streaming_cache(monkeypatch):
    clear_calls = []

    monkeypatch.setattr(streaming_mod, "clear_streaming_cache", lambda: clear_calls.append(True))

    runtime_mod.clear_unified_sdxl_runtime_component_cache()

    assert clear_calls == [True]


def test_clear_unified_component_cache_preserves_shared_vae_until_explicit_teardown(monkeypatch):
    clear_calls = []

    monkeypatch.setattr(streaming_mod, "clear_streaming_cache", lambda: clear_calls.append(True))
    monkeypatch.setattr(runtime_mod.resources, "soft_empty_cache", lambda force=False: None)

    runtime_mod._SHARED_SDXL_VAE_CACHE.clear()
    cache_key = ("D:/resolved/sdxl_vae.safetensors", "cpu", "cpu")
    shared_vae = FakeVAE()
    runtime_mod._SHARED_SDXL_VAE_CACHE[cache_key] = shared_vae

    runtime_mod.clear_unified_sdxl_runtime_component_cache(teardown=False)

    assert clear_calls == [True]
    assert runtime_mod._SHARED_SDXL_VAE_CACHE == {cache_key: shared_vae}
    assert shared_vae.patcher.detach_calls == []

    runtime_mod.clear_unified_sdxl_runtime_component_cache(teardown=True)

    assert runtime_mod._SHARED_SDXL_VAE_CACHE == {}
    assert shared_vae.patcher.detach_calls == [("vae", True)]


def test_decode_latent_detaches_shared_vae_even_for_resident_policy(monkeypatch):
    resident_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
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
    )
    runtime.unet = FakePatcher("unet")
    runtime.vae = FakeVAE()
    runtime.policy = resident_policy
    runtime._checkpoint_fingerprint = "checkpoint-fingerprint"

    monkeypatch.setattr(runtime, "load_components", lambda: 0.0)
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)
    monkeypatch.setattr(
        runtime_mod.decode,
        "decode_preloaded_vae",
        lambda vae, latent, tiled=False, tile_size=64: latent.detach().cpu().movedim(1, -1),
    )
    cache_flush_calls = []
    monkeypatch.setattr(runtime_mod.resources, "soft_empty_cache", lambda force=False: cache_flush_calls.append(force))

    assert runtime._is_vae_resident() is False

    images, _, _ = runtime.decode_latent(torch.zeros((1, 4, 8, 8), dtype=torch.float32))

    assert images.shape == (1, 8, 8, 4)
    assert runtime.vae.patcher.detach_calls == [("vae", True)]
    assert cache_flush_calls == [True]


@pytest.mark.parametrize("lora_count", [0, 1, 3])
def test_prepare_inputs_builds_cpu_artifacts_and_placeholders(monkeypatch, lora_count):
    compile_calls: list[tuple[str, int, bool]] = []

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        compile_calls.append(
            (
                getattr(patcher, "name", "?"),
                patch_count,
                bool(kwargs.get("pin_unet_host", True)),
            )
        )
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model, key_map=None: {"clip_target": "clip.weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model, key_map=None: {"unet_target": "unet.weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        clip_layer=-2,
        batch_size=1,
        lora_specs=tuple((f"lora-{index}.safetensors", 1.0) for index in range(lora_count)),
    )

    runtime = streaming_mod.SDXLStreamingRuntime(config)
    prepared, metrics = runtime.prepare_inputs()

    assert prepared.base_model is runtime.base_model
    assert prepared.compiled_unet is runtime.compiled_unet
    assert prepared.conditioning is runtime.conditioning
    assert prepared.spatial_conditioning is None
    assert prepared.gpu_attached_execution_state is runtime.execution_state
    assert prepared.gpu_attached_execution_state is not None
    assert prepared.gpu_attached_execution_state.active_phase == "prepare_inputs"
    assert "feature_boundary_placeholder" in prepared.injected_features
    assert prepared.injected_features["feature_boundary_placeholder"].feature_fingerprint is None
    assert prepared.payload["encoded_prompt_pair"]["positive"]["cond"].device.type == "cpu"
    assert prepared.payload["adm_pair"]["positive"].dtype == torch.float16
    assert prepared.conditioning.clip_layer_idx == config.clip_layer
    assert prepared.compiled_unet.execution_class == "cpu-first"
    assert prepared.compiled_unet.gpu_mb == 0.0
    assert prepared.compiled_unet.source_path == config.checkpoint_path
    assert prepared.compiled_unet.artifact_fingerprint
    assert prepared.conditioning.prompt_fingerprint
    assert metrics["lora_spec_count"] == float(lora_count)
    assert metrics["unet_patch_count"] == float(lora_count)
    assert metrics["conditioning_artifact_count"] == 1.0
    assert metrics["spatial_artifact_count"] == 0.0
    assert runtime.base_model.loaded is True

    if lora_count == 0:
        assert compile_calls == [("unet", 0, False)]
    else:
        assert compile_calls[0] == ("clip", lora_count, False)
        assert compile_calls[1] == ("unet", lora_count, False)

    runtime.close()
    assert runtime.prepared_inputs is None
    assert runtime.base_model is None
    assert runtime.execution_state is None


def test_patched_weights_for_block_returns_compiled_execution_surface(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
        )
    )
    prepared, _ = runtime.prepare_inputs()

    block_surface = runtime.patched_weights_for_block("attn2")

    assert prepared.compiled_unet is not None
    assert block_surface["block_id"] == "attn2"
    assert block_surface["artifact_fingerprint"] == prepared.compiled_unet.artifact_fingerprint
    assert block_surface["execution_unet"] is runtime.unet
    assert block_surface["attached"] is False

    runtime.close()


def test_patched_weights_for_block_prefers_attached_execution_unet(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
        )
    )
    runtime.prepare_inputs()
    attached_unet = object()
    runtime._attached_payload = {"execution_unet": attached_unet}

    block_surface = runtime.patched_weights_for_block("attn2")

    assert block_surface["execution_unet"] is attached_unet
    assert block_surface["attached"] is True

    runtime.close()


def test_prepare_inputs_can_pin_base_unet_without_lora_when_requested(monkeypatch):
    compile_calls: list[tuple[str, int, bool]] = []

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        compile_calls.append(
            (
                getattr(patcher, "name", "?"),
                patch_count,
                bool(kwargs.get("pin_unet_host", True)),
            )
        )
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        pin_base_unet_without_lora=True,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    runtime.prepare_inputs()

    assert compile_calls == [("unet", 0, True)]

    runtime.close()


def test_prepare_inputs_reuses_streaming_compiled_unet_cache_for_same_lora_stack(monkeypatch):
    compile_calls: list[tuple[str, int, bool]] = []

    def fake_load_checkpoint(*args, **kwargs):
        patcher = FakePatcher("unet")
        patcher.runtime_reload = lambda model, device: setattr(model, "device", device)
        return patcher, FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        compile_calls.append(
            (
                getattr(patcher, "name", "?"),
                patch_count,
                bool(kwargs.get("pin_unet_host", True)),
            )
        )
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    streaming_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_STREAMING_T1,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
    )

    runtime_mod.clear_unified_sdxl_runtime_component_cache()
    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model: {"fake_unet": "weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model: {"fake_clip": "weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: {"path": path})
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        lora_specs=(("unit-test-lora.safetensors", 0.75),),
        runtime_policy=streaming_policy,
    )

    runtime_a = streaming_mod.SDXLStreamingRuntime(config)
    prepared_a, _ = runtime_a.prepare_inputs()
    runtime_a.close()

    runtime_b = streaming_mod.SDXLStreamingRuntime(config)
    prepared_b, _ = runtime_b.prepare_inputs()
    runtime_b.close()

    assert prepared_a.metrics["compiled_unet_cache_hit"] == 0.0
    assert prepared_b.metrics["compiled_unet_cache_hit"] == 1.0
    assert sum(1 for name, _, _ in compile_calls if name == "unet") == 1
    assert sum(1 for name, _, _ in compile_calls if name == "clip") == 2
    assert ("unet", 1, False) in compile_calls

    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_streaming_lora_unet_parks_without_unpatching(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        patcher = FakePatcher("unet")
        patcher.runtime_reload = lambda model, device: setattr(model, "device", device)
        return patcher, FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        if getattr(patcher, "name", None) == "unet" and patch_count > 0:
            patcher.model._patched_marker = True
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    streaming_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_STREAMING_T1,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
    )

    runtime_mod.clear_unified_sdxl_runtime_component_cache()
    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model: {"fake_unet": "weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model: {"fake_clip": "weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: {"path": path})
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    runtime = streaming_mod.SDXLStreamingRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
            lora_specs=(("unit-test-lora.safetensors", 0.75),),
            runtime_policy=streaming_policy,
        )
    )

    runtime.prepare_inputs()
    runtime._attach_compiled_unet(torch.device("cuda"), budget_bytes=runtime._clean_unet_budget_bytes(torch.device("cuda")))
    runtime._park_compiled_unet_before_decode()

    assert runtime.unet.partial_unload_calls
    assert runtime.unet.detach_calls == []
    assert runtime._current_streaming_unet_signature() is not None
    assert getattr(runtime.unet.model, "_patched_marker", False) is True
    assert runtime.unet.current_loaded_device() == torch.device("cpu")

    runtime.close()
    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_streaming_compiled_unet_cache_preserves_clean_base_shell(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        patcher = FakePatcher("unet")

        def fake_runtime_reload(model, device):
            if hasattr(model, "_patched_marker"):
                delattr(model, "_patched_marker")
            model.device = device

        patcher.runtime_reload = fake_runtime_reload
        return patcher, FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        if getattr(patcher, "name", None) == "unet" and patch_count > 0:
            patcher.model._patched_marker = True
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    streaming_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_STREAMING_T1,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
    )

    runtime_mod.clear_unified_sdxl_runtime_component_cache()
    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model: {"fake_unet": "weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model: {"fake_clip": "weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: {"path": path})
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    lora_runtime = streaming_mod.SDXLStreamingRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
            lora_specs=(("unit-test-lora.safetensors", 0.75),),
            runtime_policy=streaming_policy,
        )
    )
    lora_runtime.prepare_inputs()
    lora_runtime.close()

    clean_runtime = streaming_mod.SDXLStreamingRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant="sdxl-base",
            execution_class="cpu-first",
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
            runtime_policy=streaming_policy,
        )
    )
    clean_runtime.prepare_inputs()

    assert getattr(clean_runtime.unet.model, "_patched_marker", False) is False

    clean_runtime.close()
    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_prepare_inputs_builds_spatial_conditioning_artifacts(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_pixels[:, 16:48, 16:48, :] = 1.0
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 20:44, 24:40] = 1.0

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, metrics = runtime.prepare_inputs()

    assert prepared.spatial_conditioning is runtime.spatial_conditioning
    assert prepared.spatial_conditioning is not None
    assert prepared.spatial_conditioning.spatial_mode == "inpaint"
    assert prepared.spatial_conditioning.artifact_fingerprint
    assert prepared.spatial_conditioning.mask_fingerprint
    assert prepared.spatial_conditioning.source_latent_fingerprint
    assert prepared.spatial_conditioning.masked_latent_fingerprint is None
    assert prepared.spatial_conditioning.bb_latent_fingerprint
    assert prepared.spatial_conditioning.denoise_mask_fingerprint
    assert prepared.spatial_conditioning.bb_latent_fingerprint == prepared.spatial_conditioning.source_latent_fingerprint
    assert prepared.payload["bbox"] == prepared.spatial_conditioning.bbox
    assert prepared.payload["bb_pixels"].shape[1] == prepared.spatial_conditioning.target_height
    assert prepared.payload["bb_pixels"].shape[2] == prepared.spatial_conditioning.target_width
    assert prepared.spatial_conditioning.mask_coverage > 0.0
    assert prepared.spatial_conditioning.bbox_area_ratio > 0.0

    assert prepared.payload["source_latent"].shape == prepared.payload["bb_latent"].shape
    assert prepared.payload["bb_denoise_mask"].shape[0] == 1
    assert prepared.payload["bb_denoise_mask"].shape[1] == 1
    assert prepared.payload["bb_denoise_mask"].shape[-2:] == prepared.payload["bb_latent"].shape[-2:]
    assert prepared.payload["bb_pixels"].shape[0] == 1
    assert prepared.payload["bb_mask"].shape[0] == 1
    assert metrics["spatial_artifact_count"] == 1.0
    assert metrics["bb_vae_encode_cpu"] >= 0.0
    assert metrics["inpaint_prepare_cpu"] >= 0.0
    assert metrics["spatial_mask_coverage"] == pytest.approx(prepared.spatial_conditioning.mask_coverage)

    runtime.close()


def test_prepare_inputs_uses_resolved_inpaint_context(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 4:20, 8:24] = 1.0
    resolved_context = SimpleNamespace(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        original_mask=np.zeros((64, 64), dtype=np.uint8),
        bb=(4, 20, 8, 24),
        bb_image=np.full((32, 48, 3), 17, dtype=np.uint8),
        bb_mask=np.pad(np.full((16, 24), 255, dtype=np.uint8), ((8, 8), (12, 12))),
        target_w=48,
        target_h=32,
        blend_mask=np.full((64, 64), 255, dtype=np.uint8),
    )

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
        resolved_spatial_context=resolved_context,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()

    assert prepared.spatial_conditioning is not None
    assert prepared.spatial_conditioning.bbox == (4, 20, 8, 24)
    assert prepared.spatial_conditioning.target_width == 48
    assert prepared.spatial_conditioning.target_height == 32
    assert prepared.payload["bb_pixels"].shape == (1, 32, 48, 3)
    assert prepared.payload["bb_mask"].shape == (1, 32, 48)
    assert prepared.payload["bb_latent"].shape == (1, 4, 4, 6)
    assert prepared.payload["bb_denoise_mask"].shape == (1, 1, 4, 6)
    assert float(prepared.payload["bb_pixels"].mean().item()) == pytest.approx(17.0 / 255.0)

    runtime.close()


def test_prepare_inputs_reuses_cached_inpaint_latent_artifacts(monkeypatch):
    runtime_mod.clear_unified_sdxl_runtime_component_cache()
    encode_counter = {"calls": 0}

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), CountingVAE(encode_counter)

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 4:20, 8:24] = 1.0
    resolved_context = SimpleNamespace(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        original_mask=np.zeros((64, 64), dtype=np.uint8),
        bb=(4, 20, 8, 24),
        bb_image=np.full((32, 48, 3), 17, dtype=np.uint8),
        bb_mask=np.pad(np.full((16, 24), 255, dtype=np.uint8), ((8, 8), (12, 12))),
        target_w=48,
        target_h=32,
        blend_mask=np.full((64, 64), 255, dtype=np.uint8),
    )
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
        resolved_spatial_context=resolved_context,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(config)
    prepared_a, metrics_a = runtime_a.prepare_inputs()
    runtime_a.close()

    runtime_b = runtime_mod.UnifiedSDXLRuntime(config)
    prepared_b, metrics_b = runtime_b.prepare_inputs()
    runtime_b.close()

    assert encode_counter["calls"] == 1
    assert metrics_a["spatial_cache_hit"] == 0.0
    assert metrics_b["spatial_cache_hit"] == 1.0
    assert torch.equal(prepared_a.payload["bb_latent"], prepared_b.payload["bb_latent"])
    assert torch.equal(prepared_a.payload["bb_denoise_mask"], prepared_b.payload["bb_denoise_mask"])
    assert prepared_a.spatial_conditioning.artifact_fingerprint == prepared_b.spatial_conditioning.artifact_fingerprint

    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_prepare_inputs_invalidates_spatial_cache_when_vae_changes(monkeypatch):
    runtime_mod.clear_unified_sdxl_runtime_component_cache()
    encode_counter = {"calls": 0}

    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), CountingVAE(encode_counter)

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.loader, "load_vae", lambda *args, **kwargs: CountingVAE(encode_counter))
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 4:20, 8:24] = 1.0
    resolved_context = SimpleNamespace(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        original_mask=np.zeros((64, 64), dtype=np.uint8),
        bb=(4, 20, 8, 24),
        bb_image=np.full((32, 48, 3), 17, dtype=np.uint8),
        bb_mask=np.pad(np.full((16, 24), 255, dtype=np.uint8), ((8, 8), (12, 12))),
        target_w=48,
        target_h=32,
        blend_mask=np.full((64, 64), 255, dtype=np.uint8),
    )
    common_kwargs = dict(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
        resolved_spatial_context=resolved_context,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(vae_path="vae_a.safetensors", **common_kwargs)
    )
    _, metrics_a = runtime_a.prepare_inputs()
    runtime_a.close()

    runtime_b = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(vae_path="vae_b.safetensors", **common_kwargs)
    )
    _, metrics_b = runtime_b.prepare_inputs()
    runtime_b.close()

    assert encode_counter["calls"] == 2
    assert metrics_a["spatial_cache_hit"] == 0.0
    assert metrics_b["spatial_cache_hit"] == 0.0

    runtime_mod.clear_unified_sdxl_runtime_component_cache()


def test_streamlike_budget_is_explicit_and_gpu_only():
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
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
        streamlike_budget_mb=384,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)

    assert runtime._clean_unet_budget_bytes(torch.device("cpu")) == 0
    assert runtime._clean_unet_budget_bytes(torch.device("cuda")) == 384 * 1024 * 1024


def test_spatial_artifact_fingerprint_is_prompt_independent(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    source_pixels = torch.ones((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 8:24, 8:32] = 1.0

    common_kwargs = dict(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(prompt="a red fox", **common_kwargs)
    )
    runtime_b = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(prompt="a blue whale", **common_kwargs)
    )

    prepared_a, _ = runtime_a.prepare_inputs()
    prepared_b, _ = runtime_b.prepare_inputs()

    assert prepared_a.spatial_conditioning is not None
    assert prepared_b.spatial_conditioning is not None
    assert prepared_a.spatial_conditioning.artifact_fingerprint == prepared_b.spatial_conditioning.artifact_fingerprint
    assert prepared_a.conditioning is not None
    assert prepared_b.conditioning is not None
    assert prepared_a.conditioning.prompt_fingerprint != prepared_b.conditioning.prompt_fingerprint

    runtime_a.close()
    runtime_b.close()


def test_prepare_inputs_builds_outpaint_spatial_conditioning_artifacts(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    def fake_build_outpaint_context(self, pixels, mask_2d):
        _ = self
        _ = mask_2d
        original = (pixels.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())
        working = original.copy()
        working[:8, :, :] = 32
        working_mask = torch.zeros((64, 64), dtype=torch.uint8)
        working_mask[:12, :] = 255
        bb_mask = torch.zeros((1, 32, 48), dtype=torch.float32)
        bb_mask[:, :12, :] = 1.0
        return {
            "working_pixels": working,
            "working_mask": working_mask.numpy(),
            "bb_pixels": working[:32, :48, :],
            "bb_mask": (bb_mask[0].numpy() * 255.0).astype("uint8"),
            "blend_mask": (torch.ones((64, 64), dtype=torch.uint8).numpy() * 255),
            "bbox": (0, 32, 0, 48),
            "target_width": 48,
            "target_height": 32,
            "direction": "top",
            "prepare_wall": 0.05,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_build_outpaint_context", fake_build_outpaint_context)

    source_pixels = torch.ones((1, 64, 64, 3), dtype=torch.float32)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        spatial_mode="outpaint",
        outpaint_direction="top",
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, metrics = runtime.prepare_inputs()

    assert prepared.spatial_conditioning is not None
    assert prepared.spatial_conditioning.spatial_mode == "outpaint"
    assert prepared.spatial_conditioning.artifact_fingerprint
    assert prepared.spatial_conditioning.source_latent_fingerprint
    assert prepared.spatial_conditioning.bb_latent_fingerprint == prepared.spatial_conditioning.source_latent_fingerprint
    assert prepared.spatial_conditioning.masked_latent_fingerprint is None
    assert prepared.spatial_conditioning.bbox == (0, 32, 0, 48)
    assert prepared.payload["source_latent"].shape == (1, 4, 4, 6)
    assert prepared.payload["bb_latent"].shape == (1, 4, 4, 6)
    assert prepared.payload["bb_denoise_mask"].shape == (1, 1, 4, 6)
    assert prepared.payload["bb_pixels"].shape == (1, 32, 48, 3)
    assert prepared.payload["outpaint_working_pixels"].shape == (1, 64, 64, 3)
    assert prepared.payload["outpaint_working_mask"].shape == (1, 64, 64)
    assert prepared.payload["blend_mask"].shape == (1, 64, 64)
    assert metrics["outpaint_prepare_cpu"] == pytest.approx(0.05)
    assert metrics["bb_vae_encode_cpu"] >= 0.0

    runtime.close()


def test_denoise_consumes_prepared_spatial_latent_and_mask(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    cond_records: list[dict[str, object]] = []
    sampler_records: list[dict[str, object]] = []

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = sampler
            _ = scheduler
            _ = denoise
            _ = model_options
            self.sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    def fake_prepare_direct_conds(self, *, execution_unet, noise, positive, negative, latent_image, denoise_mask, device):
        _ = self
        _ = execution_unet
        _ = noise
        _ = positive
        _ = negative
        _ = device
        cond_records.append(
            {
                "latent_shape": tuple(int(dim) for dim in latent_image.shape),
                "mask_shape": None if denoise_mask is None else tuple(int(dim) for dim in denoise_mask.shape),
            }
        )
        return {"positive": [], "negative": []}, 0.05

    def fake_sample_euler(model_fn, noise, sigmas, extra_args, callback, disable):
        _ = callback
        sampler_records.append(
            {
                "noise_shape": tuple(int(dim) for dim in noise.shape),
                "mask_shape": None if extra_args.get("denoise_mask") is None else tuple(int(dim) for dim in extra_args["denoise_mask"].shape),
                "disable": bool(disable),
            }
        )
        return model_fn(noise, sigmas[:1], **extra_args)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(runtime_mod.sampling, "KSampler", FakeKSampler)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_prepare_direct_conds", fake_prepare_direct_conds)
    monkeypatch.setattr(runtime_mod.k_diffusion, "sample_euler", fake_sample_euler)
    monkeypatch.setattr(runtime_mod.precision, "autocast_context", lambda device, enabled=True: contextlib.nullcontext())

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_pixels[:, 12:52, 12:52, :] = 1.0
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 16:48, 20:44] = 1.0

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()

    denoise_result = runtime.denoise_prepared_inputs(prepared)
    expected_latent_shape = tuple(int(dim) for dim in prepared.payload["bb_latent"].shape)
    expected_mask_shape = tuple(int(dim) for dim in prepared.payload["bb_denoise_mask"].shape)

    assert prepared.spatial_conditioning is not None
    assert prepared.spatial_conditioning.spatial_mode == "inpaint"
    assert cond_records == [{"latent_shape": expected_latent_shape, "mask_shape": expected_mask_shape}]
    assert sampler_records == [{"noise_shape": expected_latent_shape, "mask_shape": expected_mask_shape, "disable": True}]
    assert tuple(int(dim) for dim in denoise_result.samples.shape) == expected_latent_shape
    assert denoise_result.metrics["prepared_spatial_reused"] == 1.0
    assert denoise_result.metrics["denoise_mask_attached"] == 1.0
    assert any(component.startswith("spatial_conditioning:inpaint:") for component in denoise_result.execution_state.attached_component_ids)

    runtime.close()


def test_denoise_can_disable_inpaint_initial_latent(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    cond_records: list[dict[str, object]] = []

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = sampler
            _ = scheduler
            _ = denoise
            _ = model_options
            self.sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    def fake_prepare_direct_conds(self, *, execution_unet, noise, positive, negative, latent_image, denoise_mask, device):
        _ = self
        _ = execution_unet
        _ = noise
        _ = positive
        _ = negative
        _ = denoise_mask
        _ = device
        cond_records.append(
            {
                "latent_shape": tuple(int(dim) for dim in latent_image.shape),
                "latent_mean": float(latent_image.mean().item()),
            }
        )
        return {"positive": [], "negative": []}, 0.05

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(runtime_mod.sampling, "KSampler", FakeKSampler)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_prepare_direct_conds", fake_prepare_direct_conds)
    monkeypatch.setattr(runtime_mod.k_diffusion, "sample_euler", lambda model_fn, noise, sigmas, extra_args, callback, disable: model_fn(noise, sigmas[:1], **extra_args))
    monkeypatch.setattr(runtime_mod.precision, "autocast_context", lambda device, enabled=True: contextlib.nullcontext())

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 4:20, 8:24] = 1.0
    resolved_context = SimpleNamespace(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        original_mask=np.zeros((64, 64), dtype=np.uint8),
        bb=(4, 20, 8, 24),
        bb_image=np.full((32, 48, 3), 17, dtype=np.uint8),
        bb_mask=np.pad(np.full((16, 24), 255, dtype=np.uint8), ((8, 8), (12, 12))),
        target_w=48,
        target_h=32,
        blend_mask=np.full((64, 64), 255, dtype=np.uint8),
    )

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
        resolved_spatial_context=resolved_context,
        disable_initial_latent=True,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()
    assert float(prepared.payload["bb_latent"].mean().item()) > 0.0

    denoise_result = runtime.denoise_prepared_inputs(prepared)

    assert cond_records == [{"latent_shape": (1, 4, 4, 6), "latent_mean": pytest.approx(0.0)}]
    assert tuple(int(dim) for dim in denoise_result.samples.shape) == (1, 4, 4, 6)

    runtime.close()


def test_decode_latent_composes_inpaint_output(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    def fake_decode_preloaded_vae(vae, latent, tiled=False, tile_size=64):
        _ = vae
        _ = tiled
        _ = tile_size
        batch, _, height, width = latent.shape
        return torch.ones((batch, height * 8, width * 8, 3), dtype=torch.float32)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.decode, "decode_preloaded_vae", fake_decode_preloaded_vae)

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    source_mask[:, 16:48, 20:44] = 1.0

    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        source_mask=source_mask,
        spatial_mode="inpaint",
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()
    bbox = prepared.payload["bbox"]
    blend_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    y1, y2, x1, x2 = bbox
    blend_mask[:, y1:y2, x1:x2] = 1.0
    prepared.payload["blend_mask"] = blend_mask

    images, vae_attach, vae_decode = runtime.decode_latent(prepared.payload["bb_latent"])

    assert images.shape == (1, 64, 64, 3)
    assert float(images[:, y1:y2, x1:x2, :].mean().item()) == pytest.approx(1.0)
    assert float(images[:, :y1, :, :].max().item()) == pytest.approx(0.0)
    assert vae_attach >= 0.0
    assert vae_decode >= 0.0

    runtime.close()


def test_decode_latent_composes_outpaint_output(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    def fake_build_outpaint_context(self, pixels, mask_2d):
        _ = self
        _ = mask_2d
        original = (pixels.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).numpy())
        working = original.copy()
        bbox = (0, 32, 0, 48)
        bb_mask = torch.zeros((1, 32, 48), dtype=torch.float32)
        bb_mask[:, :12, :] = 1.0
        blend_mask = torch.zeros((64, 64), dtype=torch.uint8)
        blend_mask[:32, :48] = 255
        return {
            "working_pixels": working,
            "working_mask": torch.zeros((64, 64), dtype=torch.uint8).numpy(),
            "bb_pixels": working[:32, :48, :],
            "bb_mask": (bb_mask[0].numpy() * 255.0).astype("uint8"),
            "blend_mask": blend_mask.numpy(),
            "bbox": bbox,
            "target_width": 48,
            "target_height": 32,
            "direction": "top",
            "prepare_wall": 0.05,
        }

    def fake_decode_preloaded_vae(vae, latent, tiled=False, tile_size=64):
        _ = vae
        _ = tiled
        _ = tile_size
        batch, _, height, width = latent.shape
        return torch.ones((batch, height * 8, width * 8, 3), dtype=torch.float32)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_build_outpaint_context", fake_build_outpaint_context)
    monkeypatch.setattr(runtime_mod.decode, "decode_preloaded_vae", fake_decode_preloaded_vae)

    source_pixels = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=48,
        height=32,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        source_pixels=source_pixels,
        spatial_mode="outpaint",
        outpaint_direction="top",
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()

    images, _, _ = runtime.decode_latent(prepared.payload["bb_latent"])

    assert images.shape == (1, 64, 64, 3)
    assert float(images[:, :32, :48, :].mean().item()) == pytest.approx(1.0)
    assert float(images[:, 40:, 50:, :].max().item()) == pytest.approx(0.0)

    runtime.close()


def test_compiled_unet_fingerprint_is_prompt_independent(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_load_lora(header, to_load, log_missing=False):
        _ = header
        _ = log_missing
        target_key = next(iter(to_load.values()), None)
        if target_key is None:
            return {}
        return {target_key: ("diff", torch.ones(1, dtype=torch.float16))}

    def fake_compile_patcher(patcher, *args, **kwargs):
        patch_count = len(getattr(patcher, "patches", {}) or {})
        patcher.patches = {}
        return {
            "status": "compiled" if patch_count else "noop",
            "patch_count": patch_count,
            "materialized_patch_keys": patch_count,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.backend_lora, "load_lora", fake_load_lora)
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_clip", lambda model, key_map=None: {"clip_target": "clip.weight"})
    monkeypatch.setattr(runtime_mod.backend_lora, "model_lora_keys_unet", lambda model, key_map=None: {"unet_target": "unet.weight"})
    monkeypatch.setattr(runtime_mod, "SafeOpenHeaderOnly", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    common_kwargs = dict(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        negative_prompt="low quality",
        width=1024,
        height=1024,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        clip_layer=-2,
        batch_size=1,
        lora_specs=(("lora-a.safetensors", 1.0),),
    )

    runtime_a = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(prompt="a red fox", **common_kwargs)
    )
    runtime_b = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(prompt="a blue whale", **common_kwargs)
    )

    prepared_a, _ = runtime_a.prepare_inputs()
    prepared_b, _ = runtime_b.prepare_inputs()

    assert prepared_a.compiled_unet is not None
    assert prepared_b.compiled_unet is not None
    assert prepared_a.conditioning is not None
    assert prepared_b.conditioning is not None
    assert prepared_a.compiled_unet.artifact_fingerprint == prepared_b.compiled_unet.artifact_fingerprint
    assert prepared_a.conditioning.prompt_fingerprint != prepared_b.conditioning.prompt_fingerprint

    runtime_a.close()
    runtime_b.close()


def test_prepare_inputs_builds_contextual_injected_feature_artifacts(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    preprocess_calls: list[str] = []

    def fake_preprocess(img, model_path, **kwargs):
        _ = img
        _ = kwargs
        preprocess_calls.append(str(model_path))
        cond = torch.ones((1, 1, 1), dtype=torch.float16)
        uncond = torch.zeros((1, 1, 1), dtype=torch.float16)
        return ([cond, cond * 2], [uncond, uncond + 1])

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    fake_ip_adapter = SimpleNamespace(preprocess=fake_preprocess, patch_model=lambda model, tasks: model)
    fake_pulid = SimpleNamespace(preprocess=lambda *args, **kwargs: None, patch_model=lambda model, tasks: model)
    monkeypatch.setattr(
        runtime_mod.UnifiedSDXLRuntime,
        "_load_contextual_runtime_modules",
        lambda self: (fake_ip_adapter, fake_pulid),
    )

    source = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        contextual_tasks={
            flags.cn_ip: ((source[0].numpy(), 0.8, 0.7, 0.25),),
        },
        contextual_assets={
            "contextual_model_paths": {flags.cn_ip: "ip-adapter.safetensors"},
            "clip_vision_path": "clip-vision.safetensors",
            "ip_negative_path": "ip-negative.safetensors",
        },
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, metrics = runtime.prepare_inputs()

    assert preprocess_calls == ["ip-adapter.safetensors"]
    assert "feature_boundary_placeholder" not in prepared.injected_features
    assert "contextual:ImagePrompt" in prepared.injected_features
    assert prepared.injected_features["contextual:ImagePrompt"].feature_fingerprint
    assert prepared.payload["contextual_tasks"][flags.cn_ip][0][1:] == (0.8, 0.7, 0.25)
    assert metrics["injected_feature_count"] == 1.0
    assert metrics["contextual_task_count"] == 1.0
    assert metrics["contextual_imageprompt_task_count"] == 1.0

    runtime.close()


def test_prepare_inputs_builds_structural_conditioning_artifacts(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))

    hint = torch.ones((1, 32, 32, 3), dtype=torch.float32)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        structural_tasks={
            flags.cn_canny: ((hint, 0.8, 0.7),),
        },
        controlnet_paths={flags.cn_canny: "canny-controlnet.safetensors"},
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, metrics = runtime.prepare_inputs()

    assert prepared.structural_conditioning is runtime.structural_conditioning
    assert prepared.structural_conditioning is not None
    assert prepared.structural_conditioning.task_count == 1
    assert prepared.structural_conditioning.control_types == (flags.cn_canny,)
    assert prepared.payload["structural_tasks"][flags.cn_canny][0][0].shape == (1, 32, 32, 3)
    assert metrics["structural_artifact_count"] == 1.0
    assert metrics["structural_task_count"] == 1.0

    runtime.close()


def test_denoise_uses_request_local_contextual_execution_model(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = sampler
            _ = scheduler
            _ = denoise
            _ = model_options
            self.sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    patch_calls: list[int] = []
    cond_records: list[Any] = []

    def fake_patch_model(model, tasks):
        patch_calls.append(len(tasks))
        return SimpleNamespace(model_options={"transformer_options": {}}, model=model.model)

    def fake_prepare_direct_conds(self, *, execution_unet, noise, positive, negative, latent_image, denoise_mask, device):
        _ = self
        _ = noise
        _ = positive
        _ = negative
        _ = latent_image
        _ = denoise_mask
        _ = device
        cond_records.append(execution_unet)
        return {"positive": [], "negative": []}, 0.05

    def fake_sample_euler(model_fn, noise, sigmas, extra_args, callback, disable):
        _ = callback
        _ = disable
        return model_fn(noise, sigmas[:1], **extra_args)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(runtime_mod.sampling, "KSampler", FakeKSampler)
    fake_ip_adapter = SimpleNamespace(preprocess=lambda *args, **kwargs: None, patch_model=fake_patch_model)
    fake_pulid = SimpleNamespace(preprocess=lambda *args, **kwargs: None, patch_model=lambda model, tasks: model)
    monkeypatch.setattr(
        runtime_mod.UnifiedSDXLRuntime,
        "_load_contextual_runtime_modules",
        lambda self: (fake_ip_adapter, fake_pulid),
    )
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_prepare_direct_conds", fake_prepare_direct_conds)
    monkeypatch.setattr(runtime_mod.k_diffusion, "sample_euler", fake_sample_euler)
    monkeypatch.setattr(runtime_mod.precision, "autocast_context", lambda device, enabled=True: contextlib.nullcontext())

    cond = torch.ones((1, 1, 1), dtype=torch.float16)
    uncond = torch.zeros((1, 1, 1), dtype=torch.float16)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        contextual_tasks={
            flags.cn_ip: ((((cond, cond * 2), (uncond, uncond + 1)), 0.8, 0.7, 0.25),),
        },
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()
    denoise_result = runtime.denoise_prepared_inputs(prepared)

    assert patch_calls == [1]
    assert len(cond_records) == 1
    assert cond_records[0] is not runtime.unet
    assert denoise_result.metrics["prepared_conditioning_reused"] == 1.0
    assert any(component.startswith("injected_feature:contextual:ImagePrompt:") for component in denoise_result.execution_state.attached_component_ids)

    runtime.close()


def test_denoise_applies_prepared_structural_controlnets(monkeypatch):
    def fake_load_checkpoint(*args, **kwargs):
        return FakePatcher("unet"), FakeClip(), FakeVAE()

    def fake_compile_patcher(patcher, *args, **kwargs):
        _ = patcher
        _ = args
        _ = kwargs
        return {
            "status": "noop",
            "patch_count": 0,
            "materialized_patch_keys": 0,
            "host_pinned_bytes": 0,
        }

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = sampler
            _ = scheduler
            _ = denoise
            _ = model_options
            self.sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    load_calls: list[str] = []
    apply_calls: list[tuple[str, float, float, tuple[int, ...]]] = []
    cond_records: list[dict[str, Any]] = []

    def fake_load_controlnet(path):
        load_calls.append(path)
        return f"controlnet:{path}"

    def fake_apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
        apply_calls.append((control_net, strength, end_percent, tuple(int(dim) for dim in image.shape)))
        positive = [[positive[0][0], {**positive[0][1], "control_tag": control_net}]]
        negative = [[negative[0][0], {**negative[0][1], "control_tag": control_net}]]
        return positive, negative

    def fake_prepare_direct_conds(self, *, execution_unet, noise, positive, negative, latent_image, denoise_mask, device):
        _ = self
        _ = execution_unet
        _ = noise
        _ = latent_image
        _ = denoise_mask
        _ = device
        cond_records.append(
            {
                "positive_payload": dict(positive[0][1]),
                "negative_payload": dict(negative[0][1]),
            }
        )
        return {"positive": [], "negative": []}, 0.05

    def fake_sample_euler(model_fn, noise, sigmas, extra_args, callback, disable):
        _ = callback
        _ = disable
        return model_fn(noise, sigmas[:1], **extra_args)

    monkeypatch.setattr(runtime_mod.loader, "load_sdxl_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.CpuArtifactCompiler, "compile_patcher", staticmethod(fake_compile_patcher))
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(runtime_mod.sampling, "KSampler", FakeKSampler)
    monkeypatch.setattr(
        runtime_mod.UnifiedSDXLRuntime,
        "_load_structural_runtime_modules",
        lambda self: SimpleNamespace(load_controlnet=fake_load_controlnet, apply_controlnet=fake_apply_controlnet),
    )
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, "_prepare_direct_conds", fake_prepare_direct_conds)
    monkeypatch.setattr(runtime_mod.k_diffusion, "sample_euler", fake_sample_euler)
    monkeypatch.setattr(runtime_mod.precision, "autocast_context", lambda device, enabled=True: contextlib.nullcontext())

    hint = torch.ones((1, 32, 32, 3), dtype=torch.float32)
    config = runtime_mod.UnifiedSDXLRuntimeConfig(
        model_variant="sdxl-base",
        execution_class="cpu-first",
        checkpoint_path="checkpoint.safetensors",
        prompt="a red fox",
        negative_prompt="low quality",
        width=128,
        height=128,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="karras",
        seed=1234,
        structural_tasks={
            flags.cn_canny: ((hint, 0.8, 0.7),),
        },
        controlnet_paths={flags.cn_canny: "canny-controlnet.safetensors"},
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(config)
    prepared, _ = runtime.prepare_inputs()
    denoise_result = runtime.denoise_prepared_inputs(prepared)

    assert load_calls == ["canny-controlnet.safetensors"]
    assert apply_calls == [("controlnet:canny-controlnet.safetensors", 0.7, 0.8, (1, 32, 32, 3))]
    assert cond_records[0]["positive_payload"]["control_tag"] == "controlnet:canny-controlnet.safetensors"
    assert cond_records[0]["negative_payload"]["control_tag"] == "controlnet:canny-controlnet.safetensors"
    assert denoise_result.metrics["prepared_structural_reused"] == 1.0
    assert any(component.startswith("structural_conditioning:") for component in denoise_result.execution_state.attached_component_ids)

    runtime.close()


def test_decode_latent_clears_vram_before_and_after_vae_attachment(monkeypatch):
    cache_flush_calls = []

    def fake_soft_empty_cache(force=False):
        cache_flush_calls.append(force)

    policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_class=ExecutionClass.SDXL_STREAMING_T1,
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
    )

    runtime = streaming_mod.SDXLStreamingRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
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
            runtime_policy=policy,
        )
    )
    runtime.unet = FakePatcher("unet")
    runtime.vae = FakeVAE()
    runtime.policy = policy
    runtime._checkpoint_fingerprint = "checkpoint-fingerprint"

    monkeypatch.setattr(runtime, "load_components", lambda: 0.0)
    monkeypatch.setattr(runtime_mod.resources, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runtime_mod.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)
    monkeypatch.setattr(runtime_mod.resources, "soft_empty_cache", fake_soft_empty_cache)
    monkeypatch.setattr(
        runtime_mod.decode,
        "decode_preloaded_vae",
        lambda vae, latent, tiled=False, tile_size=64: latent.detach().cpu().movedim(1, -1),
    )

    assert runtime._is_vae_resident() is False

    images, _, _ = runtime.decode_latent(torch.zeros((1, 4, 8, 8), dtype=torch.float32))

    assert images.shape == (1, 8, 8, 4)
    assert runtime.vae.patcher.detach_calls == [("vae", True)]
    assert cache_flush_calls == [True, True]

    runtime.close()


def test_runtime_reload_prefers_offload_device_before_active_device():
    reload_targets = []

    class TinyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diffusion_model = torch.nn.Linear(4, 4, bias=False)
            self.device = torch.device("meta")

    class RecordingPatcher(backend_patching.NexModelPatcher):
        def _load_list(self, *, device_to=None):
            _ = device_to
            return []

    patcher = RecordingPatcher(
        TinyContainer(),
        load_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        runtime_reload=lambda model, device: reload_targets.append(device),
        runtime_release_to_meta=True,
    )

    patcher.model.device = torch.device("meta")
    patcher.load(device_to=torch.device("cuda"), lowvram_model_memory=256 * 1024 * 1024)

    assert reload_targets == [torch.device("cpu")]


def test_patcher_detach_does_not_release_to_meta():
    class TinyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diffusion_model = torch.nn.Linear(4, 4, bias=False)
            self.device = torch.device("cpu")

    reload_targets = []
    patcher = backend_patching.NexModelPatcher(
        TinyContainer(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        runtime_reload=lambda model, device: reload_targets.append(device),
        runtime_release_to_meta=True,
    )

    # Detach should NOT release weights to meta
    patcher.detach()
    assert patcher.current_loaded_device() != torch.device("meta")

    # Calling release_weights_to_meta explicitly should release weights to meta
    released = patcher.release_weights_to_meta()
    assert released is True
    assert patcher.current_loaded_device() == torch.device("meta")


def test_patcher_detach_does_not_detach_flux_scheduler_side_effects():
    class TinyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diffusion_model = torch.nn.Linear(4, 4, bias=False)
            self.device = torch.device("cpu")

    class FakeScheduler:
        def __init__(self) -> None:
            self.detach_calls = 0

        def detach(self) -> None:
            self.detach_calls += 1

    scheduler = FakeScheduler()
    patcher = backend_patching.NexModelPatcher(
        TinyContainer(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    patcher.model_options["flux_fill"] = {"streaming_scheduler": scheduler}

    patcher.detach()

    assert scheduler.detach_calls == 0


def test_release_shared_sdxl_base_components_clears_snapshots():
    class DummyModel:
        def __init__(self):
            self._nex_clean_unet_source = "unet_source"
            self._nex_resident_compile_metrics = "metrics"
            self._nex_resident_lora_signature = "sig"

    class DummyClipModel:
        def __init__(self):
            self._nex_clean_clip_source = "clip_source"

    class DummyPatcher:
        def __init__(self, model):
            self.model = model
            self.detached = False

        def detach(self):
            self.detached = True

    unet_model = DummyModel()
    clip_model = DummyClipModel()

    unet_patcher = DummyPatcher(unet_model)
    clip_patcher = DummyPatcher(clip_model)
    class DummyClipWrapper:
        def __init__(self, patcher):
            self.patcher = patcher

    entry = runtime_mod.SharedSDXLBaseComponents(
        unet=unet_patcher,
        clip=DummyClipWrapper(clip_patcher),
    )

    runtime_mod._release_shared_sdxl_base_components(entry, teardown=True)

    assert unet_patcher.detached is True
    assert clip_patcher.detached is True

    # Memory snapshots must be cleared
    assert unet_model._nex_clean_unet_source is None
    assert not hasattr(unet_model, "_nex_resident_compile_metrics")
    assert not hasattr(unet_model, "_nex_resident_lora_signature")
    assert clip_model._nex_clean_clip_source is None


def test_resolve_shared_sdxl_vae_path_ignores_small_or_corrupted_files(tmp_path, monkeypatch):
    import modules.config as config
    # Mock config.path_vae list
    monkeypatch.setattr(config, "path_vae", [str(tmp_path)])

    # 1. When file does not exist
    resolved = runtime_mod._resolve_shared_sdxl_vae_path()
    assert resolved is None

    # 2. When file is small (corrupted/empty)
    corrupted_file = tmp_path / "sdxl_vae.safetensors"
    corrupted_file.write_bytes(b"too small")

    resolved = runtime_mod._resolve_shared_sdxl_vae_path()
    assert resolved is None

    # 3. When file is valid/large
    large_size = 11 * 1024 * 1024
    with open(corrupted_file, "wb") as f:
        f.seek(large_size - 1)
        f.write(b"\0")

    resolved = runtime_mod._resolve_shared_sdxl_vae_path()
    assert resolved == str(corrupted_file)
