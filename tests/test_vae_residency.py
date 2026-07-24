import torch

from backend import encode as backend_encode
from backend import decode as backend_decode
from backend import sdxl_runtime_policy


class _DummyPatcher:
    def __init__(self, *, loaded_device="cpu"):
        self._loaded_device = torch.device(loaded_device)
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.detached = False
        self.parent = None
        self.model = self

    def current_loaded_device(self):
        return self._loaded_device

    def model_size(self):
        return 0

    def loaded_size(self):
        return 0

    def is_clone(self, other):
        return False

    def model_patches_to(self, target):
        pass

    def model_dtype(self):
        return torch.float32

    def partially_load(self, device, use_more_vram, force_patch_weights=False):
        return 0

    def partially_unload(self, device, memory_to_free):
        return 0

    def detach(self, unpatch_weights=True):
        self.detached = True

    def lowvram_patch_counter(self):
        return 0


class _DummyFirstStage:
    def __init__(self, dtype=torch.float32):
        self._dtype = dtype
        self.last_decode_dtype = None
        self.last_to_dtype = None
        self.last_to_device = None

    def parameters(self):
        return iter([torch.zeros(1, dtype=self._dtype)])

    def to(self, *, device=None, dtype=None):
        if dtype is not None:
            self._dtype = dtype
            self.last_to_dtype = dtype
        if device is not None:
            self.last_to_device = torch.device(device)
        return self

    def encode(self, batch):
        return batch.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)

    def decode(self, batch):
        self.last_decode_dtype = batch.dtype
        return batch.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


class _DummyLatentFormat:
    latent_channels = 4

    def process_in(self, tensor):
        return tensor

    def process_out(self, tensor):
        return tensor


class _DummyVAE:
    def __init__(self, *, loaded_device="cpu"):
        self.patcher = _DummyPatcher(loaded_device=loaded_device)
        self.first_stage_model = _DummyFirstStage()
        self.latent_format = _DummyLatentFormat()


class _DummyFluxLatentFormat:
    latent_channels = 16
    taesd_decoder_name = "taef1_decoder"

    def process_in(self, tensor):
        return tensor

    def process_out(self, tensor):
        return tensor


def test_encode_pixels_uses_shared_eject_boundary(monkeypatch):
    vae = _DummyVAE(loaded_device="mps")
    calls = []

    monkeypatch.setattr(backend_encode.resources, "eject_model", lambda patcher: calls.append(patcher))
    monkeypatch.setattr(backend_encode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    pixels = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    result = backend_encode.encode_pixels(vae, pixels)

    assert calls == [vae.patcher]
    assert result["samples"].shape == (1, 4, 8, 8)


def test_decode_latent_uses_shared_activation_and_does_not_eject_on_success(monkeypatch):
    vae = _DummyVAE()
    prepare_calls = []
    eject_calls = []

    monkeypatch.setattr(
        backend_decode.resources,
        "prepare_models_for_stage",
        lambda models, **kwargs: prepare_calls.append((tuple(models), kwargs)),
    )
    monkeypatch.setattr(backend_decode.resources, "eject_model", lambda patcher: eject_calls.append(patcher))
    monkeypatch.setattr(backend_decode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    result = backend_decode.decode_latent(vae, latent, tiled=False)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] == (vae.patcher,)
    assert prepare_calls[0][1]["force_full_load"] is True
    assert eject_calls == []
    assert result.shape == (1, 8, 8, 3)


def test_force_fp32_vae_decode_applies_to_sdxl_but_not_flux():
    sdxl_vae = _DummyVAE()
    flux_vae = _DummyVAE()
    flux_vae.latent_format = _DummyFluxLatentFormat()

    assert backend_decode._should_force_fp32_vae_decode(sdxl_vae) is True
    assert backend_decode._should_force_fp32_vae_decode(flux_vae) is True


def test_decode_preloaded_vae_promotes_sdxl_vae_to_fp32(monkeypatch):
    vae = _DummyVAE()
    vae.first_stage_model = _DummyFirstStage(dtype=torch.float16)

    monkeypatch.setattr(backend_decode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    result = backend_decode.decode_preloaded_vae(vae, latent, tiled=False)

    assert vae.first_stage_model.last_to_dtype == torch.float32
    assert vae.first_stage_model.last_decode_dtype == torch.float32
    assert result.shape == (1, 8, 8, 3)


def test_decode_latent_prefers_tiled_when_gpu_headroom_is_tight(monkeypatch):
    vae = _DummyVAE(loaded_device="cuda")
    prepare_calls = []
    tiled_calls = []

    monkeypatch.setattr(
        backend_decode.resources,
        "prepare_models_for_stage",
        lambda models, **kwargs: prepare_calls.append((tuple(models), kwargs)),
    )
    monkeypatch.setattr(backend_decode.resources, "get_free_memory", lambda device: 1)
    monkeypatch.setattr(backend_decode.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        backend_decode,
        "_decode_tiled",
        lambda vae_obj, samples, tile_x=64, tile_y=64, overlap=16, min_free_mem=0: tiled_calls.append(
            {
                "shape": tuple(samples.shape),
                "tile_x": tile_x,
                "tile_y": tile_y,
            }
        ) or torch.zeros((samples.shape[0], 3, samples.shape[2], samples.shape[3])),
    )

    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    result = backend_decode.decode_latent(vae, latent, tiled=False)

    assert len(prepare_calls) == 1
    assert tiled_calls
    assert result.shape == (1, 8, 8, 3)


def test_decode_tiled_promotes_sdxl_vae_to_fp32(monkeypatch):
    vae = _DummyVAE(loaded_device="cuda")
    vae.first_stage_model = _DummyFirstStage(dtype=torch.float16)
    prepare_calls = []

    monkeypatch.setattr(
        backend_decode.resources,
        "prepare_models_for_stage",
        lambda models, **kwargs: prepare_calls.append((tuple(models), kwargs)),
    )

    def _fake_tiled_scale(samples, decode_fn, *_args, **_kwargs):
        batch = samples[:1]
        decoded = decode_fn(batch)
        return torch.zeros_like(decoded)

    monkeypatch.setattr(backend_decode.utils, "tiled_scale", _fake_tiled_scale)

    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    result = backend_decode._decode_tiled(vae, latent, tile_x=64, tile_y=64)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][1]["force_full_load"] is True
    assert vae.first_stage_model.last_to_dtype == torch.float32
    assert vae.first_stage_model.last_decode_dtype == torch.float32
    assert result.shape == (1, 3, 8, 8)


def test_encode_pixels_ejects_after_success_for_resident_sdxl_policy(monkeypatch):
    vae = _DummyVAE(loaded_device="cpu")
    prepare_calls = []
    eject_calls = []
    vae.runtime_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_GPU_RESIDENT,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_GPU_RESIDENT,
        keep_clip_loaded=True,
        prefer_clip_gpu=True,
        prefer_gpu_vae_encode=True,
    )

    def _prepare(models, **kwargs):
        prepare_calls.append((tuple(models), kwargs))
        vae.patcher._loaded_device = torch.device("cuda")

    monkeypatch.setattr(backend_encode.resources, "prepare_models_for_stage", _prepare)
    monkeypatch.setattr(backend_encode.resources, "eject_model", lambda patcher: eject_calls.append(patcher))
    monkeypatch.setattr(backend_encode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    pixels = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    result = backend_encode.encode_pixels(vae, pixels)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] == (vae.patcher,)
    assert vae.runtime_policy.vae_encode_mode == sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU
    assert eject_calls == [vae.patcher]
    assert result["samples"].shape == (1, 4, 8, 8)


def test_encode_pixels_ejects_transient_gpu_vae_after_success(monkeypatch):
    vae = _DummyVAE(loaded_device="cpu")
    eject_calls = []

    vae.runtime_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
        prefer_gpu_vae_encode=True,
    )

    def _prepare(models, **kwargs):
        _ = models
        _ = kwargs
        vae.patcher._loaded_device = torch.device("cuda")

    monkeypatch.setattr(backend_encode.resources, "prepare_models_for_stage", _prepare)
    monkeypatch.setattr(backend_encode.resources, "eject_model", lambda patcher: eject_calls.append(patcher))
    monkeypatch.setattr(backend_encode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    pixels = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    result = backend_encode.encode_pixels(vae, pixels)

    assert result["samples"].shape == (1, 4, 8, 8)
    assert eject_calls == [vae.patcher]


def test_decode_latent_ejects_transient_gpu_vae_after_success(monkeypatch):
    vae = _DummyVAE(loaded_device="cpu")
    prepare_calls = []
    eject_calls = []

    vae.runtime_policy = sdxl_runtime_policy.SDXLExecutionPolicy(
        enabled=True,
        architecture="sdxl",
        execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        clip_residency_mode=sdxl_runtime_policy.CLIP_RESIDENCY_CPU_ONLY,
        vae_encode_mode=sdxl_runtime_policy.VAE_POSTURE_TRANSIENT_GPU,
        prefer_gpu_vae_encode=True,
    )

    def _prepare(models, **kwargs):
        prepare_calls.append((tuple(models), kwargs))
        vae.patcher._loaded_device = torch.device("cuda")

    monkeypatch.setattr(backend_decode.resources, "prepare_models_for_stage", _prepare)
    monkeypatch.setattr(backend_decode.resources, "eject_model", lambda patcher: eject_calls.append(patcher))
    monkeypatch.setattr(backend_decode.resources, "get_free_memory", lambda device: 1024 * 1024 * 1024)

    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    result = backend_decode.decode_latent(vae, latent, tiled=False)

    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] == (vae.patcher,)
    assert eject_calls == [vae.patcher]
    assert result.shape == (1, 8, 8, 3)
