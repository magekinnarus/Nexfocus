from __future__ import annotations

import importlib
import sys
import types

import safetensors.torch
import torch


def _load_loader(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"], raising=False)

    class _DummyTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, text, **_kwargs):
            return {"input_ids": [0, 1, 2]}

    class _DummyModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__path__ = []
    fake_transformers.__version__ = "0.0.0"
    fake_transformers.CLIPTokenizer = _DummyTokenizer
    fake_transformers.T5TokenizerFast = _DummyTokenizer
    fake_transformers.AutoTokenizer = _DummyTokenizer
    fake_transformers.AutoModel = _DummyModel
    fake_transformers.AutoModelForMaskedLM = _DummyModel
    fake_transformers.AutoConfig = type("DummyConfig", (), {})
    fake_transformers.PretrainedConfig = type("DummyPretrainedConfig", (), {})
    fake_transformers.CLIPTextModel = _DummyModel
    fake_transformers.CLIPTextConfig = type("DummyCLIPTextConfig", (), {})
    fake_transformers.CLIPVisionConfig = type("DummyCLIPVisionConfig", (), {})
    fake_transformers.CLIPVisionModelWithProjection = _DummyModel
    fake_transformers.modeling_utils = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import backend.loader as loader

    return importlib.reload(loader)


def test_clip_encode_from_tokens_forces_full_clip_activation(monkeypatch):
    loader = _load_loader(monkeypatch)

    class _DummyCondStage:
        def encode_token_weights(self, tokens):
            return "cond", "pooled"

        def reset_clip_layer(self):
            return None

        def state_dict(self):
            return {}

    clip = loader.CLIP(
        cond_stage_model=_DummyCondStage(),
        tokenizer=types.SimpleNamespace(tokenize_with_weights=lambda text, return_word_ids=False: text),
        load_device="cuda:0",
        offload_device="cpu",
    )

    captured = {}

    def fake_prepare_models_for_stage(models, **kwargs):
        captured["models"] = models
        captured["kwargs"] = kwargs

    monkeypatch.setattr(loader.resources, "prepare_models_for_stage", fake_prepare_models_for_stage)

    result = clip.encode_from_tokens("tokens", return_pooled=True)

    assert result == ("cond", "pooled")
    assert captured["models"] == [clip.patcher]
    assert captured["kwargs"]["stage_name"] == "text_encode"
    assert captured["kwargs"]["target_phase"] == loader.resources.MemoryPhase.PROMPT_ENCODE
    assert captured["kwargs"]["force_full_load"] is True


def test_load_sdxl_clip_reuses_same_bundled_source(monkeypatch):
    loader = _load_loader(monkeypatch)

    stream_calls = []

    class _DummyClipModel:
        def __init__(self, *args, **kwargs):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_sd(self, sd, force_type=None):
            raise AssertionError("dict-based CLIP load should not be used for safetensors sources")

        def state_dict(self):
            return {}

    monkeypatch.setattr(loader.clip, "NexSDXLTokenizer", lambda: object())
    monkeypatch.setattr(loader.clip, "NexSDXLClipModel", _DummyClipModel)
    monkeypatch.setattr(
        loader,
        "_load_sdxl_clip_source_into_model",
        lambda target_model, source, *, force_type=None, prefixes=None, dtype=None: stream_calls.append(
            (source, force_type, tuple(prefixes) if prefixes is not None else None, dtype)
        ),
    )
    monkeypatch.setattr(
        loader,
        "resolve_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolve_source should not be used")),
    )

    bundled_path = "D:/models/bundled_sdxl_clip.safetensors"
    loaded_clip = loader.load_sdxl_clip(bundled_path, bundled_path, dtype=None)

    assert stream_calls == [
        (bundled_path, "l", None, torch.float32),
        (bundled_path, "g", None, torch.float32),
    ]
    assert callable(loaded_clip.patcher.runtime_reload)


def test_load_sdxl_clip_defaults_to_fp32_residency(monkeypatch):
    loader = _load_loader(monkeypatch)

    captured = {}

    class _DummyClipModel:
        def __init__(self, *args, **kwargs):
            captured["dtype"] = kwargs.get("dtype")
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_sd(self, sd, force_type=None):
            raise AssertionError("dict-based CLIP load should not be used for safetensors sources")

        def state_dict(self):
            return {}

    monkeypatch.setattr(loader.clip, "NexSDXLTokenizer", lambda: object())
    monkeypatch.setattr(loader.clip, "NexSDXLClipModel", _DummyClipModel)
    monkeypatch.setattr(
        loader,
        "_load_sdxl_clip_source_into_model",
        lambda *args, **kwargs: None,
    )

    loader.load_sdxl_clip("clip_l.safetensors", "clip_g.safetensors", dtype=None)

    assert captured["dtype"] == torch.float32


def test_sdxl_clip_text_model_honors_requested_embedding_dtype(monkeypatch):
    loader = _load_loader(monkeypatch)

    model = loader.clip.CLIPTextModel_(
        {
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 1,
            "intermediate_size": 16,
            "hidden_act": "gelu",
        },
        dtype=torch.float16,
        device="cpu",
        embedding_dtype=torch.float16,
    )

    assert model.embeddings.token_embedding.weight.dtype == torch.float16
    assert model.embeddings.position_embedding.weight.dtype == torch.float16


def test_sdxl_clip_model_keeps_fp32_embeddings_for_compute_stability(monkeypatch):
    loader = _load_loader(monkeypatch)

    model = loader.clip.NexSDXLClipModel(device="cpu", dtype=torch.float16)

    assert model.clip_l.transformer.embeddings.token_embedding.weight.dtype == torch.float32
    assert model.clip_l.transformer.embeddings.position_embedding.weight.dtype == torch.float32
    assert model.clip_g.transformer.embeddings.token_embedding.weight.dtype == torch.float32
    assert model.clip_g.transformer.embeddings.position_embedding.weight.dtype == torch.float32
    assert model.clip_l.transformer.final_layer_norm.weight.dtype == torch.float16
    assert model.clip_g.transformer.final_layer_norm.weight.dtype == torch.float16


def test_load_sdxl_checkpoint_uses_fp32_clip_residency(monkeypatch):
    loader = _load_loader(monkeypatch)

    monkeypatch.setattr(loader, "_extract_prefixed_safetensors_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        loader,
        "_inspect_safetensors_vae_metadata",
        lambda *args, **kwargs: {
            "key_count": 0,
            "decoder_conv_in_shape": None,
            "post_quant_conv_shape": None,
            "has_downsample": False,
            "has_upsample": False,
        },
    )
    monkeypatch.setattr(loader, "load_vae", lambda *args, **kwargs: "vae")
    monkeypatch.setattr(loader, "_stream_load_sdxl_unet_from_checkpoint", lambda *args, **kwargs: "unet")

    captured = {}

    def fake_load_sdxl_clip(*args, **kwargs):
        captured["dtype"] = kwargs.get("dtype")
        return "clip"

    monkeypatch.setattr(loader, "load_sdxl_clip", fake_load_sdxl_clip)

    unet, clip, vae = loader.load_sdxl_checkpoint(
        "checkpoint.safetensors",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        unet_dtype=torch.float16,
        clip_load_device=torch.device("cpu"),
        clip_offload_device=torch.device("cpu"),
        vae_offload_device=torch.device("cpu"),
    )

    assert (unet, clip, vae) == ("unet", "clip", None)
    assert captured["dtype"] == torch.float32


def test_load_sdxl_checkpoint_uses_external_vae_source_when_provided(monkeypatch):
    loader = _load_loader(monkeypatch)

    monkeypatch.setattr(loader, "_extract_prefixed_safetensors_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(loader, "_stream_load_sdxl_unet_from_checkpoint", lambda *args, **kwargs: "unet")
    monkeypatch.setattr(loader, "load_sdxl_clip", lambda *args, **kwargs: "clip")

    captured = {}

    def fake_load_vae(source, *args, **kwargs):
        captured["source"] = source
        captured["kwargs"] = dict(kwargs)
        return "external-vae"

    monkeypatch.setattr(loader, "load_vae", fake_load_vae)

    unet, clip, vae = loader.load_sdxl_checkpoint(
        "checkpoint.safetensors",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        unet_dtype=torch.float16,
        clip_load_device=torch.device("cpu"),
        clip_offload_device=torch.device("cpu"),
        vae_load_device=torch.device("cpu"),
        vae_offload_device=torch.device("cpu"),
        vae_source="external_vae.safetensors",
    )

    assert (unet, clip, vae) == ("unet", "clip", "external-vae")
    assert captured["source"] == "external_vae.safetensors"
    assert captured["kwargs"]["load_device"] == torch.device("cpu")
    assert captured["kwargs"]["offload_device"] == torch.device("cpu")


def test_load_prefixed_safetensors_into_module_streams_directly(monkeypatch):
    loader = _load_loader(monkeypatch)

    class _FakeHandle:
        def __init__(self, tensors):
            self._tensors = tensors

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._tensors.keys())

        def get_tensor(self, key):
            return self._tensors[key]

    module = torch.nn.Linear(3, 2, bias=True)
    module.to(dtype=torch.float16)
    source_weight = torch.full((2, 3), 7.0, dtype=torch.float32)
    source_bias = torch.full((2,), -3.0, dtype=torch.float32)

    monkeypatch.setattr(
        loader,
        "safe_open",
        lambda *args, **kwargs: _FakeHandle(
            {
                "model.weight": source_weight,
                "model.bias": source_bias,
            }
        ),
    )

    missing, unexpected = loader._load_prefixed_safetensors_into_module(
        "checkpoint.safetensors",
        ["model."],
        module,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert missing == []
    assert unexpected == []
    assert module.weight.dtype == torch.float16
    assert module.bias.dtype == torch.float16
    assert torch.allclose(module.weight, source_weight.to(dtype=torch.float16))
    assert torch.allclose(module.bias, source_bias.to(dtype=torch.float16))


def test_load_prefixed_safetensors_into_module_chunks_pinned_targets(monkeypatch):
    loader = _load_loader(monkeypatch)

    class _FakeSlice:
        def __init__(self, tensor):
            self._tensor = tensor

        def get_shape(self):
            return list(self._tensor.shape)

        def get_dtype(self):
            return "F32"

        def __getitem__(self, item):
            return self._tensor[item]

    class _FakeHandle:
        def __init__(self, tensors):
            self._tensors = tensors

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._tensors.keys())

        def get_tensor(self, key):
            raise AssertionError("chunked pinned targets should avoid full get_tensor loads")

        def get_slice(self, key):
            return _FakeSlice(self._tensors[key])

    module = torch.nn.Linear(3, 4, bias=False)
    pinned_weight = torch.empty_like(module.weight.data, device="cpu", pin_memory=True)
    module.weight = torch.nn.Parameter(pinned_weight)
    source_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    monkeypatch.setattr(
        loader,
        "safe_open",
        lambda *args, **kwargs: _FakeHandle({"model.weight": source_weight}),
    )

    missing, unexpected = loader._load_prefixed_safetensors_into_module(
        "checkpoint.safetensors",
        ["model."],
        module,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_bytes=16,
    )

    assert missing == []
    assert unexpected == []
    assert module.weight.is_pinned()
    assert torch.equal(module.weight.detach().cpu(), source_weight)


def test_load_prefixed_safetensors_into_module_progressively_realizes_meta_targets(monkeypatch):
    loader = _load_loader(monkeypatch)

    class _FakeHandle:
        def __init__(self, tensors):
            self._tensors = tensors

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._tensors.keys())

        def get_tensor(self, key):
            return self._tensors[key]

    module = torch.nn.Linear(3, 4, bias=False, device="meta")
    source_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    empty_like_calls = []

    def fake_empty_like(tensor, *args, **kwargs):
        empty_like_calls.append(
            {
                "shape": tuple(tensor.shape),
                "device": kwargs.get("device"),
                "pin_memory": kwargs.get("pin_memory"),
                "source_device": tensor.device.type,
            }
        )
        return torch.zeros(tuple(tensor.shape), dtype=tensor.dtype)

    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loader.torch, "empty_like", fake_empty_like)
    monkeypatch.setattr(
        loader,
        "safe_open",
        lambda *args, **kwargs: _FakeHandle({"model.weight": source_weight}),
    )

    load_metrics = {}
    missing, unexpected = loader._load_prefixed_safetensors_into_module(
        "checkpoint.safetensors",
        ["model."],
        module,
        device=torch.device("cpu"),
        dtype=torch.float32,
        realize_pinned_targets=True,
        load_metrics=load_metrics,
    )

    assert missing == []
    assert unexpected == []
    assert module.weight.device.type == "cpu"
    assert torch.equal(module.weight.detach().cpu(), source_weight)
    assert load_metrics == {
        "realized_pinned_bytes": module.weight.numel() * module.weight.element_size(),
        "realized_pinned_tensor_count": 1,
    }
    assert empty_like_calls == [
        {"shape": (4, 3), "device": torch.device("cpu"), "pin_memory": True, "source_device": "meta"}
    ]


def test_load_prefixed_safetensors_into_module_raw_byte_streams_without_safe_open(monkeypatch, tmp_path):
    loader = _load_loader(monkeypatch)

    source_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    ckpt_path = tmp_path / "raw_stream.safetensors"
    safetensors.torch.save_file({"model.weight": source_weight}, str(ckpt_path))

    module = torch.nn.Linear(3, 4, bias=False, device="meta")
    empty_like_calls = []

    def fake_empty_like(tensor, *args, **kwargs):
        empty_like_calls.append(
            {
                "shape": tuple(tensor.shape),
                "device": kwargs.get("device"),
                "pin_memory": kwargs.get("pin_memory"),
                "source_device": tensor.device.type,
            }
        )
        return torch.zeros(tuple(tensor.shape), dtype=tensor.dtype)

    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loader.torch, "empty_like", fake_empty_like)
    monkeypatch.setattr(
        loader,
        "safe_open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("safe_open fallback should not be used")),
    )

    load_metrics = {}
    missing, unexpected = loader._load_prefixed_safetensors_into_module(
        str(ckpt_path),
        ["model."],
        module,
        device=torch.device("cpu"),
        dtype=None,
        chunk_bytes=5,
        realize_pinned_targets=True,
        load_metrics=load_metrics,
        raw_byte_stream=True,
    )

    assert missing == []
    assert unexpected == []
    assert module.weight.device.type == "cpu"
    assert torch.equal(module.weight.detach().cpu(), source_weight)
    assert load_metrics == {
        "realized_pinned_bytes": module.weight.numel() * module.weight.element_size(),
        "realized_pinned_tensor_count": 1,
    }
    assert empty_like_calls == [
        {"shape": (4, 3), "device": torch.device("cpu"), "pin_memory": True, "source_device": "meta"}
    ]


def test_stream_load_sdxl_unet_uses_raw_sequential_stream(monkeypatch):
    loader = _load_loader(monkeypatch)

    captured = {}

    class _DummySDXL(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.diffusion_model = torch.nn.Linear(2, 2, bias=False)

    def fake_stream_load(path, prefixes, module, **kwargs):
        captured["path"] = path
        captured["prefixes"] = prefixes
        captured["module"] = module
        captured["kwargs"] = dict(kwargs)
        load_metrics = kwargs.get("load_metrics")
        if isinstance(load_metrics, dict):
            load_metrics["realized_pinned_bytes"] = 0
            load_metrics["realized_pinned_tensor_count"] = 0
        return [], []

    monkeypatch.setattr(loader.model_base, "SDXL", _DummySDXL)
    monkeypatch.setattr(loader, "_load_prefixed_safetensors_into_module", fake_stream_load)

    patcher = loader._stream_load_sdxl_unet_from_checkpoint(
        "checkpoint.safetensors",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        dtype=torch.float16,
        reload_prefixes=["model.diffusion_model."],
        stream_chunk_bytes=64 * 1024 * 1024,
    )

    assert captured["path"] == "checkpoint.safetensors"
    assert captured["prefixes"] == ["model.diffusion_model."]
    assert captured["kwargs"]["raw_byte_stream"] is True
    assert captured["kwargs"]["chunk_bytes"] == 64 * 1024 * 1024
    assert patcher.model_options["sdxl_assembly_loader"] == {
        "direct_safetensors_load": True,
        "raw_sequential_stream": True,
        "meta_construction": True,
        "stream_chunk_bytes": 64 * 1024 * 1024,
        "realized_cpu_bytes": 0,
        "realized_cpu_tensor_count": 0,
        "realized_pinned_bytes": 0,
        "realized_pinned_tensor_count": 0,
    }


def test_reload_unet_weights_streams_safetensors_into_existing_module(monkeypatch):
    loader = _load_loader(monkeypatch)

    captured = {}

    def fake_stream_load(path, prefixes, module, **kwargs):
        captured["path"] = path
        captured["prefixes"] = prefixes
        captured["module"] = module
        captured["kwargs"] = dict(kwargs)
        return [], []

    monkeypatch.setattr(loader, "_load_prefixed_safetensors_into_module", fake_stream_load)
    monkeypatch.setattr(
        loader,
        "_extract_prefixed_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prefixed state dict should not be materialized")),
    )
    monkeypatch.setattr(
        loader,
        "resolve_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full state dict load should not be used")),
    )

    diffusion_model = torch.nn.Linear(3, 2, bias=True)
    target_model = types.SimpleNamespace(diffusion_model=diffusion_model)

    loader._reload_unet_weights(
        target_model,
        "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float16,
        prefixes=["model."],
    )

    assert captured["path"] == "checkpoint.safetensors"
    assert captured["prefixes"] == ["model."]
    assert captured["module"] is diffusion_model
    assert captured["kwargs"]["device"] == torch.device("cpu")
    assert captured["kwargs"]["dtype"] == torch.float16
    assert captured["kwargs"]["raw_byte_stream"] is True
    assert diffusion_model.weight.dtype == torch.float16
    assert diffusion_model.bias.dtype == torch.float16


def test_reload_sdxl_clip_weights_streams_safetensors_sources_directly(monkeypatch):
    loader = _load_loader(monkeypatch)

    calls = []

    class _DummyTargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((kwargs.get("device"), kwargs.get("dtype")))
            return self

        def load_sd(self, *_args, **_kwargs):
            raise AssertionError("dict-based CLIP load should not be used for safetensors sources")

    monkeypatch.setattr(
        loader,
        "_load_sdxl_clip_source_into_model",
        lambda target_model, source, *, force_type=None, prefixes=None, dtype=None: calls.append(
            (target_model, source, force_type, tuple(prefixes) if prefixes is not None else None, dtype)
        ),
    )
    monkeypatch.setattr(
        loader,
        "resolve_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolve_source should not be used")),
    )

    target_model = _DummyTargetModel()
    loader._reload_sdxl_clip_weights(
        target_model,
        "checkpoint.safetensors",
        "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        prefixes_l=["clip_l."],
        prefixes_g=["clip_g."],
    )

    assert target_model.to_calls == [(torch.device("cpu"), torch.float32)]
    assert calls == [
        (target_model, "checkpoint.safetensors", "l", ("clip_l.",), torch.float32),
        (target_model, "checkpoint.safetensors", "g", ("clip_g.",), torch.float32),
    ]


def test_load_sdxl_clip_streams_safetensors_sources_directly(monkeypatch):
    loader = _load_loader(monkeypatch)

    calls = []

    class _DummyClipModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_sd(self, *_args, **_kwargs):
            raise AssertionError("dict-based CLIP load should not be used for safetensors sources")

    monkeypatch.setattr(loader.clip, "NexSDXLTokenizer", lambda: object())
    monkeypatch.setattr(loader.clip, "NexSDXLClipModel", _DummyClipModel)
    monkeypatch.setattr(
        loader,
        "_load_sdxl_clip_source_into_model",
        lambda target_model, source, *, force_type=None, prefixes=None, dtype=None: calls.append(
            (target_model, source, force_type, tuple(prefixes) if prefixes is not None else None, dtype)
        ),
    )
    monkeypatch.setattr(
        loader,
        "resolve_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolve_source should not be used")),
    )

    loaded_clip = loader.load_sdxl_clip(
        "clips.safetensors",
        "clips.safetensors",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        dtype=torch.float32,
        reload_prefixes_l=["clip_l."],
        reload_prefixes_g=["clip_g."],
    )

    assert [call[1:] for call in calls] == [
        ("clips.safetensors", "l", ("clip_l.",), torch.float32),
        ("clips.safetensors", "g", ("clip_g.",), torch.float32),
    ]
    assert callable(loaded_clip.patcher.runtime_reload)


def test_load_vae_streams_safetensors_directly(monkeypatch):
    loader = _load_loader(monkeypatch)

    captured = {}

    class _DummyVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(
        loader,
        "_inspect_safetensors_vae_metadata",
        lambda *args, **kwargs: {
            "key_count": 2,
            "decoder_conv_in_shape": (128, 4, 3, 3),
            "post_quant_conv_shape": (4, 4, 1, 1),
            "has_downsample": True,
            "has_upsample": True,
        },
    )
    monkeypatch.setattr(loader, "AutoencoderKL", lambda *args, **kwargs: _DummyVAEModel())
    monkeypatch.setattr(
        loader,
        "_load_prefixed_safetensors_into_module",
        lambda path, prefixes, module, *, device=None, dtype=None: captured.update(
            {
                "path": path,
                "prefixes": prefixes,
                "module": module,
                "device": device,
                "dtype": dtype,
            }
        ) or ([], []),
    )
    monkeypatch.setattr(
        loader,
        "resolve_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolve_source should not be used")),
    )

    loaded_vae = loader.load_vae(
        "sdxl_vae.safetensors",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert captured["path"] == "sdxl_vae.safetensors"
    assert captured["prefixes"] == [""]
    assert captured["device"] == torch.device("cpu")
    assert captured["dtype"] == torch.float32
    assert loaded_vae.patcher.load_device == torch.device("cpu")
