from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

# Cache original args_manager if any to restore later and prevent leakage
_original_args_manager = sys.modules.get("args_manager")

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
    args_parser=SimpleNamespace(args=fake_args, parser=types.SimpleNamespace()),
)


@pytest.fixture(scope="module", autouse=True)
def _mock_args_manager_lifecycle():
    yield
    if _original_args_manager is None:
        sys.modules.pop("args_manager", None)
    else:
        sys.modules["args_manager"] = _original_args_manager


from backend import sdxl_runtime_policy
from backend.process_transition import clear_active_process_key, get_active_process_key


def _load_default_pipeline(monkeypatch, *, profile_name="colab_free", default_model_name=None):
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

    import modules.config as config

    monkeypatch.setattr(
        config,
        "resolved_memory_environment_profile",
        types.SimpleNamespace(name=profile_name),
        raising=False,
    )
    if default_model_name is not None:
        monkeypatch.setattr(config, "default_base_model_name", default_model_name, raising=False)

    import modules.default_pipeline as default_pipeline

    return importlib.reload(default_pipeline)


def _prepare_fake_pipeline_state(default_pipeline):
    dummy_model = types.SimpleNamespace(
        filename="model-a.safetensors",
        unet_with_lora=types.SimpleNamespace(model=types.SimpleNamespace(name="unet")),
        clip_with_lora=types.SimpleNamespace(fcs_cond_cache={"stale": object()}),
        vae=types.SimpleNamespace(name="vae"),
    )
    default_pipeline.model_base = dummy_model
    default_pipeline.final_unet = dummy_model.unet_with_lora
    default_pipeline.final_clip = dummy_model.clip_with_lora
    default_pipeline.final_vae = dummy_model.vae


def test_release_sdxl_runtime_state_soft_reset_clears_cached_references(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    _prepare_fake_pipeline_state(default_pipeline)
    from backend import sdxl_unified_runtime

    cache_clear_calls = []

    calls = []

    def fake_prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
        calls.append(
            {
                "current_model": current_model,
                "next_model": next_model,
                "notes": notes,
            }
        )
        if release_callback is not None:
            release_callback()
        return {"called": True}

    monkeypatch.setattr(default_pipeline.resources, "prepare_for_checkpoint_switch", fake_prepare_for_checkpoint_switch)
    monkeypatch.setattr(
        sdxl_unified_runtime,
        "clear_unified_sdxl_runtime_component_cache",
        lambda teardown=False: cache_clear_calls.append(teardown),
    )

    current_key = default_pipeline._sdxl_process_key(
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=sdxl_runtime_policy.resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model-a.safetensors",
            profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
            requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        ),
    )
    next_key = default_pipeline._sdxl_process_key(
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=sdxl_runtime_policy.resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model-a.safetensors",
            profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
            requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        ),
    )

    result = default_pipeline.release_sdxl_runtime_state(
        current_process_key=current_key,
        next_process_key=next_key,
        reason="unit_test_soft_reset",
        hard_reset=False,
    )

    assert result["hard_reset"] is False
    assert calls == []
    assert cache_clear_calls == [False]
    assert default_pipeline.final_unet is None
    assert default_pipeline.final_clip is None
    assert default_pipeline.final_vae is None
    assert default_pipeline.refresh_state["sdxl_process_key"] is None


def test_release_sdxl_runtime_state_classifies_vae_only_change_as_spatial(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    _prepare_fake_pipeline_state(default_pipeline)
    from backend import sdxl_unified_runtime
    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange

    cache_clear_calls = []
    release_calls = []

    monkeypatch.setattr(
        sdxl_unified_runtime,
        "clear_unified_sdxl_runtime_component_cache",
        lambda teardown=False: cache_clear_calls.append(teardown),
    )
    monkeypatch.setattr(
        "backend.sdxl_assembly.lifecycle_coordinator.release_for_changes",
        lambda changes, reason=None, **kwargs: release_calls.append((list(changes), reason)),
    )

    current_key = default_pipeline._sdxl_process_key(
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=sdxl_runtime_policy.resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name="model-a.safetensors",
            profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
            requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        ),
    )

    result = default_pipeline.release_sdxl_runtime_state(
        current_process_key=current_key,
        next_process_key=current_key,
        current_vae_name="vae-a.safetensors",
        next_vae_name="vae-b.safetensors",
        reason="vae_only_test",
        hard_reset=False,
    )

    assert result["hard_reset"] is False
    assert cache_clear_calls == [False]
    assert release_calls
    assert all(reason == "vae_only_test" for _, reason in release_calls)
    assert any(LifecycleChange.SPATIAL_VAE_CHANGE in changes for changes, _ in release_calls)
    assert all(LifecycleChange.MODEL_CHANGE not in changes for changes, _ in release_calls)


def test_refresh_base_model_classifies_vae_only_change_as_spatial(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange

    release_calls = []
    prepared_switches = []

    default_pipeline.model_base = types.SimpleNamespace(
        filename="model-a.safetensors",
        vae_filename="vae-a.safetensors",
        clip_filename="clip-a.safetensors",
        clip=types.SimpleNamespace(fcs_cond_cache={}),
        clip_with_lora=types.SimpleNamespace(fcs_cond_cache={}),
        unet=None,
        unet_with_lora=None,
        vae=None,
    )

    monkeypatch.setattr(default_pipeline, "get_file_from_folder_list", lambda name, _paths: name)
    monkeypatch.setattr(default_pipeline, "_apply_sdxl_policy_to_model_base", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        default_pipeline.resources,
        "prepare_for_checkpoint_switch",
        lambda **kwargs: prepared_switches.append(kwargs) or kwargs["release_callback"](),
    )
    monkeypatch.setattr(
        "backend.sdxl_assembly.lifecycle_coordinator.release_for_changes",
        lambda changes, reason=None, **kwargs: release_calls.append((list(changes), reason)),
    )
    monkeypatch.setattr(
        default_pipeline.core,
        "StableDiffusionModel",
        lambda: types.SimpleNamespace(
            filename=None,
            vae_filename=None,
            clip_filename=None,
            clip=None,
            clip_with_lora=None,
            unet=None,
            unet_with_lora=None,
            vae=None,
        ),
    )
    monkeypatch.setattr(
        default_pipeline.core,
        "load_model",
        lambda filename, vae_filename, clip_name, sdxl_policy=None: types.SimpleNamespace(
            filename=filename,
            vae_filename=vae_filename,
            clip_filename=clip_name,
            clip=None,
            clip_with_lora=None,
            unet=None,
            unet_with_lora=None,
            vae=None,
        ),
    )

    policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-a.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )

    default_pipeline.refresh_base_model(
        "model-a.safetensors",
        "vae-b.safetensors",
        "clip-a.safetensors",
        sdxl_policy=policy,
    )

    assert prepared_switches, "expected a staged release before the VAE-only refresh"
    assert release_calls
    assert all(reason == "checkpoint_switch" for _, reason in release_calls)
    assert any(LifecycleChange.SPATIAL_VAE_CHANGE in changes for changes, _ in release_calls)
    assert all(LifecycleChange.CHECKPOINT_CHANGE not in changes for changes, _ in release_calls)


def test_refresh_everything_hard_resets_on_sdxl_checkpoint_change(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    _prepare_fake_pipeline_state(default_pipeline)
    clear_active_process_key()

    current_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-a.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )
    next_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-b.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )

    default_pipeline.refresh_state = {
        "base_model_name": "model-a.safetensors",
        "loras": [],
        "base_model_additional_loras": [],
        "vae_name": "vae-a.safetensors",
        "clip_name": "clip-a.safetensors",
        "sdxl_policy": default_pipeline._policy_signature(current_policy),
        "sdxl_process_class": default_pipeline._sdxl_process_class(current_policy),
        "sdxl_process_key": default_pipeline._sdxl_process_key(
            base_model_name="model-a.safetensors",
            vae_name="vae-a.safetensors",
            clip_name="clip-a.safetensors",
            sdxl_policy=current_policy,
        ),
    }

    calls = []

    def fake_prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
        calls.append(
            {
                "current_model": current_model,
                "next_model": next_model,
                "notes": notes,
            }
        )
        if release_callback is not None:
            release_callback()
        return {"called": True}

    monkeypatch.setattr(default_pipeline.resources, "prepare_for_checkpoint_switch", fake_prepare_for_checkpoint_switch)
    monkeypatch.setattr(default_pipeline, "refresh_base_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(default_pipeline, "refresh_loras", lambda *args, **kwargs: None)
    monkeypatch.setattr(default_pipeline, "assert_model_integrity", lambda: True)
    monkeypatch.setattr(default_pipeline, "prepare_text_encoder", lambda async_call=True: None)
    monkeypatch.setattr(default_pipeline, "clear_all_caches", lambda: None)

    default_pipeline.refresh_everything(
        base_model_name="model-b.safetensors",
        loras=[],
        base_model_additional_loras=[],
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=next_policy,
    )

    assert calls, "expected a checkpoint-switch reset when the SDXL checkpoint changes"
    assert calls[0]["current_model"] == "model-a.safetensors"
    assert calls[0]["next_model"] == "model-b.safetensors"
    assert default_pipeline.refresh_state["sdxl_process_class"] == default_pipeline._sdxl_process_class(next_policy)
    assert default_pipeline.refresh_state["sdxl_process_key"].process_class == default_pipeline._sdxl_process_class(next_policy)
    assert get_active_process_key() == default_pipeline.refresh_state["sdxl_process_key"].normalized()


def test_refresh_everything_hard_resets_on_lora_stack_change(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    _prepare_fake_pipeline_state(default_pipeline)
    clear_active_process_key()

    current_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-a.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )

    default_pipeline.refresh_state = {
        "base_model_name": "model-a.safetensors",
        "loras": [("boost-a.safetensors", 0.75)],
        "base_model_additional_loras": [],
        "vae_name": "vae-a.safetensors",
        "clip_name": "clip-a.safetensors",
        "sdxl_policy": default_pipeline._policy_signature(current_policy),
        "sdxl_process_class": default_pipeline._sdxl_process_class(current_policy),
        "sdxl_process_key": default_pipeline._sdxl_process_key(
            base_model_name="model-a.safetensors",
            vae_name="vae-a.safetensors",
            clip_name="clip-a.safetensors",
            sdxl_policy=current_policy,
            loras=[("boost-a.safetensors", 0.75)],
        ),
    }

    calls = []

    def fake_prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
        calls.append(
            {
                "current_model": current_model,
                "next_model": next_model,
                "notes": notes,
            }
        )
        if release_callback is not None:
            release_callback()
        return {"called": True}

    monkeypatch.setattr(default_pipeline.resources, "prepare_for_checkpoint_switch", fake_prepare_for_checkpoint_switch)
    monkeypatch.setattr(default_pipeline, "refresh_base_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(default_pipeline, "refresh_loras", lambda *args, **kwargs: None)
    monkeypatch.setattr(default_pipeline, "assert_model_integrity", lambda: True)
    monkeypatch.setattr(default_pipeline, "prepare_text_encoder", lambda async_call=True: None)
    monkeypatch.setattr(default_pipeline, "clear_all_caches", lambda: None)

    default_pipeline.refresh_everything(
        base_model_name="model-a.safetensors",
        loras=[("boost-b.safetensors", 0.85)],
        base_model_additional_loras=[],
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=current_policy,
    )

    assert calls, "expected a checkpoint-switch reset when the active LoRA stack changes"
    assert calls[0]["current_model"] == "model-a.safetensors"
    assert calls[0]["next_model"] == "model-a.safetensors"
    assert default_pipeline.refresh_state["loras"] == [("boost-b.safetensors", 0.85)]
    assert get_active_process_key() == default_pipeline.refresh_state["sdxl_process_key"].normalized()


def test_refresh_everything_reuses_warm_sdxl_and_repairs_active_process_registry(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    _prepare_fake_pipeline_state(default_pipeline)
    clear_active_process_key()

    current_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-a.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )

    default_pipeline.refresh_state = {
        "base_model_name": "model-a.safetensors",
        "loras": [],
        "base_model_additional_loras": [],
        "vae_name": "vae-a.safetensors",
        "clip_name": "clip-a.safetensors",
        "sdxl_policy": default_pipeline._policy_signature(current_policy),
        "sdxl_process_class": default_pipeline._sdxl_process_class(current_policy),
        "sdxl_process_key": default_pipeline._sdxl_process_key(
            base_model_name="model-a.safetensors",
            vae_name="vae-a.safetensors",
            clip_name="clip-a.safetensors",
            sdxl_policy=current_policy,
        ),
    }

    default_pipeline.refresh_everything(
        base_model_name="model-a.safetensors",
        loras=[],
        base_model_additional_loras=[],
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=current_policy,
    )

    assert get_active_process_key() == default_pipeline.refresh_state["sdxl_process_key"].normalized()


def test_prepare_text_encoder_forces_full_clip_activation(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)
    default_pipeline.assert_model_integrity = lambda: True
    default_pipeline.final_clip = types.SimpleNamespace(patcher=object())

    captured = {}

    def fake_prepare_models_for_stage(models, **kwargs):
        captured["models"] = models
        captured["kwargs"] = kwargs

    monkeypatch.setattr(default_pipeline.resources, "prepare_models_for_stage", fake_prepare_models_for_stage)

    default_pipeline.prepare_text_encoder(async_call=False)

    assert captured["models"] == [default_pipeline.final_clip.patcher]
    assert captured["kwargs"]["stage_name"] == "text_encode"
    assert captured["kwargs"]["target_phase"] == default_pipeline.resources.MemoryPhase.PROMPT_ENCODE
    assert captured["kwargs"]["force_full_load"] is True

