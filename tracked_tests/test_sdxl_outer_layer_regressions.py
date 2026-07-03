from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


_original_args_manager = sys.modules.get("args_manager")
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

from modules.pipeline.inference import get_sampling_callback
from backend.process_transition import (
    PROCESS_CLASS_FLUX_FILL,
    PROCESS_CLASS_STANDARD_SDXL,
    PROCESS_FAMILY_FLUX_FILL,
    PROCESS_FAMILY_SDXL,
    build_process_key,
    clear_active_process_key,
    set_active_process_key,
)


@pytest.fixture(scope="module", autouse=True)
def _mock_args_manager_lifecycle():
    yield
    if _original_args_manager is None:
        sys.modules.pop("args_manager", None)
    else:
        sys.modules["args_manager"] = _original_args_manager


def _load_default_pipeline(monkeypatch, *, profile_name="colab_free"):
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

    import modules.default_pipeline as default_pipeline

    return importlib.reload(default_pipeline)


def test_sampling_callback_throttles_text_only_preview_events_when_preview_is_off():
    task_state = SimpleNamespace(yields=[], inpaint_context=None, current_progress=0, current_status_text='')
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 20)

    for step in range(20):
        callback(step, None, None, 20, None)

    assert [item[1][0] for item in task_state.yields] == [5, 25, 50, 75, 100]
    assert all(item[0] == 'preview' for item in task_state.yields)
    assert all(item[1][2] is None for item in task_state.yields)


def test_sampling_callback_progress_scales_correctly_when_invoked_sparsely():
    task_state = SimpleNamespace(
        yields=[],
        inpaint_context=None,
        current_progress=0,
        current_status_text='',
        callback_steps=0,
    )
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 20)

    callback(4, None, None, 20, None)
    callback(9, None, None, 20, None)
    callback(19, None, None, 20, None)

    assert [item[1][0] for item in task_state.yields] == [25, 50, 100]
    assert task_state.current_progress == 100


def test_release_sdxl_runtime_state_clears_greenfield_assembly_caches(monkeypatch):
    default_pipeline = _load_default_pipeline(monkeypatch)

    from backend import sdxl_unified_runtime
    from backend import sdxl_runtime_policy

    cache_clear_calls = []
    assembly_clear_calls = []

    monkeypatch.setattr(
        default_pipeline.resources,
        "prepare_for_checkpoint_switch",
        lambda **kwargs: kwargs["release_callback"]() if kwargs.get("release_callback") else None,
    )
    monkeypatch.setattr(
        sdxl_unified_runtime,
        "clear_unified_sdxl_runtime_component_cache",
        lambda teardown=False: cache_clear_calls.append(teardown),
    )
    monkeypatch.setattr(default_pipeline, "clear_all_caches", lambda: None)
    monkeypatch.setattr(
        "backend.sdxl_assembly.clear_all_caches",
        lambda **kwargs: assembly_clear_calls.append(kwargs.get("reason")),
    )

    current_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture="sdxl",
        base_model_name="model-a.safetensors",
        profile=types.SimpleNamespace(name="colab_free", total_vram_mb=16384.0),
        requested_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )
    current_key = default_pipeline._sdxl_process_key(
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_name="clip-a.safetensors",
        sdxl_policy=current_policy,
    )

    result = default_pipeline.release_sdxl_runtime_state(
        current_process_key=current_key,
        next_process_key=current_key,
        current_model_name="model-a.safetensors",
        next_model_name="model-a.safetensors",
        reason="tracked_test",
        hard_reset=True,
    )

    assert result["released"] is True
    assert cache_clear_calls == [False]
    assert assembly_clear_calls == ["tracked_test"]


def test_assembly_progress_callback_throttles_raw_text_only_callback():
    from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback

    forwarded_steps = []

    def raw_callback(step, _x0, _x, total_steps, _y):
        forwarded_steps.append((step, total_steps))

    raw_callback._sdxl_forward_text_only = True

    callback = SDXLAssemblyProgressCallback(SimpleNamespace(), raw_callback)
    for step in range(6):
        callback(step, None, None, 6, None)

    assert forwarded_steps == [(0, 6), (4, 6), (5, 6)]


def test_process_transition_checkpoint_release_clears_greenfield_assembly_caches(monkeypatch):
    from backend import process_transition

    clear_active_process_key()

    release_calls = []
    monkeypatch.setattr(
        "backend.sdxl_assembly.clear_all_caches",
        lambda **kwargs: release_calls.append(kwargs.get("reason")),
    )
    monkeypatch.setattr(
        "backend.resources.prepare_for_checkpoint_switch",
        lambda **kwargs: kwargs["release_callback"]() if kwargs.get("release_callback") else None,
    )
    monkeypatch.setattr(
        "backend.sdxl_unified_runtime.clear_unified_sdxl_runtime_component_cache",
        lambda teardown=False: None,
    )

    current_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )
    requested_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(("ae", "ae.safetensors"), ("conditioning", "empty.pt"), ("unet", "unet.safetensors")),
    )

    set_active_process_key(current_key)
    decision = process_transition.apply_process_transition_gate(requested_key)

    assert decision is not None
    assert decision.reset_required is True
    assert release_calls
    assert all(reason == "route_transition" for reason in release_calls)
