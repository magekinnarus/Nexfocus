import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from backend.sdxl_assembly.request_builder import _resolve_lora_channel_weights
from backend.sdxl_assembly.lifecycle_coordinator import release_domains, _get_cn_cache_counts
from backend.sdxl_assembly.runtime_state import LifecycleDomain
from modules.async_worker import AsyncTask, handler
from modules.lora_channel_policy import build_explicit_lora_channel_overrides


def test_resolve_lora_channel_weights_requires_explicit_overrides_for_speed_presets():
    speed_lora = ("D:/loras/sdxl_lightning_4step_lora.safetensors", 0.8)

    unresolved = _resolve_lora_channel_weights([speed_lora], [], lora_channel_overrides=None)
    assert unresolved == ((speed_lora[0], 0.8, 0.8),)

    overrides = build_explicit_lora_channel_overrides([speed_lora])
    resolved = _resolve_lora_channel_weights(
        [speed_lora],
        [],
        lora_channel_overrides=overrides,
    )
    assert resolved == ((speed_lora[0], 0.8, 0.0),)


def test_resolve_lora_channel_weights_zeroes_presets_with_explicit_overrides():
    normal_lora = ("my_custom_lora.safetensors", 0.7)
    lcm_lora = ("sdxl_lcm_lora.safetensors", 0.5)
    lightning_lora = ("sdxl_lightning_4step_lora.safetensors", 0.8)

    input_loras = [normal_lora, lcm_lora, lightning_lora]
    additional_loras = [("sdxl_inpaint_lora.safetensors", 1.0)]

    resolved = _resolve_lora_channel_weights(
        input_loras,
        additional_loras,
        lora_channel_overrides=build_explicit_lora_channel_overrides(input_loras),
    )

    assert len(resolved) == 4
    assert resolved[0][0] == normal_lora[0]
    assert resolved[0][1] == 0.7
    assert resolved[0][2] == 0.7
    assert resolved[1][0] == lcm_lora[0]
    assert resolved[1][1] == 0.5
    assert resolved[1][2] == 0.0
    assert resolved[2][0] == lightning_lora[0]
    assert resolved[2][1] == 0.8
    assert resolved[2][2] == 0.0
    assert resolved[3][0] == "sdxl_inpaint_lora.safetensors"
    assert resolved[3][1] == 1.0
    assert resolved[3][2] == 0.0


def test_release_domains_clears_compatibility_cn_caches_including_outer_structural_cache(monkeypatch):
    import backend.controlnet_registry as controlnet_registry
    import backend.ip_adapter as ip_adapter
    import backend.preprocessors.runtime as preprocessor_runtime
    import backend.pulid_runtime as pulid_runtime
    import modules.pipeline.image_input as image_input
    from backend.sdxl_assembly.stream_ctx_cn_worker import StreamingContextualControlWorker
    from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker
    from backend.sdxl_assembly.stream_st_preprocess_worker import StreamingStructuralPreprocessWorker

    StreamingStructuralPreprocessWorker._PREPROCESS_CACHE.clear()
    StreamingStructuralPreprocessWorker._PREPROCESS_CACHE["hint"] = MagicMock()
    StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE.clear()
    StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE["support"] = MagicMock()
    StreamingContextualControlWorker._PAYLOAD_CACHE.clear()
    StreamingContextualControlWorker._PAYLOAD_CACHE["payload"] = MagicMock()
    StreamingContextualControlWorker._CONTEXTUAL_MODELS.clear()
    StreamingContextualControlWorker._CONTEXTUAL_MODELS["context"] = {"model": MagicMock(), "ip_layers": MagicMock()}
    StreamingContextualControlWorker._CLIP_VISION_MODELS.clear()
    StreamingContextualControlWorker._CLIP_VISION_MODELS["clip"] = MagicMock()
    StreamingContextualControlWorker._IP_NEGATIVES.clear()
    StreamingContextualControlWorker._IP_NEGATIVES["neg"] = MagicMock()
    StreamingContextualControlWorker._EVA_CLIP_MODELS.clear()
    StreamingContextualControlWorker._EVA_CLIP_MODELS["eva"] = MagicMock()
    StreamingContextualControlWorker._FACE_PARSERS.clear()
    StreamingContextualControlWorker._FACE_PARSERS["face"] = MagicMock()
    StreamingContextualControlWorker._INSIGHTFACE_APPS.clear()
    StreamingContextualControlWorker._INSIGHTFACE_APPS["insight"] = MagicMock()

    image_input._STRUCTURAL_PREPROCESS_CACHE.clear()
    image_input._STRUCTURAL_PREPROCESS_CACHE[("cache",)] = MagicMock()

    monkeypatch.setattr(
        preprocessor_runtime,
        "_MODEL_CACHE",
        {"Depth": {"path": "dummy", "model": MagicMock()}},
        raising=False,
    )
    monkeypatch.setattr(
        controlnet_registry,
        "_LOADED_CONTROLNETS",
        {"controlnet": MagicMock()},
        raising=False,
    )
    monkeypatch.setattr(
        ip_adapter,
        "contextual_models",
        {"ctx": {"model": MagicMock(), "ip_layers": MagicMock()}},
        raising=False,
    )
    monkeypatch.setattr(
        ip_adapter,
        "_CONTEXTUAL_PAYLOAD_CACHE",
        OrderedDict({("payload",): ([MagicMock()], [MagicMock()])}),
        raising=False,
    )
    monkeypatch.setattr(
        ip_adapter,
        "clip_vision_models",
        {"clip": MagicMock()},
        raising=False,
    )
    monkeypatch.setattr(
        ip_adapter,
        "ip_negative",
        {"neg": MagicMock()},
        raising=False,
    )
    monkeypatch.setattr(
        ip_adapter,
        "insightface_apps",
        {"insight": MagicMock()},
        raising=False,
    )
    monkeypatch.setattr(
        pulid_runtime,
        "eva_clip_models",
        {"eva": MagicMock()},
        raising=False,
    )
    monkeypatch.setattr(
        pulid_runtime,
        "face_parsers",
        {"face": MagicMock()},
        raising=False,
    )

    counts_st = _get_cn_cache_counts(LifecycleDomain.STRUCTURAL_CN)
    assert counts_st["compat_outer_structural_preprocess"] == 1
    assert counts_st["compat_preprocessors"] == 1
    assert counts_st["compat_controlnets"] == 1

    counts_ctx = _get_cn_cache_counts(LifecycleDomain.CONTEXTUAL_CN)
    assert counts_ctx["compat_ctx_models"] == 1
    assert counts_ctx["compat_ctx_payload"] == 1
    assert counts_ctx["compat_pulid_models"] == 2

    res = release_domains(
        (LifecycleDomain.STRUCTURAL_CN, LifecycleDomain.CONTEXTUAL_CN),
        reason="test_run"
    )
    assert res.ok
    assert len(image_input._STRUCTURAL_PREPROCESS_CACHE) == 0
    assert len(preprocessor_runtime._MODEL_CACHE["Depth"]["path"] or "") == 0 if preprocessor_runtime._MODEL_CACHE["Depth"]["path"] is not None else True
    assert preprocessor_runtime._MODEL_CACHE["Depth"]["model"] is None
    assert controlnet_registry._LOADED_CONTROLNETS == {}
    assert ip_adapter.contextual_models == {}
    assert len(ip_adapter._CONTEXTUAL_PAYLOAD_CACHE) == 0
    assert ip_adapter.clip_vision_models == {}
    assert ip_adapter.ip_negative == {}
    assert ip_adapter.insightface_apps == {}
    assert pulid_runtime.eva_clip_models == {}
    assert pulid_runtime.face_parsers == {}


def test_handler_serializes_manual_controlnet_cache_release(monkeypatch):
    import backend.sdxl_assembly.lifecycle_coordinator as lifecycle_coordinator

    recorded = {}

    def fake_release_domains(domains, *, reason=None, **_kwargs):
        recorded["domains"] = tuple(domains)
        recorded["reason"] = reason
        return SimpleNamespace(errors=(), ok=True)

    monkeypatch.setattr(lifecycle_coordinator, "release_domains", fake_release_domains)

    task = AsyncTask(args={})
    task.is_valid = True
    task.is_utility = True
    task.utility_action = "release_controlnet_cache"

    handler(task)

    assert recorded["domains"] == (
        LifecycleDomain.STRUCTURAL_CN,
        LifecycleDomain.CONTEXTUAL_CN,
    )
    assert recorded["reason"] == "manual_release"
    assert task.state.processing is False
    assert task.state.current_status_text == "ControlNet Caches Released."
    assert task.state.yields[0][0] == "preview"
    assert task.state.yields[-1][0] == "preview"
