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
from backend.sdxl_assembly.lifecycle_coordinator import (
    LifecycleChange,
    release_domains,
    _get_cn_cache_counts,
)
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


def test_inspect_lora_asset_classification(monkeypatch):
    from modules.lora_channel_policy import inspect_lora_asset
    from backend.sdxl_assembly.contracts import ResolvedFileIdentity

    mock_headers = {
        "unet_only.safetensors": {
            "lora_unet_input_blocks_0_0.lora_down.weight": object(),
            "lora_unet_input_blocks_0_0.lora_up.weight": object(),
        },
        "clip_l_only.safetensors": {
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": object(),
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": object(),
        },
        "clip_g_only.safetensors": {
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": object(),
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": object(),
        },
        "dual_target.safetensors": {
            "lora_unet_input_blocks_0_0.lora_down.weight": object(),
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": object(),
        },
        "diffusers_unet.safetensors": {
            "unet.down_blocks.0.resnets.0.conv1.lora_down.weight": object(),
        },
        "diffusers_unet_unprefixed.safetensors": {
            "down_blocks.0.resnets.0.conv1.lora_down.weight": object(),
        },
        "generic_text.safetensors": {
            "text_encoders.clip_l.transformer.encoder.layers.0.mlp_fc1.lora_down.weight": object(),
        },
        "generic_text_numbered.safetensors": {
            "lora_te3_transformer_encoder_layers_0_mlp_fc1.lora_down.weight": object(),
        },
        "t5_only.safetensors": {
            "t5xxl.transformer.encoder.layers.0.self_attn.q.lora_down.weight": object(),
        },
        "unrecognized.safetensors": {
            "some_weird_unrecognized_key.lora_down.weight": object(),
        },
    }

    monkeypatch.setattr(
        "modules.lora_channel_policy.SafeOpenHeaderOnly",
        lambda path: mock_headers[os.path.basename(path)],
    )

    def _identity(name):
        return ResolvedFileIdentity(Path(name), f"sha_{name}", 100, 100)

    # 1. UNet-only
    ev = inspect_lora_asset(_identity("unet_only.safetensors"))
    assert ev.status == "recognized"
    assert ev.unet_count == 2
    assert ev.clip_l_count == 0
    assert ev.clip_g_count == 0

    # 2. CLIP-L only
    ev = inspect_lora_asset(_identity("clip_l_only.safetensors"))
    assert ev.status == "recognized"
    assert ev.unet_count == 0
    assert ev.clip_l_count == 2

    # 3. CLIP-G only
    ev = inspect_lora_asset(_identity("clip_g_only.safetensors"))
    assert ev.status == "recognized"
    assert ev.clip_g_count == 2

    # 4. Dual-target
    ev = inspect_lora_asset(_identity("dual_target.safetensors"))
    assert ev.status == "recognized"
    assert ev.unet_count == 1
    assert ev.clip_l_count == 1

    # 5. Diffusers
    ev = inspect_lora_asset(_identity("diffusers_unet.safetensors"))
    assert ev.status == "recognized"
    assert ev.unet_count == 1

    # 6. Unprefixed Diffusers UNet
    ev = inspect_lora_asset(_identity("diffusers_unet_unprefixed.safetensors"))
    assert ev.status == "recognized"
    assert ev.unet_count == 1

    # 7. Generic text encoder forms
    ev = inspect_lora_asset(_identity("generic_text.safetensors"))
    assert ev.status == "recognized"
    assert ev.generic_text_count == 1
    ev = inspect_lora_asset(_identity("generic_text_numbered.safetensors"))
    assert ev.status == "recognized"
    assert ev.generic_text_count == 1

    # 8. Generic Text/T5
    ev = inspect_lora_asset(_identity("t5_only.safetensors"))
    assert ev.status == "recognized"
    assert ev.generic_text_count == 1

    # 9. Unrecognized
    ev = inspect_lora_asset(_identity("unrecognized.safetensors"))
    assert ev.status == "unknown"
    assert ev.unrecognized_key_count == 1


def test_inspect_unknown_formats(monkeypatch):
    from modules.lora_channel_policy import inspect_lora_asset, resolve_lora_channels
    from backend.sdxl_assembly.contracts import ResolvedFileIdentity

    monkeypatch.setattr(os.path, "exists", lambda p: True)

    identity_bin = ResolvedFileIdentity(Path("lora.bin"), "sha_bin", 200, 200)
    ev = inspect_lora_asset(identity_bin)
    assert ev.status == "unknown"

    decision = resolve_lora_channels(
        file_identity=identity_bin,
        requested_unet_weight=0.7,
        requested_clip_weight=0.7,
        provenance="input",
    )
    assert decision.source == "conservative_default"
    assert decision.effective_unet_weight == 0.7
    assert decision.effective_clip_weight == 0.7
    assert decision.evidence_status == "unknown"


def test_resolve_lora_channels_precedence(monkeypatch):
    from modules.lora_channel_policy import resolve_lora_channels
    from backend.sdxl_assembly.contracts import ResolvedFileIdentity

    mock_headers = {
        "user_unet_only.safetensors": {
            "lora_unet_input_blocks_0_0.lora_down.weight": object(),
        },
        "user_dual.safetensors": {
            "lora_unet_input_blocks_0_0.lora_down.weight": object(),
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": object(),
        }
    }
    monkeypatch.setattr(
        "modules.lora_channel_policy.SafeOpenHeaderOnly",
        lambda path: mock_headers[os.path.basename(path)],
    )
    monkeypatch.setattr(os.path, "exists", lambda p: "user_" in str(p))

    identity_unet = ResolvedFileIdentity(Path("user_unet_only.safetensors"), "sha_unet", 123, 456)
    identity_dual = ResolvedFileIdentity(Path("user_dual.safetensors"), "sha_dual", 789, 101)

    # 1. Normal recognized UNet-only user LoRA freezes effective clip_weight=0
    dec = resolve_lora_channels(
        file_identity=identity_unet,
        requested_unet_weight=0.8,
        requested_clip_weight=0.8,
        provenance="input",
    )
    assert dec.source == "asset_evidence"
    assert dec.effective_unet_weight == 0.8
    assert dec.effective_clip_weight == 0.0
    assert "UNet-only" in dec.reason

    # 2. Normal recognized Dual user LoRA retains clip_weight
    dec = resolve_lora_channels(
        file_identity=identity_dual,
        requested_unet_weight=0.8,
        requested_clip_weight=0.8,
        provenance="input",
    )
    assert dec.source == "asset_evidence"
    assert dec.effective_unet_weight == 0.8
    assert dec.effective_clip_weight == 0.8
    assert "dual-target" in dec.reason

    # 3. Explicit provenance (additional) overrides asset evidence and remains UNet-only
    dec = resolve_lora_channels(
        file_identity=identity_dual,
        requested_unet_weight=1.0,
        requested_clip_weight=1.0,
        provenance="additional",
    )
    assert dec.source == "explicit"
    assert dec.effective_unet_weight == 1.0
    assert dec.effective_clip_weight == 0.0


def test_evidence_cache_behavior(monkeypatch):
    from modules.lora_channel_policy import inspect_lora_asset, _EVIDENCE_CACHE
    from backend.sdxl_assembly.contracts import ResolvedFileIdentity

    call_count = 0
    def mock_safe_open(path):
        nonlocal call_count
        call_count += 1
        return {"lora_unet_input_blocks_0_0.lora_down.weight": object()}

    monkeypatch.setattr(
        "modules.lora_channel_policy.SafeOpenHeaderOnly",
        mock_safe_open,
    )
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    _EVIDENCE_CACHE.clear()
    identity = ResolvedFileIdentity(Path("cache_test.safetensors"), "sha_cache", 111, 222)

    # First call: cache miss
    ev1 = inspect_lora_asset(identity)
    assert call_count == 1
    assert ev1.unet_count == 1

    # Second call: cache hit
    ev2 = inspect_lora_asset(identity)
    assert call_count == 1
    assert ev2.unet_count == 1

    # Call with different identity (modified mtime): cache miss
    identity_mod = ResolvedFileIdentity(Path("cache_test.safetensors"), "sha_cache", 111, 333)
    ev3 = inspect_lora_asset(identity_mod)
    assert call_count == 2


def test_signatures_and_prompt_cache_effective_only():
    from backend.sdxl_assembly.runtime_state import _clip_lora_signature, _unet_lora_signature
    from backend.sdxl_assembly.contracts import SDXLLoraSpec, ResolvedFileIdentity

    id_dual = ResolvedFileIdentity(Path("dual.safetensors"), "sha_dual", 100, 100)
    spec_dual = SDXLLoraSpec(
        file_identity=id_dual,
        unet_weight=0.8,
        clip_weight=0.8,
        requested_unet_weight=0.8,
        requested_clip_weight=0.8,
        decision_source="asset_evidence",
        decision_reason="dual-target",
    )

    id_unet = ResolvedFileIdentity(Path("unet_only.safetensors"), "sha_unet", 200, 200)
    spec_unet = SDXLLoraSpec(
        file_identity=id_unet,
        unet_weight=0.5,
        clip_weight=0.0,
        requested_unet_weight=0.5,
        requested_clip_weight=0.5,
        decision_source="asset_evidence",
        decision_reason="unet-only",
    )

    from types import SimpleNamespace
    req_dual_only = SimpleNamespace(lora_specs=(spec_dual,))
    req_both = SimpleNamespace(lora_specs=(spec_dual, spec_unet))

    # UNet signatures should differ
    assert _unet_lora_signature(req_dual_only) != _unet_lora_signature(req_both)

    # CLIP signatures should be identical (due to effective clip_weight=0 on spec_unet)
    assert _clip_lora_signature(req_dual_only) == _clip_lora_signature(req_both)

    # Removing last effective CLIP LoRA
    req_unet_only = SimpleNamespace(lora_specs=(spec_unet,))
    assert _clip_lora_signature(req_unet_only) == ()


def test_additional_lora_telemetry_uses_frozen_provenance():
    from backend.sdxl_assembly.contracts import SDXLLoraSpec, ResolvedFileIdentity
    from backend.sdxl_assembly.gateway import _summarize_additional_unet_only_loras

    user_unet_only = SDXLLoraSpec(
        file_identity=ResolvedFileIdentity(Path("stabilizer.safetensors"), "user", 1, 1),
        unet_weight=0.2,
        clip_weight=0.0,
        provenance="input",
    )
    additional_patch = SDXLLoraSpec(
        file_identity=ResolvedFileIdentity(Path("inpaint.patch"), "additional", 1, 1),
        unet_weight=1.0,
        clip_weight=0.0,
        provenance="additional",
    )

    assert _summarize_additional_unet_only_loras((user_unet_only, additional_patch)) == [
        "inpaint.patch@1",
    ]


def test_gateway_lora_transition_only_releases_text_for_effective_clip_change():
    from backend.sdxl_assembly.contracts import SDXLLoraSpec, ResolvedFileIdentity
    from backend.sdxl_assembly.gateway import _build_gateway_request_state, _calculate_gateway_changes

    checkpoint = SimpleNamespace(sha256="checkpoint")
    base = SimpleNamespace(
        checkpoint=checkpoint,
        vae=None,
        unet_posture=SimpleNamespace(value="streaming"),
        clip_posture=SimpleNamespace(value="cpu_resident"),
        vae_posture=SimpleNamespace(value="transient"),
        lora_posture=SimpleNamespace(value="streaming"),
        lora_stack_hash="base",
        prompt_payload_hash="prompt",
        spatial_context=None,
        structural_controls=(),
        contextual_controls=(),
        lora_specs=(),
    )
    unet_spec = SDXLLoraSpec(
        file_identity=ResolvedFileIdentity(Path("unet.safetensors"), "sha_unet", 1, 1),
        unet_weight=0.2,
        clip_weight=0.0,
        provenance="input",
    )
    clip_spec = SDXLLoraSpec(
        file_identity=ResolvedFileIdentity(Path("clip.safetensors"), "sha_clip", 1, 1),
        unet_weight=0.0,
        clip_weight=0.2,
        provenance="input",
    )

    with_unet = SimpleNamespace(**{**base.__dict__, "lora_specs": (unet_spec,), "lora_stack_hash": "unet"})
    with_clip = SimpleNamespace(**{**with_unet.__dict__, "lora_specs": (unet_spec, clip_spec), "lora_stack_hash": "dual"})

    unet_changes = _calculate_gateway_changes(
        _build_gateway_request_state(base),
        _build_gateway_request_state(with_unet),
    )
    assert LifecycleChange.LORA_STACK_CHANGE not in unet_changes

    clip_changes = _calculate_gateway_changes(
        _build_gateway_request_state(with_unet),
        _build_gateway_request_state(with_clip),
    )
    assert LifecycleChange.LORA_STACK_CHANGE in clip_changes
