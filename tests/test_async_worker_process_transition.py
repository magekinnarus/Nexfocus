from __future__ import annotations

import sys
import types

from backend import sdxl_runtime_policy, process_transition
from backend.process_transition import (
    PROCESS_CLASS_FLUX_FILL,
    PROCESS_CLASS_STANDARD_SDXL,
    PROCESS_FAMILY_FLUX_FILL,
    PROCESS_FAMILY_SDXL,
    build_process_key,
    clear_active_process_key,
    get_active_process_key,
    set_active_process_key,
)


def _load_async_worker(monkeypatch):
    sys.modules.pop("modules.async_worker", None)
    sys.modules.pop("modules.objr_engine", None)
    sys.modules.pop("modules.private_logger", None)

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

    fake_args = types.SimpleNamespace(
        colab=False,
        preset="",
        output_path="",
        temp_path="",
        skip_model_load=True,
    )
    fake_args_manager = types.ModuleType("args_manager")
    fake_args_manager.args = fake_args
    fake_args_manager.args_parser = types.SimpleNamespace(args=fake_args, parser=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "args_manager", fake_args_manager)

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

    fake_private_logger = types.ModuleType("modules.private_logger")
    fake_private_logger.log = lambda *args, **kwargs: None
    fake_private_logger.get_current_html_path = lambda output_format=None: "history.html"
    monkeypatch.setitem(sys.modules, "modules.private_logger", fake_private_logger)

    import modules.async_worker as async_worker

    return async_worker


def test_apply_process_transition_gate_releases_sdxl_before_flux(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    release_calls = []

    import modules.default_pipeline as default_pipeline

    monkeypatch.setattr(
        default_pipeline,
        "release_sdxl_runtime_state",
        lambda **kwargs: release_calls.append(kwargs) or {"released": True},
    )

    current_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )
    requested_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(("ae", "/tmp/ae.safetensors"), ("conditioning", "/tmp/empty.pt"), ("unet", "/tmp/unet.safetensors")),
    )

    set_active_process_key(current_key)
    decision = process_transition.apply_process_transition_gate(requested_key)

    assert decision is not None
    assert decision.reset_required is True
    assert decision.reason == "family_change"
    assert release_calls and release_calls[0]["hard_reset"] is False
    assert get_active_process_key() is None


def test_apply_process_transition_gate_clears_archived_flux_before_sdxl(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    prepare_calls = []
    release_calls = []

    def fake_prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
        prepare_calls.append(
            {
                "current_model": current_model,
                "next_model": next_model,
                "notes": notes,
            }
        )
        assert release_callback is not None
        release_callback()
        return {"released": True}

    monkeypatch.setattr(
        "backend.resources.prepare_for_checkpoint_switch",
        fake_prepare_for_checkpoint_switch,
    )
    monkeypatch.setattr(
        "backend.flux_fill_v3.release_active_flux_resident_spine",
        lambda **kwargs: release_calls.append(("spine", kwargs)) or True,
    )
    monkeypatch.setattr(
        "backend.flux_fill_v3.release_flux_latent_artifacts",
        lambda: release_calls.append(("artifacts", None)) or True,
    )

    current_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(("ae", "/tmp/ae.safetensors"), ("conditioning", "/tmp/empty.pt"), ("unet", "/tmp/unet.safetensors")),
    )
    requested_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-b.safetensors", "vae-b.safetensors", "clip-b.safetensors"),
    )

    set_active_process_key(current_key)
    decision = process_transition.apply_process_transition_gate(requested_key)

    assert decision is not None
    assert decision.reset_required is True
    assert decision.reason == "family_change"
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["current_model"] == PROCESS_FAMILY_FLUX_FILL
    assert prepare_calls[0]["next_model"] == "model-b.safetensors"
    assert prepare_calls[0]["notes"]["reason"] == "route_transition"
    assert release_calls == [
        ("spine", {"reason": "route_transition"}),
        ("artifacts", None),
    ]
    assert get_active_process_key() is None


def test_resolve_flux_fill_process_key_uses_greenfield_asset_identity(monkeypatch):
    _load_async_worker(monkeypatch)
    from modules import model_registry

    monkeypatch.setattr(model_registry, "ensure_asset", model_registry.resolve_asset_path)

    task_state = types.SimpleNamespace(
        objr_engine="flux fill",
        inpaint_route="flux",
        flux_fill_conditioning="empty",
    )

    key = process_transition.resolve_flux_fill_process_key(task_state, route_family="flux_fill")

    assert key is not None
    assert key.family == PROCESS_FAMILY_FLUX_FILL
    assert ("ae_path", model_registry.resolve_asset_path("inpaint.flux_fill.ae")) in key.authoritative_identity
    assert (
        "conditioning_cache_path",
        model_registry.resolve_asset_path("inpaint.flux_fill.empty_conditioning"),
    ) in key.authoritative_identity
    assert ("unet_spine", "streaming") in key.authoritative_identity
    assert task_state.flux_fill_unet_path != ""
    assert task_state.flux_fill_ae_path != ""
    assert task_state.flux_fill_conditioning_cache_path != ""



def test_resolve_requested_process_key_does_not_fallback_to_sdxl_for_flux_route(monkeypatch):
    _load_async_worker(monkeypatch)

    sdxl_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )

    monkeypatch.setattr(process_transition, "resolve_flux_fill_process_key", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(process_transition, "resolve_sdxl_process_key", lambda *_args, **_kwargs: sdxl_key)

    task_state = types.SimpleNamespace(
        objr_engine=None,
        sdxl_execution_policy=types.SimpleNamespace(enabled=True),
    )
    route = types.SimpleNamespace(family="flux_fill")

    assert process_transition.resolve_requested_process_key(task_state, route) is None


def test_resolve_requested_process_key_returns_none_for_plain_upscale_without_active_major_family(monkeypatch):
    _load_async_worker(monkeypatch)

    sdxl_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )

    monkeypatch.setattr(process_transition, "resolve_sdxl_process_key", lambda *_args, **_kwargs: sdxl_key)

    task_state = types.SimpleNamespace(
        objr_engine=None,
        sdxl_execution_policy=types.SimpleNamespace(enabled=True),
    )
    route = types.SimpleNamespace(family="upscale", route_id="upscale")

    assert process_transition.resolve_requested_process_key(task_state, route) is None


def test_resolve_requested_process_key_preserves_active_major_family_for_plain_upscale(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    active_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(("ae", "/tmp/ae.safetensors"), ("conditioning", "/tmp/empty.pt"), ("unet", "/tmp/unet.safetensors")),
    )
    set_active_process_key(active_key)

    task_state = types.SimpleNamespace(
        objr_engine=None,
        sdxl_execution_policy=types.SimpleNamespace(enabled=True),
    )
    route = types.SimpleNamespace(family="upscale", route_id="upscale")

    try:
        assert process_transition.resolve_requested_process_key(task_state, route) == active_key.normalized()
    finally:
        clear_active_process_key()


def test_resolve_requested_process_key_keeps_sdxl_identity_for_color_and_super_upscale(monkeypatch):
    _load_async_worker(monkeypatch)

    sdxl_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )

    monkeypatch.setattr(process_transition, "resolve_sdxl_process_key", lambda *_args, **_kwargs: sdxl_key)

    task_state = types.SimpleNamespace(
        objr_engine=None,
        sdxl_execution_policy=types.SimpleNamespace(enabled=True),
    )

    color_route = types.SimpleNamespace(family="upscale", route_id="color_enhanced_upscale")
    super_route = types.SimpleNamespace(family="upscale", route_id="super_upscale")

    assert process_transition.resolve_requested_process_key(task_state, color_route) == sdxl_key
    assert process_transition.resolve_requested_process_key(task_state, super_route) == sdxl_key


def test_resolve_requested_process_key_preserves_active_major_family_for_auxiliary_removal(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    active_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )
    set_active_process_key(active_key)

    task_state = types.SimpleNamespace(
        objr_engine="mat",
        sdxl_execution_policy=types.SimpleNamespace(enabled=True),
    )
    route = types.SimpleNamespace(family="removal", route_id="removal")

    try:
        assert process_transition.resolve_requested_process_key(task_state, route) == active_key.normalized()
    finally:
        clear_active_process_key()


def test_resolve_sdxl_process_key_uses_standard_route_family(monkeypatch):
    _load_async_worker(monkeypatch)

    task_state = types.SimpleNamespace(
        base_model_name="model-b.safetensors",
        vae_name="vae-b.safetensors",
        clip_model_name="clip-b.safetensors",
        sdxl_execution_policy=sdxl_runtime_policy.SDXLExecutionPolicy(
            enabled=True,
            architecture="sdxl",
            runtime_family="unified_sdxl",
            execution_mode="resident",
            hardware_tier="NORMAL_VRAM",
        ),
    )

    key = process_transition.resolve_sdxl_process_key(task_state)

    assert key is not None
    assert key.family == PROCESS_FAMILY_SDXL
    assert key.process_class == PROCESS_CLASS_STANDARD_SDXL
    assert key.route_family == "sdxl"


def test_resolve_sdxl_process_key_keeps_standard_route_family(monkeypatch):
    _load_async_worker(monkeypatch)

    task_state = types.SimpleNamespace(
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_model_name="clip-a.safetensors",
        sdxl_execution_policy=sdxl_runtime_policy.SDXLExecutionPolicy(
            enabled=True,
            architecture="sdxl",
            runtime_family="unified_sdxl",
            execution_mode="resident",
            hardware_tier="NORMAL_VRAM",
        ),
    )

    key = process_transition.resolve_sdxl_process_key(task_state)

    assert key is not None
    assert key.family == PROCESS_FAMILY_SDXL
    assert key.process_class == PROCESS_CLASS_STANDARD_SDXL
    assert key.route_family == "sdxl"

    alt_task_state = types.SimpleNamespace(
        base_model_name="model-a.safetensors",
        vae_name="vae-b.safetensors",
        clip_model_name="clip-a.safetensors",
        sdxl_execution_policy=task_state.sdxl_execution_policy,
    )
    alt_key = process_transition.resolve_sdxl_process_key(alt_task_state)

    assert alt_key is not None
    assert alt_key.normalized() == key.normalized()


def test_sync_route_process_activation_clears_registry_for_flux_route(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()
    route = types.SimpleNamespace(family="flux_fill", route_id="flux_inpaint")
    task_state = types.SimpleNamespace()
    stale_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )

    set_active_process_key(stale_key)
    result = process_transition.sync_route_process_activation(route, task_state, None)

    assert result is None
    assert get_active_process_key() is None


def test_sync_route_process_activation_preserves_active_registry_for_auxiliary_only_route(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    active_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(("ae", "/tmp/ae.safetensors"), ("conditioning", "/tmp/empty.pt"), ("unet", "/tmp/unet.safetensors")),
    )
    set_active_process_key(active_key)

    route = types.SimpleNamespace(family="upscale", route_id="upscale")
    task_state = types.SimpleNamespace(objr_engine=None, sdxl_execution_policy=types.SimpleNamespace(enabled=True))

    try:
        result = process_transition.sync_route_process_activation(route, task_state, active_key)
        assert result is None
        assert get_active_process_key() == active_key.normalized()
    finally:
        clear_active_process_key()


def test_apply_process_transition_gate_releases_greenfield_flux_runtime_on_none_request(monkeypatch):
    _load_async_worker(monkeypatch)
    clear_active_process_key()

    prepare_calls = []
    release_calls = []

    def fake_prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
        prepare_calls.append(
            {
                "current_model": current_model,
                "next_model": next_model,
                "notes": notes,
            }
        )
        assert release_callback is not None
        release_callback()
        return {"released": True}

    current_key = build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(
            ("ae_path", "ae.safetensors"),
            ("conditioning_cache_path", "empty.pt"),
            ("model_variant", "flux_fill_fp8"),
            ("unet_path", "unet.safetensors"),
            ("unet_spine", "resident"),
        ),
        route_family="flux_fill",
    )
    set_active_process_key(current_key)
    monkeypatch.setattr(
        "backend.resources.prepare_for_checkpoint_switch",
        fake_prepare_for_checkpoint_switch,
    )
    monkeypatch.setattr(
        "backend.flux_fill_v3.release_active_flux_resident_spine",
        lambda **kwargs: release_calls.append(("spine", kwargs)) or True,
    )
    monkeypatch.setattr(
        "backend.flux_fill_v3.release_flux_latent_artifacts",
        lambda: release_calls.append(("artifacts", None)) or True,
    )

    decision = process_transition.apply_process_transition_gate(None)

    assert decision is None
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["current_model"] == "unet.safetensors"
    assert prepare_calls[0]["next_model"] is None
    assert prepare_calls[0]["notes"]["reason"] == "route_transition"
    assert release_calls == [
        ("spine", {"reason": "route_transition"}),
        ("artifacts", None),
    ]
    assert get_active_process_key() is None



def test_resolve_preflight_additional_loras_inpaint(monkeypatch):
    _load_async_worker(monkeypatch)

    import modules.config as config
    monkeypatch.setattr(config, "downloading_inpaint_models", lambda engine: "fake_inpaint_patch.safetensors")

    task_state = types.SimpleNamespace(
        input_image_checkbox=True,
        current_tab="inpaint",
        inpaint_input_image=object(),
        inpaint_engine="v2.6",
        inpaint_route="sdxl",
        base_model_name="model-a.safetensors",
        vae_name="vae-a.safetensors",
        clip_model_name="clip-a.safetensors",
        sdxl_execution_policy=sdxl_runtime_policy.SDXLExecutionPolicy(
            enabled=True,
            architecture="sdxl",
            runtime_family="unified_sdxl",
            execution_mode="resident",
            hardware_tier="NORMAL_VRAM",
        ),
    )

    additional_loras = process_transition.resolve_preflight_additional_loras(task_state)
    assert additional_loras == [("fake_inpaint_patch.safetensors", 1.0)]

    task_state.base_model_additional_loras = additional_loras
    key = process_transition.resolve_sdxl_process_key(task_state)
    assert key is not None
    assert "fake_inpaint_patch.safetensors" in str(key.authoritative_identity)


def test_resolve_preflight_additional_loras_ignores_stale_inpaint_mix_without_controlnet_tasks(monkeypatch):
    _load_async_worker(monkeypatch)

    import modules.config as config
    monkeypatch.setattr(config, "downloading_inpaint_models", lambda engine: "fake_inpaint_patch.safetensors")

    task_state = types.SimpleNamespace(
        input_image_checkbox=True,
        current_tab="ip",
        inpaint_input_image=object(),
        inpaint_mask_image=object(),
        inpaint_context_mask_image=None,
        inpaint_bb_image=None,
        inpaint_engine="v2.6",
        inpaint_route="sdxl",
        mixing_image_prompt_and_inpaint=True,
        mixing_image_prompt_and_outpaint=False,
        outpaint_input_image=None,
        outpaint_mask_image=None,
        outpaint_step2_checkbox=False,
        outpaint_selections=[],
        cn_tasks={},
        get_cn_tasks_for_channel=lambda *_args, **_kwargs: {},
    )

    additional_loras = process_transition.resolve_preflight_additional_loras(task_state)
    assert additional_loras == []
