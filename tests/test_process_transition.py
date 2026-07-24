from backend.process_transition import (
    PROCESS_CLASS_FLUX_FILL,
    PROCESS_CLASS_STANDARD_SDXL,
    PROCESS_FAMILY_FLUX_FILL,
    PROCESS_FAMILY_SDXL,
    SharedProcessRegistry,
    build_process_key,
    clear_active_process_key,
    describe_process_key,
    evaluate_process_transition,
    get_active_process_key,
    set_active_process_key,
)


def test_build_process_key_normalizes_family_and_class():
    key = build_process_key(
        family="SDXL",
        process_class="standard sdxl",
        authoritative_identity=("model-a", "uuid-1"),
    )

    assert key.family == PROCESS_FAMILY_SDXL
    assert key.process_class == PROCESS_CLASS_STANDARD_SDXL
    assert key.authoritative_identity == ("model-a", "uuid-1")


def test_build_process_key_infers_flux_class():
    key = build_process_key(
        family="flux",
        authoritative_identity=("flux-unet", "flux-session", "flux-ae"),
    )

    assert key.family == PROCESS_FAMILY_FLUX_FILL
    assert key.process_class == PROCESS_CLASS_FLUX_FILL


def test_evaluate_transition_allows_same_identity_reuse():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class="standard sdxl",
        authoritative_identity="sdxl-model-1",
    )
    requested = build_process_key(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity="sdxl-model-1",
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reuse"
    assert decision.reset_required is False
    assert decision.reason == "same_process_identity"


def test_evaluate_transition_resets_on_family_change():
    registry = SharedProcessRegistry()
    registry.set_active_key(
        build_process_key(
            family="sdxl",
            process_class=PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity="sdxl-model-1",
        )
    )

    decision = registry.evaluate_transition(
        build_process_key(
            family="flux_fill",
            process_class=PROCESS_CLASS_FLUX_FILL,
            authoritative_identity=("flux-unet", "flux-session", "flux-ae"),
        )
    )

    assert decision.action == "reset"
    assert decision.reset_required is True
    assert decision.reason == "family_change"


def test_evaluate_transition_resets_on_process_class_change():
    registry = SharedProcessRegistry()
    registry.set_active_key(
        build_process_key(
            family="sdxl",
            process_class=PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity="sdxl-model-1",
        )
    )

    decision = registry.evaluate_transition(
        build_process_key(
            family="sdxl",
            process_class="alternate_sdxl",
            authoritative_identity="sdxl-model-1",
        )
    )

    assert decision.action == "reset"
    assert decision.reset_required is True
    assert decision.reason == "process_class_change"


def test_evaluate_transition_resets_on_identity_change():
    registry = SharedProcessRegistry()
    registry.set_active_key(
        build_process_key(
            family="sdxl",
            process_class=PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity="sdxl-model-1",
        )
    )

    decision = registry.evaluate_transition(
        build_process_key(
            family="sdxl",
            process_class=PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity="sdxl-model-2",
        )
    )

    assert decision.action == "reset"
    assert decision.reset_required is True
    assert decision.reason == "identity_change"


def test_evaluate_transition_allows_lora_stack_change_reuse():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "vae-1", "clip-1"),
    )
    requested = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "vae-1", "clip-1", "lora-a:1.0"),
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reuse"
    assert decision.reset_required is False
    assert decision.reason == "lora_stack_change"


def test_evaluate_transition_preserves_unet_when_text_posture_is_not_process_identity():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "vae-1", "lora-a:1.0"),
        residency_class="full_resident",
    )
    requested = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "vae-1", "lora-b:1.0"),
        residency_class="full_resident",
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reuse"
    assert decision.reset_required is False
    assert decision.reason == "lora_stack_change"


def test_module_registry_round_trip_and_clear():
    clear_active_process_key()
    assert get_active_process_key() is None

    key = build_process_key(
        family="flux_fill",
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=("flux-unet", "flux-session", "flux-ae"),
    )
    set_active_process_key(key)

    active_key = get_active_process_key()
    assert active_key == key.normalized()
    assert "family=flux_fill" in describe_process_key(active_key)
    assert "class=flux_fill" in describe_process_key(active_key)

    clear_active_process_key()
    assert get_active_process_key() is None


def test_clear_active_process_key_clears_runtime_metadata_too():
    from backend.process_transition import (
        clear_active_runtime,
        get_active_family,
        get_active_route_owner,
        is_safe_to_retain,
        set_active_runtime,
    )

    clear_active_runtime()
    key = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity="sdxl-model-1",
    )
    set_active_runtime(
        family="sdxl",
        key=key,
        route_owner="txt2img",
        safe_to_retain=True,
    )

    clear_active_process_key()

    assert get_active_process_key() is None
    assert get_active_family() is None
    assert get_active_route_owner() is None
    assert is_safe_to_retain() is False


def test_set_active_process_key_resets_extended_runtime_metadata():
    from backend.process_transition import (
        clear_active_runtime,
        get_active_family,
        get_active_route_owner,
        is_safe_to_retain,
        set_active_runtime,
    )

    clear_active_runtime()
    set_active_runtime(
        family="sdxl",
        key=build_process_key(
            family="sdxl",
            process_class=PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity="sdxl-model-1",
        ),
        route_owner="txt2img",
        safe_to_retain=True,
    )

    flux_key = build_process_key(
        family="flux_fill",
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=("flux-unet", "flux-session", "flux-ae"),
    )
    set_active_process_key(flux_key)

    assert get_active_process_key() == flux_key.normalized()
    assert get_active_family() == "flux_fill"
    assert get_active_route_owner() is None
    assert is_safe_to_retain() is False


def test_evaluate_transition_start_state_does_not_require_reset():
    clear_active_process_key()
    decision = evaluate_process_transition(
        build_process_key(
            family="sdxl",
            process_class="full_resident",
            authoritative_identity="sdxl-model-1",
        )
    )

    assert decision.action == "start"
    assert decision.reset_required is False
    assert decision.current_key is None


def test_registry_metadata_tracking():
    from backend.process_transition import (
        get_active_family,
        get_active_route_owner,
        is_safe_to_retain,
        set_active_runtime,
        clear_active_runtime,
        get_active_process_key,
    )
    clear_active_runtime()
    assert get_active_process_key() is None
    assert get_active_family() is None
    assert get_active_route_owner() is None
    assert is_safe_to_retain() is False

    key = build_process_key(
        family="SDXL",
        process_class="standard sdxl",
        authoritative_identity="sdxl-model-1",
    )
    set_active_runtime(
        family="sdxl",
        key=key,
        route_owner="txt2img",
        safe_to_retain=True,
    )

    assert get_active_process_key() == key.normalized()
    assert get_active_family() == "sdxl"
    assert get_active_route_owner() == "txt2img"
    assert is_safe_to_retain() is True

    clear_active_runtime()
    assert get_active_process_key() is None
    assert get_active_family() is None
    assert get_active_route_owner() is None
    assert is_safe_to_retain() is False


def test_evaluate_transition_uses_base_and_clip_identity_for_sdxl():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "clip-a"),
        route_family="sdxl",
    )
    requested = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "clip-a", "lora-a:1.0"),
        route_family="sdxl",
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reuse"
    assert decision.reset_required is False
    assert decision.reason == "lora_stack_change"


def test_evaluate_transition_dynamic_base_len_sdxl_decoupled_vae():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "clip-1"),
        route_family="sdxl",
    )
    requested = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "clip-1", "lora-a:1.0"),
        route_family="sdxl",
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reuse"
    assert decision.reset_required is False
    assert decision.reason == "lora_stack_change"


def test_evaluate_transition_reset_on_base_components_change():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-1", "clip-1"),
        route_family="sdxl",
    )
    requested = build_process_key(
        family="sdxl",
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("sdxl-model-2", "clip-1"),
        route_family="sdxl",
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reset"
    assert decision.reset_required is True
    assert decision.reason == "identity_change"


def test_evaluate_transition_resets_on_flux_spine_change():
    registry = SharedProcessRegistry()
    current = build_process_key(
        family="flux_fill",
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=(
            ("ae_path", "ae.safetensors"),
            ("conditioning_cache_path", "empty.pt"),
            ("model_variant", "flux_fill_fp8"),
            ("unet_path", "unet.safetensors"),
            ("unet_spine", "streaming"),
        ),
        route_family="flux_fill",
    )
    requested = build_process_key(
        family="flux_fill",
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

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == "reset"
    assert decision.reset_required is True
    assert decision.reason == "identity_change"
