"""Focused P4-M18-W12d queue-frozen workflow-plan contract tests."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import modules.flags as flags
from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL
from modules.pipeline.routes import build_generation_route, describe_route
from modules.pipeline.workflow_compiler import compile_workflow_plan, expected_stage_ids
from modules.pipeline.workflow_contracts import (
    AUXILIARY_BACKGROUND_REMOVAL,
    AUXILIARY_GAN_UPSCALE,
    AUXILIARY_MAT_INPAINT,
    MAIN_FAMILY_FLUX_FILL,
    MAIN_FAMILY_SDXL,
    FrozenWorkflowSelection,
)
from modules.pipeline.workflow_legacy_adapter import (
    capture_controlnet_slot_inputs,
    capture_workflow_selection,
)
from modules.task_state import TaskState


def _cn_image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _prepared_inpaint_state(*, bbox):
    source = np.zeros((96, 128, 3), dtype=np.uint8)
    return SimpleNamespace(
        inpaint_input_image=source,
        inpaint_context_mask_image=None,
        inpaint_bb_image=np.full((64, 64, 3), 40, dtype=np.uint8),
        inpaint_mask_image=np.full((64, 64), 255, dtype=np.uint8),
        inpaint_bbox=json.dumps(bbox) if bbox is not None else '',
        inpaint_step2_checkbox=True,
        context_mask=None,
        inpaint_strength=0.85,
        debugging_inpaint_preprocessor=False,
    ), source


def _state(surface: str, *, mixing: bool = False, cn: bool = True) -> TaskState:
    kwargs = {
        "input_image_checkbox": surface in {"controlnet", "inpaint", "outpaint"},
        "current_tab": {
            "normal_generate": "generate",
            "controlnet": "ip",
            "inpaint": "inpaint",
            "outpaint": "outpaint",
            "removal": "remove",
        }.get(surface, "uov"),
        "requested_source_surface": surface,
        "inpaint_input_image": _cn_image() if surface == "inpaint" else None,
        "inpaint_mask_image": np.zeros((8, 8), dtype=np.uint8) if surface == "inpaint" else None,
        "outpaint_input_image": _cn_image() if surface == "outpaint" else None,
        "mixing_image_prompt_and_inpaint": mixing if surface == "inpaint" else False,
        "mixing_image_prompt_and_outpaint": mixing if surface == "outpaint" else False,
        "remove_bg_enabled": surface == "removal",
    }
    state = TaskState(**kwargs)
    if cn:
        state.add_cn_task(flags.cn_canny, [_cn_image(), 0.85, 0.7, 0.15, 2])
    return state


def _compile_state(state: TaskState):
    """Test helper that keeps compatibility capture separate from pure compile."""
    return compile_workflow_plan(
        capture_workflow_selection(state),
        capture_controlnet_slot_inputs(state.cn_tasks),
    )


@pytest.mark.parametrize(
    ("surface", "mixing", "cn", "route", "overlay", "stages"),
    (
        (
            "normal_generate", False, True, "txt2img", False,
            ("prompt_encode", "diffusion_batch"),
        ),
        (
            "controlnet", False, False, "txt2img", False,
            ("prompt_encode", "diffusion_batch"),
        ),
        (
            "controlnet", False, True, "txt2img", True,
            ("image_input_prepare", "controlnet_support_load", "prompt_encode", "structural_controlnet", "diffusion_batch"),
        ),
        (
            "inpaint", False, True, "inpaint", False,
            ("image_input_prepare", "inpaint_prepare", "prompt_encode", "diffusion_batch"),
        ),
        (
            "inpaint", True, False, "inpaint", False,
            ("image_input_prepare", "inpaint_prepare", "prompt_encode", "diffusion_batch"),
        ),
        (
            "inpaint", True, True, "inpaint", True,
            ("image_input_prepare", "controlnet_support_load", "inpaint_prepare", "prompt_encode", "structural_controlnet", "diffusion_batch"),
        ),
        (
            "outpaint", False, True, "outpaint", False,
            ("image_input_prepare", "outpaint_prepare", "prompt_encode", "diffusion_batch"),
        ),
        (
            "outpaint", True, False, "outpaint", False,
            ("image_input_prepare", "outpaint_prepare", "prompt_encode", "diffusion_batch"),
        ),
        (
            "outpaint", True, True, "outpaint", True,
            ("image_input_prepare", "controlnet_support_load", "outpaint_prepare", "prompt_encode", "structural_controlnet", "diffusion_batch"),
        ),
        (
            "removal", True, True, "removal", False,
            ("removal",),
        ),
        (
            "upscale", True, True, "upscale", False,
            ("image_input_prepare", "prompt_encode", "upscale"),
        ),
        (
            "super_upscale", True, True, "super_upscale", False,
            ("image_input_prepare", "prompt_encode", "upscale"),
        ),
        (
            "color_enhanced_upscale", True, True, "color_enhanced_upscale", False,
            ("image_input_prepare", "color_enhanced_upscale"),
        ),
    ),
)
def test_truth_table_freezes_base_route_and_exact_overlay_stages(
    surface, mixing, cn, route, overlay, stages
):
    state = _state(surface, mixing=mixing, cn=cn)
    plan = _compile_state(state)
    state.set_workflow_plan(plan)

    assert plan.route_id == route
    assert plan.controlnet_overlay.enabled is overlay
    assert plan.ordered_stage_ids == stages
    assert expected_stage_ids(plan) == stages
    assert describe_route(build_generation_route(state)) == list(stages)

    if not overlay:
        assert "controlnet_support_load" not in stages
        assert "structural_controlnet" not in stages
        assert "contextual_controlnet" not in stages


@pytest.mark.parametrize(
    ("selection", "route", "main_family", "auxiliary", "step_ids"),
    (
        (FrozenWorkflowSelection("normal_generate"), "txt2img", MAIN_FAMILY_SDXL, (), ("sdxl",)),
        (FrozenWorkflowSelection("controlnet"), "txt2img", MAIN_FAMILY_SDXL, (), ("sdxl",)),
        (FrozenWorkflowSelection("inpaint"), "inpaint", MAIN_FAMILY_SDXL, (), ("sdxl",)),
        (FrozenWorkflowSelection("outpaint"), "outpaint", MAIN_FAMILY_SDXL, (), ("sdxl",)),
        (
            FrozenWorkflowSelection("upscale"),
            "upscale", None, (AUXILIARY_GAN_UPSCALE,), ("gan_upscale",),
        ),
        (
            FrozenWorkflowSelection("removal", remove_background=True, remove_object=True),
            "removal", None,
            (AUXILIARY_BACKGROUND_REMOVAL, AUXILIARY_MAT_INPAINT),
            ("background_removal", "mat_inpaint"),
        ),
        (
            FrozenWorkflowSelection("color_enhanced_upscale"),
            "color_enhanced_upscale", MAIN_FAMILY_SDXL,
            (),
            ("sdxl_color_pass",),
        ),
        (
            FrozenWorkflowSelection("super_upscale"),
            "super_upscale", MAIN_FAMILY_SDXL, (), ("sdxl",),
        ),
        (
            FrozenWorkflowSelection("inpaint", inpaint_route="flux"),
            "flux_inpaint", MAIN_FAMILY_FLUX_FILL, (), ("flux_fill",),
        ),
        (
            FrozenWorkflowSelection(
                "removal",
                object_removal_engine=OBJR_ENGINE_FLUX_FILL,
                remove_object=True,
            ),
            "flux_removal", MAIN_FAMILY_FLUX_FILL, (), ("flux_fill",),
        ),
    ),
)
def test_every_route_declares_exact_main_and_auxiliary_work(
    selection, route, main_family, auxiliary, step_ids
):
    plan = compile_workflow_plan(selection)

    assert plan.route_id == route
    assert plan.execution_declaration.main_family == main_family
    assert plan.execution_declaration.ordered_auxiliary_requirements == auxiliary
    assert tuple(step.step_id for step in plan.execution_declaration.ordered_steps) == step_ids


def test_disabled_image_checkbox_wins_over_stale_tab_surface_and_slots_at_queue_capture():
    args = {
        "input_image_checkbox": False,
        "current_tab": "inpaint",
        "requested_source_surface": "outpaint",
        "mixing_image_prompt_and_inpaint": True,
        "mixing_image_prompt_and_outpaint": True,
        "cn_0_image": _cn_image(),
        "cn_0_type": flags.cn_canny,
    }

    selection = capture_workflow_selection(args, queue_capture=True)
    assert selection.source_surface == "normal_generate"

    from modules.async_worker import AsyncTask

    queued = AsyncTask(args)
    assert queued.state.workflow_selection == selection
    assert queued.state.workflow_plan.source_surface == "normal_generate"
    assert queued.state.workflow_plan.route_id == "txt2img"
    assert not queued.state.workflow_plan.controlnet_overlay.enabled


def test_flux_inpaint_with_active_controlnet_mixing_fails_closed():
    selection = FrozenWorkflowSelection(
        "inpaint",
        inpaint_route="flux",
        allow_inpaint_controlnet=True,
    )
    slots = capture_controlnet_slot_inputs({
        flags.cn_canny: [[_cn_image(), 1.0, 1.0, 0.0, 0]],
    })

    with pytest.raises(ValueError, match="not supported for Flux Fill inpaint"):
        compile_workflow_plan(selection, slots)


def test_controlnet_surface_stays_txt2img_when_mixing_checkboxes_change():
    state = _state("controlnet", mixing=True)
    state.mixing_image_prompt_and_inpaint = True
    state.mixing_image_prompt_and_outpaint = True
    plan = _compile_state(state)

    assert plan.route_id == "txt2img"
    assert plan.route_family == "txt2img"
    assert plan.controlnet_overlay.activation_source == "controlnet_tab"


def test_hidden_slots_are_not_plan_data_and_do_not_change_plan_identity():
    state = _state("outpaint", mixing=False)
    plan = _compile_state(state)
    state.set_workflow_plan(plan)
    identity = plan.identity()

    state.current_tab = "ip"
    state.mixing_image_prompt_and_outpaint = True
    state.cn_tasks[flags.cn_cpds] = [[_cn_image(), 1.0, 1.0, 0.0, 0]]

    assert state.workflow_plan is plan
    assert plan.identity() == identity
    assert plan.controlnet_overlay.active_slot_descriptors == ()
    assert describe_route(build_generation_route(state)) == list(plan.ordered_stage_ids)


def test_plan_copies_literal_slot_and_payload_parameters():
    image = _cn_image()
    state = _state("controlnet", mixing=False, cn=False)
    state.add_cn_task(flags.cn_depth, [image, 0.91, 0.42, 0.17, 3])
    plan = _compile_state(state)
    descriptor = plan.controlnet_overlay.active_slot_descriptors[0]

    image[0, 0, 0] = 255
    state.cn_tasks[flags.cn_depth][0][1] = 0.1
    state.cn_tasks[flags.cn_depth][0][4] = 0

    assert descriptor.ui_slot_index == 3
    assert descriptor.end_percent == pytest.approx(0.91)
    assert descriptor.weight == pytest.approx(0.42)
    assert descriptor.start_percent == pytest.approx(0.17)
    assert descriptor.input_image[0, 0, 0] == 0


def test_plan_is_stable_under_later_task_mutation_and_repeated_planning():
    state = _state("inpaint", mixing=True)
    selection = capture_workflow_selection(state)
    slots = capture_controlnet_slot_inputs(state.cn_tasks)
    plan = compile_workflow_plan(selection, slots)
    state.set_workflow_plan(plan)
    first = plan.telemetry_record()

    state.current_tab = "outpaint"
    state.mixing_image_prompt_and_inpaint = False
    state.mixing_image_prompt_and_outpaint = False
    state.cn_tasks[flags.cn_canny].clear()

    assert state.workflow_plan.telemetry_record() == first
    assert compile_workflow_plan(selection, slots).identity() == plan.identity()
    assert state.workflow_plan.route_id == "inpaint"


def test_streaming_cpu_text_and_gpu_text_share_workflow_plan_truth():
    states = []
    for posture in ("streaming", "auto", "gpu_text"):
        state = _state("outpaint", mixing=False)
        state.sdxl_assembly_posture = posture
        plan = _compile_state(state)
        states.append(plan)

    assert [plan.route_id for plan in states] == ["outpaint"] * 3
    assert [plan.ordered_stage_ids for plan in states] == [states[0].ordered_stage_ids] * 3
    assert [plan.controlnet_overlay.enabled for plan in states] == [False] * 3


def test_inactive_asset_maps_are_ignored_by_admission():
    state = _state("outpaint", mixing=False)
    state.set_workflow_plan(_compile_state(state))
    state.base_model_name = "test_model.safetensors"

    with patch("backend.environment_profile.detect_total_vram_mb", return_value=4096.0), \
         patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list", return_value="checkpoint.safetensors"), \
         patch("backend.sdxl_assembly.request_builder.os.path.exists", return_value=True), \
         patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as taxonomy:
        from types import SimpleNamespace
        from modules.model_taxonomy import ARCHITECTURE_SDXL

        taxonomy.return_value = SimpleNamespace(architecture=ARCHITECTURE_SDXL)
        from backend.sdxl_assembly.request_builder import determine_eligibility

        eligible, reason = determine_eligibility(
            state,
            controlnet_paths={flags.cn_cpds: "missing-hidden-controlnet.safetensors"},
            contextual_assets={"contextual_model_paths": {"FaceID V2": "missing-hidden.model"}},
        )

    assert eligible, reason


def test_active_unresolved_controlnet_fails_closed():
    state = _state("controlnet", cn=True)
    state.set_workflow_plan(_compile_state(state))
    state.base_model_name = "test_model.safetensors"

    with patch("backend.environment_profile.detect_total_vram_mb", return_value=4096.0), \
         patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list", return_value="checkpoint.safetensors"), \
         patch("backend.sdxl_assembly.request_builder.os.path.exists", return_value=True), \
         patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as taxonomy:
        from types import SimpleNamespace
        from modules.model_taxonomy import ARCHITECTURE_SDXL

        taxonomy.return_value = SimpleNamespace(architecture=ARCHITECTURE_SDXL)
        from backend.sdxl_assembly.request_builder import determine_eligibility

        eligible, reason = determine_eligibility(state, controlnet_paths={})

    assert not eligible
    assert "requires a resolved checkpoint path" in reason


def test_active_unsupported_controlnet_fails_closed_at_plan_boundary():
    state = _state("controlnet", cn=False)
    state.cn_tasks["UnknownControl"] = [[_cn_image(), 1.0, 1.0, 0.0, 4]]

    with pytest.raises(ValueError, match="Unsupported active ControlNet"):
        _compile_state(state)


def test_contextual_overlay_adds_only_contextual_stage():
    state = _state("controlnet", cn=False)
    state.add_cn_task(flags.cn_ip, [_cn_image(), 0.8, 0.6, 0.1, 5])

    plan = _compile_state(state)

    assert plan.controlnet_overlay.structural_descriptors == ()
    assert plan.controlnet_overlay.contextual_descriptors[0].ui_slot_index == 5
    assert plan.ordered_stage_ids == (
        "image_input_prepare",
        "controlnet_support_load",
        "prompt_encode",
        "contextual_controlnet",
        "diffusion_batch",
    )


def test_bound_plan_cannot_be_replaced_after_queue_compilation():
    state = _state("normal_generate", cn=False)
    first = _compile_state(state)
    replacement = compile_workflow_plan(FrozenWorkflowSelection("outpaint"))
    state.set_workflow_plan(first)

    assert state.set_workflow_plan(first) is first
    with pytest.raises(RuntimeError, match="already bound"):
        state.set_workflow_plan(replacement)


def test_route_and_gateway_fail_closed_without_queue_bound_plan():
    state = TaskState()

    with pytest.raises(RuntimeError, match="workflow plan is missing"):
        build_generation_route(state)

    from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly

    eligible, reason = is_eligible_for_sdxl_assembly(state, loras=[])
    assert not eligible
    assert "Invalid frozen workflow plan" in reason


def test_production_admission_does_not_call_legacy_plan_adapter():
    state = _state("normal_generate", cn=False)
    plan = _compile_state(state)
    state.set_workflow_plan(plan)
    state.base_model_name = "test_model.safetensors"

    with patch(
        "backend.sdxl_assembly.request_builder.bind_legacy_workflow_plan",
        side_effect=AssertionError("legacy adapter must not run"),
    ), patch(
        "backend.environment_profile.detect_total_vram_mb", return_value=4096.0
    ), patch(
        "backend.sdxl_assembly.request_builder.get_file_from_folder_list",
        return_value="checkpoint.safetensors",
    ), patch(
        "backend.sdxl_assembly.request_builder.os.path.exists", return_value=True
    ), patch(
        "backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy"
    ) as taxonomy:
        from types import SimpleNamespace
        from modules.model_taxonomy import ARCHITECTURE_SDXL
        from backend.sdxl_assembly.request_builder import determine_eligibility

        taxonomy.return_value = SimpleNamespace(architecture=ARCHITECTURE_SDXL)
        eligible, reason = determine_eligibility(
            state,
            workflow_plan=plan,
            allow_legacy_adapter=False,
        )

    assert eligible, reason


def test_hidden_controlnet_slots_do_not_resolve_assets():
    state = _state("normal_generate", cn=True)
    state.set_workflow_plan(_compile_state(state))

    with patch(
        "modules.model_registry.ensure_asset",
        side_effect=AssertionError("inactive ControlNet asset was resolved"),
    ):
        from modules.pipeline.image_input import apply_image_input

        result = apply_image_input(state, base_model_additional_loras=[])

    assert result["controlnet_paths"] == {}
    assert result["contextual_assets"]["contextual_model_paths"] == {}


def test_central_runtime_publisher_attaches_plan_composition_identity(monkeypatch):
    from backend import process_transition
    from modules.pipeline import inference

    state = _state("normal_generate", cn=False)
    plan = _compile_state(state)
    state.set_workflow_plan(plan)
    raw_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        execution_family="standard_sdxl",
        residency_class="full_resident",
        route_family="sdxl",
    )
    publications = []
    monkeypatch.setattr(
        inference,
        "resolve_unified_sdxl_process_key",
        lambda *args, **kwargs: raw_key,
    )
    monkeypatch.setattr(
        process_transition,
        "set_active_runtime",
        lambda **kwargs: publications.append(kwargs),
    )

    published = process_transition.publish_sdxl_runtime(
        state,
        workflow_plan=plan,
        route_owner="test-route",
        safe_to_retain=True,
    )

    assert published.composition_identity == plan.identity()
    assert publications == [{
        "family": process_transition.PROCESS_FAMILY_SDXL,
        "key": published,
        "route_owner": "test-route",
        "safe_to_retain": True,
    }]


def test_composition_only_transition_reuses_sdxl_model_residency():
    from backend import process_transition

    registry = process_transition.SharedProcessRegistry()
    current = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint', 'clip', 'user-lora', 'inpaint-patch'),
        residency_class='resident_unet_gpu_text',
        composition_identity=('outpaint', False),
    )
    requested = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint', 'clip', 'user-lora', 'inpaint-patch'),
        residency_class='resident_unet_gpu_text',
        composition_identity=('inpaint', True, 'depth', 'cpds', 'pulid'),
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == 'reuse'
    assert decision.reset_required is False
    assert decision.reason == 'workflow_composition_change'


def test_model_identity_change_wins_over_simultaneous_composition_change():
    from backend import process_transition

    registry = process_transition.SharedProcessRegistry()
    current = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint-a', 'clip'),
        composition_identity=('outpaint', False),
    )
    requested = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint-b', 'clip'),
        composition_identity=('inpaint', True),
    )

    registry.set_active_key(current)
    decision = registry.evaluate_transition(requested)

    assert decision.action == 'reset'
    assert decision.reset_required is True
    assert decision.reason == 'identity_change'


def test_composition_only_release_boundary_is_a_noop(monkeypatch):
    from backend import process_transition

    current = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint', 'clip', 'inpaint-patch'),
        composition_identity=('outpaint', False),
    )
    requested = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=('checkpoint', 'clip', 'inpaint-patch'),
        composition_identity=('inpaint', True, 'depth'),
    )
    monkeypatch.setattr(
        'backend.resources.prepare_for_checkpoint_switch',
        lambda **_kwargs: pytest.fail('composition-only transition reached model release'),
    )

    result = process_transition.release_process_boundary(current, requested)

    assert result['released'] is False
    assert result['reason'] == 'no_model_boundary'


def test_preview_stitching_is_disabled_only_for_inpaint_route():
    from modules.pipeline.inference import _resolve_preview_stitch_context

    context = object()
    inpaint_state = _state('inpaint', cn=False)
    inpaint_state.inpaint_context = context
    inpaint_plan = _compile_state(inpaint_state)
    outpaint_state = _state('outpaint', cn=False)
    outpaint_state.inpaint_context = context
    outpaint_plan = _compile_state(outpaint_state)

    assert _resolve_preview_stitch_context(inpaint_state, inpaint_plan) is None
    assert _resolve_preview_stitch_context(outpaint_state, outpaint_plan) is context


def test_tiled_refinement_registration_uses_central_plan_aware_publisher(monkeypatch):
    from backend import process_transition
    from modules.pipeline import tiled_refinement

    state = _state("super_upscale", cn=False)
    plan = _compile_state(state)
    state.set_workflow_plan(plan)
    state.runtime_route_id = "super_upscale"
    calls = []
    monkeypatch.setattr(
        process_transition,
        "publish_sdxl_runtime",
        lambda task_state, **kwargs: calls.append((task_state, kwargs)),
    )

    tiled_refinement._register_active_unified_sdxl_process(state)

    assert calls == [(state, {
        "workflow_plan": plan,
        "route_owner": "super_upscale",
        "safe_to_retain": False,
    })]


def test_prepared_inpaint_uses_frozen_source_bbox_and_fails_closed_without_it():
    from modules.pipeline.image_input import apply_inpaint
    from modules.pipeline.inpaint import InpaintPipeline

    bbox = (16, 80, 32, 96)
    state, source = _prepared_inpaint_state(bbox=bbox)
    apply_inpaint(state, source, np.zeros(source.shape[:2], dtype=np.uint8))
    assert state.inpaint_context.bb == bbox

    generated = np.full((64, 64, 3), 200, dtype=np.uint8)
    stitched = InpaintPipeline().paste_back(state.inpaint_context, generated)
    assert np.all(stitched[16:80, 32:96] == 200)
    assert np.all(stitched[:16] == 0)
    assert np.all(stitched[:, :32] == 0)

    missing_bbox_state, source = _prepared_inpaint_state(bbox=None)
    with pytest.raises(ValueError, match='bbox is missing'):
        apply_inpaint(missing_bbox_state, source, np.zeros(source.shape[:2], dtype=np.uint8))


def test_additional_lora_channel_is_unet_only():
    from backend.sdxl_assembly.request_builder import _resolve_lora_channel_weights

    assert _resolve_lora_channel_weights(
        [('user.safetensors', 0.7)],
        [('inpaint_v26.fooocus.patch', 1.0)],
    ) == (
        ('user.safetensors', 0.7, 0.7),
        ('inpaint_v26.fooocus.patch', 1.0, 0.0),
    )


def test_flux_artifact_worker_isolates_private_cli_before_backend_import(monkeypatch):
    from tools import generate_flux_t5_fp16_stream_artifact as worker

    monkeypatch.setattr(
        worker,
        '_parse_args',
        lambda: Namespace(
            prompt='',
            output='unused.pt',
            clip_l='clip.safetensors',
            fp16_t5='t5.safetensors',
            embedding_directory=None,
            metrics_json=None,
            disk_paged_t5_gc_interval=None,
            traceback=False,
        ),
    )
    monkeypatch.setattr(sys, 'argv', ['worker.py', '--prompt', 'private flag'])
    with pytest.raises(ValueError, match='non-empty'):
        worker.main()
    assert sys.argv == ['worker.py']
