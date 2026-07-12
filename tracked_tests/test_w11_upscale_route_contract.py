"""Tracked W11a upscale route-contract coverage."""

import os
import sys
import pytest
import numpy as np
from types import SimpleNamespace

# Setup sys.path
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.pipeline.routes import UpscaleStage, PipelineRouteContext
from backend import process_transition
import backend.resources as resources
import modules.pipeline.tiled_refinement as tiled_refinement
import modules.pipeline.output as pipeline_output


def test_upscale_route_light_vs_super(monkeypatch) -> None:
    """Verify that light upscale and super-upscale execute through GanUpscaleWorker truthfully."""

    # 1. Track worker lifecycle calls
    worker_calls = []

    class DummyWorker:
        def __init__(self):
            worker_calls.append("init")
        def load(self, model_name):
            worker_calls.append(("load", model_name))
        def infer(self, img, scale_override=None):
            worker_calls.append(("infer", scale_override))
            return np.ones((512, 512, 3), dtype=np.uint8)
        def teardown(self):
            worker_calls.append("teardown")

    monkeypatch.setattr("backend.auxiliary_workers.gan_upscale_worker.GanUpscaleWorker", DummyWorker)

    def unexpected_global_cleanup(*args, **kwargs):
        raise AssertionError("Auxiliary GAN execution must not clear another assembly's lifecycle domains.")

    monkeypatch.setattr(resources, "cleanup_memory", unexpected_global_cleanup)
    monkeypatch.setattr(resources, "teardown_active_runtime", unexpected_global_cleanup)

    # Track tiled refinement calls
    tiled_refine_calls = []
    def dummy_apply_tiled_refinement(task_state, img, progress_cb, prompt_task=None):
        tiled_refine_calls.append(("apply", img.shape))
        return np.ones((1024, 1024, 3), dtype=np.uint8)
    monkeypatch.setattr(tiled_refinement, "apply_tiled_diffusion_refinement", dummy_apply_tiled_refinement)

    # Mock save_and_log to avoid actual output writing
    monkeypatch.setattr(pipeline_output, "save_and_log", lambda *args, **kwargs: ["/mock/path.png"])

    # 2. Test Light Upscale Route Contract
    task_state_light = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="upscale",
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        upscale_model="4xNomos2_otf_esrgan.pth",
        upscale_scale_override=0,
        height=128,
        width=128,
        prompt="beautiful landscape",
        negative_prompt="",
        style_selections=[],
        seed=42,
        use_expansion=False,
        loras=[],
    )

    context_light = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state_light,
        route_id="upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=lambda *args: None
    )

    stage = UpscaleStage()
    worker_calls.clear()
    tiled_refine_calls.clear()

    result_light = stage.execute(context_light)

    # Assertions for Light Upscale
    assert result_light.route_complete is True
    assert "init" in worker_calls
    assert ("load", "4xNomos2_otf_esrgan.pth") in worker_calls
    assert ("infer", None) in worker_calls
    assert "teardown" in worker_calls
    assert len(tiled_refine_calls) == 0  # Should NOT proceed to tiled refinement
    assert task_state_light.uov_input_image.shape == (512, 512, 3)

    # 3. Test Super Upscale Route Contract (Stage 1 GAN -> Stage 2 Tiled Refinement handoff)
    task_state_super = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="super-upscale",
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        upscale_model="DAT-4x.pth",  # Should be forced to 4xNomos2 for initial GAN stage
        upscale_scale_override=0,
        height=128,
        width=128,
        prompt="beautiful landscape",
        negative_prompt="",
        style_selections=[],
        seed=42,
        use_expansion=False,
        loras=[],
    )

    context_super = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state_super,
        route_id="upscale",
        route_family="upscale",
        prompt_tasks=[SimpleNamespace()],
        progressbar_callback=lambda *args: None
    )

    worker_calls.clear()
    tiled_refine_calls.clear()

    result_super = stage.execute(context_super)

    # Assertions for Super Upscale
    assert result_super.route_complete is True
    assert "init" in worker_calls
    assert ("load", "4xNomos2_otf_esrgan.pth") in worker_calls  # Forced Nomos2
    assert ("infer", None) in worker_calls
    assert "teardown" in worker_calls
    # Should proceed to tiled refinement using GAN output
    assert len(tiled_refine_calls) == 1
    assert tiled_refine_calls[0] == ("apply", (512, 512, 3))
    assert task_state_super.uov_input_image.shape == (1024, 1024, 3)


def test_plain_upscale_preserves_active_major_family_without_publishing_synthetic_sdxl_identity(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(process_transition, "resolve_sdxl_process_key", lambda *_args, **_kwargs: sentinel)

    task_state = SimpleNamespace(
        objr_engine=None,
        sdxl_execution_policy=SimpleNamespace(enabled=True),
    )

    active_key = process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )

    process_transition.clear_active_runtime()
    try:
        assert process_transition.resolve_requested_process_key(
            task_state,
            SimpleNamespace(family="upscale", route_id="upscale"),
        ) is None

        process_transition.set_active_runtime(
            family=process_transition.PROCESS_FAMILY_SDXL,
            key=active_key,
            route_owner="txt2img",
            safe_to_retain=False,
        )
        assert process_transition.resolve_requested_process_key(
            task_state,
            SimpleNamespace(family="upscale", route_id="upscale"),
        ) == active_key.normalized()
        assert process_transition.resolve_requested_process_key(
            task_state,
            SimpleNamespace(family="upscale", route_id="super_upscale"),
        ) is sentinel
        assert process_transition.resolve_requested_process_key(
            task_state,
            SimpleNamespace(family="upscale", route_id="color_enhanced_upscale"),
        ) is sentinel
    finally:
        process_transition.clear_active_runtime()
