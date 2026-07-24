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

@pytest.fixture(autouse=True)
def _stub_model_file_existence(monkeypatch):
    from pathlib import Path
    original_path_exists = Path.exists
    def smart_path_exists(self):
        p = str(self)
        if 'auth.json' in p:
            return False
        if 'model.safetensors' in p or 'boost.safetensors' in p:
            return True
        return original_path_exists(self)
    monkeypatch.setattr(Path, 'exists', smart_path_exists)
    
    import os
    original_exists = os.path.exists
    def smart_exists(path):
        p = str(path)
        if 'auth.json' in p:
            return False
        if 'model.safetensors' in p or 'boost.safetensors' in p:
            return True
        return original_exists(path)
    monkeypatch.setattr(os.path, 'exists', smart_exists)
    
    import backend.sdxl_assembly.request_builder as rb
    monkeypatch.setattr(rb, 'get_file_from_folder_list', lambda model_name, folders: 'D:/resolved/model.safetensors')
    monkeypatch.setattr(rb, 'get_file_identity', lambda path: SimpleNamespace(path=Path(path), sha256='123'))
    
    import modules.config as modules_config
    def smart_resolve_model_taxonomy(path):
        p = str(path).lower()
        if 'sd15' in p or 'sd1.5' in p or 'sd1_5' in p:
            class DummySD15:
                architecture = 'sd15'
            return DummySD15()
        class DummySDXL:
            architecture = 'sdxl'
        return DummySDXL()
    monkeypatch.setattr(modules_config, 'resolve_model_taxonomy', smart_resolve_model_taxonomy)
    
    yield



def test_upscale_route_light_vs_super(monkeypatch) -> None:
    """Verify light upscale uses GAN and Super-Upscale consumes a provided target truthfully."""

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

    # 3. Test Super Upscale Route Contract (provided target -> tiled refinement)
    task_state_super = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="super-upscale",
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        upscale_gan_output_image=np.ones((640, 768, 3), dtype=np.uint8),
        upscale_model="DAT-4x.pth",
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
    assert worker_calls == []
    # Should proceed to tiled refinement using the provided target image
    assert len(tiled_refine_calls) == 1
    assert tiled_refine_calls[0] == ("apply", (640, 768, 3))
    assert task_state_super.uov_input_image.shape == (1024, 1024, 3)


def test_super_upscale_requires_provided_target(monkeypatch) -> None:
    from modules.pipeline.image_input import apply_upscale

    task_state = SimpleNamespace(
        current_progress=0,
        uov_method="super-upscale",
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        upscale_gan_output_image=None,
        width=128,
        height=128,
    )

    with pytest.raises(ValueError, match="provided upscaled target image"):
        apply_upscale(task_state)


def test_super_upscale_ui_exposes_target_and_hides_gan_controls() -> None:
    from modules.ui_logic import uov_method_change

    updates = uov_method_change("Super-Upscale")

    assert len(updates) == 7
    assert updates[0]["visible"] is True
    assert updates[1]["visible"] is False
    assert updates[2]["visible"] is False
    assert updates[3]["visible"] is False
    assert updates[4]["visible"] is True
    assert updates[5]["visible"] is False
    assert updates[6]["visible"] is False


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
