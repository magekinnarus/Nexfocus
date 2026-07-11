"""Tracked W11c color-enhanced-upscale smoke tests."""

import os
import sys
import pytest
import numpy as np
from types import SimpleNamespace

# Setup sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.pipeline.routes import ColorEnhancedUpscaleStage, PipelineRouteContext
import backend.resources as resources


def test_w11c_color_enhanced_upscale_smoke(monkeypatch) -> None:
    """Smoke test running the entire Color Enhancement pipeline flow with mocked workers."""

    # Mock resources phases to prevent CUDA operations or real model caching checks
    monkeypatch.setattr(resources, "get_torch_device", lambda: "cpu")

    # Mock SDXL assembly execution
    def dummy_run_sdxl_assembly_task(
        task_state,
        task_dict,
        current_task_id,
        total_count,
        all_steps,
        preparation_steps,
        denoising_strength,
        final_scheduler_name,
        **kwargs
    ):
        h, w = task_state.source_pixels.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    monkeypatch.setattr("backend.sdxl_assembly.gateway.run_sdxl_assembly_task", dummy_run_sdxl_assembly_task)

    # Mock save_and_log to avoid actual file system writes
    monkeypatch.setattr("modules.pipeline.output.save_and_log", lambda *args, **kwargs: ["/mock/output.png"])

    task_state = SimpleNamespace(
        goals=["upscale"],
        current_progress=0,
        uov_method="Color Enhancement",
        uov_input_image=np.zeros((128, 128, 3), dtype=np.uint8),
        upscale_gan_output_image=np.zeros((512, 512, 3), dtype=np.uint8),
        upscale_model="None",
        upscale_scale_override=0,
        seed=42,
        prompt="beautiful art",
        negative_prompt="",
        scheduler_name="karras",
        steps=20,
        cfg_scale=7.0,
        style_selections=[],
        use_expansion=False,
        loras=[],
    )

    context = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=task_state,
        route_id="color_enhanced_upscale",
        route_family="upscale",
        prompt_tasks=[],
        progressbar_callback=lambda *args, **kwargs: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    stage = ColorEnhancedUpscaleStage()
    result = stage.execute(context)

    assert result.route_complete is True
    assert result.notes['completed'] is True
    # Required donor dimensions define the final output dimensions.
    assert task_state.uov_input_image.shape == (512, 512, 3)
