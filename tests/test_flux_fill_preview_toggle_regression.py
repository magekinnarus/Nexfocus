import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.mark.parametrize(
    ("disable_preview", "expect_preview_transform"),
    [
        (True, False),
        (False, True),
    ],
)
def test_flux_fill_inpaint_stage_honors_disable_preview_toggle(monkeypatch, disable_preview, expect_preview_transform):
    import modules.pipeline.routes as routes
    import modules.pipeline.inference as inference
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly
    from backend.flux_fill_v3.contracts import FluxFillResult

    import backend.flux_fill_v3.activation as activation
    monkeypatch.setattr(activation, "resolve_flux_fill_assets", lambda state: FluxFillActivationAssets(
        unet_path="unet.safetensors",
        ae_path="ae.safetensors",
        conditioning_cache_path="empty_conditioning.pt",
        model_variant="flux_fill_fp8",
        conditioning_kind="empty",
        clip_l_path="clip_l.safetensors",
        t5_path="t5xxl.safetensors",
        prompt="replace the statue\na garden",
    ))

    captured_preview_transforms = []

    def mock_get_sampling_callback(*args, **kwargs):
        captured_preview_transforms.append(kwargs.get("preview_transform"))
        return object()

    monkeypatch.setattr(inference, "get_sampling_callback", mock_get_sampling_callback)
    monkeypatch.setattr(
        FluxAssembly,
        "execute",
        lambda self, req, callback=None: FluxFillResult(
            output_image=np.ones((req.image.shape[0], req.image.shape[1], 3), dtype=np.uint8),
            seed=req.seed,
            width=req.image.shape[1],
            height=req.image.shape[0],
        ),
    )

    import modules.pipeline.output as pipeline_output
    monkeypatch.setattr(pipeline_output, "save_and_log", lambda *args, **kwargs: ["mock_output_path.png"])

    stage = routes.FluxFillInpaintStage()
    task_state = SimpleNamespace(
        goals=['inpaint'],
        current_progress=0,
        image_number=1,
        disable_seed_increment=False,
        disable_preview=disable_preview,
        seed=42,
        steps=12,
        sampler_name='dpmpp_2m',
        scheduler_name='karras',
        prompt='a garden',
        inpaint_additional_prompt='replace the statue',
        negative_prompt='',
        style_selections=[],
        loras=[],
        width=16,
        height=16,
        disable_intermediate_results=False,
        flux_fill_conditioning='empty',
        flux_fill_prompt_cache='permanent',
        inpaint_context=None,
        prefetch_depth=0,
        prefetch_chunk_mb=64,
    )
    context = SimpleNamespace(
        task_state=task_state,
        image_input_result={
            'inpaint_image': np.zeros((24, 32, 3), dtype=np.uint8),
            'inpaint_mask': np.zeros((24, 32), dtype=np.uint8),
        },
        progressbar_callback=lambda *args: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = stage.execute(context)

    assert res.route_complete is True
    assert len(captured_preview_transforms) == 1
    assert (captured_preview_transforms[0] is not None) is expect_preview_transform
