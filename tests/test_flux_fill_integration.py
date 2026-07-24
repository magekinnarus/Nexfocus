import os
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.objr_engine as objr_engine
import modules.pipeline.inference as inference_module
from modules.pipeline.inference import get_sampling_callback
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
from modules.parameter_registry import PARAM_REGISTRY
from modules.task_state import TaskState


@pytest.fixture(autouse=True)
def _refresh_objr_engine_module_reference():
    global objr_engine
    objr_engine = importlib.import_module("modules.objr_engine")
    yield
    objr_engine = importlib.import_module("modules.objr_engine")


def test_task_state_and_parameter_registry_include_objr_engine():
    params = {param.name: param for param in PARAM_REGISTRY}

    state = TaskState()
    assert state.objr_engine == objr_engine.OBJR_ENGINE_MAT
    assert state.flux_fill_conditioning == 'empty'
    assert state.remove_prompt == ''
    assert state.flux_fill_prompt_cache == 'temp'
    assert state.inpaint_route == 'sdxl'
    assert state.objr_mask_blur == 6
    assert state.objr_blend_mode == 'morphological'
    assert state.prefetch_depth == 1
    assert state.flux_fill_runtime_posture == 'auto'
    assert state.flux_fill_disk_paged_t5_gc_interval == 'auto'
    assert params['objr_engine'].task_field == 'objr_engine'
    assert params['objr_engine'].default == objr_engine.OBJR_ENGINE_MAT
    assert params['flux_fill_conditioning'].task_field == 'flux_fill_conditioning'
    assert params['flux_fill_conditioning'].default == 'empty'
    assert params['flux_fill_runtime_posture'].task_field == 'flux_fill_runtime_posture'
    assert params['flux_fill_runtime_posture'].default == 'auto'
    assert params['flux_fill_disk_paged_t5_gc_interval'].task_field == 'flux_fill_disk_paged_t5_gc_interval'
    assert params['flux_fill_disk_paged_t5_gc_interval'].default == 'auto'
    assert params['inpaint_route'].task_field == 'inpaint_route'
    assert params['inpaint_route'].default == 'sdxl'
    assert params['remove_prompt'].task_field == 'remove_prompt'
    assert params['flux_fill_prompt_cache'].default == 'temp'
    assert params['objr_mask_blur'].default == 6
    assert params['objr_blend_mode'].default == 'morphological'
    assert params['prefetch_depth'].default == 1
    assert 'flux_fill_runtime_posture' in params


def test_flux_fill_inpaint_stage_describes_greenfield_runtime_resources():
    import modules.pipeline.routes as routes

    resources = routes.FluxFillInpaintStage().describe_resources(SimpleNamespace(task_state=SimpleNamespace()))

    runtime_resource = next(resource for resource in resources if resource.resource_id == 'flux_fill_runtime')
    assert runtime_resource.owner == 'backend.flux_fill_v3'
    assert 'greenfield flux fill runtime' in runtime_resource.description.lower()
    assert all(resource.resource_id != 'flux_session' for resource in resources)


def test_flux_fill_tier_selection_tracks_nex_fp8_rollout_policy():
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.select_flux_fill_tier(SimpleNamespace(name='colab_pro', total_ram_mb=53248, total_vram_mb=23000))


def test_flux_fill_hardware_inspection_distinguishes_resident_and_hybrid_postures():
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.inspect_flux_fill_hardware(
            SimpleNamespace(name='colab_free', total_ram_mb=16384, total_vram_mb=15360, is_colab=True)
        )


def test_flux_fill_high_ram_colab_pro_can_still_plan_streaming_when_vram_is_overridden():
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.inspect_flux_fill_hardware(
            SimpleNamespace(name='colab_pro', total_ram_mb=57344, total_vram_mb=12288, is_colab=True)
        )


def test_flux_fill_hardware_inspection_surfaces_acceleration_class_from_profile_notes():
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.inspect_flux_fill_hardware(
            SimpleNamespace(
                name='local_normal',
                total_ram_mb=32768,
                total_vram_mb=15360,
                is_colab=False,
                notes={
                    'gpu_name': 'Tesla T4',
                    'cuda_capability': '7.5',
                    'flux_acceleration_class': 'tensor_core_accelerated',
                    'tensor_core_accelerated': True,
                },
            )
        )


def test_sampling_callback_ignores_tensor_preview_payloads():
    task_state = SimpleNamespace(yields=[], inpaint_context=None, current_progress=0, current_status_text='')
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 10)

    callback(0, None, None, 10, torch.zeros((1, 4, 8, 8)))

    assert task_state.yields
    assert task_state.current_progress == 10
    assert task_state.current_status_text == "Sampling step 1/10 (10%)"
    assert task_state.yields[0][0] == 'preview'
    assert task_state.yields[0][1][2] is None


def test_sampling_callback_applies_preview_transform_before_yield():
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(
        task_state,
        None,
        0,
        1,
        0,
        10,
        preview_transform=lambda preview: np.full((8, 8, 3), 7, dtype=np.uint8) if isinstance(preview, torch.Tensor) else preview,
    )

    callback(0, None, None, 10, torch.zeros((1, 4, 8, 8)))

    assert isinstance(task_state.yields[0][1][2], np.ndarray)
    assert task_state.yields[0][1][2].shape == (8, 8, 3)


def test_sampling_callback_pastes_preview_without_morphological_blend():
    from modules.pipeline.inpaint import InpaintContext

    task_state = SimpleNamespace(yields=[])
    context = InpaintContext(
        original_image=np.zeros((16, 16, 3), dtype=np.uint8),
        original_mask=np.zeros((16, 16), dtype=np.uint8),
        bb=(4, 12, 4, 12),
        bb_image=np.zeros((8, 8, 3), dtype=np.uint8),
        bb_mask=np.zeros((8, 8), dtype=np.uint8),
        target_w=8,
        target_h=8,
        blend_mask=np.full((16, 16), 255, dtype=np.uint8),
    )
    callback = get_sampling_callback(
        task_state,
        None,
        0,
        1,
        0,
        10,
        preview_transform=lambda preview: np.full((8, 8, 3), 9, dtype=np.uint8),
        preview_stitch_context=context,
    )

    callback(0, None, None, 10, torch.zeros((1, 4, 8, 8)))

    preview = task_state.yields[0][1][2]
    assert preview.shape == (16, 16, 3)
    assert np.all(preview[:4] == 0)
    assert np.all(preview[4:12, 4:12] == 9)


def test_sampling_callback_emits_preview_images_on_configured_step_interval():
    task_state = SimpleNamespace(
        yields=[],
        inpaint_context=None,
        preview_update_interval=3,
    )
    callback = get_sampling_callback(
        task_state,
        None,
        0,
        1,
        0,
        10,
        preview_transform=lambda preview: np.full((8, 8, 3), 9, dtype=np.uint8),
    )

    callback(0, None, None, 10, torch.zeros((1, 4, 8, 8)))
    callback(2, None, None, 10, torch.zeros((1, 4, 8, 8)))
    callback(9, None, None, 10, torch.zeros((1, 4, 8, 8)))

    assert task_state.yields[0][1][2] is None
    assert task_state.yields[1][1][2].shape == (8, 8, 3)
    assert task_state.yields[2][1][2].shape == (8, 8, 3)


def test_sampling_callback_prints_step_timing(capsys):
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 10)

    callback(0, None, None, 10, None)

    stdout = capsys.readouterr().out
    assert "Sampling:" in stdout
    assert "Step 1/10" in stdout


def test_sampling_callback_emits_step_progress(capsys):
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 20)

    for step in range(20):
        callback(step, None, None, 20, None)

    stdout = capsys.readouterr().out
    assert "[Nex] Sampling: [" in stdout
    assert "Step 20/20" in stdout


def test_sampling_callback_uses_live_newlines_for_non_tty_stream(monkeypatch, capsys):
    monkeypatch.setattr(inference_module, "_supports_inline_console_progress", lambda: False)
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 3)

    for step in range(3):
        callback(step, None, None, 3, None)

    stdout = capsys.readouterr().out
    assert "\r" not in stdout
    assert stdout.count("[Nex] Sampling: [") == 3
    assert "Step 1/3" in stdout
    assert "Step 3/3" in stdout


def test_sampling_callback_keeps_inline_updates_for_tty_stream(monkeypatch, capsys):
    monkeypatch.setattr(inference_module, "_supports_inline_console_progress", lambda: True)
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 3)

    for step in range(3):
        callback(step, None, None, 3, None)

    stdout = capsys.readouterr().out
    assert stdout.count("\r") == 2
    assert stdout.endswith("\n")


def test_sampling_callback_silent_when_pbar_enabled(capsys):
    task_state = SimpleNamespace(yields=[], inpaint_context=None)
    callback = get_sampling_callback(task_state, None, 0, 1, 0, 10, disable_pbar=False)

    callback(0, None, None, 10, None)

    stdout = capsys.readouterr().out
    assert "Sampling step 1/10" not in stdout


def test_taesd_supports_flux_decoder_channel_inference_from_taef1_name():
    from ldm_patched.taesd.taesd import TAESD

    model = TAESD(encoder_path=None, decoder_path=None, latent_channels=16)

    assert model.decoder[1].in_channels == 16
    assert model.taesd_decoder[1].in_channels == 16
    assert model.guess_latent_channels_and_arch("taef1_decoder.pth") == (16, None)


def test_taesd_preserves_sd_default_channel_shape():
    from ldm_patched.taesd.taesd import TAESD

    model = TAESD(encoder_path=None, decoder_path=None)

    assert model.decoder[1].in_channels == 4
    assert model.taesd_decoder[1].in_channels == 4


def test_flux_fill_inpaint_stage_handles_skip_interrupt(monkeypatch):
    import modules.pipeline.routes as routes
    from backend import resources
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly

    # Mock activation to return dummy paths
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

    called_requests = []
    def mock_execute(self, req, callback=None):
        called_requests.append(req)
        # Raise interrupt exception to simulate skip/stop
        raise resources.InterruptProcessingException()

    monkeypatch.setattr(FluxAssembly, "execute", mock_execute)

    stage = routes.FluxFillInpaintStage()
    task_state = SimpleNamespace(
        goals=['inpaint'],
        current_progress=0,
        image_number=1,
        disable_seed_increment=False,
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
        flux_fill_prompt_cache='temp',
        inpaint_context=None,
        last_stop='skip',
        prefetch_depth=2,
        prefetch_chunk_mb=128,
        flux_fill_disk_paged_t5_gc_interval='16',
    )
    context = SimpleNamespace(
        task_state=task_state,
        image_input_result={'inpaint_image': np.zeros((24, 32, 3), dtype=np.uint8), 'inpaint_mask': np.zeros((24, 32), dtype=np.uint8)},
        progressbar_callback=None,
        yield_result_callback=None,
    )

    res = stage.execute(context)
    assert res.route_complete is True
    assert res.notes.get('tasks_processed') == 0
    assert task_state.last_stop is False
    assert len(called_requests) == 1
    assert called_requests[0].prefetch_depth == 2
    assert called_requests[0].prefetch_chunk_mb == 128
    assert called_requests[0].disk_paged_t5_gc_interval == 16


def test_flux_fill_t5_selection_uses_profile_and_override_policy(monkeypatch):
    profile = SimpleNamespace(name='colab_free', total_ram_mb=16384, total_vram_mb=15360, runtime_posture='streaming')
    variant = objr_engine.select_flux_fill_t5_variant(profile)
    assert variant == "fp16"
    with pytest.raises(ValueError, match="native Flux Fill fp16"):
        objr_engine.select_flux_fill_t5_variant(profile, variant="legacy")


def test_flux_fill_text_encoder_residency_policy_stays_disk_paged_on_32gb_ram():
    profile1 = SimpleNamespace(name='local_normal', total_ram_mb=32768, total_vram_mb=16384, is_colab=False, runtime_posture='streaming')
    res1 = objr_engine.evaluate_flux_fill_text_encoder_residency(profile1)
    assert res1["keep_resident"] is False

    profile2 = SimpleNamespace(name='local_normal', total_ram_mb=32768, total_vram_mb=16384, is_colab=False, runtime_posture='resident')
    res2 = objr_engine.evaluate_flux_fill_text_encoder_residency(profile2)
    assert res2["keep_resident"] is False


def test_flux_fill_text_encoder_residency_policy_ignores_next_route_headroom():
    profile = SimpleNamespace(name='colab_pro', total_ram_mb=53248, free_ram_mb=40960, total_vram_mb=23000, is_colab=True, runtime_posture='streaming')
    res = objr_engine.evaluate_flux_fill_text_encoder_residency(profile)
    assert res["keep_resident"] is False


def test_flux_fill_text_encoder_residency_policy_never_requests_resident_t5():
    profile = SimpleNamespace(name='local_normal', total_ram_mb=49152, free_ram_mb=30000, total_vram_mb=24576, is_colab=False, runtime_posture='streaming')
    res = objr_engine.evaluate_flux_fill_text_encoder_residency(profile)
    assert res["keep_resident"] is False


def test_generate_flux_fill_prompt_conditioning_uses_native_runtime_boundary(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.generate_flux_fill_prompt_conditioning('garden scene', progress=False)


def test_reconcile_active_flux_fill_session_clears_flux_prompt_cache_when_leaving_for_sdxl(monkeypatch):
    result = objr_engine.reconcile_active_flux_fill_session(
        route_family='image_input',
        selected_engine=objr_engine.OBJR_ENGINE_MAT,
        conditioning='empty',
        progress=False,
    )
    assert result.decision == 'ignored'
    assert result.text_encoder_action == 'cleared'


def test_generate_flux_fill_prompt_conditioning_rejects_legacy_runtime(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.generate_flux_fill_prompt_conditioning('garden scene', progress=False)


def test_generate_flux_fill_prompt_conditioning_trims_host_memory_when_non_resident(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.generate_flux_fill_prompt_conditioning('garden scene', progress=False)


def test_generate_flux_fill_prompt_conditioning_cache_uses_native_prompt_generator(monkeypatch, tmp_path):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.generate_flux_fill_prompt_conditioning_cache(
            'garden scene',
            next_route_family='inpaint',
            progress=False,
        )


def test_generate_flux_fill_prompt_conditioning_cache_trims_host_memory_when_non_resident(monkeypatch, tmp_path):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.generate_flux_fill_prompt_conditioning_cache('garden scene', progress=False)


def test_resolve_flux_fill_asset_paths_downloads_selected_unet_ae_and_conditioning(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.resolve_flux_fill_asset_paths(conditioning='empty', progress=False)


def test_resolve_flux_fill_asset_paths_uses_generated_prompt_cache_without_conditioning_asset(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.resolve_flux_fill_asset_paths(
            conditioning='empty',
            conditioning_cache_path='/tmp/prompt_conditioning.pt',
            progress=False,
        )


def test_resolve_flux_fill_asset_paths_rejects_legacy_runtime(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.resolve_flux_fill_asset_paths(conditioning='empty', progress=False)


def test_build_generation_route_uses_flux_fill_for_inpaint_when_selected():
    import modules.pipeline.routes as routes

    task_state = SimpleNamespace(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        inpaint_mask_image=None,
        inpaint_context_mask_image=None,
        inpaint_bb_image=None,
        inpaint_route='flux',
        inpaint_step2_checkbox=False,
        mixing_image_prompt_and_inpaint=False,
        mixing_image_prompt_and_outpaint=False,
        outpaint_input_image=None,
        outpaint_mask_image=None,
        outpaint_step2_checkbox=False,
        outpaint_selections=[],
        cn_tasks={},
        goals=[],
    )

    bind_legacy_workflow_plan(task_state)
    route = routes.build_generation_route(task_state)

    assert route.route_id == 'flux_inpaint'
    assert route.family == 'flux_fill'
    assert [stage.stage_id for stage in route.stages] == ['image_input_prepare', 'flux_inpaint']


def test_build_generation_route_reverts_to_txt2img_when_input_image_checkbox_disabled():
    import modules.pipeline.routes as routes

    task_state = SimpleNamespace(
        input_image_checkbox=False,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        inpaint_mask_image=None,
        inpaint_context_mask_image=None,
        inpaint_bb_image=None,
        inpaint_route='flux',
        inpaint_step2_checkbox=False,
        mixing_image_prompt_and_inpaint=False,
        mixing_image_prompt_and_outpaint=False,
        outpaint_input_image=None,
        outpaint_mask_image=None,
        outpaint_step2_checkbox=False,
        outpaint_selections=[],
        cn_tasks={},
        goals=[],
    )

    bind_legacy_workflow_plan(task_state)
    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert route.family == 'txt2img'


def test_apply_image_input_skips_sdxl_inpaint_patch_loading_for_flux_route(monkeypatch):
    import modules.pipeline.image_input as image_input

    task_state = SimpleNamespace(
        current_tab='inpaint',
        input_image_checkbox=True,
        uov_method='Disabled',
        uov_input_image=None,
        inpaint_input_image=np.zeros((16, 16, 3), dtype=np.uint8),
        inpaint_context_mask_image=None,
        inpaint_mask_image=None,
        inpaint_bb_image=None,
        inpaint_route='flux',
        inpaint_step2_checkbox=False,
        inpaint_erode_or_dilate=0,
        inpaint_disable_initial_latent=False,
        inpaint_strength=0.5,
        context_mask=None,
        goals=[],
        cn_tasks={},
        mixing_image_prompt_and_inpaint=False,
        mixing_image_prompt_and_outpaint=False,
        outpaint_input_image=None,
        outpaint_mask_image=None,
        outpaint_step2_checkbox=False,
        outpaint_selections=[],
        remove_bg_enabled=False,
        remove_obj_enabled=False,
        skipping_cn_preprocessor=False,
        outpaint_direction=None,
    )

    called = {'download': False}

    monkeypatch.setattr(image_input.config, 'downloading_inpaint_models', lambda *args, **kwargs: called.__setitem__('download', True))

    bind_legacy_workflow_plan(task_state)
    result = image_input.apply_image_input(task_state, base_model_additional_loras=[], progressbar_callback=None)

    assert called['download'] is False
    assert result['skip_prompt_processing'] is True
    assert 'inpaint' in task_state.goals


def test_flux_fill_inpaint_stage_raises_archived_error(monkeypatch):
    import modules.pipeline.routes as routes
    from backend.flux_fill_v3.contracts import FluxFillResult
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly

    # Mock activation to return dummy paths
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

    called_requests = []
    def mock_execute(self, req, callback=None):
        called_requests.append(req)
        target_h = req.image.shape[0]
        target_w = req.image.shape[1]
        return FluxFillResult(
            output_image=np.ones((target_h, target_w, 3), dtype=np.uint8),
            seed=req.seed,
            width=target_w,
            height=target_h,
        )

    monkeypatch.setattr(FluxAssembly, "execute", mock_execute)

    # Mock save_and_log to avoid saving to file system
    import modules.pipeline.output as pipeline_output
    monkeypatch.setattr(pipeline_output, "save_and_log", lambda *args, **kwargs: ["mock_output_path.png"])

    stage = routes.FluxFillInpaintStage()
    task_state = SimpleNamespace(
        goals=['inpaint'],
        current_progress=0,
        image_number=2,
        disable_seed_increment=False,
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
        image_input_result={'inpaint_image': np.zeros((24, 32, 3), dtype=np.uint8), 'inpaint_mask': np.zeros((24, 32), dtype=np.uint8)},
        progressbar_callback=lambda *args: None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = stage.execute(context)
    assert res.route_complete is True
    assert res.notes.get('tasks_processed') == 2
    assert len(called_requests) == 2
    assert called_requests[0].prefetch_depth == 0
    assert called_requests[0].prefetch_chunk_mb == 64


def test_flux_fill_inpaint_stage_requests_host_cleanup_on_low_vram_profiles(monkeypatch):
    import modules.pipeline.routes as routes
    import backend.resources as routed_resources
    from backend import environment_profile as environment_profiles
    from backend.flux_fill_v3.contracts import FluxFillResult
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly

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
    monkeypatch.setattr(
        routed_resources,
        "active_memory_environment_profile",
        lambda: SimpleNamespace(name=environment_profiles.PROFILE_LOCAL_LOW_VRAM),
    )

    cleanup_calls = []
    monkeypatch.setattr(
        routed_resources,
        "cleanup_memory",
        lambda reason, **kwargs: cleanup_calls.append((reason, kwargs)),
    )

    stage = routes.FluxFillInpaintStage()
    task_state = SimpleNamespace(
        goals=['inpaint'],
        current_progress=0,
        image_number=1,
        disable_seed_increment=False,
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
        prefetch_depth=1,
        prefetch_chunk_mb=64,
    )
    context = SimpleNamespace(
        task_state=task_state,
        route_id='flux_inpaint',
        image_input_result={'inpaint_image': np.zeros((24, 32, 3), dtype=np.uint8), 'inpaint_mask': np.zeros((24, 32), dtype=np.uint8)},
        progressbar_callback=None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = stage.execute(context)

    assert res.route_complete is True
    flux_cleanup = next(kwargs for reason, kwargs in cleanup_calls if reason == 'flux_inpaint_image_complete')
    assert flux_cleanup['gc_collect'] is True
    assert flux_cleanup['trim_host'] is True


def test_remove_object_with_engine_dispatches_mat_and_flux(monkeypatch):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mat_result = np.full_like(image, 11)
    calls = []

    monkeypatch.setattr(objr_engine, 'remove_object', lambda *args, **kwargs: calls.append(('mat', kwargs)) or mat_result)

    assert objr_engine.remove_object_with_engine(image, mask, engine=objr_engine.OBJR_ENGINE_MAT).mean() == 11
    assert calls[0][0] == 'mat'

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_with_engine(image, mask, engine=objr_engine.OBJR_ENGINE_FLUX_FILL)


def test_remove_object_flux_fill_expands_mask_before_runtime(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[31:33, 31:33] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(image, mask, seed=123, progress=False)


def test_remove_object_flux_fill_threads_runtime_policy_to_pipeline_config(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(image, mask, seed=11, progress=False)


def test_remove_object_flux_fill_uses_generated_prompt_conditioning_cache(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(
            image,
            mask,
            seed=123,
            conditioning='empty',
            prompt='empty patio, plants, natural background',
            prompt_cache='permanent',
            progress=False,
        )


def test_prepare_flux_fill_prompt_conditioning_cache_path_raises_archived_error(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.prepare_flux_fill_prompt_conditioning_cache_path(
            'repair statue',
            cache_mode='permanent',
            next_route_family='inpaint',
            progress=False,
        )


def test_remove_object_flux_fill_active_session_uses_generated_prompt_conditioning_cache(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(
            image,
            mask,
            prompt='repair statue',
            prompt_cache='permanent',
            progress=False,
        )


def test_flux_fill_mask_preparation_defaults_are_deterministic():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 255

    default_mask = objr_engine.prepare_flux_fill_mask(mask)
    explicit_mask = objr_engine.prepare_flux_fill_mask(mask, grow=objr_engine.FLUX_FILL_MASK_GROW, blur=objr_engine.FLUX_FILL_MASK_BLUR)

    assert np.array_equal(default_mask, explicit_mask)
    assert int(default_mask.sum()) > int(mask.sum())


def test_remove_object_flux_fill_accepts_explicit_mode_override(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(image, mask, seed=321, progress=False, mode='scaled')


def test_ensure_active_flux_fill_session_reuses_existing_session(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.ensure_active_flux_fill_session(progress=False)


def test_reconcile_active_flux_fill_session_is_archived_noop(monkeypatch):
    result = objr_engine.reconcile_active_flux_fill_session(
        route_family='txt2img',
        selected_engine=objr_engine.OBJR_ENGINE_MAT,
        progress=False,
    )
    assert result.decision == 'ignored'
    assert result.text_encoder_action == 'cleared'


def test_removal_stage_rejects_archived_flux_engine_before_runtime_calls(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    from backend import resources
    from backend.flux_fill_v3.contracts import FluxFillResult
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly
    from PIL import Image

    monkeypatch.setattr(resources, "cleanup_memory", lambda *args, **kwargs: None)

    # Mock activation to return dummy paths
    import backend.flux_fill_v3.activation as activation
    monkeypatch.setattr(activation, "resolve_flux_fill_assets", lambda state: FluxFillActivationAssets(
        unet_path="unet.safetensors",
        ae_path="ae.safetensors",
        conditioning_cache_path="empty_conditioning.pt",
        model_variant="flux_fill_fp8",
        conditioning_kind="empty",
        clip_l_path="clip_l.safetensors",
        t5_path="t5xxl.safetensors",
        prompt="",
    ))

    # Mock Image.open to return a dummy image
    class DummyImage:
        def __init__(self, size):
            self.size = size
        def convert(self, mode):
            if mode == 'RGB':
                return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            else:
                return np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(Image, "open", lambda path: DummyImage((32, 24)))

    called_requests = []
    def mock_execute(self, req, callback=None):
        called_requests.append(req)
        return FluxFillResult(
            output_image=np.ones((req.image.shape[0], req.image.shape[1], 3), dtype=np.uint8),
            seed=req.seed,
            width=req.image.shape[1],
            height=req.image.shape[0],
        )

    monkeypatch.setattr(FluxAssembly, "execute", mock_execute)

    # Mock save logged output
    monkeypatch.setattr(routes, "_save_logged_output", lambda *args, **kwargs: "mock_removal.png")

    task_state = SimpleNamespace(
        goals=[flags.remove_obj],
        remove_base_image='image.png',
        remove_mask_image='mask.png',
        inpaint_context='stale-context',
        seed=42,
        steps=18,
        sampler_name='dpmpp_2m',
        scheduler_name='karras',
        objr_mask_dilate=9,
        objr_engine=objr_engine.OBJR_ENGINE_FLUX_FILL,
        flux_fill_conditioning='empty',
        remove_prompt='',
        flux_fill_prompt_cache='temp',
        objr_mask_blur=6,
        objr_blend_mode='morphological',
        prefetch_depth=1,
        prefetch_chunk_mb=128,
    )
    context = SimpleNamespace(
        task_state=task_state,
        progressbar_callback=None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = routes.RemovalStage().execute(context)
    assert res.route_complete is True
    assert len(called_requests) == 1
    assert task_state.inpaint_context is None
    assert called_requests[0].prefetch_depth == 1
    assert called_requests[0].prefetch_chunk_mb == 128


def test_flux_fill_removal_stage_uses_adapter_diffusion_preflight(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    from backend import resources
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly
    from backend.flux_fill_v3.contracts import FluxFillResult
    from PIL import Image

    cleanup_calls = []
    called_requests = []

    monkeypatch.setattr(resources, "cleanup_memory", lambda reason, **kwargs: cleanup_calls.append((reason, kwargs.get("target_phase"))))

    import backend.flux_fill_v3.activation as activation
    monkeypatch.setattr(activation, "resolve_flux_fill_assets", lambda state: FluxFillActivationAssets(
        unet_path="unet.safetensors",
        ae_path="ae.safetensors",
        conditioning_cache_path="empty_conditioning.pt",
        model_variant="flux_fill_fp8",
        conditioning_kind="empty",
        clip_l_path="clip_l.safetensors",
        t5_path="t5xxl.safetensors",
        prompt="repair statue",
    ))

    class DummyImage:
        def __init__(self, size):
            self.size = size
        def convert(self, mode):
            if mode == "RGB":
                return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            return np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(Image, "open", lambda path: DummyImage((32, 24)))
    monkeypatch.setattr(objr_engine, "prepare_flux_fill_mask", lambda mask, grow, blur: mask)

    def mock_execute(self, req, callback=None):
        called_requests.append(req)
        return FluxFillResult(
            output_image=np.ones((req.image.shape[0], req.image.shape[1], 3), dtype=np.uint8),
            seed=req.seed,
            width=req.image.shape[1],
            height=req.image.shape[0],
        )

    monkeypatch.setattr(FluxAssembly, "execute", mock_execute)
    monkeypatch.setattr(routes, "_save_logged_output", lambda *args, **kwargs: "mock_removal.png")

    task_state = SimpleNamespace(
        goals=[flags.remove_obj],
        remove_base_image="image.png",
        remove_mask_image="mask.png",
        inpaint_context=None,
        seed=42,
        steps=18,
        sampler_name="dpmpp_2m",
        scheduler_name="karras",
        objr_mask_dilate=9,
        objr_engine=objr_engine.OBJR_ENGINE_FLUX_FILL,
        flux_fill_conditioning="empty",
        remove_prompt="repair statue",
        flux_fill_prompt_cache="temp",
        objr_mask_blur=6,
        objr_blend_mode="morphological",
        prefetch_depth=1,
        prefetch_chunk_mb=128,
        process_transition_previous_family="flux_fill",
        process_transition_reuse_allowed=True,
    )
    context = SimpleNamespace(
        task_state=task_state,
        progressbar_callback=None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = routes.RemovalStage().execute(context)

    assert res.route_complete is True
    assert len(called_requests) == 1
    reasons = [reason for reason, _phase in cleanup_calls]
    assert "removal_preflight" not in reasons
    assert "flux_removal_preflight" not in reasons
    assert ("flux_removal_image_complete", resources.MemoryPhase.DIFFUSION) in cleanup_calls


def test_flux_fill_removal_republishes_runtime_after_stage_cleanup(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    import backend.flux_fill_v3.removal_adapter as removal_adapter
    from backend import process_transition, resources
    from backend.flux_fill_v3.activation import FluxFillActivationAssets
    from backend.flux_fill_v3.assembly import FluxAssembly
    from backend.flux_fill_v3.contracts import FluxFillResult
    from backend.process_transition import PROCESS_FAMILY_FLUX_FILL
    from PIL import Image

    process_transition.clear_active_runtime()

    def fake_assets(_state):
        return FluxFillActivationAssets(
            unet_path="unet.safetensors",
            ae_path="ae.safetensors",
            conditioning_cache_path="empty_conditioning.pt",
            model_variant="flux_fill_fp8",
            conditioning_kind="empty",
            clip_l_path="clip_l.safetensors",
            t5_path="t5xxl.safetensors",
            prompt="repair statue",
        )

    import backend.flux_fill_v3.activation as activation
    monkeypatch.setattr(activation, "resolve_flux_fill_assets", fake_assets)
    monkeypatch.setattr(removal_adapter, "resolve_flux_fill_assets", fake_assets)
    monkeypatch.setattr(resources, "cleanup_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "_save_logged_output", lambda *args, **kwargs: "mock_removal.png")
    monkeypatch.setattr(objr_engine, "prepare_flux_fill_mask", lambda mask, grow, blur: mask)

    class DummyImage:
        def __init__(self, size):
            self.size = size
        def convert(self, mode):
            if mode == "RGB":
                return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            return np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(Image, "open", lambda path: DummyImage((32, 24)))

    def mock_execute(self, req, callback=None):
        return FluxFillResult(
            output_image=np.ones((req.image.shape[0], req.image.shape[1], 3), dtype=np.uint8),
            seed=req.seed,
            width=req.image.shape[1],
            height=req.image.shape[0],
        )

    monkeypatch.setattr(FluxAssembly, "execute", mock_execute)

    task_state = SimpleNamespace(
        goals=[flags.remove_obj],
        remove_base_image="image.png",
        remove_mask_image="mask.png",
        inpaint_context=None,
        seed=42,
        steps=18,
        sampler_name="dpmpp_2m",
        scheduler_name="karras",
        objr_mask_dilate=9,
        objr_engine=objr_engine.OBJR_ENGINE_FLUX_FILL,
        flux_fill_conditioning="empty",
        flux_fill_runtime_posture="streaming",
        flux_fill_unet_spine="streaming",
        remove_prompt="repair statue",
        flux_fill_prompt_cache="temp",
        objr_mask_blur=6,
        objr_blend_mode="morphological",
        prefetch_depth=1,
        prefetch_chunk_mb=128,
        process_transition_previous_family="sdxl",
        process_transition_reuse_allowed=False,
        negative_prompt="",
    )
    context = SimpleNamespace(
        task_state=task_state,
        route_id="flux_removal",
        progressbar_callback=None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    try:
        res = routes.RemovalStage().execute(context)
        active_key = process_transition.get_active_process_key()

        assert res.route_complete is True
        assert active_key is not None
        assert active_key.family == PROCESS_FAMILY_FLUX_FILL
        assert ("unet_path", "unet.safetensors") in active_key.authoritative_identity
        assert process_transition.get_active_route_owner() == "flux_removal"
    finally:
        process_transition.clear_active_runtime()
        from backend.flux_fill_v3.runtime_state import (
            release_active_flux_resident_spine,
            release_flux_latent_artifacts,
        )
        release_active_flux_resident_spine(reason="test_cleanup")
        release_flux_latent_artifacts()


def test_flux_fill_removal_stage_aggressively_cleans_when_preceding_process_is_non_flux(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    from backend import resources
    from types import SimpleNamespace as ResultNamespace

    cleanup_calls = []

    monkeypatch.setattr(resources, "cleanup_memory", lambda reason, **kwargs: cleanup_calls.append((reason, kwargs.get("target_phase"))))

    import backend.flux_fill_v3.removal_adapter as removal_adapter
    monkeypatch.setattr(
        removal_adapter,
        "execute_flux_fill_removal",
        lambda context, progress_percent_start=10: ResultNamespace(output_image=np.ones((24, 32, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(routes, "_save_logged_output", lambda *args, **kwargs: "mock_removal.png")

    task_state = SimpleNamespace(
        goals=[flags.remove_obj],
        remove_base_image="image.png",
        remove_mask_image="mask.png",
        inpaint_context=None,
        seed=42,
        steps=18,
        sampler_name="dpmpp_2m",
        scheduler_name="karras",
        objr_mask_dilate=9,
        objr_engine=objr_engine.OBJR_ENGINE_FLUX_FILL,
        flux_fill_conditioning="empty",
        remove_prompt="repair statue",
        flux_fill_prompt_cache="temp",
        objr_mask_blur=6,
        objr_blend_mode="morphological",
        prefetch_depth=1,
        prefetch_chunk_mb=128,
        process_transition_previous_family="sdxl",
        process_transition_reuse_allowed=False,
        negative_prompt="",
    )
    context = SimpleNamespace(
        task_state=task_state,
        progressbar_callback=None,
        yield_result_callback=lambda *args, **kwargs: None,
    )

    res = routes.RemovalStage().execute(context)

    assert res.route_complete is True
    assert ("removal_preflight", resources.MemoryPhase.REMOVAL) in cleanup_calls


def test_request_interrupt_does_not_restore_archived_flux_runtime(monkeypatch):
    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.ensure_active_flux_fill_session(progress=False)


def test_removal_stage_persists_background_and_object_outputs(monkeypatch):
    import modules.flags as flags
    import modules.pipeline.routes as routes
    import backend.auxiliary_workers as auxiliary_workers
    from backend import resources

    yielded = []
    persisted = []
    task_state = SimpleNamespace(
        goals=[flags.remove_bg, flags.remove_obj],
        remove_base_image='image.png',
        remove_mask_image='initial-mask.png',
        seed=7,
        steps=24,
        sampler_name='euler',
        scheduler_name='normal',
        objr_mask_dilate=16,
        objr_engine=objr_engine.OBJR_ENGINE_MAT,
        flux_fill_conditioning='empty',
        remove_prompt='',
        flux_fill_prompt_cache='temp',
        objr_mask_blur=6,
        objr_blend_mode='morphological',
        bgr_threshold=0.4,
        bgr_jit=True,
    )
    context = SimpleNamespace(
        task_state=task_state,
        progressbar_callback=None,
        yield_result_callback=lambda task, paths, pct, do_not_show_finished_images=False: yielded.append((paths, pct, do_not_show_finished_images)),
    )

    monkeypatch.setattr(resources, 'begin_memory_phase', lambda *args, **kwargs: None)
    monkeypatch.setattr(resources, 'end_memory_phase', lambda *args, **kwargs: None)
    monkeypatch.setattr(resources, 'cleanup_memory', lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, '_load_removal_array', lambda filepath, mode: np.zeros((8, 8, 3), dtype=np.uint8) if mode == 'RGB' else np.zeros((8, 8), dtype=np.uint8))
    monkeypatch.setattr(auxiliary_workers, 'run_background_removal', lambda *args, **kwargs: (np.zeros((8, 8, 4), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)))
    monkeypatch.setattr(auxiliary_workers, 'run_mat_inpaint', lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(routes, '_save_removal_temp', lambda payload: 'char-temp.png' if np.asarray(payload).ndim == 3 and np.asarray(payload).shape[-1] == 4 else 'mask-temp.png' if np.asarray(payload).ndim == 2 else 'obj-temp.png')

    def fake_save_logged_output(context, payload, description, **kwargs):
        persisted.append((payload, description, kwargs))
        return f'saved::{description}'

    monkeypatch.setattr(routes, '_save_logged_output', fake_save_logged_output)

    result = routes.RemovalStage().execute(context)

    assert result.route_complete is True
    assert task_state.remove_mask_image == 'mask-temp.png'
    assert persisted == [
        ('char-temp.png', 'Background Removal Subject', {'seed': 7, 'workflow': 'bgr_subject'}),
        ('mask-temp.png', 'Background Removal Mask', {'seed': 7, 'workflow': 'bgr_mask'}),
        ('obj-temp.png', 'Object Removal', {'prompt_text': '', 'negative_prompt': '', 'seed': 7, 'workflow': 'remove_mat'}),
    ]
    assert yielded == [
        (['saved::Background Removal Subject', 'saved::Background Removal Mask'], 50, True),
        (['saved::Object Removal'], 100, True),
    ]


def test_objr_engine_change_sets_flux_dilate_default(monkeypatch):
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    import modules.ui_logic as ui_logic

    assert ui_logic.objr_engine_change(objr_engine.OBJR_ENGINE_FLUX_FILL)['value'] == 16
    assert ui_logic.objr_engine_change(objr_engine.OBJR_ENGINE_MAT)['value'] == 16


def test_select_flux_fill_mode_prefers_native_baseline_and_non_native_context_crop():
    non_native = np.zeros((960, 1280, 3), dtype=np.uint8)
    assert objr_engine._select_flux_fill_mode(non_native) == 'context_crop'


def test_remove_object_flux_fill_defaults_to_baseline_for_native_sdxl(monkeypatch):
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[400:624, 400:624] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(image, mask, seed=321, progress=False)


def test_remove_object_flux_fill_accepts_explicit_context_crop_override(monkeypatch):
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    mask = np.zeros((960, 1280), dtype=np.uint8)
    mask[320:520, 520:720] = 255

    with pytest.raises(objr_engine.LegacyFluxArchivedError):
        objr_engine.remove_object_flux_fill(image, mask, seed=321, progress=False, mode='context_crop')


def _prepared_inpaint_state(*, bbox):
    import json

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


def test_prepared_inpaint_uses_frozen_source_bbox_for_stitching():
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


def test_prepared_inpaint_fails_closed_without_frozen_bbox():
    from modules.pipeline.image_input import apply_inpaint

    state, source = _prepared_inpaint_state(bbox=None)
    with pytest.raises(ValueError, match='bbox is missing'):
        apply_inpaint(state, source, np.zeros(source.shape[:2], dtype=np.uint8))


def test_additional_lora_channel_policy_is_unet_only():
    from modules.lora_channel_policy import resolve_lora_channels

    user_decision = resolve_lora_channels(
        file_identity=None,
        requested_unet_weight=0.7,
        requested_clip_weight=0.7,
        provenance='input',
        raw_path='user.safetensors',
    )
    additional_decision = resolve_lora_channels(
        file_identity=None,
        requested_unet_weight=1.0,
        requested_clip_weight=0.0,
        provenance='additional',
        raw_path='inpaint_v26.fooocus.patch',
    )

    assert (user_decision.effective_unet_weight, user_decision.effective_clip_weight) == (0.7, 0.7)
    assert (additional_decision.effective_unet_weight, additional_decision.effective_clip_weight) == (1.0, 0.0)


def test_color_enhancement_declares_only_the_sdxl_color_pass():
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection

    plan = compile_workflow_plan(FrozenWorkflowSelection(source_surface='color_enhanced_upscale'))
    declaration = plan.execution_declaration
    assert declaration.main_family == 'sdxl'
    assert declaration.ordered_auxiliary_requirements == ()
    assert tuple(step.step_id for step in declaration.ordered_steps) == ('sdxl_color_pass',)
