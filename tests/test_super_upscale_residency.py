import os
import sys
import types
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.argv = [sys.argv[0]]

mock_args = types.SimpleNamespace(
    colab=False,
    preset=None,
    output_path=None,
    temp_path=None,
    skip_model_load=False,
    disable_preset_selection=False,
    disable_image_log=False,
)

sys.modules['args_manager'] = types.ModuleType('args_manager')
sys.modules['args_manager'].args = mock_args

import numpy as np
import pytest
import torch

from backend import sdxl_runtime_policy, process_transition, resources
from modules import flags
from modules.pipeline import inference, tiled_refinement
from modules.task_state import TaskState
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
import backend.sdxl_unified_runtime as runtime_mod
import backend.sdxl_streaming_runtime as streaming_mod


@pytest.fixture(autouse=True)
def _clear_process_and_interrupt_state(monkeypatch):
    process_transition.clear_active_process_key()
    resources.interrupt_current_processing(False)
    
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
    monkeypatch.setattr(rb, 'get_file_identity', lambda path: types.SimpleNamespace(path=Path(path), sha256='123'))
    
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
    process_transition.clear_active_process_key()
    resources.interrupt_current_processing(False)


def _build_task_state():
    task_state = TaskState(
        prompt='prompt',
        negative_prompt='negative',
        width=64,
        height=64,
        steps=3,
        cfg_scale=5.0,
        sampler_name='euler',
        scheduler_name='karras',
        seed=123,
        base_model_name='model.safetensors',
        clip_model_name='None',
        vae_name=flags.default_vae,
        input_image_checkbox=False,
        current_tab='txt2img',
        sdxl_execution_policy=types.SimpleNamespace(
            enabled=True,
            execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
            residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
        ),
        sdxl_execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_STANDARD,
        sdxl_residency_class=sdxl_runtime_policy.SDXL_RESIDENCY_CLASS_FULL,
    )
    task_state.current_progress = 0
    task_state.image_number = 1
    task_state.disable_intermediate_results = True
    return task_state


def _build_super_upscale_task_state():
    task_state = _build_task_state()
    task_state.input_image_checkbox = True
    task_state.current_tab = 'uov'
    task_state.uov_method = 'super-upscale'
    task_state.uov_input_image = np.zeros((64, 64, 3), dtype=np.uint8)
    task_state.upscale_gan_output_image = np.zeros((128, 128, 3), dtype=np.uint8)
    task_state.goals = ['upscale']
    # Tiled refinement is a queue-bound consumer and now requires the frozen
    # workflow plan that production queue capture supplies.
    bind_legacy_workflow_plan(task_state)
    return task_state


def test_normalize_pathish_resolves_paths_without_name_error():
    normalized = inference._normalize_pathish('models/example.safetensors')
    assert normalized.endswith(os.path.join('models', 'example.safetensors'))


def test_should_retain_sdxl_warm_state_policy_disabled():
    task_state = _build_super_upscale_task_state()
    task_state.sdxl_execution_policy = None
    assert tiled_refinement.should_retain_sdxl_warm_state(task_state) is False

    task_state.sdxl_execution_policy = types.SimpleNamespace(enabled=False)
    assert tiled_refinement.should_retain_sdxl_warm_state(task_state) is False


def test_should_retain_sdxl_warm_state_requires_active_process(monkeypatch):
    task_state = _build_super_upscale_task_state()
    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'D:/resolved/model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)

    assert tiled_refinement.should_retain_sdxl_warm_state(task_state) is False


def test_should_retain_sdxl_warm_state_uses_processed_loras(monkeypatch):
    task_state = _build_super_upscale_task_state()
    task_state.loras = [('D:/raw/slot.safetensors', 1.0)]
    task_state.loras_processed = [('D:/resolved/prompt_lora.safetensors', 0.75)]
    task_state.base_model_additional_loras = [('D:/resolved/additional.safetensors', 1.0)]

    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'D:/resolved/model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)

    processed_loras = task_state.loras_processed
    task_state.loras_processed = None
    raw_key = inference.resolve_unified_sdxl_process_key(task_state)
    task_state.loras_processed = processed_loras

    process_transition.set_active_process_key(raw_key)
    assert tiled_refinement.should_retain_sdxl_warm_state(task_state) is False

    expected_key = inference.resolve_unified_sdxl_process_key(task_state)
    process_transition.set_active_process_key(expected_key)
    assert tiled_refinement.should_retain_sdxl_warm_state(task_state) is True


def _patch_assembly_mocks(monkeypatch, seam_encode_calls=None, tile_callback_trigger=None):
    import types
    import torch
    import numpy as np
    from backend.sdxl_assembly.director import SDXLAssemblyDirector
    from backend.sdxl_assembly.assembly import SDXLAssembly

    class MockVaeEncodeWorker:
        def __init__(self, request):
            pass
        def encode(self, prepared):
            if seam_encode_calls is not None:
                seam_encode_calls.append(tuple(int(dim) for dim in prepared.original_pixels.shape))
            return types.SimpleNamespace(route_latent=torch.ones((1, 4, 8, 8), dtype=torch.float32))

    class MockVaeDecodeWorker:
        def __init__(self, request):
            pass
        def decode(self, latent, device):
            return np.zeros((64, 64, 3), dtype=np.uint8), 0.0, 0.0

    class MockUnetSpine:
        def __init__(self, request, lora_worker=None):
            pass
        def start(self):
            pass
        def denoise(self, latent, conditioning, callback=None):
            if tile_callback_trigger is not None:
                tile_callback_trigger(callback)
            return torch.zeros((1, 4, 8, 8), dtype=torch.float16)
        def end(self):
            pass

    class MockLoraWorker:
        def __init__(self, request):
            pass
        def materialize_patches(self):
            return []

    class MockTextEncodeWorker:
        def __init__(self, request, lora_worker=None):
            pass
        def get_conditioning(self):
            return {}

    def fake_select_assembly(request, **_kwargs):
        return SDXLAssembly(
            lora_worker=MockLoraWorker(request),
            text_encode_worker=MockTextEncodeWorker(request),
            unet_spine=MockUnetSpine(request),
            vae_decode_worker=MockVaeDecodeWorker(request),
            vae_encode_worker=MockVaeEncodeWorker(request),
            st_preprocess_worker=None,
            st_control_worker=None,
            ctx_control_worker=None,
        )

    monkeypatch.setattr(SDXLAssemblyDirector, 'select_assembly', fake_select_assembly)


def test_tiled_refinement_uses_runtime_resident_vae_encode_seam(monkeypatch):
    task_state = _build_super_upscale_task_state()
    seam_encode_calls = []

    _patch_assembly_mocks(monkeypatch, seam_encode_calls=seam_encode_calls)
    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)
    import modules.pipeline.preprocessing as preprocessing
    monkeypatch.setattr(preprocessing, 'patch_samplers', lambda *_args, **_kwargs: 'karras')
    monkeypatch.setattr(tiled_refinement.resources, 'memory_phase_scope', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(tiled_refinement.resources, 'cleanup_memory', lambda *args, **kwargs: None)
    monkeypatch.setattr(tiled_refinement.resources, 'load_models_gpu', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_models_gpu should not run for resident VAE seam")))
    monkeypatch.setattr(tiled_refinement, 'select_tile_resolution', lambda *_args, **_kwargs: ((64, 64), 1, 1, 0, 0))
    tiles = [
        tiled_refinement.TileInfo(crop=(0, 0, 64, 64), tile_image=np.ones((64, 64, 3), dtype=np.uint8), x=0, y=0, w=64, h=64),
    ]
    monkeypatch.setattr(tiled_refinement, 'split_into_tiles', lambda *_args, **_kwargs: tiles)
    monkeypatch.setattr(tiled_refinement, 'stitch_tiles', lambda tiles_list, *_args, **_kwargs: tiles_list[0].tile_image)

    result = tiled_refinement.apply_tiled_diffusion_refinement(task_state, task_state.uov_input_image)

    assert result.shape == (64, 64, 3)
    assert seam_encode_calls == [(1, 64, 64, 3)]


def test_unload_all_models_clears_active_sdxl_process_key(monkeypatch):
    task_state = _build_super_upscale_task_state()
    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'D:/resolved/model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resources, 'free_memory', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resources, 'get_torch_device', lambda: torch.device('cpu'))

    process_transition.set_active_process_key(inference.resolve_unified_sdxl_process_key(task_state))
    assert process_transition.get_active_process_key() is not None

    # unload_all_models should preserve warmth (active process key is kept)
    resources.unload_all_models()
    assert process_transition.get_active_process_key() is not None
    assert process_transition.get_active_family() == 'sdxl'

    # teardown_active_runtime should clear it
    resources.teardown_active_runtime("test_teardown")
    assert process_transition.get_active_process_key() is None
    assert process_transition.get_active_family() is None


def test_tiled_refinement_stop_interrupt(monkeypatch):
    task_state = _build_super_upscale_task_state()
    task_state.last_stop = 'stop'
    resources.interrupt_current_processing(True)
    import backend.sdxl_assembly.gateway as assembly_gateway
    monkeypatch.setattr(
        assembly_gateway,
        'run_sdxl_assembly_task',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stop interrupt must not enter the assembly gateway")),
    )

    with pytest.raises(resources.InterruptProcessingException):
        tiled_refinement.apply_tiled_diffusion_refinement(task_state, task_state.uov_input_image)

    assert resources.processing_interrupted() is False


def test_tiled_refinement_skip_interrupt(monkeypatch):
    task_state = _build_super_upscale_task_state()
    task_state.last_stop = 'skip'
    resources.interrupt_current_processing(True)
    import backend.sdxl_assembly.gateway as assembly_gateway
    monkeypatch.setattr(
        assembly_gateway,
        'run_sdxl_assembly_task',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip interrupt must not enter the assembly gateway")),
    )

    target_image = np.ones((96, 128, 3), dtype=np.uint8) * 23
    result = tiled_refinement.apply_tiled_diffusion_refinement(task_state, target_image)

    assert np.array_equal(result, target_image)
    assert result is not target_image
    assert task_state.last_stop is False
    assert resources.processing_interrupted() is False


def test_tiled_refinement_posture_teardown_behavior(monkeypatch):
    task_state = _build_super_upscale_task_state()
    
    teardown_calls = []
    cleanup_calls = []

    monkeypatch.setattr(resources, 'teardown_active_runtime', lambda reason=None: teardown_calls.append(reason))
    monkeypatch.setattr(resources, 'cleanup_memory', lambda reason, **kwargs: cleanup_calls.append((reason, kwargs.get('unload_models'))))
    monkeypatch.setattr(resources, 'load_models_gpu', lambda *args, **kwargs: None)
    
    _patch_assembly_mocks(monkeypatch)

    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)
    import modules.pipeline.preprocessing as preprocessing
    monkeypatch.setattr(preprocessing, 'patch_samplers', lambda *_args, **_kwargs: 'karras')
    monkeypatch.setattr(tiled_refinement.resources, 'memory_phase_scope', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(tiled_refinement, 'select_tile_resolution', lambda *_args, **_kwargs: ((64, 64), 1, 1, 0, 0))
    tiles = [
        tiled_refinement.TileInfo(crop=(0, 0, 64, 64), tile_image=np.ones((64, 64, 3), dtype=np.uint8), x=0, y=0, w=64, h=64),
    ]
    monkeypatch.setattr(tiled_refinement, 'split_into_tiles', lambda *_args, **_kwargs: tiles)
    monkeypatch.setattr(tiled_refinement, 'stitch_tiles', lambda tiles_list, *_args, **_kwargs: tiles_list[0].tile_image)

    # Case 1: retain_warm is False -> should call teardown_active_runtime at pre-flight and finalization
    monkeypatch.setattr(tiled_refinement, 'should_retain_sdxl_warm_state', lambda *args: False)
    
    tiled_refinement.apply_tiled_diffusion_refinement(task_state, task_state.uov_input_image)
    
    assert 'upscale_preflight' in teardown_calls
    assert 'upscale_finalization' in teardown_calls
    # cleanup_memory should be called with unload_models=False on tile complete, and not for preflight/finalize
    assert any(reason == 'tiled_refine_tile_complete' for reason, _ in cleanup_calls)
    assert not any(reason == 'tiled_refine_preflight' for reason, _ in cleanup_calls)
    assert not any(reason == 'tiled_refine_finalize' for reason, _ in cleanup_calls)

    # Reset lists
    teardown_calls.clear()
    cleanup_calls.clear()

    # Case 2: retain_warm is True -> should call cleanup_memory with unload_models=False instead of teardown_active_runtime
    monkeypatch.setattr(tiled_refinement, 'should_retain_sdxl_warm_state', lambda *args: True)
    
    tiled_refinement.apply_tiled_diffusion_refinement(task_state, task_state.uov_input_image)
    
    assert len(teardown_calls) == 0
    assert any(reason == 'tiled_refine_preflight' and unload is False for reason, unload in cleanup_calls)
    assert any(reason == 'tiled_refine_finalize' and unload is False for reason, unload in cleanup_calls)
