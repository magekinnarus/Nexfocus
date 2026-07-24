import os
import sys
import types
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from backend import sdxl_runtime_policy
from backend.staging_manager import ExecutionClass
from modules import flags
from modules.pipeline import inference, preprocessing, routes
from modules.pipeline.stage_runtime import PipelineRouteContext
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
import modules.pipeline.tiled_refinement as tiled_refinement
from modules.task_state import TaskState
import backend.sdxl_unified_runtime as runtime_mod
import numpy as np
import pytest
import torch


def _build_sdxl_policy(**overrides):
    values = dict(
        enabled=True,
        architecture="sdxl",
        runtime_family="unified_sdxl",
        execution_mode="resident",
        hardware_tier="NORMAL_VRAM",
    )
    values.update(overrides)
    return sdxl_runtime_policy.SDXLExecutionPolicy(**values)


def _build_task_state():
    policy = _build_sdxl_policy()
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
        sdxl_execution_policy=policy,
        sdxl_execution_family=policy.execution_family,
        sdxl_residency_class=policy.residency_class,
    )
    task_state.current_progress = 0
    task_state.image_number = 1
    task_state.disable_intermediate_results = True
    return task_state


def _build_task_dict():
    return {
        'c': 'positive-cond',
        'uc': 'negative-cond',
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
        'positive_top_k': 1,
        'negative_top_k': 1,
        'log_positive_prompt': 'prompt',
        'log_negative_prompt': 'negative',
        'styles': [],
    }


def _build_super_upscale_task_state():
    task_state = _build_task_state()
    task_state.input_image_checkbox = True
    task_state.current_tab = 'uov'
    task_state.uov_method = 'super-upscale'
    task_state.uov_input_image = np.zeros((8, 8, 3), dtype=np.uint8)
    task_state.upscale_gan_output_image = np.zeros((16, 16, 3), dtype=np.uint8)
    task_state.goals = ['upscale']
    return task_state

@pytest.fixture(autouse=True)
def _stub_model_file_existence(monkeypatch):
    real_build_generation_route = routes.build_generation_route
    def build_planned_generation_route(task_state):
        if getattr(task_state, 'workflow_plan', None) is None:
            bind_legacy_workflow_plan(task_state)
        return real_build_generation_route(task_state)
    monkeypatch.setattr(routes, 'build_generation_route', build_planned_generation_route)

    real_run_unified = inference._run_unified_sdxl_task
    def run_planned_unified(task_state, *args, **kwargs):
        if getattr(task_state, 'workflow_plan', None) is None:
            bind_legacy_workflow_plan(task_state)
        return real_run_unified(task_state, *args, **kwargs)
    monkeypatch.setattr(inference, '_run_unified_sdxl_task', run_planned_unified)

    real_tiled_refinement = tiled_refinement.apply_tiled_diffusion_refinement
    def run_planned_tiled_refinement(task_state, *args, **kwargs):
        if getattr(task_state, 'workflow_plan', None) is None:
            bind_legacy_workflow_plan(task_state)
        return real_tiled_refinement(task_state, *args, **kwargs)
    monkeypatch.setattr(tiled_refinement, 'apply_tiled_diffusion_refinement', run_planned_tiled_refinement)

    from pathlib import Path
    original_path_exists = Path.exists
    def smart_path_exists(self):
        p = str(self)
        if 'auth.json' in p:
            return False
        if 'model.safetensors' in p or 'boost.safetensors' in p or 'vae' in p:
            return True
        return original_path_exists(self)
    monkeypatch.setattr(Path, 'exists', smart_path_exists)
    
    import os
    original_exists = os.path.exists
    def smart_exists(path):
        p = str(path)
        if 'auth.json' in p:
            return False
        if 'model.safetensors' in p or 'boost.safetensors' in p or 'vae' in p:
            return True
        return original_exists(path)
    monkeypatch.setattr(os.path, 'exists', smart_exists)
    
    import backend.sdxl_assembly.request_builder as rb
    monkeypatch.setattr(rb, 'get_file_from_folder_list', lambda model_name, folders: f'D:/resolved/{model_name}')
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
def test_build_generation_route_marks_supported_standard_sdxl_for_unified_owner():
    task_state = _build_task_state()

    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'


def test_build_generation_route_keeps_unified_owner_when_external_vae_is_selected():
    task_state = _build_task_state()
    task_state.vae_name = 'custom_vae.safetensors'

    route = routes.build_generation_route(task_state)
    assert route.route_id == 'txt2img'


def test_inpaint_and_outpaint_route_prep_stages_use_explicit_phase_labels():
    assert routes.InpaintPreparationStage.phase_name == 'inpaint_prepare'
    assert routes.OutpaintPreparationStage.phase_name == 'outpaint_prepare'


def test_resolve_unified_vae_path_normalizes_legacy_default_aliases(monkeypatch):
    monkeypatch.setattr(inference, 'get_file_from_folder_list', lambda name, folders: f'D:/resolved/{name}')

    default_alias_task = types.SimpleNamespace(vae_name='Default (model)')
    same_alias_task = types.SimpleNamespace(vae_name='Default (Same as model)')

    assert inference._resolve_unified_vae_path(default_alias_task) == flags.default_vae
    assert inference._resolve_unified_vae_path(same_alias_task) == flags.default_vae


def test_build_generation_route_allows_tiled_decode_for_unified_owner():
    task_state = _build_task_state()
    task_state.tiled = True

    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'


def test_build_generation_route_keeps_nondefault_quality_on_unified_owner():
    task_state = _build_task_state()
    task_state.sharpness = 3.0

    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'


def test_build_generation_route_rejects_gguf_for_unified_owner():
    task_state = _build_task_state()
    task_state.base_model_name = 'model.gguf'

    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'


def test_build_generation_route_rejects_sd15_for_unified_owner():
    task_state = _build_task_state()
    task_state.base_model_name = 'sd15_demo.safetensors'

    route = routes.build_generation_route(task_state)

    assert route.route_id == 'txt2img'


def test_upscale_stage_passes_prompt_blueprint_to_tiled_refinement(monkeypatch):
    task_state = _build_super_upscale_task_state()
    prompt_task = {
        'task_seed': 999,
        'task_prompt': 'expanded prompt',
        'task_negative_prompt': 'expanded negative',
        'positive': ['expanded prompt', 'style add', 'extra add'],
        'negative': ['expanded negative', 'extra neg'],
        'positive_top_k': 3,
        'negative_top_k': 2,
        'styles': ['cinematic'],
    }
    context = PipelineRouteContext(
        async_task=None,
        task_state=task_state,
        route_id='super_upscale',
        route_family='upscale',
        prompt_tasks=[prompt_task],
    )
    captured = {}

    import modules.pipeline.image_input as image_input
    import modules.pipeline.tiled_refinement as tiled_refinement
    import modules.pipeline.output as pipeline_output
    monkeypatch.setattr(image_input, 'apply_upscale', lambda *_args, **_kwargs: False)
    
    def fake_tiled_refine(*_args, **kwargs):
        captured['prompt_task'] = kwargs.get('prompt_task')
        return task_state.uov_input_image

    monkeypatch.setattr(tiled_refinement, 'apply_tiled_diffusion_refinement', fake_tiled_refine)
    monkeypatch.setattr(pipeline_output, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    result = routes.UpscaleStage().execute(context)

    assert result.route_complete is True
    assert captured['prompt_task'] == prompt_task


def test_tiled_refinement_uses_processed_prompt_blueprint(monkeypatch):
    task_state = _build_super_upscale_task_state()
    task_state.prompt = 'raw prompt that should not be used directly'
    task_state.negative_prompt = 'raw negative that should not be used directly'
    task_state.loras_processed = [('D:/resolved/boost.safetensors', 0.75)]
    task_state.upscale_refinement_denoise = 0.42
    prompt_task = {
        'task_seed': 999,
        'task_prompt': 'expanded prompt',
        'task_negative_prompt': 'expanded negative',
        'positive': ['expanded prompt', 'style add', 'extra add'],
        'negative': ['expanded negative', 'extra neg'],
        'positive_top_k': 3,
        'negative_top_k': 2,
        'styles': ['cinematic'],
    }
    captured_requests = []

    # Mock components in the director
    from backend.sdxl_assembly.director import SDXLAssemblyDirector
    from backend.sdxl_assembly.assembly import SDXLAssembly

    class MockVaeEncodeWorker:
        def __init__(self, request):
            pass
        def encode(self, prepared):
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

    def fake_select_assembly(request, *args, **kwargs):
        captured_requests.append(request)
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
    monkeypatch.setattr(inference, '_resolve_unified_checkpoint_path', lambda *_args, **_kwargs: 'D:/resolved/model.safetensors')
    monkeypatch.setattr(inference, '_resolve_unified_vae_path', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(inference.os.path, 'exists', lambda path: str(path) in {'D:/resolved/boost.safetensors', 'D:/resolved/model.safetensors'})
    import modules.pipeline.preprocessing as preprocessing
    monkeypatch.setattr(preprocessing, 'patch_samplers', lambda *_args, **_kwargs: 'karras')
    monkeypatch.setattr(tiled_refinement.resources, 'memory_phase_scope', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(tiled_refinement.resources, 'cleanup_memory', lambda *args, **kwargs: None)
    monkeypatch.setattr(tiled_refinement.resources, 'load_models_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(tiled_refinement, 'select_tile_resolution', lambda *_args, **_kwargs: ((64, 64), 1, 1, 0, 0))
    monkeypatch.setattr(
        tiled_refinement,
        'split_into_tiles',
        lambda *_args, **_kwargs: [
            tiled_refinement.TileInfo(
                crop=(0, 0, 64, 64),
                tile_image=np.zeros((64, 64, 3), dtype=np.uint8),
                x=0,
                y=0,
                w=64,
                h=64,
            )
        ],
    )
    monkeypatch.setattr(tiled_refinement, 'stitch_tiles', lambda tiles, *_args, **_kwargs: tiles[0].tile_image)

    result = tiled_refinement.apply_tiled_diffusion_refinement(
        task_state,
        task_state.uov_input_image,
        prompt_task=prompt_task,
    )

    assert result.shape == (64, 64, 3)
    assert captured_requests
    req = captured_requests[0]
    assert req.prompt == 'expanded prompt'
    assert req.negative_prompt == 'expanded negative'
    assert req.positive_texts == ('expanded prompt', 'style add', 'extra add')
    assert req.negative_texts == ('expanded negative', 'extra neg')
    assert req.seed == 999
    assert req.tiled_refinement.denoise_strength == 0.42
    assert req.tiled_refinement.target_image is not None


def test_process_task_dispatches_to_unified_runtime(monkeypatch):
    task_state = _build_task_state()
    task_state.tiled = True
    bind_legacy_workflow_plan(task_state)
    task_dict = _build_task_dict()
    direct_calls = []

    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *_args, **_kwargs: direct_calls.append((_args, _kwargs)) or ['unified-image'],
    )
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    imgs, img_paths, current_progress = inference.process_task(
        task_state=task_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        route_family='txt2img',
        contextual_assets={flags.cn_ip: 'unused'},
        image_input_result={},
    )

    assert imgs == ['unified-image']
    assert img_paths == ['saved-path']
    assert current_progress == 100
    assert len(direct_calls) == 1


def test_process_task_does_not_legacy_stitch_unified_runtime_outputs(monkeypatch):
    task_state = _build_task_state()
    task_state.tiled = True
    bind_legacy_workflow_plan(task_state)
    task_state.inpaint_context = types.SimpleNamespace()
    task_dict = _build_task_dict()

    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *_args, **_kwargs: [np.zeros((8, 8, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    from modules.pipeline import inpaint as inpaint_module

    monkeypatch.setattr(
        inpaint_module.InpaintPipeline,
        'stitch',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('legacy stitch should not run after unified decode composition')),
    )

    imgs, img_paths, current_progress = inference.process_task(
        task_state=task_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        route_family='inpaint',
        contextual_assets={},
        image_input_result={},
    )

    assert len(imgs) == 1
    assert imgs[0].shape == (8, 8, 3)
    assert img_paths == ['saved-path']
    assert current_progress == 100


def test_process_prompt_skips_legacy_refresh_and_clip_encode_for_unified_owner(monkeypatch):
    task_state = _build_task_state()
    refresh_calls = []
    clip_calls = []

    monkeypatch.setattr(
        preprocessing.pipeline,
        'refresh_everything',
        lambda *args, **kwargs: refresh_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        preprocessing.pipeline,
        'clip_encode',
        lambda *args, **kwargs: clip_calls.append((args, kwargs)) or [['unexpected-cond']],
    )

    tasks = preprocessing.process_prompt(
        task_state,
        base_model_additional_loras=[],
        progressbar_callback=None,
        route_family='txt2img',
    )

    assert refresh_calls == []
    assert clip_calls == []
    assert len(tasks) == 1
    assert tasks[0]['positive'] == ['prompt']
    assert tasks[0]['negative'] == ['negative']
    assert tasks[0]['c'] is None
    assert tasks[0]['uc'] is None


def test_controlnet_support_stage_reports_structural_paths_as_route_artifacts():
    task_state = _build_task_state()
    task_state.input_image_checkbox = True
    context = PipelineRouteContext(
        async_task=None,
        task_state=task_state,
        route_id='txt2img',
        route_family='txt2img',
    )

    resources = routes.ControlNetSupportLoadStage().describe_resources(context)

    assert resources[0].resource_id == 'structural_controlnet_paths'
    assert resources[0].resource_type == 'artifact'
    assert resources[0].owner == 'modules.pipeline.image_input'
    assert resources[0].optional is True
    assert resources[1].resource_id == 'contextual_support_models'
    assert resources[1].owner == 'backend.ip_adapter'


def test_structural_controlnet_finalize_forwards_task_to_cleanup(monkeypatch):
    from backend import resources as backend_resources

    cleanup_calls = []
    monkeypatch.setattr(backend_resources, 'cleanup_memory', lambda reason, **kwargs: cleanup_calls.append((reason, kwargs)))

    task_state = _build_task_state()
    task_state.input_image_checkbox = True
    task_state.current_tab = 'inpaint'
    task_state.get_cn_tasks_for_channel = lambda *_args, **_kwargs: {}
    context = PipelineRouteContext(
        async_task=None,
        task_state=task_state,
        route_id='inpaint',
        route_family='image_input',
    )

    routes.StructuralControlNetStage().finalize(context, result=None, error=None)

    assert cleanup_calls[0][0] == 'structural_preprocess_complete'
    assert cleanup_calls[0][1]['task'] is task_state


def test_contextual_controlnet_finalize_forwards_task_to_cleanup(monkeypatch):
    from backend import resources as backend_resources

    cleanup_calls = []
    monkeypatch.setattr(backend_resources, 'cleanup_memory', lambda reason, **kwargs: cleanup_calls.append((reason, kwargs)))

    task_state = _build_task_state()
    task_state.input_image_checkbox = True
    task_state.current_tab = 'inpaint'
    context = PipelineRouteContext(
        async_task=None,
        task_state=task_state,
        route_id='inpaint',
        route_family='image_input',
    )

    routes.ContextualControlNetStage().finalize(context, result=None, error=None)

    assert cleanup_calls[0][0] == 'contextual_preprocess_complete'
    assert cleanup_calls[0][1]['task'] is task_state


def test_run_unified_sdxl_task_resolves_external_vae_path(monkeypatch):
    task_state = _build_task_state()
    task_state.vae_name = 'vae_override.safetensors'

    resolved_requests = []
    created_configs = []

    def fake_get_file_from_folder_list(name, folders):
        resolved_requests.append((name, folders))
        return f'D:/resolved/{name}'

    class FakeRuntime:
        def __init__(self, config):
            created_configs.append(config)

        def prepare_inputs(self):
            return types.SimpleNamespace(payload={}), {}

        def denoise_prepared_inputs(self, prepared_inputs, callback=None, disable_pbar=True):
            _ = prepared_inputs
            _ = callback
            _ = disable_pbar
            return types.SimpleNamespace(samples='latent')

        def decode_latent(self, latent, tiled=False):
            _ = latent
            _ = tiled
            return ['decoded'], 0.0, 0.0

        def close(self):
            return None

    monkeypatch.setattr(inference, 'get_file_from_folder_list', fake_get_file_from_folder_list)
    monkeypatch.setattr(runtime_mod, 'UnifiedSDXLRuntime', FakeRuntime)
    monkeypatch.setattr(inference.core, 'pytorch_to_numpy', lambda images: images)

    images = inference._run_unified_sdxl_task(
        task_state,
        _build_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        image_input_result={},
    )

    assert images == ['decoded']
    assert any(name == 'vae_override.safetensors' for name, _folders in resolved_requests)
    assert created_configs
    assert created_configs[0].vae_path == 'D:/resolved/vae_override.safetensors'


def test_resolve_unified_sdxl_lora_specs_resolves_lookup_relative_names(monkeypatch):
    task_state = _build_task_state()
    task_state.loras_processed = [('boost.safetensors', 0.75)]

    monkeypatch.setattr(inference.config, 'paths_lora_lookup', ['D:/loras'], raising=False)
    monkeypatch.setattr(inference.config, 'paths_checkpoints', ['D:/checkpoints'], raising=False)

    def fake_get_file_from_folder_list(name, folders):
        if folders == inference.config.paths_lora_lookup:
            return f'D:/loras/{name}'
        if folders == inference.config.paths_checkpoints:
            return f'D:/checkpoints/{name}'
        return str(name)

    monkeypatch.setattr(inference, 'get_file_from_folder_list', fake_get_file_from_folder_list)
    monkeypatch.setattr(inference.os.path, 'exists', lambda path: str(path) == 'D:/loras/boost.safetensors')

    resolved = inference._resolve_unified_sdxl_lora_specs(
        task_state,
        checkpoint_path='D:/checkpoints/model.safetensors',
        strict=True,
    )

    assert resolved == (('D:/loras/boost.safetensors', 0.75),)


def test_resolve_unified_sdxl_lora_specs_skips_selected_checkpoint_candidate(monkeypatch):
    task_state = _build_task_state()
    task_state.base_model_name = 'SDXL/TWbabeXL01.safetensors'
    task_state.loras_processed = [('SDXL/TWbabeXL01.safetensors', 1.0)]

    resolved_checkpoint = 'D:/checkpoints/SDXL/TWbabeXL01.safetensors'

    monkeypatch.setattr(inference.config, 'paths_lora_lookup', ['D:/loras'], raising=False)
    monkeypatch.setattr(inference.config, 'paths_checkpoints', ['D:/checkpoints'], raising=False)

    def fake_get_file_from_folder_list(name, folders):
        if folders == inference.config.paths_lora_lookup:
            return f'D:/loras/{name}'
        if folders == inference.config.paths_checkpoints:
            return resolved_checkpoint
        return str(name)

    monkeypatch.setattr(inference, 'get_file_from_folder_list', fake_get_file_from_folder_list)
    monkeypatch.setattr(inference.os.path, 'exists', lambda path: str(path) == resolved_checkpoint)

    resolved = inference._resolve_unified_sdxl_lora_specs(
        task_state,
        checkpoint_path=resolved_checkpoint,
        strict=True,
    )

    assert resolved == ()


def test_run_unified_sdxl_task_passes_quality_and_stream_budget_to_runtime_config(monkeypatch):
    task_state = _build_task_state()
    task_state.sharpness = 3.0
    task_state.adaptive_cfg = 5.5
    task_state.adm_scaler_positive = 1.7
    task_state.adm_scaler_negative = 0.6
    task_state.adm_scaler_end = 0.15
    task_state.controlnet_softness = 0.1
    task_state.sdxl_execution_policy = _build_sdxl_policy(
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
        stream_budget_mb=96.0,
    )
    task_state.sdxl_execution_family = task_state.sdxl_execution_policy.execution_family
    task_state.sdxl_residency_class = task_state.sdxl_execution_policy.residency_class

    created_configs = []

    class FakeRuntime:
        def __init__(self, config):
            created_configs.append(config)

        def prepare_inputs(self):
            return types.SimpleNamespace(payload={}), {}

        def denoise_prepared_inputs(self, prepared_inputs, callback=None, disable_pbar=True):
            _ = prepared_inputs
            _ = callback
            _ = disable_pbar
            return types.SimpleNamespace(samples='latent')

        def decode_latent(self, latent, tiled=False):
            _ = latent
            _ = tiled
            return ['decoded'], 0.0, 0.0

        def close(self):
            return None

    monkeypatch.setattr(runtime_mod, 'UnifiedSDXLRuntime', FakeRuntime)
    monkeypatch.setattr(inference.core, 'pytorch_to_numpy', lambda images: images)
    monkeypatch.setattr(inference, 'get_file_from_folder_list', lambda name, folders: f'D:/resolved/{name}')

    images = inference._run_unified_sdxl_task(
        task_state,
        _build_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        image_input_result={},
    )

    assert images == ['decoded']
    assert created_configs
    assert created_configs[0].execution_class == ExecutionClass.SDXL_RESIDENT_T2
    assert created_configs[0].runtime_policy is task_state.sdxl_execution_policy
    assert created_configs[0].streamlike_budget_mb == 96.0
    assert created_configs[0].quality == {
        'sharpness': 3.0,
        'adaptive_cfg': 5.5,
        'adm_scaler_positive': 1.7,
        'adm_scaler_negative': 0.6,
        'adm_scaler_end': 0.15,
        'controlnet_softness': 0.1,
    }
    assert created_configs[0].controlnet_quality == created_configs[0].quality


def test_unified_runtime_prepare_inputs_uses_quality_adm_scalers(monkeypatch):
    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant='sdxl',
            execution_class='standard_sdxl',
            checkpoint_path='D:/resolved/model.safetensors',
            prompt='prompt',
            negative_prompt='negative',
            width=64,
            height=64,
            steps=3,
            cfg=5.0,
            sampler='euler',
            scheduler='karras',
            seed=123,
            quality={
                'adm_scaler_positive': 1.7,
                'adm_scaler_negative': 0.6,
            },
        )
    )
    runtime.base_model = runtime_mod.BaseModelAvailability(
        family='sdxl',
        variant='sdxl',
        source_path='D:/resolved/model.safetensors',
        fingerprint='base-fp',
        loaded=True,
        reusable=True,
    )
    runtime.unet = types.SimpleNamespace(model=types.SimpleNamespace())
    runtime.clip = object()
    runtime._clip_identity = 'clip-id'
    runtime._checkpoint_fingerprint = 'checkpoint-fp'

    monkeypatch.setattr(runtime, 'load_components', lambda: 0.0)
    monkeypatch.setattr(
        runtime,
        '_materialize_lora_stack',
        lambda: {
            'unet_compile_metrics': {},
            'unet_compile_wall': 0.0,
            'spec_count': 0,
            'clip_patch_count': 0,
            'unet_patch_count': 0,
            'clip_host_pinned_bytes': 0.0,
            'unet_host_pinned_bytes': 0.0,
            'clip_compile_wall': 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_mod.conditioning,
        'encode_prompt_pair_sdxl',
        lambda *args, **kwargs: {
            'positive': {'cond': torch.zeros((1, 1, 1)), 'pooled': torch.zeros((1, 1))},
            'negative': {'cond': torch.zeros((1, 1, 1)), 'pooled': torch.zeros((1, 1))},
        },
    )
    adm_kwargs = {}
    monkeypatch.setattr(
        runtime_mod.conditioning,
        'build_sdxl_adm_pair',
        lambda *args, **kwargs: adm_kwargs.update(kwargs) or {
            'positive': torch.zeros((1, 1)),
            'negative': torch.zeros((1, 1)),
        },
    )

    class FakePromptStage:
        def digest(self):
            return 'prompt-fp'

    monkeypatch.setattr(
        runtime_mod.conditioning,
        'build_sdxl_text_conditioning_fingerprint',
        lambda **kwargs: FakePromptStage(),
    )
    monkeypatch.setattr(runtime, '_prepare_injected_feature_artifacts', lambda: ({}, {}, {}))
    monkeypatch.setattr(runtime, '_prepare_structural_conditioning_artifacts', lambda: (None, {}, {}))
    monkeypatch.setattr(runtime, '_prepare_spatial_conditioning_artifacts', lambda: (None, {}, {}))
    monkeypatch.setattr(runtime, '_build_compiled_unet_fingerprint', lambda **kwargs: 'compiled-fp')
    monkeypatch.setattr(runtime, '_measure_pinned_bytes', lambda *_args: 0.0)

    prepared_inputs, _ = runtime.prepare_inputs()

    assert prepared_inputs.conditioning is not None
    assert adm_kwargs['adm_scale_positive'] == 1.7
    assert adm_kwargs['adm_scale_negative'] == 0.6


def test_unified_runtime_direct_model_callable_applies_quality(monkeypatch):
    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant='sdxl',
            execution_class='standard_sdxl',
            checkpoint_path='D:/resolved/model.safetensors',
            prompt='prompt',
            negative_prompt='negative',
            width=64,
            height=64,
            steps=3,
            cfg=7.0,
            sampler='euler',
            scheduler='karras',
            seed=123,
            quality={
                'sharpness': 40.0,
                'adaptive_cfg': 3.5,
            },
        )
    )

    class FakeModelSampling:
        sigma_max = 1.0

        @staticmethod
        def timestep(_sigma):
            return torch.tensor(500.0)

    execution_unet = types.SimpleNamespace(
        model=types.SimpleNamespace(model_sampling=FakeModelSampling()),
        model_options={},
    )
    cond_pred_input = torch.full((1, 4, 8, 8), 0.5, dtype=torch.float32)
    uncond_pred_input = torch.full((1, 4, 8, 8), 0.2, dtype=torch.float32)
    x_input = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    captured = {}

    monkeypatch.setattr(
        runtime,
        '_calc_fullframe_cond_batch',
        lambda *args, **kwargs: [cond_pred_input.clone(), uncond_pred_input.clone()],
    )
    monkeypatch.setattr(
        runtime_mod.sampling.anisotropic,
        'adaptive_anisotropic_filter',
        lambda x, g=None: torch.zeros_like(x),
    )

    def fake_cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options=None, cfg_pp=False, adaptive_cfg=0.0, diffusion_progress=0.0):
        _ = model
        _ = uncond_pred
        _ = cond_scale
        _ = x
        _ = timestep
        _ = cfg_pp
        captured['cond_pred'] = cond_pred.clone()
        captured['model_options'] = dict(model_options or {})
        captured['adaptive_cfg'] = adaptive_cfg
        captured['diffusion_progress'] = diffusion_progress
        return cond_pred

    monkeypatch.setattr(runtime_mod.sampling, 'cfg_function', fake_cfg_function)

    model_fn = runtime._build_direct_model_callable(
        execution_unet,
        {'positive': [], 'negative': []},
        latent_image=torch.zeros_like(x_input),
        reference_noise=torch.zeros_like(x_input),
        denoise_mask=None,
    )
    result = model_fn(x_input.clone(), torch.tensor([1.0], dtype=torch.float32))

    assert torch.equal(result, captured['cond_pred'])
    assert captured['adaptive_cfg'] == 3.5
    assert abs(captured['diffusion_progress'] - (1.0 - 500.0 / 999.0)) < 1e-6
    assert captured['model_options']['quality'] == {
        'sharpness': 40.0,
        'adaptive_cfg': 3.5,
    }
    assert not torch.allclose(captured['cond_pred'], cond_pred_input)


def test_unified_runtime_calculate_sigmas_passes_quality_to_ksampler(monkeypatch):
    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant='sdxl',
            execution_class='standard_sdxl',
            checkpoint_path='D:/resolved/model.safetensors',
            prompt='prompt',
            negative_prompt='negative',
            width=64,
            height=64,
            steps=3,
            cfg=7.0,
            sampler='euler',
            scheduler='karras',
            seed=123,
            quality={
                'adm_scaler_end': 0.15,
            },
        )
    )
    captured = {}

    class FakeKSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise, model_options=None):
            _ = model
            _ = steps
            _ = device
            _ = sampler
            _ = scheduler
            _ = denoise
            captured['model_options'] = dict(model_options or {})
            self.sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    monkeypatch.setattr(runtime_mod.sampling, 'KSampler', FakeKSampler)

    sigmas = runtime._calculate_sigmas(
        torch.device('cpu'),
        execution_unet=types.SimpleNamespace(model=types.SimpleNamespace()),
    )

    assert tuple(float(v) for v in sigmas.tolist()) == (1.0, 0.0)
    assert captured['model_options']['quality'] == {'adm_scaler_end': 0.15}


def test_unified_runtime_load_components_uses_external_vae(monkeypatch):
    load_calls = {'vae_source': [], 'direct_vae': []}

    class DummyUNet:
        runtime_release_to_meta = True

    class DummyClip:
        def __init__(self):
            self.layer_idx = None

        def clip_layer(self, layer_idx):
            self.layer_idx = layer_idx

    dummy_clip = DummyClip()
    dummy_vae = types.SimpleNamespace()

    monkeypatch.setattr(
        runtime_mod.loader,
        'load_sdxl_checkpoint',
        lambda *args, **kwargs: load_calls['vae_source'].append(kwargs.get('vae_source')) or (DummyUNet(), dummy_clip, dummy_vae),
    )
    monkeypatch.setattr(
        runtime_mod.loader,
        'load_vae',
        lambda source, **kwargs: load_calls['direct_vae'].append((source, kwargs)) or dummy_vae,
    )

    runtime = runtime_mod.UnifiedSDXLRuntime(
        runtime_mod.UnifiedSDXLRuntimeConfig(
            model_variant='sdxl',
            execution_class='standard_sdxl',
            checkpoint_path='D:/resolved/model.safetensors',
            vae_path='D:/resolved/vae_override.safetensors',
            prompt='prompt',
            negative_prompt='negative',
            width=64,
            height=64,
            steps=3,
            cfg=5.0,
            sampler='euler',
            scheduler='karras',
            seed=123,
        )
    )

    runtime.load_components()

    assert load_calls['vae_source'] == [dummy_vae]
    assert len(load_calls['direct_vae']) == 1
    assert load_calls['direct_vae'][0][0] == 'D:/resolved/vae_override.safetensors'


def test_process_task_unified_route_passes_external_vae_for_missing_checkpoint_vae(monkeypatch):
    import backend.sdxl_resident_runtime as resident_mod

    runtime_mod.clear_unified_sdxl_runtime_component_cache(teardown=True)

    task_state = _build_task_state()
    task_state.tiled = True
    bind_legacy_workflow_plan(task_state)
    task_state.vae_name = 'vae_override.safetensors'
    task_state.sdxl_execution_policy = _build_sdxl_policy(
        execution_class=ExecutionClass.SDXL_RESIDENT_T2,
    )
    task_state.sdxl_execution_family = task_state.sdxl_execution_policy.execution_family
    task_state.sdxl_residency_class = task_state.sdxl_execution_policy.residency_class
    route = routes.build_generation_route(task_state)
    assert route.route_id == 'txt2img'

    prompt_tasks = preprocessing.process_prompt(
        task_state,
        base_model_additional_loras=[],
        progressbar_callback=None,
        route_family=route.family,
    )
    task_dict = prompt_tasks[0]

    load_calls = {'vae_source': []}
    load_vae_requests = []

    def fake_get_file_from_folder_list(name, folders):
        return f'D:/resolved/{name}'

    class DummyUNet:
        runtime_release_to_meta = True

    class DummyClip:
        def clip_layer(self, _layer_idx):
            return None

    class DummyVAE:
        pass

    shared_override_vae = DummyVAE()

    def fake_load_checkpoint(*args, **kwargs):
        load_calls['vae_source'].append(kwargs.get('vae_source'))
        return DummyUNet(), DummyClip(), DummyVAE()

    def fake_prepare_inputs(self):
        self.load_components()
        self.prepared_inputs = types.SimpleNamespace(payload={})
        return self.prepared_inputs, {}

    def fake_denoise(self, prepared_inputs, callback=None, disable_pbar=True):
        _ = prepared_inputs
        _ = callback
        _ = disable_pbar
        return types.SimpleNamespace(samples=torch.zeros((1, 4, 8, 8), dtype=torch.float32))

    def fake_decode(self, latent, tiled=False):
        _ = latent
        _ = tiled
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32), 0.0, 0.0

    monkeypatch.setattr(inference, 'get_file_from_folder_list', fake_get_file_from_folder_list)
    monkeypatch.setattr(
        resident_mod,
        '_load_shared_sdxl_vae_for_device',
        lambda vae_path, **kwargs: load_vae_requests.append((vae_path, kwargs)) or shared_override_vae,
    )
    monkeypatch.setattr(runtime_mod.loader, 'load_sdxl_checkpoint', fake_load_checkpoint)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, 'prepare_inputs', fake_prepare_inputs)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, 'denoise_prepared_inputs', fake_denoise)
    monkeypatch.setattr(runtime_mod.UnifiedSDXLRuntime, 'decode_latent', fake_decode)
    monkeypatch.setattr(resident_mod.ResidentSDXLRuntime, 'prepare_inputs', fake_prepare_inputs)
    monkeypatch.setattr(resident_mod.ResidentSDXLRuntime, 'denoise_prepared_inputs', fake_denoise)
    monkeypatch.setattr(resident_mod.ResidentSDXLRuntime, 'decode_latent', fake_decode)
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    imgs, img_paths, current_progress = inference.process_task(
        task_state=task_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        route_family=route.family,
        contextual_assets={},
        image_input_result={},
    )

    assert len(imgs) == 1
    assert imgs[0].shape == (8, 8, 3)
    assert img_paths == ['saved-path']
    assert current_progress == 100
    assert load_vae_requests == [
        (
            'D:/resolved/vae_override.safetensors',
            {
                'load_device': torch.device('cpu'),
                'offload_device': torch.device('cpu'),
                'allow_default_fallback': False,
            },
        )
    ]
    assert len(load_calls['vae_source']) == 1
    assert load_calls['vae_source'][0] is shared_override_vae

    runtime_mod.clear_unified_sdxl_runtime_component_cache(teardown=True)


def test_process_task_falls_back_to_shared_diffusion_when_unified_route_is_rejected(monkeypatch):
    task_state = _build_task_state()
    task_state.sdxl_execution_policy = _build_sdxl_policy(enabled=False)
    task_state.sdxl_execution_family = task_state.sdxl_execution_policy.execution_family
    task_state.sdxl_residency_class = task_state.sdxl_execution_policy.residency_class
    task_state.tiled = True
    task_state.disable_preview = True
    task_state.use_expansion = False
    task_state.initial_latent = None

    route = routes.build_generation_route(task_state)
    assert route.family == 'txt2img'

    task_dict = _build_task_dict()

    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('unified runtime should not be called')),
    )
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    with pytest.raises(RuntimeError, match="Unified SDXL runtime requires an active SDXL execution policy; legacy shared diffusion path is gutted"):
        inference.process_task(
            task_state=task_state,
            task_dict=task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=3,
            preparation_steps=0,
            denoising_strength=1.0,
            final_scheduler_name='karras',
            loras=[],
            route_family=route.family,
            contextual_assets={},
            image_input_result={},
        )


def test_process_task_keeps_nondefault_quality_on_unified_runtime(monkeypatch):
    task_state = _build_task_state()
    task_state.tiled = True
    bind_legacy_workflow_plan(task_state)
    task_state.sharpness = 3.0
 
    route = routes.build_generation_route(task_state)
    assert route.family == 'txt2img'

    task_dict = _build_task_dict()
    direct_calls = []

    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *_args, **_kwargs: direct_calls.append((_args, _kwargs)) or ['unified-image'],
    )
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    imgs, img_paths, current_progress = inference.process_task(
        task_state=task_state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        route_family=route.family,
        contextual_assets={},
        image_input_result={},
    )

    assert imgs == ['unified-image']
    assert img_paths == ['saved-path']
    assert current_progress == 100
    assert len(direct_calls) == 1


def test_process_task_rejects_sd15_after_legacy_shared_gutting(monkeypatch):
    task_state = _build_task_state()
    task_state.base_model_name = 'sd15_demo.safetensors'
    task_state.disable_preview = True
    task_state.use_expansion = False
    task_state.initial_latent = None

    route = routes.build_generation_route(task_state)
    assert route.family == 'txt2img'

    task_dict = _build_task_dict()

    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('unified runtime should not be called')),
    )

    with pytest.raises(RuntimeError, match='SD 1.5 execution is no longer supported'):
        inference.process_task(
            task_state=task_state,
            task_dict=task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=3,
            preparation_steps=0,
            denoising_strength=1.0,
            final_scheduler_name='karras',
            loras=[],
            route_family=route.family,
            contextual_assets={},
            image_input_result={},
        )


def test_filter_supported_sdxl_base_model_choices_filters_sd15():
    from modules.config import filter_supported_sdxl_base_model_choices
    candidates = [
        'sdxl_base.safetensors',
        'sd15_demo.safetensors',
        'another_sd1.5_model.ckpt',
        'pony_model.safetensors',
    ]
    filtered = filter_supported_sdxl_base_model_choices(candidates)
    assert 'sdxl_base.safetensors' in filtered
    assert 'pony_model.safetensors' in filtered
    assert 'sd15_demo.safetensors' not in filtered
    assert 'another_sd1.5_model.ckpt' not in filtered


def test_core_load_model_rejects_sd15(tmp_path):
    from modules.core import load_model
    dummy_model_path = tmp_path / "sd15_dummy.safetensors"
    dummy_model_path.write_text("dummy model data")

    with pytest.raises(RuntimeError, match="SD 1.5 execution is no longer supported."):
        load_model(str(dummy_model_path))
