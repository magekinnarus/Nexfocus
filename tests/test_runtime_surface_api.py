from __future__ import annotations
from io import BytesIO
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

# Pre-emptively mock args_manager to avoid pytest argparse errors
fake_args = types.SimpleNamespace(
    colab=False,
    preset="",
    output_path="",
    temp_path="",
    skip_model_load=True,
    disable_metadata=False,
    disable_preset_selection=False,
)
fake_args_manager = types.ModuleType("args_manager")
fake_args_manager.args = fake_args
fake_args_manager.args_parser = types.SimpleNamespace(args=fake_args, parser=types.SimpleNamespace())
sys.modules["args_manager"] = fake_args_manager

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
sys.modules["transformers"] = fake_transformers

import modules.async_worker as worker
import modules.runtime_surface_state as runtime_surface_state
from modules.runtime_surface_api import runtime_surface_router


app = FastAPI()
app.include_router(runtime_surface_router)
client = TestClient(app)


def setup_function():
    worker.async_tasks.clear()
    worker.set_active_task(None)
    runtime_surface_state.reset_runtime_surface_state()


def teardown_function():
    worker.async_tasks.clear()
    worker.set_active_task(None)
    runtime_surface_state.reset_runtime_surface_state()


def test_runtime_surface_state_returns_encoded_completed_image_urls():
    runtime_surface_state.completed_tasks_history.append(
        runtime_surface_state.CompletedTaskRecord(
            task_id='done1234',
            prompt='Prompt',
            model_name='Model',
            seed=7,
            images=[r'D:\AI\Fooocus_Nex\outputs\my image #1.png'],
        )
    )

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'success'
    assert payload['state']['completed'][0]['image_urls'][0] == '/runtime_surface_api/completed_image/done1234/0'


def test_runtime_surface_completed_image_endpoint_serves_file(tmp_path):
    image_path = tmp_path / 'completed-thumb.png'
    image_bytes = b'\x89PNG\r\n\x1a\nfakepng'
    image_path.write_bytes(image_bytes)

    runtime_surface_state.completed_tasks_history.append(
        runtime_surface_state.CompletedTaskRecord(
            task_id='thumb1234',
            prompt='Prompt',
            model_name='Model',
            seed=7,
            images=[str(image_path)],
        )
    )

    response = client.get('/runtime_surface_api/completed_image/thumb1234/0')

    assert response.status_code == 200
    assert response.content == image_bytes


def test_runtime_surface_delete_completed_action_removes_history():
    runtime_surface_state.completed_tasks_history.append(
        runtime_surface_state.CompletedTaskRecord(
            task_id='done5678',
            prompt='Prompt',
            model_name='Model',
            seed=8,
            images=['out.png'],
        )
    )

    response = client.post('/runtime_surface_api/action', json={'action': 'delete_completed', 'task_id': 'done5678'})

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'success'
    assert payload['state']['completed'] == []
    assert runtime_surface_state.completed_tasks_history == []


def test_runtime_surface_skip_action_interrupts_active_task(monkeypatch):
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 99})
    worker.set_active_task(task)

    interrupted = {'called': False}

    def fake_request_interrupt(action, task_obj=None):
        interrupted['called'] = action == 'skip' and task_obj is task

    monkeypatch.setattr(worker, 'request_interrupt', fake_request_interrupt)

    response = client.post('/runtime_surface_api/action', json={'action': 'skip'})

    assert response.status_code == 200
    assert interrupted['called'] is True


def test_runtime_surface_state_drains_running_preview_status():
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 101})
    task.yields.append(['preview', (37, 'Loading models ...', None)])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    state = response.json()['state']
    assert state['running']['task_id'] == task.task_id
    assert state['progress']['visible'] is True
    assert state['progress']['number'] == 37
    assert state['progress']['text'] == 'Loading models ...'


def test_runtime_surface_state_reports_preview_revision_and_url():
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 202})
    task.yields.append(['preview', (51, 'Sampling step 5/10 ...', np.zeros((12, 20, 3), dtype=np.uint8))])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    state = response.json()['state']
    assert state['preview']['available'] is True
    assert state['preview']['revision'] == 1
    assert state['preview']['image_url'] == '/runtime_surface_api/preview_image?revision=1'


def test_runtime_surface_state_reports_flux_inpaint_prompt_and_engine():
    task = worker.AsyncTask({
        'prompt': 'base prompt',
        'image_seed': 303,
        'input_image_checkbox': True,
        'current_tab': 'inpaint',
        'inpaint_input_image': 'input.png',
        'inpaint_route': 'flux',
        'inpaint_additional_prompt': 'mask prompt',
        'base_model': 'sdxl\\illustrious\\model.safetensors',
    })
    task.yields.append(['preview', (21, 'Sampling step 2/10 ...', None)])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    running = response.json()['state']['running']
    assert running['workflow_name'] == 'Flux Inpaint'
    assert running['prompt_label'] == 'Inpaint Prompt'
    assert running['show_prompt'] is True
    assert running['prompt_preview'].startswith('mask prompt')
    assert running['model_label'] == 'Engine'
    assert running['model_name'] == 'Flux Fill'


def test_runtime_surface_state_reports_background_removal_without_main_prompt():
    task = worker.AsyncTask({
        'prompt': 'txt2img prompt should not appear',
        'image_seed': 404,
        'current_tab': 'remove',
        'input_image_checkbox': True,
        'remove_bg_enabled': True,
        'remove_base_image': 'source.png',
        'base_model_name': 'sdxl\\illustrious\\model.safetensors',
    })
    task.yields.append(['preview', (12, 'Background Removal Starting...', None)])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    running = response.json()['state']['running']
    assert running['workflow_name'] == 'Background Removal'
    assert running['show_prompt'] is False
    assert running['prompt_preview'] == ''
    assert running['model_label'] == 'Engine'
    assert running['model_name'] == 'Background Removal'


def test_runtime_surface_state_does_not_mislabel_pending_txt2img_as_upscale_from_stale_uov_fields():
    task = worker.AsyncTask({
        'prompt': 'txt2img prompt',
        'image_seed': 454,
        'current_tab': 'uov',
        'input_image_checkbox': False,
        'uov_method': 'Upscale',
        'uov_input_image': 'stale-image.png',
        'base_model': 'sdxl\\illustrious\\model.safetensors',
    })
    worker.async_tasks.append(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    pending = response.json()['state']['pending']
    assert len(pending) == 1
    assert pending[0]['workflow_name'] == 'Txt2Img'
    assert pending[0]['model_name'] == 'sdxl\\illustrious\\model.safetensors'


def test_runtime_surface_state_reports_queue_selected_color_enhancement():
    task = worker.AsyncTask({
        'prompt': 'main prompt should not appear',
        'upscale_prompt': 'local color prompt',
        'image_seed': 4541,
        'current_tab': 'uov',
        'input_image_checkbox': True,
        'uov_method': 'Color Enhancement',
        'requested_route_id': 'color_enhanced_upscale',
        'requested_route_family': 'upscale',
        'base_model': 'sdxl\\illustrious\\model.safetensors',
    })
    worker.async_tasks.append(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    pending = response.json()['state']['pending']
    assert len(pending) == 1
    assert pending[0]['workflow_name'] == 'Color Enhancement'
    assert pending[0]['prompt_label'] == 'Upscale Prompt'
    assert pending[0]['show_prompt'] is True
    assert pending[0]['prompt_preview'].startswith('local color prompt')
    assert pending[0]['model_label'] == 'Model'
    assert pending[0]['model_name'] == 'sdxl\\illustrious\\model.safetensors'


def test_runtime_surface_state_reports_super_upscale_as_target_plus_sdxl_model():
    task = worker.AsyncTask({
        'prompt': 'detail prompt',
        'image_seed': 4542,
        'current_tab': 'uov',
        'input_image_checkbox': True,
        'uov_method': 'Super-Upscale',
        'requested_route_id': 'super_upscale',
        'requested_route_family': 'upscale',
        'base_model': 'sdxl\\illustrious\\model.safetensors',
    })
    worker.async_tasks.append(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    pending = response.json()['state']['pending']
    assert len(pending) == 1
    assert pending[0]['workflow_name'] == 'Super Upscale'
    assert pending[0]['prompt_label'] == 'Prompt'
    assert pending[0]['show_prompt'] is True
    assert pending[0]['model_label'] == 'Pipeline'
    assert pending[0]['model_name'] == 'Provided Upscale Target + sdxl\\illustrious\\model.safetensors'


def test_runtime_surface_state_does_not_mislabel_pending_txt2img_from_stale_inpaint_mix_fields():
    task = worker.AsyncTask({
        'prompt': 'txt2img prompt',
        'image_seed': 455,
        'current_tab': 'txt2img',
        'input_image_checkbox': True,
        'mixing_image_prompt_and_inpaint': True,
        'inpaint_input_image': 'stale-image.png',
        'inpaint_mask_image': 'stale-mask.png',
        'base_model': 'sdxl\\illustrious\\model.safetensors',
    })
    worker.async_tasks.append(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    pending = response.json()['state']['pending']
    assert len(pending) == 1
    assert pending[0]['workflow_name'] == 'Txt2Img'
    assert pending[0]['model_name'] == 'sdxl\\illustrious\\model.safetensors'


def test_runtime_surface_state_prefers_resolved_route_over_stale_remove_flags():
    task = worker.AsyncTask({
        'prompt': 'base prompt',
        'image_seed': 505,
        'current_tab': 'inpaint',
        'input_image_checkbox': True,
        'inpaint_input_image': 'input.png',
        'inpaint_route': 'flux',
        'inpaint_additional_prompt': 'edit prompt',
        'remove_bg_enabled': True,
        'remove_obj_enabled': True,
        'runtime_route_id': 'flux_inpaint',
        'runtime_route_family': 'flux_fill',
        'runtime_route_display_name': 'Flux Inpaint',
    })
    task.yields.append(['preview', (18, 'Flux Fill Inpaint 1/1 ...', None)])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    running = response.json()['state']['running']
    assert running['workflow_name'] == 'Flux Inpaint'
    assert running['model_name'] == 'Flux Fill'
    assert running['prompt_label'] == 'Inpaint Prompt'
    assert running['show_prompt'] is True
    assert running['prompt_preview'].startswith('edit prompt')


def test_runtime_surface_state_reports_queue_selected_running_removal_task():
    task = worker.AsyncTask({
        'prompt': 'txt2img prompt should not appear',
        'image_seed': 506,
        'current_tab': 'remove',
        'input_image_checkbox': True,
        'remove_bg_enabled': True,
        'remove_base_image': 'source.png',
        'requested_route_id': 'removal',
        'requested_route_family': 'removal',
        'goals': ['remove_bg'],
        'base_model_name': 'sdxl\\illustrious\\model.safetensors',
    })
    task.yields.append(['preview', (12, 'Background Removal Starting...', None)])
    worker.set_active_task(task)

    response = client.get('/runtime_surface_api/state')

    assert response.status_code == 200
    running = response.json()['state']['running']
    assert running['workflow_name'] == 'Background Removal'
    assert running['model_name'] == 'Background Removal'


def test_runtime_surface_preview_image_endpoint_serves_numpy_preview_payload():
    runtime_surface_state._set_preview_value(np.full((8, 8, 3), 127, dtype=np.uint8))

    response = client.get('/runtime_surface_api/preview_image?revision=1')

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert response.headers['cache-control'] == 'no-store, max-age=0'
    assert response.headers['x-nex-preview-revision'] == '1'
    assert response.content.startswith(b'\x89PNG\r\n\x1a\n')


def test_runtime_surface_preview_image_endpoint_fits_numpy_preview_to_requested_bounds():
    runtime_surface_state._set_preview_value(np.full((600, 400, 3), 127, dtype=np.uint8))

    response = client.get('/runtime_surface_api/preview_image?revision=1&max_width=120&max_height=120')

    assert response.status_code == 200
    preview_image = Image.open(BytesIO(response.content))
    assert preview_image.size == (80, 120)


def test_runtime_surface_preview_image_endpoint_serves_file_preview(tmp_path):
    image_path = tmp_path / 'preview.png'
    image_bytes = b'\x89PNG\r\n\x1a\nfilepreview'
    image_path.write_bytes(image_bytes)
    runtime_surface_state._set_preview_value(str(image_path))

    response = client.get('/runtime_surface_api/preview_image?revision=1')

    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store, max-age=0'
    assert response.headers['x-nex-preview-revision'] == '1'
    assert response.content == image_bytes


def test_runtime_surface_preview_image_endpoint_fits_file_preview_to_requested_bounds(tmp_path):
    image_path = tmp_path / 'preview-fit.png'
    Image.new('RGB', (300, 600), color=(32, 64, 96)).save(image_path, format='PNG')
    runtime_surface_state._set_preview_value(str(image_path))

    response = client.get('/runtime_surface_api/preview_image?revision=1&max_width=120&max_height=120')

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    preview_image = Image.open(BytesIO(response.content))
    assert preview_image.size == (300, 600)
