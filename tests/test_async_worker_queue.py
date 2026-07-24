from __future__ import annotations

import sys
import time
import types
import pytest
import gradio as gr

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

import modules.async_worker as worker
import modules.flags as flags
import modules.ui_logic as ui_logic
from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL
from modules.route_intent import resolve_route_intent

@pytest.fixture(autouse=True)
def sync_modules():
    global worker, ui_logic
    import sys
    worker = sys.modules["modules.async_worker"]
    ui_logic = sys.modules["modules.ui_logic"]


@pytest.fixture(autouse=True)
def reset_queue_ui_state():
    worker.async_tasks.clear()
    worker.set_active_task(None)
    ui_logic.runtime_surface_state.reset_runtime_surface_state()
    ui_logic.completed_tasks_history.clear()
    ui_logic._last_seen_active_task = None
    ui_logic._last_rendered_completed_queue_html = None
    ui_logic._last_rendered_progress_state = None
    ui_logic._last_rendered_running_task_html = None
    ui_logic._last_rendered_running_progress_value = None
    ui_logic._last_rendered_running_status_html = None
    ui_logic._last_rendered_running_skip_interactive = None
    ui_logic._last_rendered_pending_queue_html = None
    ui_logic._last_rendered_queue_len = -1
    yield
    worker.async_tasks.clear()
    worker.set_active_task(None)
    ui_logic.runtime_surface_state.reset_runtime_surface_state()

def test_get_tasks_snapshot_parameter_isolation():
    # Setup args registry and parameters
    args = {
        'prompt': 'A beautiful sunset',
        'image_seed': 12345,
        'image_number': 3,
        'disable_seed_increment': False,
    }
    
    # Force ctrls keys alignment
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())
    
    # Call get_tasks to fan out
    tasks = ui_logic.get_tasks(None, *args.values())
    
    assert len(tasks) == 1
    
    # Assert parameters are frozen in the queued task
    assert tasks[0].state.prompt == 'A beautiful sunset'
    assert tasks[0].state.seed == 12345
    assert tasks[0].state.image_number == 1
    assert tasks[0].state.generate_image_grid is False
    assert tasks[0].task_id is not None
    
    # If we mutate args now, the already created tasks are not affected
    args['prompt'] = 'A completely different prompt'
    assert tasks[0].state.prompt == 'A beautiful sunset'


def test_get_tasks_freezes_requested_removal_route_and_goals():
    args = {
        'prompt': 'remove prompt',
        'image_seed': 9001,
        'input_image_checkbox': True,
        'current_tab': 'remove',
        'remove_bg_enabled': True,
        'remove_obj_enabled': True,
        'remove_base_image': 'source.png',
    }

    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    tasks = ui_logic.get_tasks(None, *args.values())

    assert len(tasks) == 1
    task = tasks[0]

    assert task.state.requested_route_id == 'removal'
    assert task.state.requested_route_family == 'removal'
    assert task.state.goals == [flags.remove_bg, flags.remove_obj]

    intent = resolve_route_intent(task.state)
    assert intent.route_id == 'removal'
    assert intent.wants_removal is True


def test_get_tasks_keeps_txt2img_route_when_stale_remove_flags_exist():
    args = {
        'prompt': 'plain prompt',
        'image_seed': 9002,
        'input_image_checkbox': False,
        'current_tab': 'uov',
        'uov_method': 'Disabled',
        'remove_bg_enabled': True,
        'remove_obj_enabled': True,
        'remove_base_image': 'stale-source.png',
    }

    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    tasks = ui_logic.get_tasks(None, *args.values())

    assert len(tasks) == 1
    task = tasks[0]

    assert task.state.requested_route_id == 'txt2img'
    assert task.state.requested_route_family == 'txt2img'
    assert task.state.goals == []

    intent = resolve_route_intent(task.state)
    assert intent.route_id == 'txt2img'
    assert intent.wants_removal is False


def test_get_tasks_freezes_requested_flux_removal_route_and_goals():
    args = {
        'prompt': 'remove prompt',
        'image_seed': 9003,
        'input_image_checkbox': True,
        'current_tab': 'remove',
        'remove_obj_enabled': True,
        'remove_base_image': 'source.png',
        'remove_mask_image': 'mask.png',
        'objr_engine': OBJR_ENGINE_FLUX_FILL,
    }

    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    tasks = ui_logic.get_tasks(None, *args.values())

    assert len(tasks) == 1
    task = tasks[0]

    assert task.state.requested_route_id == 'flux_removal'
    assert task.state.requested_route_family == 'flux_fill'
    assert task.state.goals == [flags.remove_obj]

    intent = resolve_route_intent(task.state)
    assert intent.route_id == 'flux_removal'
    assert intent.wants_removal is True


def test_get_tasks_returns_guidance_for_flux_inpaint_with_controlnet():
    args = {
        'input_image_checkbox': True,
        'current_tab': 'inpaint',
        'inpaint_route': 'flux',
        'mixing_image_prompt_and_inpaint': True,
        'cn_0_image': 'control.png',
        'cn_0_type': flags.cn_canny,
    }
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    task = ui_logic.get_tasks(None, *args.values())[0]

    assert task.is_valid is False
    assert task.state.workflow_plan is None
    assert task.validation_message == (
        "Flux Fill does not support ControlNet guidance. Select SDXL in Inpaint, "
        "or turn off 'Add ControlNet to Inpaint', then Generate again."
    )


def test_get_tasks_returns_upload_readiness_guidance_for_color_enhancement():
    args = {
        'input_image_checkbox': True,
        'current_tab': 'uov',
        'uov_method': 'Color Enhancement',
        'uov_input_image': 'source.png',
        'upscale_gan_output_image': None,
    }
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    task = ui_logic.get_tasks(None, *args.values())[0]

    assert task.is_valid is False
    assert task.validation_message == (
        'Color Enhancement is not ready. Upload the Upscale Target and wait for it '
        'to finish loading, then Generate again.'
    )


@pytest.mark.parametrize(
    ('args', 'expected_message'),
    [
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': False,
            },
            'Prepare Inpaint first so the BB image and BB mask are ready, then Generate again.',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': True,
                'inpaint_input_image': 'source.png',
                'inpaint_bbox': '[0, 64, 0, 64]',
                'inpaint_bb_image': 'bb.png',
                'inpaint_mask_image': None,
            },
            (
                'Inpaint is not ready. Missing: BB Mask. Complete Inpaint preparation and '
                'wait for the prepared image slots to finish loading, then Generate again.'
            ),
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'outpaint',
                'outpaint_step2_checkbox': True,
                'outpaint_input_image': 'source.png',
                'outpaint_bb_image': None,
                'outpaint_mask_image': 'mask.png',
            },
            (
                'Outpaint is not ready. Missing: BB Image. Complete Outpaint preparation and '
                'wait for the prepared image slots to finish loading, then Generate again.'
            ),
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Upscale',
                'uov_input_image': None,
            },
            (
                'Upscale is not ready. Upload the source Image and wait for it '
                'to finish loading, then Generate again.'
            ),
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Super-Upscale',
                'uov_input_image': 'source.png',
                'upscale_gan_output_image': None,
            },
            (
                'Super-Upscale is not ready. Upload the Upscale Target and wait for it '
                'to finish loading, then Generate again.'
            ),
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'remove',
                'remove_bg_enabled': True,
                'remove_base_image': None,
            },
            (
                'Removal is not ready. Upload the Base Image and wait for it '
                'to finish loading, then Generate again.'
            ),
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'remove',
                'remove_obj_enabled': True,
                'remove_base_image': 'source.png',
                'remove_mask_image': None,
            },
            (
                'Removal is not ready. Upload the Mask and wait for it '
                'to finish loading, then Generate again.'
            ),
        ),
    ],
)
def test_required_image_slot_preflight_returns_actionable_guidance(args, expected_message):
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    task = ui_logic.get_tasks(None, *args.values())[0]

    assert task.is_valid is False
    assert task.state.workflow_plan is None
    assert task.validation_message == expected_message


@pytest.mark.parametrize(
    ('args', 'missing_key', 'expected_label'),
    [
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': True,
                'inpaint_input_image': 'source.png',
                'inpaint_bbox': '[0, 64, 0, 64]',
                'inpaint_bb_image': 'bb.png',
                'inpaint_mask_image': 'mask.png',
            },
            'inpaint_input_image',
            'Base Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': True,
                'inpaint_input_image': 'source.png',
                'inpaint_bbox': '[0, 64, 0, 64]',
                'inpaint_bb_image': 'bb.png',
                'inpaint_mask_image': 'mask.png',
            },
            'inpaint_bbox',
            'BB Selection',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': True,
                'inpaint_input_image': 'source.png',
                'inpaint_bbox': '[0, 64, 0, 64]',
                'inpaint_bb_image': 'bb.png',
                'inpaint_mask_image': 'mask.png',
            },
            'inpaint_bb_image',
            'BB Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'inpaint',
                'inpaint_route': 'sdxl',
                'inpaint_step2_checkbox': True,
                'inpaint_input_image': 'source.png',
                'inpaint_bbox': '[0, 64, 0, 64]',
                'inpaint_bb_image': 'bb.png',
                'inpaint_mask_image': 'mask.png',
            },
            'inpaint_mask_image',
            'BB Mask',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'outpaint',
                'outpaint_step2_checkbox': True,
                'outpaint_input_image': 'source.png',
                'outpaint_bb_image': 'bb.png',
                'outpaint_mask_image': 'mask.png',
            },
            'outpaint_input_image',
            'Base Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'outpaint',
                'outpaint_step2_checkbox': True,
                'outpaint_input_image': 'source.png',
                'outpaint_bb_image': 'bb.png',
                'outpaint_mask_image': 'mask.png',
            },
            'outpaint_bb_image',
            'BB Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'outpaint',
                'outpaint_step2_checkbox': True,
                'outpaint_input_image': 'source.png',
                'outpaint_bb_image': 'bb.png',
                'outpaint_mask_image': 'mask.png',
            },
            'outpaint_mask_image',
            'BB Mask',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Upscale',
                'uov_input_image': 'source.png',
            },
            'uov_input_image',
            'source Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Super-Upscale',
                'uov_input_image': 'source.png',
                'upscale_gan_output_image': 'target.png',
            },
            'upscale_gan_output_image',
            'Upscale Target',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Color Enhancement',
                'uov_input_image': 'source.png',
                'upscale_gan_output_image': 'target.png',
            },
            'uov_input_image',
            'source Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'uov',
                'uov_method': 'Color Enhancement',
                'uov_input_image': 'source.png',
                'upscale_gan_output_image': 'target.png',
            },
            'upscale_gan_output_image',
            'Upscale Target',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'remove',
                'remove_bg_enabled': True,
                'remove_base_image': 'source.png',
            },
            'remove_base_image',
            'Base Image',
        ),
        (
            {
                'input_image_checkbox': True,
                'current_tab': 'remove',
                'remove_obj_enabled': True,
                'remove_base_image': 'source.png',
                'remove_mask_image': 'mask.png',
            },
            'remove_mask_image',
            'Mask',
        ),
    ],
)
def test_each_required_route_input_is_covered_by_preflight(args, missing_key, expected_label):
    args[missing_key] = None

    message = ui_logic.validate_user_correctable_generate_request(args)

    assert expected_label in message
    assert message.endswith('then Generate again.')


def test_removal_background_output_can_supply_object_removal_mask():
    args = {
        'input_image_checkbox': True,
        'current_tab': 'remove',
        'remove_bg_enabled': True,
        'remove_obj_enabled': True,
        'remove_base_image': 'source.png',
        'remove_mask_image': None,
    }
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    task = ui_logic.get_tasks(None, *args.values())[0]

    assert task.is_valid is True
    assert task.state.goals == [flags.remove_bg, flags.remove_obj]


def test_inactive_stale_image_slots_do_not_block_txt2img():
    args = {
        'input_image_checkbox': False,
        'current_tab': 'inpaint',
        'inpaint_step2_checkbox': True,
        'inpaint_input_image': None,
        'inpaint_bb_image': None,
        'inpaint_mask_image': None,
        'uov_method': 'Super-Upscale',
        'uov_input_image': None,
        'upscale_gan_output_image': None,
    }
    ui_logic.ctrls_keys = ['_currentTask'] + list(args.keys())

    task = ui_logic.get_tasks(None, *args.values())[0]

    assert task.is_valid is True
    assert task.state.requested_route_id == 'txt2img'


def test_invalid_task_feedback_is_visible_and_not_queued(capsys):
    message = 'Correct this request before generating.'
    invalid_task = ui_logic._invalid_task(message)

    task, progress_update, preview_update = (
        ui_logic.enqueue_tasks_with_ui_feedback([invalid_task])
    )
    snapshot = ui_logic.runtime_surface_state.get_runtime_snapshot()
    stdout = capsys.readouterr().out

    assert task is invalid_task
    assert worker.async_tasks == []
    assert stdout == f'[Nex] {message}\n'
    assert progress_update['visible'] is True
    assert preview_update['visible'] is False
    assert snapshot['progress'] == {
        'visible': True,
        'number': 0,
        'text': message,
    }


def test_idle_validation_notice_survives_previous_task_handoff():
    previous_task = worker.AsyncTask({'prompt': 'Previous Task', 'image_seed': 88})
    worker.set_active_task(previous_task)
    ui_logic.runtime_surface_state.get_runtime_snapshot()
    worker.set_active_task(None)

    ui_logic.runtime_surface_state.set_idle_notice('Wait for the upload to finish.')
    snapshot = ui_logic.runtime_surface_state.get_runtime_snapshot()

    assert snapshot['progress']['visible'] is True
    assert snapshot['progress']['text'] == 'Wait for the upload to finish.'


def test_invalid_queued_request_does_not_overwrite_active_task_status(monkeypatch):
    active_task = worker.AsyncTask({'prompt': 'Active Task', 'image_seed': 89})
    monkeypatch.setattr(
        ui_logic.runtime_surface_state.worker,
        'get_active_task',
        lambda: active_task,
    )
    ui_logic.runtime_surface_state.get_runtime_snapshot()
    ui_logic.runtime_surface_state.set_progress_state(
        visible=True,
        number=42,
        text='Sampling step 8/20 (40%)',
    )

    ui_logic.runtime_surface_state.set_idle_notice('Correct the next request.')
    snapshot = ui_logic.runtime_surface_state.get_runtime_snapshot()

    assert snapshot['progress'] == {
        'visible': True,
        'number': 42,
        'text': 'Sampling step 8/20 (40%)',
    }


def test_cancel_pending_task():
    # Clean queue
    worker.async_tasks.clear()
    
    task1 = worker.AsyncTask({'prompt': 'Task 1'})
    task2 = worker.AsyncTask({'prompt': 'Task 2'})
    
    worker.async_tasks.append(task1)
    worker.async_tasks.append(task2)
    
    assert len(worker.async_tasks) == 2
    
    # Cancel task 2 (pending)
    cancelled = worker.cancel_task(task2.task_id)
    assert cancelled is True
    assert len(worker.async_tasks) == 1
    assert worker.async_tasks[0].task_id == task1.task_id
    
    # Check that task2 yields has finish event to unblock UI wait loop
    assert len(task2.yields) == 1
    assert task2.yields[0][0] == 'finish'
    assert task2.yields[0][1] == []


def test_cancel_active_task(monkeypatch):
    worker.async_tasks.clear()
    
    task1 = worker.AsyncTask({'prompt': 'Task 1'})
    worker.set_active_task(task1)
    
    interrupt_called = False
    def mock_request_interrupt(action, task):
        nonlocal interrupt_called
        if action == 'stop' and task.task_id == task1.task_id:
            interrupt_called = True
            
    monkeypatch.setattr(worker, 'request_interrupt', mock_request_interrupt)
    
    cancelled = worker.cancel_task(task1.task_id)
    assert cancelled is True
    assert interrupt_called is True
    
    worker.set_active_task(None)


def test_poll_active_task_status_deduplicates_results_and_finish_events():
    worker.async_tasks.clear()

    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 11})
    worker.set_active_task(task)

    task.yields.append(['results', ['img-1.png']])
    task.yields.append(['finish', ['img-1.png']])

    result = ui_logic.poll_active_task_status([], None, None, False)

    assert result[9] == ['img-1.png']
    assert task.ui_delivered_result_count == 1

    worker.set_active_task(None)


def test_poll_active_task_status_only_toggles_layout_when_task_changes():
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 22})
    worker.set_active_task(task)

    first = ui_logic.poll_active_task_status([], None, None, False)
    second = ui_logic.poll_active_task_status([], None, task.task_id, False)

    assert first[12].get('visible') is True
    assert first[13].get('visible') is False
    assert second[12] == gr.skip()
    assert second[13] == gr.skip()


def test_progressbar_updates_current_progress_and_yields_preview():
    task_state = types.SimpleNamespace(yields=[], current_progress=0, current_status_text='')

    worker.progressbar(task_state, 37, 'Working ...')

    assert task_state.current_progress == 37
    assert task_state.current_status_text == 'Working ...'
    assert task_state.yields == [['preview', (37, 'Working ...', None)]]


def test_get_pending_queue_html_escapes_prompt_without_shadowing_html_module():
    task = worker.AsyncTask({'prompt': '<danger & prompt>', 'image_seed': 123})
    worker.async_tasks.append(task)

    rendered = ui_logic.get_pending_queue_html(worker.async_tasks, active_task=None)

    assert '&lt;danger &amp; prompt&gt;' in rendered
    assert '<danger & prompt>' not in rendered


def test_poll_active_task_status_creates_completed_task_record():
    task = worker.AsyncTask({'prompt': 'Test Prompt', 'image_seed': 42})
    worker.set_active_task(task)

    task.yields.append(['finish', ['img-output.png']])

    # Call poll status (new_gallery, last_preview, active_id, disable_preview)
    result = ui_logic.poll_active_task_status([], None, None, False)

    # Assert completed task history has 1 entry
    assert len(ui_logic.completed_tasks_history) == 1
    record = ui_logic.completed_tasks_history[0]
    assert record.task_id == task.task_id
    assert record.prompt == 'Test Prompt'
    assert record.seed == 42
    assert record.images == ['img-output.png']

    # Assert completed html output is in the 15th output (index 14)
    completed_html = result[14]
    assert completed_html != gr.skip()
    assert 'Completed' in completed_html
    assert task.task_id in completed_html
    assert 'img-output.png' in completed_html
    assert 'stageAllImages' in completed_html


def test_poll_active_task_status_preserves_preview_when_active_id_resets():
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 101})
    worker.set_active_task(task)

    result = ui_logic.poll_active_task_status([], 'keep-preview.png', None, False)

    assert result[6] == gr.skip()
    assert result[10] == 'keep-preview.png'


def test_poll_active_task_status_records_completion_across_task_handoff():
    first_task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 111})
    worker.set_active_task(first_task)

    ui_logic.poll_active_task_status([], None, None, False)

    first_task.yields.append(['finish', ['final-a.png']])
    second_task = worker.AsyncTask({'prompt': 'Task 2', 'image_seed': 222})
    worker.set_active_task(second_task)

    result = ui_logic.poll_active_task_status([], None, first_task.task_id, False)

    assert len(ui_logic.completed_tasks_history) == 1
    assert ui_logic.completed_tasks_history[0].task_id == first_task.task_id
    assert result[10] == 'final-a.png'
    assert result[11] == second_task.task_id
    assert 'final-a.png' in result[14]


def test_poll_active_task_status_skips_duplicate_progress_html_updates():
    task = worker.AsyncTask({'prompt': 'Task 1', 'image_seed': 333})
    worker.set_active_task(task)

    task.yields.append(['preview', (12, 'Loading models ...', None)])
    first = ui_logic.poll_active_task_status([], None, None, False)

    task.yields.append(['preview', (12, 'Loading models ...', None)])
    second = ui_logic.poll_active_task_status([], None, task.task_id, False)

    assert first[5] != gr.skip()
    assert second[5] == gr.skip()


def test_enqueue_tasks_does_not_reset_progress_when_another_task_is_running():
    active_task = worker.AsyncTask({'prompt': 'Active Task', 'image_seed': 444})
    queued_task = worker.AsyncTask({'prompt': 'Queued Task', 'image_seed': 555})
    worker.set_active_task(active_task)
    ui_logic.runtime_surface_state.get_runtime_snapshot()
    ui_logic.runtime_surface_state.set_progress_state(visible=True, number=37, text='Sampling step 3/20 ...')

    result_task = ui_logic.enqueue_tasks([queued_task], False, 'uov', False, False)
    snapshot = ui_logic.runtime_surface_state.get_runtime_snapshot()

    assert result_task is queued_task
    assert snapshot['progress']['number'] == 37
    assert snapshot['progress']['text'] == 'Sampling step 3/20 ...'


def test_get_completed_queue_html_encodes_file_urls():
    record = ui_logic.CompletedTaskRecord(
        task_id='task1234',
        prompt='Prompt',
        model_name='Model',
        seed=1,
        images=[r'D:\AI\Fooocus_Nex\outputs\my image #1.png'],
    )
    ui_logic.completed_tasks_history.append(record)

    rendered = ui_logic.get_completed_queue_html()

    assert '/file=D%3A%5CAI%5CFooocus_Nex%5Coutputs%5Cmy%20image%20%231.png' in rendered
