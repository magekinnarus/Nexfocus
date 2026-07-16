import os
import sys

import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.flags as flags
from modules.pipeline.routes import build_generation_route as _build_generation_route, describe_route
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
from modules.task_state import TaskState


def build_generation_route(task_state):
    bind_legacy_workflow_plan(task_state)
    return _build_generation_route(task_state)


def test_build_generation_route_maps_default_txt2img_path():
    task_state = TaskState()

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == ['prompt_encode', 'diffusion_batch']


def test_build_generation_route_maps_inpaint_family():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'inpaint'
    assert describe_route(route) == [
        'image_input_prepare',
        'inpaint_prepare',
        'prompt_encode',
        'diffusion_batch',
    ]


def test_build_generation_route_maps_flux_fill_inpaint_family():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        inpaint_route='flux',
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'flux_inpaint'
    assert route.family == 'flux_fill'
    assert describe_route(route) == [
        'image_input_prepare',
        'flux_inpaint',
    ]


def test_build_generation_route_maps_controlnet_extensions_explicitly():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='ip',
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == [
        'image_input_prepare',
        'controlnet_support_load',
        'prompt_encode',
        'structural_controlnet',
        'diffusion_batch',
    ]


def test_build_generation_route_maps_upscale_family():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='uov',
        uov_method='super-upscale',
        uov_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'super_upscale'
    assert describe_route(route) == ['image_input_prepare', 'prompt_encode', 'upscale']


def test_build_generation_route_ignores_stale_inpaint_mix_without_live_controlnet_tasks():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='ip',
        mixing_image_prompt_and_inpaint=True,
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        inpaint_mask_image=np.zeros((8, 8), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == ['prompt_encode', 'diffusion_batch']


def test_build_generation_route_does_not_add_image_input_stages_for_stale_txt2img_checkbox_state():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='txt2img',
        mixing_image_prompt_and_inpaint=True,
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        inpaint_mask_image=np.zeros((8, 8), dtype=np.uint8),
        uov_method='upscale',
        uov_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == ['prompt_encode', 'diffusion_batch']


def test_build_generation_route_outpaint_no_controlnet_when_checkbox_off():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='outpaint',
        outpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        mixing_image_prompt_and_outpaint=False,
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'outpaint'
    assert 'structural_controlnet' not in describe_route(route)
    assert 'contextual_controlnet' not in describe_route(route)


def test_build_generation_route_outpaint_with_controlnet_when_checkbox_on():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='outpaint',
        outpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        mixing_image_prompt_and_outpaint=True,
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'outpaint'
    assert 'structural_controlnet' in describe_route(route)
    assert 'contextual_controlnet' not in describe_route(route)


def test_build_generation_route_inpaint_no_controlnet_when_checkbox_off():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        mixing_image_prompt_and_inpaint=False,
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'inpaint'
    assert 'structural_controlnet' not in describe_route(route)
    assert 'contextual_controlnet' not in describe_route(route)


def test_build_generation_route_inpaint_with_controlnet_when_checkbox_on():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
        mixing_image_prompt_and_inpaint=True,
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'inpaint'
    assert 'structural_controlnet' in describe_route(route)
    assert 'contextual_controlnet' not in describe_route(route)
