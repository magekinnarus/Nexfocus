import os
import sys
import types

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

import numpy as np
import pytest
import torch

from modules import flags
from modules.pipeline import image_input
from modules.task_state import TaskState


def _rgb(size=8):
    return np.zeros((size, size, 3), dtype=np.uint8)


def test_preprocess_contextual_controlnets_defers_shared_unet_patch_for_unified_owner(monkeypatch):
    task_state = TaskState(width=8, height=8, skipping_cn_preprocessor=True)
    task_state.set_cn_tasks(flags.cn_ip, [[_rgb(), 0.8, 0.7]])

    patch_calls = {"count": 0}

    monkeypatch.setattr(image_input.mask_proc, 'unpack_gradio_data', lambda raw: raw)
    monkeypatch.setattr(image_input, 'resize_image', lambda img, **kwargs: img)
    monkeypatch.setattr(
        image_input.contextual_ip_adapter,
        'preprocess',
        lambda *args, **kwargs: ([torch.ones(1, 1, 1)], [torch.zeros(1, 1, 1)]),
    )
    monkeypatch.setattr(
        image_input.contextual_ip_adapter,
        'patch_model',
        lambda model, tasks: patch_calls.__setitem__("count", patch_calls["count"] + 1) or model,
    )
    monkeypatch.setattr(image_input.pulid_runtime, 'patch_model', lambda model, tasks: model)
    monkeypatch.setattr(image_input, 'pipeline', types.SimpleNamespace(final_unet=object()), raising=False)

    image_input.preprocess_contextual_controlnets(
        task_state,
        contextual_assets={'contextual_model_paths': {flags.cn_ip: 'dummy-model'}},
    )

    assert patch_calls["count"] == 0
    assert task_state.prepared_contextual_cn_tasks[flags.cn_ip][0][3] == pytest.approx(0.0)
    assert isinstance(task_state.prepared_contextual_cn_tasks[flags.cn_ip][0][0], tuple)


def test_preprocess_structural_controlnets_records_prepared_tasks(monkeypatch):
    task_state = TaskState(width=8, height=8, skipping_cn_preprocessor=True)
    task_state.set_cn_tasks(flags.cn_canny, [[_rgb(), 0.8, 0.7]])

    monkeypatch.setattr(image_input.mask_proc, 'unpack_gradio_data', lambda raw: raw)
    monkeypatch.setattr(image_input, 'resize_image', lambda img, **kwargs: img)

    image_input.preprocess_structural_controlnets(task_state, structural_preprocessor_paths={})

    assert len(task_state.prepared_structural_cn_tasks[flags.cn_canny]) == 1
    hint, cn_stop, cn_weight, cn_start, slot_index = task_state.prepared_structural_cn_tasks[flags.cn_canny][0]
    assert isinstance(hint, torch.Tensor)
    assert tuple(int(dim) for dim in hint.shape) == (1, 8, 8, 3)
    assert cn_stop == pytest.approx(0.8)
    assert cn_weight == pytest.approx(0.7)
    assert cn_start == pytest.approx(0.0)
    assert slot_index == 0
