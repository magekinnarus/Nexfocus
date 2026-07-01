from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.flags as flags
import modules.model_taxonomy as model_taxonomy
from backend.sdxl_assembly.contracts import (
    ResolvedFileIdentity,
    SDXLAssemblyRequest,
    SDXLAssemblyValidationError,
)
from backend.sdxl_assembly.request_builder import build_assembly_request
from modules.pipeline import inference


def _task_state(**overrides):
    state = SimpleNamespace(
        last_stop=False,
        base_model_name='sdxl_base.safetensors',
        vae_name=flags.default_vae,
        goals=['txt2img'],
        tiled=False,
        prepared_structural_cn_tasks={},
        prepared_contextual_cn_tasks={},
        initial_latent=None,
        prompt='prompt',
        negative_prompt='negative',
        width=64,
        height=64,
        steps=3,
        cfg_scale=5.0,
        sampler_name='euler',
        clip_skip=1,
        style_selections=[],
        sdxl_execution_policy=None,
        sharpness=2.0,
        adaptive_cfg=7.0,
        adm_scaler_positive=1.5,
        adm_scaler_negative=0.8,
        adm_scaler_end=0.3,
        use_expansion=False,
        disable_intermediate_results=True,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _task_dict():
    return {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }


def test_request_validation_rejects_missing_checkpoint():
    request = SDXLAssemblyRequest(
        request_id='req_1',
        route_id='txt2img_assembly',
        image_index=0,
        image_count=1,
        checkpoint=ResolvedFileIdentity(
            path=Path(os.path.abspath('missing_checkpoint.safetensors')),
            sha256='sha',
            size_bytes=1,
            modified_ns=1,
        ),
        vae=None,
        model_variant_key='sdxl',
        prompt='prompt',
        negative_prompt='negative',
        positive_texts=('prompt',),
        negative_texts=('negative',),
        width=64,
        height=64,
        steps=3,
        cfg=5.0,
        sampler='euler',
        scheduler='karras',
        seed=123,
    )

    try:
        request.validate()
    except SDXLAssemblyValidationError:
        return

    raise AssertionError('missing checkpoint should fail request validation')


def test_build_assembly_request_smoke_resolves_frozen_snapshot(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'w02 checkpoint identity')

    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.get_file_from_folder_list',
        lambda _name, _folders: str(checkpoint_path),
    )
    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy',
        lambda _path: SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL),
    )

    request = build_assembly_request(
        task_state=_task_state(),
        task_dict=_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
    )

    assert request.request_id.startswith('req_0_')
    assert request.checkpoint.path == checkpoint_path
    assert request.prompt == 'prompt'
    assert request.negative_prompt == 'negative'
    assert request.prefetch_chunk_mb == 64
    assert request.prefetch_depth == 1


def test_process_task_keeps_unified_runtime_until_w04_even_when_w02_seam_is_eligible(monkeypatch):
    from backend.sdxl_assembly import gateway

    unified_calls = []

    def fail_if_assembly_runs(*_args, **_kwargs):
        raise AssertionError('W02/W03 must not execute the assembly route from process_task')

    monkeypatch.setattr(
        gateway,
        'is_eligible_for_sdxl_assembly',
        lambda **_kwargs: (True, None),
    )
    monkeypatch.setattr(gateway, 'run_sdxl_assembly_task', fail_if_assembly_runs)
    monkeypatch.setattr(inference, '_ensure_supported_unified_runtime_request', lambda _state: None)
    monkeypatch.setattr(
        inference,
        '_run_unified_sdxl_task',
        lambda *args, **kwargs: unified_calls.append((args, kwargs)) or [np.zeros((64, 64, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    imgs, img_paths, current_progress = inference.process_task(
        task_state=_task_state(),
        task_dict=_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        controlnet_paths={},
        contextual_assets={},
        image_input_result={},
    )

    assert len(unified_calls) == 1
    assert len(imgs) == 1
    assert img_paths == ['saved-path']
    assert current_progress == 100
