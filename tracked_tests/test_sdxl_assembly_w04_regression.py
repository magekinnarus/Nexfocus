from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.model_taxonomy as model_taxonomy
from backend import resources
from backend.sdxl_assembly.assembly import SDXLAssembly
from backend.sdxl_assembly.contracts import ResolvedFileIdentity, SDXLAssemblyRequest
from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback
from backend.sdxl_assembly.request_builder import build_assembly_request
from modules.pipeline import inference


def _task_state(**overrides):
    state = SimpleNamespace(
        last_stop=False,
        base_model_name='sdxl_base.safetensors',
        vae_name='Default (model)',
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
        prefetch_depth=1,
        prefetch_chunk_mb=64,
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


def _identity(path: Path, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=path,
        sha256=sha,
        size_bytes=path.stat().st_size,
        modified_ns=path.stat().st_mtime_ns,
    )


def test_build_assembly_request_uses_task_state_streaming_settings_and_processed_loras(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')
    inline_lora_path = tmp_path / 'inline_lora.safetensors'
    inline_lora_path.write_bytes(b'lora')

    def fake_resolve(name, _folders):
        lookup = {
            'sdxl_base.safetensors': checkpoint_path,
            'inline_lora.safetensors': inline_lora_path,
        }
        return str(lookup[name])

    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.get_file_from_folder_list',
        fake_resolve,
    )
    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy',
        lambda _path: SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL),
    )

    request = build_assembly_request(
        task_state=_task_state(
            prefetch_depth=2,
            prefetch_chunk_mb=128,
            loras_processed=[('inline_lora.safetensors', 0.7)],
            sdxl_execution_policy=SimpleNamespace(execution_mode='resident'),
        ),
        task_dict=_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
    )

    assert request.prefetch_depth == 2
    assert request.prefetch_chunk_mb == 128
    assert request.unet_posture.value == 'streaming'
    assert len(request.lora_specs) == 1
    assert request.lora_specs[0].file_identity.path == inline_lora_path


def test_run_sdxl_assembly_task_preserves_interrupts(monkeypatch):
    import backend.sdxl_assembly.gateway as gateway

    close_calls = []
    request = object()

    class FakeAssembly:
        def execute(self, _request, callback=None):
            raise resources.InterruptProcessingException()

        def close(self):
            close_calls.append('closed')

    monkeypatch.setattr(gateway, 'build_assembly_request', lambda *args, **kwargs: request)
    monkeypatch.setattr(gateway.SDXLAssemblyDirector, 'select_assembly', lambda _request: FakeAssembly())

    with pytest.raises(resources.InterruptProcessingException):
        run_sdxl_assembly_task(
            _task_state(),
            _task_dict(),
            current_task_id=0,
            total_count=1,
            all_steps=3,
            preparation_steps=0,
            denoising_strength=1.0,
            final_scheduler_name='karras',
            loras=[],
        )

    assert close_calls == ['closed']


def test_sdxl_assembly_execute_preserves_interrupts(tmp_path):
    checkpoint_path = tmp_path / 'checkpoint.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    request = SDXLAssemblyRequest(
        request_id='req_interrupt',
        route_id='txt2img_assembly',
        image_index=0,
        image_count=1,
        checkpoint=_identity(checkpoint_path, 'checkpoint_sha'),
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
        device='cpu',
    )

    lora_worker = SimpleNamespace(materialize_patches=lambda: None, teardown_assembly_order=lambda: None)
    text_worker = SimpleNamespace(get_conditioning=lambda: {}, teardown_assembly_order=lambda: None)
    vae_worker = SimpleNamespace(
        prepare_latents=lambda _device: SimpleNamespace(samples=torch.zeros((1, 4, 8, 8))),
        teardown_assembly_order=lambda: None,
    )
    unet_spine = SimpleNamespace(
        start=lambda: None,
        denoise=lambda *args, **kwargs: (_ for _ in ()).throw(resources.InterruptProcessingException()),
        end=lambda: None,
        teardown_assembly_order=lambda: None,
    )

    assembly = SDXLAssembly(unet_spine, text_worker, vae_worker, lora_worker)

    with pytest.raises(resources.InterruptProcessingException):
        assembly.execute(request)


def test_process_task_propagates_interrupt_processing_exception(monkeypatch):
    from backend.sdxl_assembly import gateway

    save_calls = []

    monkeypatch.setattr(gateway, 'is_eligible_for_sdxl_assembly', lambda **_kwargs: (True, None))
    monkeypatch.setattr(
        gateway,
        'run_sdxl_assembly_task',
        lambda *args, **kwargs: (_ for _ in ()).throw(resources.InterruptProcessingException()),
    )
    monkeypatch.setattr(
        inference,
        'save_and_log',
        lambda *args, **kwargs: save_calls.append((args, kwargs)) or ['saved-path'],
    )

    with pytest.raises(resources.InterruptProcessingException):
        inference.process_task(
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

    assert save_calls == []


def test_process_task_assembly_route_emits_preview_image(monkeypatch):
    from backend.sdxl_assembly import gateway

    task_state = _task_state(disable_preview=False, yields=[])

    monkeypatch.setattr(gateway, 'is_eligible_for_sdxl_assembly', lambda **_kwargs: (True, None))
    monkeypatch.setattr(inference, 'save_and_log', lambda *args, **kwargs: ['saved-path'])

    latent_format = SimpleNamespace(
        latent_rgb_factors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )

    def fake_run(*args, **kwargs):
        holder = kwargs.get('preview_runtime_holder')
        assert isinstance(holder, dict)
        holder['assembly'] = SimpleNamespace(
            unet_spine=SimpleNamespace(
                unet=SimpleNamespace(
                    load_device=torch.device('cpu'),
                    model=SimpleNamespace(latent_format=latent_format),
                )
            ),
            vae_worker=SimpleNamespace(vae=None),
        )
        kwargs['progressbar_callback'](
            0,
            None,
            None,
            1,
            torch.zeros((1, 4, 1, 1), dtype=torch.float32),
        )
        return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(gateway, 'run_sdxl_assembly_task', fake_run)

    imgs, paths, progress = inference.process_task(
        task_state=task_state,
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

    assert imgs[0].shape == (1, 1, 3)
    assert paths == ['saved-path']
    assert any(
        item[0] == 'preview' and isinstance(item[1][2], np.ndarray)
        for item in task_state.yields
    )


def test_assembly_progress_callback_throttles_full_memory_telemetry(monkeypatch):
    import backend.sdxl_assembly.progress as progress

    full_snapshots = []
    lightweight_debug = []

    monkeypatch.setattr(
        progress,
        'log_telemetry',
        lambda event, extra_msg='': full_snapshots.append((event, extra_msg)),
    )
    monkeypatch.setattr(
        progress.logger,
        'debug',
        lambda msg, *args, **kwargs: lightweight_debug.append(msg % args if args else msg),
    )

    callback = SDXLAssemblyProgressCallback(SimpleNamespace(), None)
    for step in range(6):
        callback(step, None, None, 6, None)

    assert [event for event, _extra in full_snapshots] == [
        'spine_stream_step',
        'spine_stream_step',
        'spine_stream_step',
    ]
    assert 'step=0 total_steps=6' in full_snapshots[0][1]
    assert 'step=4 total_steps=6' in full_snapshots[1][1]
    assert 'step=5 total_steps=6' in full_snapshots[2][1]
    assert any('step=1 total_steps=6' in line for line in lightweight_debug)
    assert any('step=2 total_steps=6' in line for line in lightweight_debug)
    assert any('step=3 total_steps=6' in line for line in lightweight_debug)
