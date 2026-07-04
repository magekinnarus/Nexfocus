from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.sdxl_assembly.contracts import (
    ResolvedFileIdentity,
    SDXLAssemblyRequest,
    SDXLLoraSpec,
)
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.runtime_state import (
    acquire_text_encoder_component,
    acquire_unet_component,
    acquire_vae_component,
    clear_all_caches,
)
from modules.pipeline import inference


class CloneCounter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.clone_count = 0
        self.patcher = MagicMock()

    def clone(self):
        self.clone_count += 1
        return SimpleNamespace(
            component=self.name,
            clone_index=self.clone_count,
            patcher=self.patcher,
        )


def _identity(name: str, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=Path(name),
        sha256=sha,
        size_bytes=1,
        modified_ns=1,
    )


def _request(**overrides) -> SDXLAssemblyRequest:
    kwargs = {
        'request_id': 'req_w03_regression',
        'route_id': 'txt2img_assembly',
        'image_index': 0,
        'image_count': 1,
        'checkpoint': _identity('checkpoint.safetensors', 'checkpoint_sha'),
        'vae': _identity('vae.safetensors', 'vae_sha'),
        'model_variant_key': 'sdxl',
        'prompt': 'prompt',
        'negative_prompt': 'negative',
        'positive_texts': ('prompt',),
        'negative_texts': ('negative',),
        'width': 64,
        'height': 64,
        'steps': 3,
        'cfg': 5.0,
        'sampler': 'euler',
        'scheduler': 'karras',
        'seed': 123,
        'device': 'cpu',
    }
    kwargs.update(overrides)
    return SDXLAssemblyRequest(**kwargs)


def test_component_acquisition_loads_only_requested_component(monkeypatch):
    import backend.loader as loader

    clear_all_caches()
    unet = CloneCounter('unet')
    clip = CloneCounter('clip')
    vae = SimpleNamespace(component='vae')
    load_calls = []

    def fake_load_unet(*args, **kwargs):
        load_calls.append(('unet', args, kwargs))
        return unet

    def fake_load_clip(*args, **kwargs):
        load_calls.append(('clip', args, kwargs))
        return clip

    def fake_load_vae(*args, **kwargs):
        load_calls.append(('vae', args, kwargs))
        return vae

    monkeypatch.setattr(loader, '_stream_load_sdxl_unet_from_checkpoint', fake_load_unet)
    monkeypatch.setattr(loader, 'load_sdxl_clip', fake_load_clip)
    monkeypatch.setattr(loader, 'load_vae', fake_load_vae)
    request = _request()

    text_component = acquire_text_encoder_component(request)
    assert text_component.component == 'clip'
    assert clip.clone_count == 1
    assert unet.clone_count == 0
    assert [call[0] for call in load_calls] == ['clip']

    vae_component = acquire_vae_component(request)
    assert vae_component is vae
    assert clip.clone_count == 1
    assert unet.clone_count == 0
    assert [call[0] for call in load_calls] == ['clip', 'vae']

    unet_component = acquire_unet_component(request)
    assert unet_component is unet
    assert unet.clone_count == 0
    assert clip.clone_count == 1
    assert [call[0] for call in load_calls] == ['clip', 'vae', 'unet']
    unet_kwargs = load_calls[-1][2]
    assert unet_kwargs['raw_byte_stream'] is True
    assert unet_kwargs['stream_chunk_bytes'] == request.prefetch_chunk_mb * 1024 * 1024

    second_text_component = acquire_text_encoder_component(request)
    assert second_text_component.component == 'clip'
    assert clip.clone_count == 2
    assert [call[0] for call in load_calls] == ['clip', 'vae', 'unet']

    clear_all_caches()


def test_sdxl_unet_loader_records_raw_sequential_stream(monkeypatch):
    import backend.loader as loader

    captured = {}

    class DummySDXL(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.diffusion_model = torch.nn.Linear(2, 2, bias=False)

    def fake_stream_load(path, prefixes, module, **kwargs):
        captured['path'] = path
        captured['prefixes'] = prefixes
        captured['module'] = module
        captured['kwargs'] = dict(kwargs)
        load_metrics = kwargs.get('load_metrics')
        if isinstance(load_metrics, dict):
            load_metrics['realized_pinned_bytes'] = 0
            load_metrics['realized_pinned_tensor_count'] = 0
        return [], []

    monkeypatch.setattr(loader.model_base, 'SDXL', DummySDXL)
    monkeypatch.setattr(loader, '_load_prefixed_safetensors_into_module', fake_stream_load)

    patcher = loader._stream_load_sdxl_unet_from_checkpoint(
        'checkpoint.safetensors',
        load_device=torch.device('cpu'),
        offload_device=torch.device('cpu'),
        dtype=torch.float16,
        reload_prefixes=['model.diffusion_model.'],
        stream_chunk_bytes=64 * 1024 * 1024,
    )

    assert captured['path'] == 'checkpoint.safetensors'
    assert captured['prefixes'] == ['model.diffusion_model.']
    assert captured['kwargs']['raw_byte_stream'] is True
    assert captured['kwargs']['chunk_bytes'] == 64 * 1024 * 1024
    assert patcher.model_options['sdxl_assembly_loader'] == {
        'direct_safetensors_load': True,
        'raw_sequential_stream': True,
        'meta_construction': True,
        'stream_chunk_bytes': 64 * 1024 * 1024,
        'realized_cpu_bytes': 0,
        'realized_cpu_tensor_count': 0,
        'realized_pinned_bytes': 0,
        'realized_pinned_tensor_count': 0,
    }


def test_sdxl_telemetry_sink_receives_worker_events():
    from backend.sdxl_assembly.progress import log_telemetry, telemetry_sink

    snapshots = []
    with telemetry_sink(lambda snapshot: snapshots.append(snapshot)):
        log_telemetry('w03_probe_event', 'phase=test')

    assert len(snapshots) == 1
    assert snapshots[0]['event'] == 'w03_probe_event'
    assert snapshots[0]['extra'] == 'phase=test'
    assert 'proc_rss_mb' in snapshots[0]

    log_telemetry('w03_probe_event_after_unregister')
    assert len(snapshots) == 1


def test_director_wires_one_lora_worker_to_text_and_unet_workers():
    clear_all_caches()
    assembly = SDXLAssemblyDirector.select_assembly(_request(lora_stack_hash='stack_a'))
    try:
        assert assembly.text_encode_worker.lora_worker is assembly.lora_worker
        assert assembly.unet_spine.lora_worker is assembly.lora_worker
    finally:
        assembly.close()
        clear_all_caches()


def test_streaming_unet_host_pinning_is_explicit_request_metadata(monkeypatch):
    from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine
    from backend.cpu_compiler import CpuArtifactCompiler

    compile_calls = []
    fake_unet = SimpleNamespace(runtime_release_to_meta=True)
    fake_lora_worker = SimpleNamespace(apply_unet_patches=lambda _unet: None)

    monkeypatch.setattr(
        'backend.sdxl_assembly.streaming_unet.acquire_unet_component',
        lambda _request: fake_unet,
    )
    monkeypatch.setattr(
        CpuArtifactCompiler,
        'compile_patcher',
        lambda _unet, *, pin_unet_host=True: compile_calls.append(pin_unet_host),
    )

    StreamingUnetSpine(_request(), lora_worker=fake_lora_worker).start()
    StreamingUnetSpine(
        _request(metadata={'pin_unet_host': True}),
        lora_worker=fake_lora_worker,
    ).start()

    assert compile_calls == [False, True]


def test_lora_worker_caches_zero_patch_clip_results(monkeypatch):
    import backend.sdxl_assembly.cpu_lora_worker as cpu_lora_worker

    clear_all_caches()
    cpu_lora_worker._PARSED_LORA_CACHE.clear()

    load_calls = []

    class DummyClipModel:
        pass

    clip = SimpleNamespace(
        patcher=SimpleNamespace(
            model=DummyClipModel(),
            add_patches=lambda _patch_dict, _weight: None,
        )
    )

    monkeypatch.setattr(cpu_lora_worker, "SafeOpenHeaderOnly", lambda _path: object())
    monkeypatch.setattr(cpu_lora_worker.backend_lora, "model_lora_keys_clip", lambda _model: {})
    monkeypatch.setattr(
        cpu_lora_worker.backend_lora,
        "load_lora",
        lambda _header, _key_map, log_missing=False: load_calls.append(log_missing) or {},
    )

    worker = cpu_lora_worker.CpuLoraWorker(
        _request(
            lora_stack_hash="stack_with_zero_clip_patch",
            lora_specs=(
                SDXLLoraSpec(
                    file_identity=_identity("twbabe.safetensors", "lora_sha"),
                    unet_weight=1.0,
                    clip_weight=1.0,
                    enabled=True,
                ),
            ),
        )
    )

    assert worker.apply_clip_patches(clip) == 0
    assert worker.apply_clip_patches(clip) == 0
    assert len(load_calls) == 1

    cpu_lora_worker._PARSED_LORA_CACHE.clear()
    clear_all_caches()


def test_process_task_keeps_unified_runtime_until_w04(monkeypatch):
    from backend.sdxl_assembly import gateway

    assembly_calls = []

    task_state = SimpleNamespace(
        last_stop=False,
        steps=3,
        height=64,
        width=64,
        use_expansion=False,
        disable_intermediate_results=True,
    )
    task_dict = {'task_seed': 123}

    monkeypatch.setattr(gateway, 'is_eligible_for_sdxl_assembly', lambda **_kwargs: (True, None))
    monkeypatch.setattr(
        gateway,
        'run_sdxl_assembly_task',
        lambda *args, **kwargs: assembly_calls.append((args, kwargs)) or np.zeros((64, 64, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(inference, '_ensure_supported_unified_runtime_request', lambda _state: None)
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
        controlnet_paths={},
        contextual_assets={},
        image_input_result={},
    )

    assert len(assembly_calls) == 1
    assert len(imgs) == 1
    assert img_paths == ['saved-path']
    assert current_progress == 100
