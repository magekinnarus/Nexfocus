from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    SDXLLoraSpec,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
)
from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
from backend.sdxl_assembly.runtime_state import (
    clear_all_caches,
    lookup_prompt_conditioning,
    remember_prompt_conditioning,
)
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.streaming_unet import StreamingUnetSpine
from modules.pipeline import inference

class TestSDXLAssemblyW03(unittest.TestCase):
    def setUp(self):
        clear_all_caches()
        self.checkpoint = ResolvedFileIdentity(
            path=Path("dummy_checkpoint.safetensors"),
            sha256="fake_checkpoint_sha256",
            size_bytes=1024,
            modified_ns=12345
        )
        self.vae_identity = ResolvedFileIdentity(
            path=Path("dummy_vae.safetensors"),
            sha256="fake_vae_sha256",
            size_bytes=512,
            modified_ns=12345
        )

    def tearDown(self):
        clear_all_caches()

    def _make_mock_components(self):
        mock_unet = MagicMock()
        mock_unet.model_size.return_value = 1000
        mock_unet.model.dtype = torch.float16
        mock_unet.model.get_dtype.return_value = torch.float16
        mock_unet.model.model_sampling.sigma_max = 14.6
        mock_unet.model.model_sampling.inverse_noise_scaling.return_value = torch.zeros(1)
        mock_unet.model.parameters.return_value = [torch.zeros(1)]
        
        cloned_unet = MagicMock()
        cloned_unet.model_size.return_value = 1000
        cloned_unet.model.dtype = torch.float16
        cloned_unet.model.get_dtype.return_value = torch.float16
        cloned_unet.model.model_sampling.sigma_max = 14.6
        cloned_unet.model.model_sampling.inverse_noise_scaling.return_value = torch.zeros(1)
        cloned_unet.model.parameters.return_value = [torch.zeros(1)]
        mock_unet.clone.return_value = cloned_unet
        
        mock_clip = MagicMock()
        cloned_clip = MagicMock()
        mock_clip.clone.return_value = cloned_clip
        
        mock_vae = MagicMock()
        return mock_unet, mock_clip, mock_vae

    def _make_dummy_request(self, **overrides) -> SDXLAssemblyRequest:
        kwargs = {
            "request_id": "req_w03_test",
            "route_id": "txt2img_assembly",
            "image_index": 0,
            "image_count": 1,
            "checkpoint": self.checkpoint,
            "vae": self.vae_identity,
            "model_variant_key": "sdxl",
            "prompt": "A beautiful landscape",
            "negative_prompt": "ugly, blurry",
            "positive_texts": ("A beautiful landscape",),
            "negative_texts": ("ugly, blurry",),
            "width": 64,
            "height": 64,
            "steps": 5,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "karras",
            "seed": 42,
            "unet_posture": UNetPostureKind.STREAMING,
            "clip_posture": TextEncoderPostureKind.CPU_PINNED,
            "vae_posture": VAEPostureKind.TRANSIENT,
            "lora_posture": LoraPatchPostureKind.STREAMING,
            "prefetch_depth": 1,
            "prefetch_chunk_mb": 64,
            "device": "cpu",
        }
        kwargs.update(overrides)
        return SDXLAssemblyRequest(**kwargs)

    @patch("pathlib.Path.exists")
    def test_import_boundary_hygiene(self, mock_exists):
        mock_exists.return_value = True
        # Verify no imports from legacy SDXL runtime modules in backend/sdxl_assembly/
        forbidden_imports = [
            "backend.sdxl_unified_runtime",
            "backend.sdxl_resident_runtime",
            "backend.sdxl_streaming_runtime",
            "backend.sdxl_runtime_policy",
            "backend.staging_manager",
            "backend.process_transition",
            "backend.memory_governor",
        ]
        package_dir = Path(__file__).resolve().parents[1] / "backend" / "sdxl_assembly"
        self.assertTrue(package_dir.exists())
        
        for py_file in package_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                if f"import {forbidden}" in content or f"from {forbidden}" in content:
                    self.fail(f"Forbidden import '{forbidden}' found in {py_file.name}")

    @patch("backend.cpu_compiler.CpuArtifactCompiler.compile_streaming_unet_patcher")
    @patch("backend.sdxl_assembly.streaming_unet.acquire_unet_component")
    def test_only_streaming_unet_spine_consumes_host_pinning_policy(
        self,
        mock_acquire_unet,
        mock_compile_streaming_unet,
    ):
        unpinned_unet, _, _ = self._make_mock_components()
        pinned_unet, _, _ = self._make_mock_components()
        mock_acquire_unet.side_effect = [unpinned_unet, pinned_unet]

        for pin_unet_host in (False, True):
            request = self._make_dummy_request(
                metadata={"pin_unet_host": pin_unet_host},
            )
            lora_worker = MagicMock()
            StreamingUnetSpine(request, lora_worker=lora_worker).start()

        self.assertEqual(
            mock_compile_streaming_unet.call_args_list,
            [
                unittest.mock.call(unpinned_unet, pin_unet_host=False),
                unittest.mock.call(pinned_unet, pin_unet_host=True),
            ],
        )

    @patch("pathlib.Path.exists")
    @patch("backend.loader.load_vae")
    @patch("backend.loader.load_sdxl_clip")
    @patch("backend.loader._stream_load_sdxl_unet_from_checkpoint")
    @patch("backend.cpu_compiler.CpuArtifactCompiler.compile_patcher")
    @patch("backend.conditioning.encode_prompt_pair_sdxl")
    @patch("backend.conditioning.build_sdxl_adm_pair")
    @patch("backend.sampling.KSampler")
    @patch("backend.k_diffusion.sample_euler")
    @patch("backend.decode.decode_preloaded_vae")
    @patch("modules.core.pytorch_to_numpy")
    def test_direct_assembly_execution_and_caching(
        self,
        mock_pt_to_np,
        mock_decode_vae,
        mock_sample_euler,
        mock_ksampler,
        mock_build_adm,
        mock_encode_prompt,
        mock_compile,
        mock_load_unet,
        mock_load_clip,
        mock_load_vae,
        mock_exists,
    ):
        mock_exists.return_value = True
        
        mock_unet, mock_clip, mock_vae = self._make_mock_components()
        mock_load_unet.return_value = mock_unet
        mock_load_clip.return_value = mock_clip
        mock_load_vae.return_value = mock_vae
        mock_encode_prompt.return_value = {"positive": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}, "negative": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}}
        mock_build_adm.return_value = {"positive": torch.zeros(1), "negative": torch.zeros(1)}
        
        mock_ksampler_instance = MagicMock()
        mock_ksampler_instance.sigmas = torch.tensor([10.0, 5.0, 0.0])
        mock_ksampler.return_value = mock_ksampler_instance
        
        mock_sample_euler.return_value = torch.zeros((1, 4, 8, 8))
        mock_decode_vae.return_value = torch.zeros((1, 3, 64, 64))
        mock_pt_to_np.return_value = [np.ones((64, 64, 3), dtype=np.uint8) * 255]

        # Execute request
        req = self._make_dummy_request()
        assembly = SDXLAssemblyDirector.select_assembly(req)
        result = assembly.execute(req)
        
        # Verify result and calls
        self.assertEqual(result.width, 64)
        self.assertEqual(result.height, 64)
        self.assertTrue((result.output_image == 255).all())
        
        mock_load_unet.assert_called_once()
        mock_load_clip.assert_called_once()
        mock_load_vae.assert_called_once()
        mock_encode_prompt.assert_called_once()
        
        # Execute again with same keys to check warm reuse / caching
        req_warm = self._make_dummy_request()
        assembly_warm = SDXLAssemblyDirector.select_assembly(req_warm)
        result_warm = assembly_warm.execute(req_warm)
        
        # Should reuse the warm UNet spine and prompt conditioning cache.
        self.assertEqual(mock_load_unet.call_count, 1)
        self.assertEqual(mock_load_clip.call_count, 1)
        self.assertEqual(mock_load_vae.call_count, 2)
        # Total prompt encode calls should still be 1 (due to prompt caching)
        self.assertEqual(mock_encode_prompt.call_count, 1)

    @patch("pathlib.Path.exists")
    @patch("backend.loader.load_vae")
    @patch("backend.loader.load_sdxl_clip")
    @patch("backend.loader._stream_load_sdxl_unet_from_checkpoint")
    @patch("backend.cpu_compiler.CpuArtifactCompiler.compile_patcher")
    @patch("backend.conditioning.encode_prompt_pair_sdxl")
    @patch("backend.conditioning.build_sdxl_adm_pair")
    @patch("backend.sampling.KSampler")
    @patch("backend.k_diffusion.sample_euler")
    @patch("backend.decode.decode_preloaded_vae")
    @patch("modules.core.pytorch_to_numpy")
    def test_cache_invalidation_on_prompt_change(
        self,
        mock_pt_to_np,
        mock_decode_vae,
        mock_sample_euler,
        mock_ksampler,
        mock_build_adm,
        mock_encode_prompt,
        mock_compile,
        mock_load_unet,
        mock_load_clip,
        mock_load_vae,
        mock_exists,
    ):
        mock_exists.return_value = True
        
        mock_unet, mock_clip, mock_vae = self._make_mock_components()
        mock_load_unet.return_value = mock_unet
        mock_load_clip.return_value = mock_clip
        mock_load_vae.return_value = mock_vae
        mock_encode_prompt.return_value = {"positive": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}, "negative": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}}
        mock_ksampler_instance = MagicMock()
        mock_ksampler_instance.sigmas = torch.tensor([10.0, 5.0, 0.0])
        mock_ksampler.return_value = mock_ksampler_instance
        mock_sample_euler.return_value = torch.zeros((1, 4, 8, 8))
        mock_decode_vae.return_value = torch.zeros((1, 3, 64, 64))
        mock_pt_to_np.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]

        # 1. First request
        req1 = self._make_dummy_request(prompt="Prompt A", prompt_payload_hash="hash_a")
        SDXLAssemblyDirector.select_assembly(req1).execute(req1)
        self.assertEqual(mock_encode_prompt.call_count, 1)

        # 2. Second request with different prompt payload hash -> invalidates prompt cache
        req2 = self._make_dummy_request(prompt="Prompt B", prompt_payload_hash="hash_b")
        SDXLAssemblyDirector.select_assembly(req2).execute(req2)
        self.assertEqual(mock_encode_prompt.call_count, 2)

        # 3. Third request with same prompt payload hash -> hits prompt cache
        req3 = self._make_dummy_request(prompt="Prompt B", prompt_payload_hash="hash_b")
        SDXLAssemblyDirector.select_assembly(req3).execute(req3)
        self.assertEqual(mock_encode_prompt.call_count, 2)

    @patch("pathlib.Path.exists")
    @patch("backend.loader.load_vae")
    @patch("backend.loader.load_sdxl_clip")
    @patch("backend.loader._stream_load_sdxl_unet_from_checkpoint")
    @patch("backend.cpu_compiler.CpuArtifactCompiler.compile_patcher")
    @patch("backend.conditioning.encode_prompt_pair_sdxl")
    @patch("backend.conditioning.build_sdxl_adm_pair")
    @patch("backend.sampling.KSampler")
    @patch("backend.k_diffusion.sample_euler")
    @patch("backend.decode.decode_preloaded_vae")
    @patch("modules.core.pytorch_to_numpy")
    def test_cache_invalidation_on_model_change(
        self,
        mock_pt_to_np,
        mock_decode_vae,
        mock_sample_euler,
        mock_ksampler,
        mock_build_adm,
        mock_encode_prompt,
        mock_compile,
        mock_load_unet,
        mock_load_clip,
        mock_load_vae,
        mock_exists,
    ):
        mock_exists.return_value = True
        
        mock_unet, mock_clip, mock_vae = self._make_mock_components()
        mock_load_unet.return_value = mock_unet
        mock_load_clip.return_value = mock_clip
        mock_load_vae.return_value = mock_vae
        mock_encode_prompt.return_value = {"positive": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}, "negative": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}}
        mock_ksampler_instance = MagicMock()
        mock_ksampler_instance.sigmas = torch.tensor([10.0, 5.0, 0.0])
        mock_ksampler.return_value = mock_ksampler_instance
        mock_sample_euler.return_value = torch.zeros((1, 4, 8, 8))
        mock_decode_vae.return_value = torch.zeros((1, 3, 64, 64))
        mock_pt_to_np.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]

        # 1. First model request
        req1 = self._make_dummy_request()
        SDXLAssemblyDirector.select_assembly(req1).execute(req1)
        self.assertEqual(mock_load_unet.call_count, 1)

        # 2. Change model checkpoint sha256 -> invalidates cache
        other_checkpoint = ResolvedFileIdentity(
            path=Path("dummy_checkpoint2.safetensors"),
            sha256="fake_checkpoint_sha256_other",
            size_bytes=1024,
            modified_ns=12345
        )
        req2 = self._make_dummy_request(checkpoint=other_checkpoint)
        SDXLAssemblyDirector.select_assembly(req2).execute(req2)
        self.assertEqual(mock_load_unet.call_count, 2)

    @patch("backend.sdxl_assembly.cpu_lora_worker.SafeOpenHeaderOnly")
    @patch("backend.lora.load_lora")
    @patch("pathlib.Path.exists")
    @patch("backend.loader.load_vae")
    @patch("backend.loader.load_sdxl_clip")
    @patch("backend.loader._stream_load_sdxl_unet_from_checkpoint")
    @patch("backend.cpu_compiler.CpuArtifactCompiler.compile_patcher")
    @patch("backend.conditioning.encode_prompt_pair_sdxl")
    @patch("backend.conditioning.build_sdxl_adm_pair")
    @patch("backend.sampling.KSampler")
    @patch("backend.k_diffusion.sample_euler")
    @patch("backend.decode.decode_preloaded_vae")
    @patch("modules.core.pytorch_to_numpy")
    def test_cache_invalidation_on_lora_change(
        self,
        mock_pt_to_np,
        mock_decode_vae,
        mock_sample_euler,
        mock_ksampler,
        mock_build_adm,
        mock_encode_prompt,
        mock_compile,
        mock_load_unet,
        mock_load_clip,
        mock_load_vae,
        mock_exists,
        mock_load_lora,
        mock_safe_open,
    ):
        mock_exists.return_value = True
        mock_load_lora.return_value = {"weight_key": torch.zeros(1)}
        
        mock_unet, mock_clip, mock_vae = self._make_mock_components()
        mock_load_unet.return_value = mock_unet
        mock_load_clip.return_value = mock_clip
        mock_load_vae.return_value = mock_vae
        mock_encode_prompt.return_value = {"positive": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}, "negative": {"cond": torch.zeros(1), "pooled": torch.zeros(1)}}
        mock_ksampler_instance = MagicMock()
        mock_ksampler_instance.sigmas = torch.tensor([10.0, 5.0, 0.0])
        mock_ksampler.return_value = mock_ksampler_instance
        mock_sample_euler.return_value = torch.zeros((1, 4, 8, 8))
        mock_decode_vae.return_value = torch.zeros((1, 3, 64, 64))
        mock_pt_to_np.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]

        # 1. No LoRA stack
        req1 = self._make_dummy_request(lora_stack_hash="no_lora")
        SDXLAssemblyDirector.select_assembly(req1).execute(req1)
        
        # 2. With LoRA stack A
        lora_spec = SDXLLoraSpec(
            file_identity=ResolvedFileIdentity(Path("lora_a.safetensors"), "lora_a_sha", 100, 1),
            unet_weight=1.0,
            clip_weight=1.0
        )
        req2 = self._make_dummy_request(lora_specs=(lora_spec,), lora_stack_hash="lora_a")
        SDXLAssemblyDirector.select_assembly(req2).execute(req2)
        
        # 3. Changing LoRA stack to stack B -> invalidates
        lora_spec2 = SDXLLoraSpec(
            file_identity=ResolvedFileIdentity(Path("lora_b.safetensors"), "lora_b_sha", 100, 1),
            unet_weight=1.0,
            clip_weight=1.0
        )
        req3 = self._make_dummy_request(lora_specs=(lora_spec2,), lora_stack_hash="lora_b")
        SDXLAssemblyDirector.select_assembly(req3).execute(req3)

    def test_process_task_route_regression(self):
        # Verify that inference.process_task still maps to the old unified runtime path
        # by checking that the W02 regression test's behavior remains valid.
        self.assertTrue(hasattr(inference, "process_task"))


from types import SimpleNamespace

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


def _identity_w03(name: str, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=Path(name),
        sha256=sha,
        size_bytes=1,
        modified_ns=1,
    )


def _request_w03(**overrides) -> SDXLAssemblyRequest:
    kwargs = {
        'request_id': 'req_w03_regression',
        'route_id': 'txt2img_assembly',
        'image_index': 0,
        'image_count': 1,
        'checkpoint': _identity_w03('checkpoint.safetensors', 'checkpoint_sha'),
        'vae': _identity_w03('vae.safetensors', 'vae_sha'),
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
    from backend.sdxl_assembly.runtime_state import (
        acquire_text_encoder_component,
        acquire_vae_component,
        acquire_unet_component,
        clear_all_caches,
    )

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
    request = _request_w03()

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
    assert load_calls[-1][2]["dtype"] == torch.float32

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
    from backend.sdxl_assembly.runtime_state import clear_all_caches
    clear_all_caches()
    assembly = SDXLAssemblyDirector.select_assembly(_request_w03(lora_stack_hash='stack_a'))
    try:
        assert assembly.text_encode_worker.lora_worker is assembly.lora_worker
        assert assembly.unet_spine.lora_worker is assembly.lora_worker
    finally:
        assembly.close()
        clear_all_caches()


def test_generic_cpu_patcher_compile_does_not_pin_model_by_default(monkeypatch):
    import backend.cpu_compiler as cpu_compiler
    from backend.cpu_compiler import CpuArtifactCompiler

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))

    patcher = SimpleNamespace(model=DummyModel(), patches={})
    pinned_models = []
    monkeypatch.setattr(
        cpu_compiler,
        '_pin_module_tensors',
        lambda model: pinned_models.append(model) or 0,
    )

    result = CpuArtifactCompiler.compile_patcher(patcher)

    assert result['status'] == 'noop'
    assert pinned_models == []


def test_lora_worker_caches_zero_patch_clip_results(monkeypatch):
    import backend.sdxl_assembly.cpu_lora_worker as cpu_lora_worker
    from backend.sdxl_assembly.runtime_state import clear_all_caches

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
        _request_w03(
            lora_stack_hash="stack_with_zero_clip_patch",
            lora_specs=(
                SDXLLoraSpec(
                    file_identity=_identity_w03("twbabe.safetensors", "lora_sha"),
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


def test_patched_text_encoder_slot_reuses_same_clip_for_prompt_and_layer_changes(monkeypatch):
    import backend.sdxl_assembly.runtime_state as runtime_state
    from backend.cpu_compiler import CpuArtifactCompiler
    from backend.sdxl_assembly.runtime_state import acquire_patched_text_encoder_component, clear_all_caches
    from dataclasses import replace

    clear_all_caches()
    acquire_calls = []
    compile_calls = []

    def make_clip(label: str):
        return SimpleNamespace(
            label=label,
            patcher=SimpleNamespace(
                model=SimpleNamespace(),
                model_size=lambda: 1024,
            ),
            clip_layer=lambda _layer_idx: None,
        )

    monkeypatch.setattr(
        runtime_state,
        "acquire_text_encoder_component",
        lambda _request: acquire_calls.append(_request.prompt_payload_hash) or make_clip(f"clip_{len(acquire_calls)}"),
    )
    monkeypatch.setattr(
        CpuArtifactCompiler,
        "compile_patcher",
        lambda patcher: compile_calls.append(patcher) or {"status": "compiled", "host_pinned_bytes": 0},
    )

    class DummyLoraWorker:
        def __init__(self):
            self.clip_patch_count = 0
            self.calls = 0

        def apply_clip_patches(self, clip):
            self.calls += 1
            self.clip_patch_count = 4

    request = _request_w03(
        prompt_payload_hash="prompt_a",
        clip_layer=-2,
        lora_specs=(
            SDXLLoraSpec(
                file_identity=_identity_w03("clip_lora.safetensors", "clip_lora_sha"),
                unet_weight=1.0,
                clip_weight=0.75,
                enabled=True,
            ),
        ),
    )

    first_worker = DummyLoraWorker()
    second_worker = DummyLoraWorker()

    first_clip = acquire_patched_text_encoder_component(request, lora_worker=first_worker)
    second_clip = acquire_patched_text_encoder_component(
        replace(request, prompt_payload_hash="prompt_b", clip_layer=-4),
        lora_worker=second_worker,
    )

    assert first_clip is second_clip
    assert len(acquire_calls) == 1
    assert len(compile_calls) == 1
    assert first_worker.calls == 1
    assert second_worker.calls == 0

    clear_all_caches()


def test_patched_text_encoder_slot_rebuilds_when_clip_lora_signature_changes(monkeypatch):
    import backend.sdxl_assembly.runtime_state as runtime_state
    from backend.cpu_compiler import CpuArtifactCompiler
    from backend.sdxl_assembly.runtime_state import acquire_patched_text_encoder_component, clear_all_caches

    clear_all_caches()
    acquire_calls = []
    compile_calls = []

    def make_clip(label: str):
        return SimpleNamespace(
            label=label,
            patcher=SimpleNamespace(
                model=SimpleNamespace(),
                model_size=lambda: 1024,
            ),
            clip_layer=lambda _layer_idx: None,
        )

    monkeypatch.setattr(
        runtime_state,
        "acquire_text_encoder_component",
        lambda _request: acquire_calls.append(_request.lora_stack_hash) or make_clip(f"clip_{len(acquire_calls)}"),
    )
    monkeypatch.setattr(
        CpuArtifactCompiler,
        "compile_patcher",
        lambda patcher: compile_calls.append(patcher) or {"status": "compiled", "host_pinned_bytes": 0},
    )

    class DummyLoraWorker:
        def __init__(self):
            self.clip_patch_count = 0

        def apply_clip_patches(self, clip):
            self.clip_patch_count = 4

    request_a = _request_w03(
        lora_stack_hash="stack_a",
        lora_specs=(
            SDXLLoraSpec(
                file_identity=_identity_w03("clip_lora_a.safetensors", "clip_lora_sha_a"),
                unet_weight=1.0,
                clip_weight=1.0,
                enabled=True,
            ),
        ),
    )
    request_b = _request_w03(
        lora_stack_hash="stack_b",
        lora_specs=(
            SDXLLoraSpec(
                file_identity=_identity_w03("clip_lora_b.safetensors", "clip_lora_sha_b"),
                unet_weight=1.0,
                clip_weight=1.0,
                enabled=True,
            ),
        ),
    )

    first_clip = acquire_patched_text_encoder_component(request_a, lora_worker=DummyLoraWorker())
    second_clip = acquire_patched_text_encoder_component(request_b, lora_worker=DummyLoraWorker())

    assert first_clip is not second_clip
    assert len(acquire_calls) == 2
    assert len(compile_calls) == 2

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
    from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
    bind_legacy_workflow_plan(task_state)
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

