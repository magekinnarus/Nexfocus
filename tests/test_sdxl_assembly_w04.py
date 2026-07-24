from __future__ import annotations

import unittest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
)
from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
from backend.sdxl_assembly.progress import add_telemetry_sink
from modules.pipeline import inference

class TestSDXLAssemblyW04(unittest.TestCase):
    def setUp(self):
        # Setup dummy task_state and task_dict
        self.task_state = MagicMock()
        self.task_state.base_model_name = "dummy_model"
        self.task_state.vae_name = "Default (model)"
        self.task_state.loras = []
        self.task_state.width = 1024
        self.task_state.height = 1024
        self.task_state.steps = 20
        self.task_state.cfg_scale = 7.0
        self.task_state.sampler_name = "euler"
        self.task_state.scheduler_name = "karras"
        self.task_state.clip_skip = 1
        self.task_state.last_stop = False
        self.task_state.use_expansion = False
        self.task_state.disable_intermediate_results = True
        self.task_state.input_image_checkbox = False
        self.task_state.current_tab = "generate"
        self.task_state.requested_source_surface = ""
        self.task_state.requested_route_id = ""
        self.task_state.sdxl_assembly_posture = "streaming"
        self.task_state.sdxl_execution_policy = None
        from modules.pipeline.workflow_compiler import compile_workflow_plan
        from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
        self.task_state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

        self.task_dict = {
            "task_prompt": "anime girl",
            "task_negative_prompt": "ugly",
            "task_seed": 42,
        }

        self.telemetry_events: list[dict] = []
        self.unregister_sink = add_telemetry_sink(self.telemetry_events.append)

    def tearDown(self):
        self.unregister_sink()

    @patch("backend.sdxl_assembly.gateway.is_eligible_for_sdxl_assembly")
    @patch("backend.sdxl_assembly.gateway.run_sdxl_assembly_task")
    @patch("modules.pipeline.inference._run_unified_sdxl_task")
    @patch("modules.pipeline.inference.save_and_log")
    def test_route_cutover_dispatch(
        self, mock_save_log, mock_run_legacy, mock_run_assembly, mock_eligible
    ):
        mock_eligible.return_value = (True, None)
        mock_run_assembly.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mock_save_log.return_value = ["output_path.png"]

        # 1. Eligible dispatch routes to run_sdxl_assembly_task
        imgs, paths, progress = inference.process_task(
            self.task_state,
            self.task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="karras",
            loras=[],
        )

        mock_run_assembly.assert_called_once()
        mock_run_legacy.assert_not_called()
        self.assertEqual(len(imgs), 1)
        self.assertEqual(paths, ["output_path.png"])

        # Check telemetry
        begin_events = [e for e in self.telemetry_events if e["event"] == "assembly_route_begin"]
        complete_events = [e for e in self.telemetry_events if e["event"] == "assembly_route_complete"]
        self.assertEqual(len(begin_events), 1)
        self.assertEqual(len(complete_events), 1)
        self.assertTrue("proc_rss_mb" in begin_events[0])

        # Reset mocks
        mock_run_assembly.reset_mock()
        mock_run_legacy.reset_mock()
        self.telemetry_events.clear()

        # 2. Ineligible dispatch fails closed when the old shared lane is gutted.
        mock_eligible.return_value = (False, "ControlNet is active")
        mock_run_legacy.return_value = [np.zeros((1024, 1024, 3), dtype=np.uint8)]

        with self.assertRaisesRegex(RuntimeError, "active SDXL execution policy"):
            inference.process_task(
                self.task_state,
                self.task_dict,
                current_task_id=0,
                total_count=1,
                all_steps=20,
                preparation_steps=0,
                denoising_strength=None,
                final_scheduler_name="karras",
                loras=[],
            )

        mock_run_assembly.assert_not_called()
        mock_run_legacy.assert_not_called()

        # Check bypass telemetry
        bypass_events = [e for e in self.telemetry_events if e["event"] == "assembly_route_legacy_bypass"]
        self.assertEqual(len(bypass_events), 1)
        self.assertEqual(bypass_events[0]["extra"], "reason=ControlNet is active")

    @patch("backend.sdxl_assembly.gateway.is_eligible_for_sdxl_assembly")
    @patch("backend.sdxl_assembly.gateway.run_sdxl_assembly_task")
    @patch("modules.pipeline.inference._run_unified_sdxl_task")
    def test_assembly_failure_no_fallback(self, mock_run_legacy, mock_run_assembly, mock_eligible):
        mock_eligible.return_value = (True, None)
        mock_run_assembly.side_effect = RuntimeError("Worker execution failed: GPU OOM")

        with self.assertRaises(RuntimeError) as context:
            inference.process_task(
                self.task_state,
                self.task_dict,
                current_task_id=0,
                total_count=1,
                all_steps=20,
                preparation_steps=0,
                denoising_strength=None,
                final_scheduler_name="karras",
                loras=[],
            )

        self.assertIn("Worker execution failed: GPU OOM", str(context.exception))
        mock_run_legacy.assert_not_called()

        # Check telemetry for route failure
        fail_events = [e for e in self.telemetry_events if e["event"] == "assembly_route_failure"]
        self.assertEqual(len(fail_events), 1)
        self.assertIn("GPU OOM", fail_events[0]["extra"])

    @patch("pathlib.Path.exists")
    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.determine_eligibility")
    def test_request_snapshot_immutability(self, mock_eligible, mock_find_file, mock_identity, mock_exists):
        from backend.sdxl_assembly.request_builder import build_assembly_request

        mock_exists.return_value = True
        mock_eligible.return_value = (True, None)
        mock_find_file.side_effect = lambda name, list_dir: f"D:\\fake\\path\\{name}.safetensors"
        
        dummy_id = ResolvedFileIdentity(Path("D:\\fake\\path\\dummy_model.safetensors"), "fake_sha", 100, 1)
        mock_identity.return_value = dummy_id

        # Build request
        req = build_assembly_request(
            self.task_state,
            self.task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="karras",
            loras=[],
        )

        # Confirm request fields captured correctly
        self.assertEqual(req.prompt, "anime girl")
        self.assertEqual(req.width, 1024)

        # Mutate the source dictionary and state
        self.task_dict["task_prompt"] = "different prompt"
        self.task_state.width = 512

        # Verify frozen request remained unchanged
        self.assertEqual(req.prompt, "anime girl")
        self.assertEqual(req.width, 1024)

    @patch("pathlib.Path.exists")
    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.determine_eligibility")
    def test_user_streaming_settings_propagation(self, mock_eligible, mock_find_file, mock_identity, mock_exists):
        from backend.sdxl_assembly.request_builder import build_assembly_request

        mock_exists.return_value = True
        mock_eligible.return_value = (True, None)
        mock_find_file.side_effect = lambda name, list_dir: f"D:\\fake\\path\\{name}.safetensors"
        mock_identity.return_value = ResolvedFileIdentity(Path("D:\\fake\\path\\dummy_model.safetensors"), "fake_sha", 100, 1)

        # 1. Test defaults
        self.task_state.sdxl_execution_policy = None
        req = build_assembly_request(
            self.task_state,
            self.task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="karras",
            loras=[],
        )
        self.assertEqual(req.prefetch_chunk_mb, 64)
        self.assertEqual(req.prefetch_depth, 1)
        self.assertFalse(req.metadata.get("pin_unet_host"))
        self.assertFalse(req.metadata.get("release_warm_unet_after_task"))
        self.assertFalse(req.metadata.get("release_text_encoder_after_task"))

        # 2. Test explicit user settings
        class DummyPolicy:
            execution_mode = "streaming"
            prefetch_depth = 2
            prefetch_chunk_mb = 128
            pin_unet_host = True
            release_warm_unet_after_task = True
            release_text_encoder_after_task = True

        self.task_state.sdxl_execution_policy = DummyPolicy()
        req2 = build_assembly_request(
            self.task_state,
            self.task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="karras",
            loras=[],
        )
        self.assertEqual(req2.prefetch_chunk_mb, 128)
        self.assertEqual(req2.prefetch_depth, 2)
        self.assertTrue(req2.metadata.get("pin_unet_host"))
        self.assertTrue(req2.metadata.get("release_warm_unet_after_task"))
        self.assertTrue(req2.metadata.get("release_text_encoder_after_task"))


def test_build_assembly_request_uses_task_state_streaming_settings_and_processed_loras(tmp_path, monkeypatch):
    from backend.sdxl_assembly.request_builder import build_assembly_request
    from backend.sdxl_assembly.contracts import SDXLLoraSpec
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
    import modules.model_taxonomy as model_taxonomy

    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')
    inline_lora_path = tmp_path / 'inline_lora.safetensors'
    inline_lora_path.write_bytes(b'lora')
    shared_vae_path = tmp_path / 'sdxl' / 'sdxl_vae.safetensors'
    shared_vae_path.parent.mkdir(parents=True, exist_ok=True)
    shared_vae_path.write_bytes(b'vae')

    def fake_resolve(name, _folders):
        lookup = {
            'sdxl_base.safetensors': checkpoint_path,
            'inline_lora.safetensors': inline_lora_path,
            'sdxl_vae.safetensors': shared_vae_path,
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
        sdxl_execution_policy=SimpleNamespace(
            execution_mode='streaming',
            prefetch_depth=2,
            prefetch_chunk_mb=128,
            pin_unet_host=True,
            release_warm_unet_after_task=True,
            release_text_encoder_after_task=True,
        ),
        sharpness=2.0,
        adaptive_cfg=7.0,
        adm_scaler_positive=1.5,
        adm_scaler_negative=0.8,
        adm_scaler_end=0.3,
        prefetch_depth=2,
        prefetch_chunk_mb=128,
        use_expansion=False,
        disable_intermediate_results=True,
        loras_processed=[('inline_lora.safetensors', 0.7)],
    )
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    request = build_assembly_request(
        task_state=state,
        task_dict=task_dict,
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


def test_build_assembly_request_resolves_default_shared_vae_path(tmp_path, monkeypatch):
    from backend.sdxl_assembly.request_builder import build_assembly_request
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
    import modules.model_taxonomy as model_taxonomy

    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')
    shared_vae_path = tmp_path / 'vae' / 'sdxl' / 'sdxl_vae.safetensors'
    shared_vae_path.parent.mkdir(parents=True, exist_ok=True)
    shared_vae_path.write_bytes(b'vae')

    def fake_resolve(name, _folders):
        lookup = {
            'sdxl_base.safetensors': checkpoint_path,
            'sdxl_vae.safetensors': shared_vae_path,
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
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    request = build_assembly_request(
        task_state=state,
        task_dict=task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
    )

    assert request.vae is not None
    assert request.vae.path == shared_vae_path
    assert request.vae.sha256


def test_run_sdxl_assembly_task_preserves_interrupts(monkeypatch):
    from backend.sdxl_assembly import gateway
    from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
    from backend import resources
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
    import pytest

    request = SimpleNamespace(
        checkpoint=SimpleNamespace(sha256='checkpoint_sha'),
        vae=None,
        unet_posture=SimpleNamespace(value='streaming'),
        clip_posture=SimpleNamespace(value='cpu_pinned'),
        vae_posture=SimpleNamespace(value='transient'),
        lora_posture=SimpleNamespace(value='streaming'),
        lora_stack_hash='lora_hash',
        prompt_payload_hash='prompt_hash',
        spatial_context=None,
        structural_controls=(),
        contextual_controls=(),
        lora_specs=(),
    )

    close_calls = []

    class FakeAssembly:
        def execute(self, _request, callback=None):
            raise resources.InterruptProcessingException()

        def close(self):
            close_calls.append('closed')

    monkeypatch.setattr(gateway, 'build_assembly_request', lambda *args, **kwargs: request)
    monkeypatch.setattr(gateway.SDXLAssemblyDirector, 'select_assembly', lambda _request, **_kwargs: FakeAssembly())

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
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    with pytest.raises(resources.InterruptProcessingException):
        run_sdxl_assembly_task(
            state,
            task_dict,
            current_task_id=0,
            total_count=1,
            all_steps=3,
            preparation_steps=0,
            denoising_strength=1.0,
            final_scheduler_name='karras',
            loras=[],
        )

    assert close_calls == ['closed']


def test_run_sdxl_assembly_task_logs_additional_unet_only_loras(tmp_path, monkeypatch, capsys, caplog):
    from backend.sdxl_assembly import gateway
    from backend.sdxl_assembly.gateway import run_sdxl_assembly_task
    from backend.sdxl_assembly.contracts import SDXLLoraSpec, ResolvedFileIdentity
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
    from pathlib import Path

    checkpoint_path = tmp_path / 'checkpoint.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')
    user_lora_path = tmp_path / 'user_lora.safetensors'
    user_lora_path.write_bytes(b'user-lora')
    patch_lora_path = tmp_path / 'inpaint_v26.fooocus.patch'
    patch_lora_path.write_bytes(b'patch-lora')

    def _identity(path: Path, sha: str) -> ResolvedFileIdentity:
        return ResolvedFileIdentity(
            path=path,
            sha256=sha,
            size_bytes=path.stat().st_size,
            modified_ns=path.stat().st_mtime_ns,
        )

    request = SimpleNamespace(
        request_id='req_lora_admission',
        route_id='inpaint_assembly',
        seed=123,
        width=64,
        height=64,
        steps=3,
        checkpoint=_identity(checkpoint_path, 'checkpoint_sha'),
        vae=None,
        unet_posture=SimpleNamespace(value='resident'),
        clip_posture=SimpleNamespace(value='gpu_pinned'),
        vae_posture=SimpleNamespace(value='transient'),
        lora_posture=SimpleNamespace(value='resident'),
        lora_stack_hash='lora_hash',
        prompt_payload_hash='prompt_hash',
        spatial_context=None,
        structural_controls=(),
        contextual_controls=(),
        lora_specs=(
            SDXLLoraSpec(file_identity=_identity(user_lora_path, 'user_sha'), unet_weight=0.7, clip_weight=0.7),
            SDXLLoraSpec(file_identity=_identity(patch_lora_path, 'patch_sha'), unet_weight=1.0, clip_weight=0.0, provenance="additional"),
        ),
    )

    class FakeAssembly:
        def execute(self, _request, callback=None):
            return SimpleNamespace(output_image=np.zeros((1, 1, 3), dtype=np.uint8))

        def close(self):
            return None

    monkeypatch.setattr(gateway, '_LAST_REQUEST_STATE', None)
    monkeypatch.setattr(gateway, 'build_assembly_request', lambda *args, **kwargs: request)
    monkeypatch.setattr(gateway.SDXLAssemblyDirector, 'select_assembly', lambda _request, **_kwargs: FakeAssembly())

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
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    caplog.set_level(logging.DEBUG)
    output = run_sdxl_assembly_task(
        state,
        task_dict,
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
    )

    captured = capsys.readouterr().out
    assert output.shape == (1, 1, 3)
    assert '[SDXL LORA ADMISSION]' not in captured
    assert '[SDXL LORA ADMISSION]' in caplog.text
    assert 'Additional UNet-only LoRAs (1)' in caplog.text
    assert 'inpaint_v26.fooocus.patch@1' in caplog.text


def test_sdxl_assembly_execute_preserves_interrupts(tmp_path):
    from backend.sdxl_assembly.assembly import SDXLAssembly
    from backend.sdxl_assembly.contracts import SDXLAssemblyRequest, ResolvedFileIdentity
    from backend import resources
    from types import SimpleNamespace
    from pathlib import Path
    import pytest

    checkpoint_path = tmp_path / 'checkpoint.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    def _identity(path: Path, sha: str) -> ResolvedFileIdentity:
        return ResolvedFileIdentity(
            path=path,
            sha256=sha,
            size_bytes=path.stat().st_size,
            modified_ns=path.stat().st_mtime_ns,
        )

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
    text_encode_worker = SimpleNamespace(get_conditioning=lambda **_kwargs: {}, teardown_assembly_order=lambda: None)
    vae_decode_worker = SimpleNamespace(
        prepare_latents=lambda _device: SimpleNamespace(samples=torch.zeros((1, 4, 8, 8))),
        teardown_assembly_order=lambda: None,
    )
    unet_spine = SimpleNamespace(
        start=lambda **_kwargs: None,
        denoise=lambda latent, *args, **kwargs: (_ for _ in ()).throw(resources.InterruptProcessingException()),
        end=lambda: None,
        teardown_assembly_order=lambda: None,
    )

    assembly = SDXLAssembly(
        unet_spine=unet_spine,
        text_encode_worker=text_encode_worker,
        vae_decode_worker=vae_decode_worker,
        lora_worker=lora_worker,
    )

    with pytest.raises(resources.InterruptProcessingException):
        assembly.execute(request)


def test_process_task_propagates_interrupt_processing_exception(monkeypatch):
    from backend.sdxl_assembly import gateway
    from backend import resources
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
    import pytest

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
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    with pytest.raises(resources.InterruptProcessingException):
        inference.process_task(
            task_state=state,
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

    assert save_calls == []


def test_process_task_assembly_route_emits_preview_image(monkeypatch):
    from backend.sdxl_assembly import gateway
    from types import SimpleNamespace
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_contracts import FrozenWorkflowSelection

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
        disable_preview=False,
        yields=[],
    )
    state.workflow_plan = compile_workflow_plan(FrozenWorkflowSelection("normal_generate"))

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
            vae_decode_worker=SimpleNamespace(vae=None),
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

    task_dict = {
        'task_seed': 123,
        'task_prompt': 'prompt',
        'task_negative_prompt': 'negative',
        'positive': ['prompt'],
        'negative': ['negative'],
    }

    imgs, paths, progress = inference.process_task(
        task_state=state,
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

    assert imgs[0].shape == (1, 1, 3)
    assert paths == ['saved-path']
    assert any(
        item[0] == 'preview' and isinstance(item[1][2], np.ndarray)
        for item in state.yields
    )


def test_assembly_progress_callback_preserves_interrupt_processing_exception():
    from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback
    from backend import resources
    from types import SimpleNamespace
    import pytest

    callback = SDXLAssemblyProgressCallback(
        SimpleNamespace(),
        lambda *args, **kwargs: (_ for _ in ()).throw(resources.InterruptProcessingException()),
    )

    with pytest.raises(resources.InterruptProcessingException):
        callback(0, None, None, 4, None)


def test_assembly_progress_callback_throttles_full_memory_telemetry(monkeypatch):
    import backend.sdxl_assembly.progress as progress
    from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback
    from types import SimpleNamespace

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
