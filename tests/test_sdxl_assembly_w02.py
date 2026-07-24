from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import modules.model_taxonomy as model_taxonomy

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLAssemblyResult,
    ResolvedFileIdentity,
    SDXLLoraSpec,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SDXLAssemblyEligibilityError,
    SDXLAssemblyValidationError,
)
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
from backend.sdxl_assembly.request_builder import determine_eligibility, build_assembly_request
from modules.pipeline.workflow_compiler import compile_workflow_plan
from modules.pipeline.workflow_contracts import FrozenWorkflowSelection

class TestSDXLAssemblyW02Contracts(unittest.TestCase):
    def test_resolved_file_identity(self):
        identity = ResolvedFileIdentity(
            path=Path("dummy_checkpoint.safetensors"),
            sha256="fake_sha256",
            size_bytes=1024,
            modified_ns=12345
        )
        self.assertEqual(identity.sha256, "fake_sha256")
        self.assertEqual(identity.size_bytes, 1024)

    @patch("pathlib.Path.exists")
    def test_request_validation(self, mock_exists):
        mock_exists.return_value = True
        checkpoint = ResolvedFileIdentity(
            path=Path("dummy_checkpoint.safetensors"),
            sha256="fake_sha256",
            size_bytes=1024,
            modified_ns=12345
        )
        req = SDXLAssemblyRequest(
            request_id="req_1",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="A photo of a cat",
            negative_prompt="",
            positive_texts=("A photo of a cat",),
            negative_texts=("",),
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=42,
        )
        # Should validate successfully
        req.validate()

    @patch("pathlib.Path.exists")
    def test_request_validation_failures(self, mock_exists):
        mock_exists.return_value = False
        checkpoint = ResolvedFileIdentity(
            path=Path("dummy_checkpoint.safetensors"),
            sha256="fake_sha256",
            size_bytes=1024,
            modified_ns=12345
        )
        # Checkpoint path doesn't exist
        req = SDXLAssemblyRequest(
            request_id="req_1",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="A photo of a cat",
            negative_prompt="",
            positive_texts=("A photo of a cat",),
            negative_texts=("",),
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=42,
        )
        with self.assertRaises(SDXLAssemblyValidationError):
            req.validate()

        # Step validation failure
        mock_exists.return_value = True
        req_bad_steps = SDXLAssemblyRequest(
            request_id="req_1",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="A photo of a cat",
            negative_prompt="",
            positive_texts=("A photo of a cat",),
            negative_texts=("",),
            width=512,
            height=512,
            steps=0, # Invalid
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=42,
        )
        with self.assertRaises(SDXLAssemblyValidationError):
            req_bad_steps.validate()

class TestSDXLAssemblyW02Eligibility(unittest.TestCase):
    def _make_mock_task_state(self):
        task_state = MagicMock()
        task_state.base_model_name = "sdxl_base.safetensors"
        task_state.goals = ["txt2img"]
        task_state.tiled = False
        task_state.prepared_structural_cn_tasks = {}
        task_state.prepared_contextual_cn_tasks = {}
        task_state.initial_latent = None
        task_state.input_image_checkbox = False
        task_state.current_tab = "generate"
        task_state.requested_source_surface = ""
        task_state.requested_route_id = ""
        task_state.sdxl_assembly_posture = "streaming"
        task_state.sdxl_execution_policy = None
        task_state.objr_engine = None
        task_state.uov_method = ""
        task_state.inpaint_route = "sdxl"
        task_state.remove_bg_enabled = False
        task_state.remove_obj_enabled = False
        task_state.mixing_image_prompt_and_inpaint = False
        task_state.mixing_image_prompt_and_outpaint = False
        task_state.workflow_plan = compile_workflow_plan(
            FrozenWorkflowSelection(source_surface="normal_generate")
        )
        return task_state

    def test_eligible_plain_txt2img(self):
        task_state = self._make_mock_task_state()

        with patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as mock_taxonomy, \
             patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list") as mock_get_file, \
             patch("os.path.exists") as mock_exists:
            mock_taxonomy.return_value = MagicMock(architecture=model_taxonomy.ARCHITECTURE_SDXL)
            mock_get_file.return_value = "sdxl_base.safetensors"
            mock_exists.return_value = True

            eligible, reason = determine_eligibility(task_state)
            self.assertTrue(eligible)
            self.assertIsNone(reason)

    def test_frozen_plan_ignores_late_goal_mutation(self):
        task_state = self._make_mock_task_state()
        task_state.goals = ["inpaint"]

        with patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as mock_taxonomy, \
             patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list") as mock_get_file, \
             patch("os.path.exists", return_value=True):
            mock_taxonomy.return_value = MagicMock(architecture=model_taxonomy.ARCHITECTURE_SDXL)
            mock_get_file.return_value = "sdxl_base.safetensors"

            eligible, reason = determine_eligibility(task_state)
            self.assertTrue(eligible)
            self.assertIsNone(reason)

    def test_unplanned_controlnet_paths_do_not_override_frozen_plan(self):
        task_state = self._make_mock_task_state()

        with patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as mock_taxonomy, \
             patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list") as mock_get_file, \
             patch("os.path.exists", return_value=True):
            mock_taxonomy.return_value = MagicMock(architecture=model_taxonomy.ARCHITECTURE_SDXL)
            mock_get_file.return_value = "sdxl_base.safetensors"

            eligible, reason = determine_eligibility(
                task_state,
                controlnet_paths={"canny": "canny_path"},
            )
            self.assertTrue(eligible)
            self.assertIsNone(reason)

    def test_ineligible_due_to_tiled(self):
        task_state = self._make_mock_task_state()
        task_state.tiled = True
        eligible, reason = determine_eligibility(task_state)
        self.assertFalse(eligible)
        self.assertIn("tiled", reason.lower())

class TestSDXLAssemblyDirector(unittest.TestCase):
    def test_unsupported_unet_posture_fails(self):
        checkpoint = ResolvedFileIdentity(
            path=Path("dummy_checkpoint.safetensors"),
            sha256="fake_sha256",
            size_bytes=1024,
            modified_ns=12345
        )
        req = SDXLAssemblyRequest(
            request_id="req_1",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="A photo of a cat",
            negative_prompt="",
            positive_texts=("A photo of a cat",),
            negative_texts=("",),
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=42,
            unet_posture=UNetPostureKind.RESIDENT, # Unsupported
        )
        with self.assertRaises(NotImplementedError):
            SDXLAssemblyDirector.select_assembly(req)

class TestSDXLAssemblyImportBoundary(unittest.TestCase):
    def test_import_boundary_hygiene(self):
        # Scan files in backend/sdxl_assembly and check their imports
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
        
        # Ensure we scan __init__.py and other files
        self.assertTrue(package_dir.exists())
        
        for py_file in package_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                # Basic check for direct import/from patterns in python files
                if f"import {forbidden}" in content or f"from {forbidden}" in content:
                    self.fail(f"Forbidden import '{forbidden}' found in {py_file.name}")
                    
                # Check for relative imports or aliased imports if any
                short_name = forbidden.split(".")[-1]
                if f"from backend import {short_name}" in content:
                    self.fail(f"Forbidden import '{forbidden}' found in {py_file.name}")


def test_build_assembly_request_smoke_resolves_frozen_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
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

    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_legacy_adapter import capture_workflow_selection
    def _bind_plan(state):
        state.workflow_plan = compile_workflow_plan(capture_workflow_selection(state))
        return state

    def _task_state(**overrides):
        import modules.flags as flags
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
        return _bind_plan(state)

    def _task_dict():
        return {
            'task_seed': 123,
            'task_prompt': 'prompt',
            'task_negative_prompt': 'negative',
            'positive': ['prompt'],
            'negative': ['negative'],
        }

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
    from types import SimpleNamespace
    from backend.sdxl_assembly import gateway
    from modules.pipeline.workflow_compiler import compile_workflow_plan
    from modules.pipeline.workflow_legacy_adapter import capture_workflow_selection
    from modules.pipeline import inference
    def _bind_plan(state):
        state.workflow_plan = compile_workflow_plan(capture_workflow_selection(state))
        return state

    def _task_state(**overrides):
        import modules.flags as flags
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
        return _bind_plan(state)

    def _task_dict():
        return {
            'task_seed': 123,
            'task_prompt': 'prompt',
            'task_negative_prompt': 'negative',
            'positive': ['prompt'],
            'negative': ['negative'],
        }

    assembly_calls = []

    monkeypatch.setattr(
        gateway,
        'is_eligible_for_sdxl_assembly',
        lambda **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        gateway,
        'run_sdxl_assembly_task',
        lambda *args, **kwargs: assembly_calls.append((args, kwargs)) or np.zeros((64, 64, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(inference, '_ensure_supported_unified_runtime_request', lambda _state: None)
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

    assert len(assembly_calls) == 1
    assert len(imgs) == 1
    assert img_paths == ['saved-path']
    assert current_progress == 100
