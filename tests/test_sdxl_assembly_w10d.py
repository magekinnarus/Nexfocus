import unittest
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import replace
import numpy as np
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SDXLStructuralControlDescriptor,
    SDXLContextualControlDescriptor,
)
from backend.sdxl_assembly import gateway
from backend.sdxl_assembly.gateway import (
    _structural_control_signature,
    _contextual_control_signature,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
)
from modules.pipeline.routes import (
    StructuralControlNetStage,
    ContextualControlNetStage,
    PipelineRouteContext,
)
import modules.flags as flags
import modules.model_taxonomy as model_taxonomy
from modules.task_state import TaskState
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan


class TestSDXLAssemblyW10d(unittest.TestCase):
    def setUp(self):
        gateway.clear_gateway_state()
        self.dummy_path = Path(__file__).resolve()
        self.checkpoint_id = ResolvedFileIdentity(
            path=self.dummy_path,
            sha256="fake_checkpoint_sha",
            size_bytes=1000,
            modified_ns=12345
        )
        self.vae_id = ResolvedFileIdentity(
            path=self.dummy_path,
            sha256="fake_vae_sha",
            size_bytes=500,
            modified_ns=67890
        )

        self.base_request = SDXLAssemblyRequest(
            request_id="req_base",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=self.checkpoint_id,
            vae=self.vae_id,
            model_variant_key="sdxl",
            prompt="A sunset",
            negative_prompt="",
            positive_texts=("A sunset",),
            negative_texts=("",),
            width=1024,
            height=1024,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            seed=42,
            clip_layer=-1,
            style_selections=(),
            prompt_payload_hash="prompt_hash_base",
            lora_specs=(),
            lora_stack_hash="lora_hash_base",
            unet_posture=UNetPostureKind.STREAMING,
            clip_posture=TextEncoderPostureKind.CPU_PINNED,
            vae_posture=VAEPostureKind.TRANSIENT,
            lora_posture=LoraPatchPostureKind.STREAMING,
            prefetch_depth=1,
            prefetch_chunk_mb=64,
            device="cpu",
            tiled=False,
            denoise_strength=None,
            sharpness=2.0,
            adaptive_cfg=7.0,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            metadata={},
            spatial_context=None,
            structural_controls=(),
            contextual_controls=(),
        )

    def tearDown(self):
        gateway.clear_gateway_state()

    def _make_task_state(self, **kwargs):
        state = types.SimpleNamespace(
            base_model_name="sdxl_base.safetensors",
            vae_name="Default (model)",
            prompt="prompt",
            negative_prompt="neg",
            width=1024,
            height=1024,
            steps=20,
            cfg_scale=7.0,
            sampler_name="euler",
            clip_skip=1,
            style_selections=[],
            tiled=False,
            goals=["txt2img"],
            initial_latent=None,
            sdxl_execution_policy=types.SimpleNamespace(execution_mode="streaming"),
            cn_tasks={t: [] for t in flags.cn_all_types},
            prepared_structural_cn_tasks={},
            prepared_contextual_cn_tasks={},
            sharpness=2.0,
            adaptive_cfg=7.0,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            input_image_checkbox=False,
            current_tab="generate",
            loras=[]
        )
        for k, v in kwargs.items():
            setattr(state, k, v)
        state.get_cn_tasks_for_channel = lambda channel: {
            t: tasks for t, tasks in state.cn_tasks.items()
            if flags.get_cn_channel(t) == channel
        }
        return state

    def test_gateway_signature_preserves_slot_zero(self):
        # 1. Structural
        struct_desc = SDXLStructuralControlDescriptor(
            slot_index=0,
            control_type="canny",
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fp1",
            preprocessor_id="canny",
            preprocessor_path=None,
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=self.dummy_path,
            checkpoint_sha256="sha1",
            checkpoint_type="controlnet",
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )
        sig = _structural_control_signature(struct_desc)
        # Ensure slot index 0 is preserved as 0, not -1
        self.assertEqual(sig[0], 0)

        # 2. Contextual
        ctx_desc = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fp2",
            source_image_role="face_image",
            model_path=self.dummy_path,
            model_sha256="sha2",
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )
        sig_ctx = _contextual_control_signature(ctx_desc)
        # Ensure ui slot index 0 is preserved as 0, not -1
        self.assertEqual(sig_ctx[0], 0)

    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_build_request_structural_slot_index_fallback(self, mock_exists, mock_get_file, mock_get_identity):
        mock_exists.return_value = True
        mock_get_file.return_value = str(self.dummy_path)
        mock_get_identity.return_value = self.checkpoint_id

        state = self._make_task_state()
        # Task only has 3 elements (no slot_index specified)
        state.cn_tasks[flags.cn_canny] = [
            [np.zeros((64, 64, 3)), 1.0, 0.8]
        ]

        req = build_assembly_request(
            task_state=state,
            task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
            current_task_id=1,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="normal",
            loras=[],
            controlnet_paths={flags.cn_canny: str(self.dummy_path)},
            contextual_assets={}
        )

        self.assertEqual(len(req.structural_controls), 1)
        # Fallback should result in 0-based slot index (i.e. 0 instead of 1)
        self.assertEqual(req.structural_controls[0].slot_index, 0)

    @patch("backend.sdxl_assembly.request_builder.mask_processing.unpack_gradio_data")
    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_build_request_structural_normalizes_filebacked_control_image(
        self,
        mock_exists,
        mock_get_file,
        mock_get_identity,
        mock_unpack,
    ):
        mock_exists.return_value = True
        mock_get_file.return_value = str(self.dummy_path)
        mock_get_identity.return_value = self.checkpoint_id
        mock_unpack.return_value = np.full((32, 32, 3), 255, dtype=np.uint8)

        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [
            ["control.png", 1.0, 0.8, 0.0, 0]
        ]

        req = build_assembly_request(
            task_state=state,
            task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
            current_task_id=1,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="normal",
            loras=[],
            controlnet_paths={flags.cn_canny: str(self.dummy_path)},
            contextual_assets={}
        )

        self.assertEqual(len(req.structural_controls), 1)
        self.assertEqual(req.structural_controls[0].image_pixels.shape, (1, 32, 32, 3))
        self.assertEqual(req.structural_controls[0].slot_index, 0)
        mock_unpack.assert_called_once_with("control.png")

    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_build_request_structural_keeps_base_checkpoint_identity(
        self,
        mock_exists,
        mock_get_file,
        mock_get_identity,
    ):
        base_checkpoint_path = str(self.dummy_path)
        controlnet_path = str((Path(__file__).resolve().parent.parent / "backend" / "sdxl_assembly" / "request_builder.py").resolve())
        base_identity = ResolvedFileIdentity(
            path=Path(base_checkpoint_path),
            sha256="base_sha",
            size_bytes=100,
            modified_ns=1,
        )
        controlnet_identity = ResolvedFileIdentity(
            path=Path(controlnet_path),
            sha256="controlnet_sha",
            size_bytes=200,
            modified_ns=2,
        )

        mock_exists.return_value = True
        mock_get_file.return_value = base_checkpoint_path
        mock_get_identity.side_effect = lambda path: {
            base_checkpoint_path: base_identity,
            controlnet_path: controlnet_identity,
            flags.default_vae: self.vae_id,
        }[str(path)]

        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="inpaint",
            goals=["inpaint", "cn"],
            inpaint_input_image=np.zeros((64, 64, 3), dtype=np.uint8),
            inpaint_mask_image=np.zeros((64, 64), dtype=np.uint8),
        )
        state.cn_tasks[flags.cn_canny] = [
            [np.zeros((64, 64, 3), dtype=np.uint8), 1.0, 0.8, 0.0, 0]
        ]
        state.inpaint_context = types.SimpleNamespace(
            bb=(0, 64, 0, 64),
            target_w=64,
            target_h=64,
            bb_image=np.zeros((64, 64, 3), dtype=np.uint8),
            bb_mask=np.zeros((64, 64), dtype=np.uint8),
            blend_mask=np.zeros((64, 64), dtype=np.uint8),
        )

        req = build_assembly_request(
            task_state=state,
            task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
            current_task_id=1,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=1.0,
            final_scheduler_name="normal",
            loras=[],
            controlnet_paths={flags.cn_canny: controlnet_path},
            contextual_assets={},
        )

        self.assertEqual(req.checkpoint.path, Path(base_checkpoint_path))
        self.assertEqual(req.checkpoint.sha256, "base_sha")
        self.assertEqual(req.structural_controls[0].checkpoint_path, Path(controlnet_path))
        self.assertEqual(req.structural_controls[0].checkpoint_sha256, "controlnet_sha")

    @patch("backend.sdxl_assembly.request_builder.mask_processing.unpack_gradio_data")
    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_build_request_contextual_normalizes_filebacked_control_image(
        self,
        mock_exists,
        mock_get_file,
        mock_get_identity,
        mock_unpack,
    ):
        mock_exists.return_value = True
        mock_get_file.return_value = str(self.dummy_path)
        mock_get_identity.return_value = self.checkpoint_id
        mock_unpack.return_value = np.full((24, 24, 3), 127, dtype=np.uint8)

        state = self._make_task_state(goals=["cn"])
        state.cn_tasks[flags.cn_ip] = [
            ["image_prompt.png", 1.0, 0.75, 0.1, 0]
        ]

        req = build_assembly_request(
            task_state=state,
            task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
            current_task_id=1,
            total_count=1,
            all_steps=20,
            preparation_steps=0,
            denoising_strength=None,
            final_scheduler_name="normal",
            loras=[],
            controlnet_paths={},
            contextual_assets={
                "contextual_model_paths": {flags.cn_ip: str(self.dummy_path)},
                "clip_vision_path": str(self.dummy_path),
                "ip_negative_path": str(self.dummy_path),
            }
        )

        self.assertEqual(len(req.contextual_controls), 1)
        self.assertEqual(req.contextual_controls[0].image_pixels.shape, (1, 24, 24, 3))
        self.assertEqual(req.contextual_controls[0].ui_slot_index, 0)
        mock_unpack.assert_called_once_with("image_prompt.png")

    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    def test_build_request_resolves_default_shared_vae_asset(
        self,
        mock_get_file,
        mock_get_identity,
        mock_taxonomy,
    ):
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_checkpoint_path = Path(tmp_dir) / "sdxl_base.safetensors"
            base_checkpoint_path.write_bytes(b"checkpoint")
            shared_vae_path = Path(tmp_dir) / "vae" / "sdxl" / "sdxl_vae.safetensors"
            shared_vae_path.parent.mkdir(parents=True, exist_ok=True)
            shared_vae_path.write_bytes(b"vae")

            base_identity = ResolvedFileIdentity(
                path=base_checkpoint_path,
                sha256="base_sha",
                size_bytes=base_checkpoint_path.stat().st_size,
                modified_ns=base_checkpoint_path.stat().st_mtime_ns,
            )
            shared_vae_identity = ResolvedFileIdentity(
                path=shared_vae_path,
                sha256="shared_vae_sha",
                size_bytes=shared_vae_path.stat().st_size,
                modified_ns=shared_vae_path.stat().st_mtime_ns,
            )

            mock_get_file.side_effect = lambda name, _folders: {
                "sdxl_base.safetensors": str(base_checkpoint_path),
                flags.default_vae: str(shared_vae_path),
            }[str(name)]
            mock_get_identity.side_effect = lambda path: {
                str(base_checkpoint_path): base_identity,
                str(shared_vae_path): shared_vae_identity,
            }[str(path)]

            req = build_assembly_request(
                task_state=self._make_task_state(vae_name="Default (model)"),
                task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
                current_task_id=1,
                total_count=1,
                all_steps=20,
                preparation_steps=0,
                denoising_strength=None,
                final_scheduler_name="normal",
                loras=[],
                controlnet_paths={},
                contextual_assets={},
            )

            self.assertIsNotNone(req.vae)
            self.assertEqual(req.vae.path, shared_vae_path)
            self.assertEqual(req.vae.sha256, "shared_vae_sha")

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_fails_closed_missing_spatial_assets(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # 1. Inpaint route with missing assets (inpaint_input_image is None)
        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="inpaint",
            inpaint_input_image=None,
            goals=["inpaint"]
        )
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("requires valid inpaint image and mask assets", reason)

        # 2. Outpaint route with missing assets (outpaint_input_image is None)
        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="outpaint",
            outpaint_input_image=None,
            goals=["outpaint"]
        )
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("requires valid outpaint image asset", reason)

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_keeps_frozen_txt2img_snapshot_when_ui_tab_is_stale(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        state = self._make_task_state(
            input_image_checkbox=False,
            current_tab="inpaint",
            inpaint_input_image=np.zeros((64, 64, 3)),
            inpaint_mask_image=None,
            goals=["txt2img"],
            requested_route_id="txt2img",
            requested_route_family="txt2img",
        )

        eligible, reason = determine_eligibility(state)
        self.assertTrue(eligible, f"Expected frozen txt2img snapshot to remain eligible: {reason}")

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_fails_closed_missing_controlnet_input_image(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # ControlNet task is active, but task has None for input image
        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [
            [None, 1.0, 0.8, 0.0, 0]
        ]

        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("missing its input image asset", reason)

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_edits_update_payload_without_teardown(self, mock_select, mock_build, mock_release):
        base_struct = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type="canny",
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fp1",
            preprocessor_id="canny",
            preprocessor_path=None,
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=self.dummy_path,
            checkpoint_sha256="sha1",
            checkpoint_type="controlnet",
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )
        changed_struct = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type="canny",
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fp1",
            preprocessor_id="canny",
            preprocessor_path=None,
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=self.dummy_path,
            checkpoint_sha256="sha1",
            checkpoint_type="controlnet",
            weight=0.5,           # Application-only weight change
            start_percent=0.2,    # Application-only start change
            end_percent=0.8,      # Application-only end change
        )

        first_request = replace(self.base_request, structural_controls=(base_struct,))
        second_request = replace(self.base_request, structural_controls=(changed_struct,))

        mock_select.return_value = MagicMock()
        mock_build.side_effect = [first_request, second_request]
        task_state = TaskState()
        bind_legacy_workflow_plan(task_state)

        # 1st run
        gateway.run_sdxl_assembly_task(
            task_state, {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        # 2nd run with application-only updates
        gateway.run_sdxl_assembly_task(
            task_state, {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        # Assert no releases were triggered
        mock_release.assert_not_called()

    @patch("backend.sdxl_assembly.gateway.is_eligible_for_sdxl_assembly")
    def test_pipeline_stages_bypass_legacy_preprocessing(self, mock_eligible):
        mock_eligible.return_value = (True, None)

        context = MagicMock(spec=PipelineRouteContext)
        context.task_state = self._make_task_state(goals=["cn"])
        context.task_state.cn_tasks[flags.cn_canny] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        context.task_state.cn_tasks[flags.cn_ip] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        context.image_input_result = {
            "controlnet_paths": {},
            "contextual_assets": {},
        }

        # 1. Structural
        stage_struct = StructuralControlNetStage()
        with patch("modules.pipeline.image_input.preprocess_structural_controlnets") as mock_prep:
            res = stage_struct.execute(context)
            self.assertEqual(res.notes.get("status"), "assembly_delegated")
            mock_prep.assert_not_called()

        # 2. Contextual
        stage_ctx = ContextualControlNetStage()
        with patch("modules.pipeline.image_input.preprocess_contextual_controlnets") as mock_prep:
            res = stage_ctx.execute(context)
            self.assertEqual(res.notes.get("status"), "assembly_delegated")
            mock_prep.assert_not_called()
