import unittest
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    SDXLStructuralControlDescriptor,
    SDXLContextualControlDescriptor,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
)
from backend.sdxl_assembly.assembly import SDXLAssembly
import modules.flags as flags
import modules.model_taxonomy as model_taxonomy

class TestSDXLAssemblyW09(unittest.TestCase):
    def setUp(self):
        # Setup common mock objects using an existing path
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
        # Setup channel lists based on cn_tasks
        state.get_cn_tasks_for_channel = lambda channel: {
            t: tasks for t, tasks in state.cn_tasks.items()
            if flags.get_cn_channel(t) == channel
        }
        return state

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_supported_routes(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # txt2img is eligible
        state = self._make_task_state()
        eligible, reason = determine_eligibility(state)
        self.assertTrue(eligible, f"Expected txt2img to be eligible: {reason}")

        # inpaint is eligible
        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="inpaint",
            inpaint_input_image=np.zeros((64, 64, 3)),
            inpaint_mask_image=np.zeros((64, 64)),
            goals=["inpaint"]
        )
        eligible, reason = determine_eligibility(state)
        self.assertTrue(eligible, f"Expected inpaint to be eligible: {reason}")

        # outpaint is eligible
        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="outpaint",
            outpaint_input_image=np.zeros((64, 64, 3)),
            outpaint_mask_image=np.zeros((64, 64)),
            outpaint_step2_checkbox=True,
            outpaint_bb_image=np.zeros((64, 64, 3)),
            goals=["outpaint"]
        )
        eligible, reason = determine_eligibility(state)
        self.assertTrue(eligible, f"Expected outpaint to be eligible: {reason}")

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_bypasses(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # Tiled refinement -> legacy bypass
        state = self._make_task_state(tiled=True)
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("Tiled refinement", reason)

        # Execution policy is no longer the admission switch.  The explicit
        # assembly posture/workflow plan controls lane selection, so a legacy
        # resident policy field does not reject an otherwise valid request.
        state = self._make_task_state(
            sdxl_execution_policy=types.SimpleNamespace(execution_mode="resident"),
            sdxl_assembly_posture="streaming",
        )
        eligible, reason = determine_eligibility(state)
        self.assertTrue(eligible)
        self.assertIsNone(reason)

        # Non-SDXL architecture -> legacy bypass
        mock_taxonomy.return_value = types.SimpleNamespace(architecture="non_sdxl")
        state = self._make_task_state()
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("is not SDXL", reason)
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # Unsupported route family (e.g. removal) -> legacy bypass
        state = self._make_task_state(
            input_image_checkbox=True,
            current_tab="remove",
            remove_bg_enabled=True,
            goals=["remove_bg"]
        )
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("not eligible", reason)

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_controlnets(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        # Supported ControlNet: PyraCanny
        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(
            state,
            controlnet_paths={flags.cn_canny: "dummy_canny_controlnet.safetensors"},
        )
        self.assertTrue(eligible, f"Expected PyraCanny to be eligible: {reason}")

        # Unsupported ControlNet type
        state = self._make_task_state()
        state.cn_tasks["UnsupportedCN"] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("not supported", reason)

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_determine_eligibility_requires_live_cn_asset_truth(self, mock_exists, mock_taxonomy, mock_get_file):
        mock_exists.return_value = True
        mock_get_file.return_value = "dummy_checkpoint.safetensors"
        mock_taxonomy.return_value = types.SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL)

        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("requires a resolved checkpoint path", reason)

        state = self._make_task_state()
        state.cn_tasks[flags.cn_ip] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 1]]
        eligible, reason = determine_eligibility(
            state,
            contextual_assets={"contextual_model_paths": {}},
        )
        self.assertFalse(eligible)
        self.assertIn("requires a resolved model path", reason)

        state = self._make_task_state()
        state.cn_tasks[flags.cn_depth] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 2]]
        eligible, reason = determine_eligibility(
            state,
            controlnet_paths={flags.cn_depth: "dummy_depth_controlnet.safetensors"},
            image_input_result={"structural_preprocessor_paths": {}},
        )
        self.assertFalse(eligible)
        self.assertIn("requires a resolved preprocessor path", reason)

    def test_determine_eligibility_retirements(self):
        # FaceID V2 retirement
        state = self._make_task_state()
        state.cn_tasks["FaceID V2"] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("FaceID V2 / FaceSwap is explicitly retired", reason)

        # FaceSwap retirement
        state = self._make_task_state()
        state.cn_tasks["FaceSwap"] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("FaceID V2 / FaceSwap is explicitly retired", reason)

        # MLSD retirement
        state = self._make_task_state()
        state.cn_tasks["MLSD"] = [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]]
        eligible, reason = determine_eligibility(state)
        self.assertFalse(eligible)
        self.assertIn("MLSD is explicitly retired", reason)

    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_sparse_slot_continuity_preservation(self, mock_exists, mock_get_file, mock_get_identity):
        mock_exists.return_value = True
        mock_get_file.return_value = str(self.dummy_path)
        mock_get_identity.return_value = self.checkpoint_id

        # Setup sparse active slots (e.g. slot 0 and slot 2 are active, slot 1 is empty)
        # Structural ControlNet
        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [
            [np.zeros((64, 64, 3)), 1.0, 0.8, 0.1, 0], # Slot 0
            [np.zeros((64, 64, 3)), 0.9, 0.7, 0.2, 2], # Slot 2
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

        self.assertEqual(len(req.structural_controls), 2)
        # Verify slot indices are exactly preserved
        self.assertEqual(req.structural_controls[0].slot_index, 0)
        self.assertEqual(req.structural_controls[1].slot_index, 2)
        # Verify weight/start/stop values
        self.assertEqual(req.structural_controls[0].weight, 0.8)
        self.assertEqual(req.structural_controls[0].start_percent, 0.1)
        self.assertEqual(req.structural_controls[0].end_percent, 1.0)
        self.assertEqual(req.structural_controls[1].weight, 0.7)
        self.assertEqual(req.structural_controls[1].start_percent, 0.2)
        self.assertEqual(req.structural_controls[1].end_percent, 0.9)

    @patch("backend.sdxl_assembly.request_builder.get_file_identity")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    @patch("backend.sdxl_assembly.request_builder.os.path.exists")
    def test_build_request_refuses_to_drop_active_controlnet_slots(self, mock_exists, mock_get_file, mock_get_identity):
        mock_exists.return_value = True
        mock_get_file.return_value = str(self.dummy_path)
        mock_get_identity.return_value = self.checkpoint_id

        state = self._make_task_state()
        state.cn_tasks[flags.cn_canny] = [[np.zeros((64, 64, 3)), 1.0, 0.8, 0.1, 0]]

        from backend.sdxl_assembly.contracts import SDXLAssemblyEligibilityError

        with self.assertRaisesRegex(SDXLAssemblyEligibilityError, "checkpoint path is missing"):
            build_assembly_request(
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
                contextual_assets={},
                force_eligible=True,
            )

    def test_assembly_close_keeps_warm_controlnet_support_domains_alive(self):
        class _ControlWorker:
            def __init__(self):
                self.end_calls = 0
                self.release_calls = 0

            def end(self):
                self.end_calls += 1

            def release_owned_resources(self):
                self.release_calls += 1

        st_worker = _ControlWorker()
        ctx_worker = _ControlWorker()
        noop_worker = types.SimpleNamespace(teardown_assembly_order=lambda: None)

        assembly = SDXLAssembly(
            unet_spine=noop_worker,
            text_encode_worker=noop_worker,
            vae_decode_worker=noop_worker,
            lora_worker=noop_worker,
            st_control_worker=st_worker,
            ctx_control_worker=ctx_worker,
        )

        assembly.close()

        self.assertEqual(st_worker.end_calls, 1)
        self.assertEqual(ctx_worker.end_calls, 1)
        self.assertEqual(st_worker.release_calls, 0)
        self.assertEqual(ctx_worker.release_calls, 0)

if __name__ == "__main__":
    unittest.main()
