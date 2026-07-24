import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataclasses import replace
import torch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
    SpatialContextDescriptor,
    SpatialImageDescriptor,
    SDXLStructuralControlDescriptor,
    SDXLContextualControlDescriptor,
)
from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange, LifecycleDomain
from backend.sdxl_assembly import gateway
from backend import process_transition

class TestSDXLOuterWiringW10c(unittest.TestCase):
    def setUp(self):
        gateway.clear_gateway_state()

        # Patch require_workflow_plan to return a mock FrozenWorkflowPlan
        from modules.pipeline.workflow_contracts import FrozenWorkflowPlan
        self.require_plan_patcher = patch(
            "backend.sdxl_assembly.gateway.require_workflow_plan",
            return_value=MagicMock(spec=FrozenWorkflowPlan)
        )
        self.require_plan_patcher.start()

        self.dummy_file = ResolvedFileIdentity(
            path=Path("dummy_model.safetensors"),
            sha256="dummy_sha256",
            size_bytes=1000,
            modified_ns=0,
        )

        self.dummy_vae = ResolvedFileIdentity(
            path=Path("dummy_vae.safetensors"),
            sha256="vae_sha256",
            size_bytes=500,
            modified_ns=0,
        )

        self.spatial_pixels = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        self.spatial_image = SpatialImageDescriptor(
            fingerprint="img_fingerprint",
            pixels=self.spatial_pixels,
        )

        self.spatial_context = SpatialContextDescriptor(
            mode="inpaint",
            source_image=self.spatial_image,
        )

        self.base_request = SDXLAssemblyRequest(
            request_id="req_base",
            route_id="txt2img_assembly",
            image_index=0,
            image_count=1,
            checkpoint=self.dummy_file,
            vae=self.dummy_vae,
            model_variant_key="sdxl",
            prompt="A beautiful sunset",
            negative_prompt="",
            positive_texts=("A beautiful sunset",),
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
        self.require_plan_patcher.stop()

    def _make_structural_control(self, **overrides):
        payload = {
            "slot_index": 1,
            "control_type": "canny",
            "image_pixels": torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            "image_fingerprint": "struct_fingerprint",
            "preprocessor_id": "canny",
            "preprocessor_path": None,
            "preprocessor_params": {},
            "target_width": 1024,
            "target_height": 1024,
            "checkpoint_path": Path("canny_model.safetensors"),
            "checkpoint_sha256": "canny_sha256",
            "checkpoint_type": "controlnet",
        }
        payload.update(overrides)
        return SDXLStructuralControlDescriptor(**payload)

    def _make_contextual_control(self, **overrides):
        payload = {
            "ui_slot_index": 0,
            "control_type": "ImagePrompt",
            "image_pixels": torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            "image_fingerprint": "ctx_fingerprint",
            "source_image_role": "source",
            "model_path": Path("adapter.safetensors"),
            "model_sha256": "adapter_sha256",
            "clip_vision_path": Path("clip_vision.safetensors"),
            "clip_vision_sha256": "clip_vision_sha256",
            "ip_negative_path": Path("ip_negative.safetensors"),
            "ip_negative_sha256": "ip_negative_sha256",
            "preprocess_params": {},
        }
        payload.update(overrides)
        return SDXLContextualControlDescriptor(**payload)

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_same_family_prompt_change(self, mock_select, mock_build, mock_release):
        mock_build.return_value = self.base_request
        mock_select.return_value = MagicMock()

        # Initial call
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        self.assertIsNotNone(gateway._LAST_REQUEST_STATE)
        mock_release.assert_not_called()

        # Prompt change call
        changed_request = replace(self.base_request, prompt_payload_hash="prompt_hash_changed")
        mock_build.return_value = changed_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_called_once()
        called_args, called_kwargs = mock_release.call_args
        self.assertIn(LifecycleChange.PROMPT_CHANGE, called_args[0])
        self.assertEqual(called_kwargs.get("reason"), "gateway_transition")

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_same_family_lora_change(self, mock_select, mock_build, mock_release):
        mock_build.return_value = self.base_request
        mock_select.return_value = MagicMock()

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        changed_request = replace(self.base_request, lora_stack_hash="lora_hash_changed")
        mock_build.return_value = changed_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_called_once()
        called_args = mock_release.call_args[0][0]
        self.assertIn(LifecycleChange.LORA_STACK_CHANGE, called_args)

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_same_family_checkpoint_change(self, mock_select, mock_build, mock_release):
        mock_build.return_value = self.base_request
        mock_select.return_value = MagicMock()

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        changed_model = ResolvedFileIdentity(
            path=Path("dummy_model.safetensors"),
            sha256="different_sha256",
            size_bytes=1000,
            modified_ns=0,
        )
        changed_request = replace(self.base_request, checkpoint=changed_model)
        mock_build.return_value = changed_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_called_once()
        called_args = mock_release.call_args[0][0]
        self.assertIn(LifecycleChange.CHECKPOINT_CHANGE, called_args)

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_targeted_structural_contextual_spatial_change(self, mock_select, mock_build, mock_release):
        mock_build.return_value = self.base_request
        mock_select.return_value = MagicMock()

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        # Trigger spatial change
        spatial_request = replace(self.base_request, spatial_context=self.spatial_context)
        mock_build.return_value = spatial_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        called_args = mock_release.call_args[0][0]
        self.assertIn(LifecycleChange.SPATIAL_VAE_CHANGE, called_args)

        # Reset gateway and test structural ControlNet change
        mock_release.reset_mock()
        gateway.clear_gateway_state()
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        dummy_structural = self._make_structural_control()
        struct_request = replace(self.base_request, structural_controls=(dummy_structural,))
        mock_build.return_value = struct_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        called_args = mock_release.call_args[0][0]
        self.assertIn(LifecycleChange.STRUCTURAL_CN_CHANGE, called_args)

        # Reset gateway and test contextual ControlNet change
        mock_release.reset_mock()
        gateway.clear_gateway_state()
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        dummy_contextual = self._make_contextual_control()
        contextual_request = replace(self.base_request, contextual_controls=(dummy_contextual,))
        mock_build.return_value = contextual_request

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        called_args = mock_release.call_args[0][0]
        self.assertIn(LifecycleChange.CONTEXTUAL_CN_CHANGE, called_args)

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_equivalent_tensor_backed_spatial_state_does_not_release(self, mock_select, mock_build, mock_release):
        first_context = SpatialContextDescriptor(
            mode="inpaint",
            source_image=SpatialImageDescriptor(
                fingerprint="img_fingerprint",
                pixels=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            ),
        )
        second_context = SpatialContextDescriptor(
            mode="inpaint",
            source_image=SpatialImageDescriptor(
                fingerprint="img_fingerprint",
                pixels=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            ),
        )
        first_request = replace(self.base_request, spatial_context=first_context)
        second_request = replace(self.base_request, spatial_context=second_context)

        mock_select.return_value = MagicMock()
        mock_build.side_effect = [first_request, second_request]

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_not_called()

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_contextual_application_only_changes_do_not_release(self, mock_select, mock_build, mock_release):
        base_contextual = self._make_contextual_control(weight=1.0, start_percent=0.0, end_percent=1.0)
        changed_contextual = self._make_contextual_control(weight=0.5, start_percent=0.25, end_percent=0.9)

        first_request = replace(self.base_request, contextual_controls=(base_contextual,))
        second_request = replace(self.base_request, contextual_controls=(changed_contextual,))

        mock_select.return_value = MagicMock()
        mock_build.side_effect = [first_request, second_request]

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_not_called()

    @patch("backend.sdxl_assembly.gateway.release_for_changes")
    @patch("backend.sdxl_assembly.gateway.build_assembly_request")
    @patch("backend.sdxl_assembly.gateway.SDXLAssemblyDirector.select_assembly")
    def test_gateway_structural_application_only_changes_do_not_release(self, mock_select, mock_build, mock_release):
        base_structural = self._make_structural_control(weight=1.0, start_percent=0.0, end_percent=1.0)
        changed_structural = self._make_structural_control(weight=0.5, start_percent=0.25, end_percent=0.9)

        first_request = replace(self.base_request, structural_controls=(base_structural,))
        second_request = replace(self.base_request, structural_controls=(changed_structural,))

        mock_select.return_value = MagicMock()
        mock_build.side_effect = [first_request, second_request]

        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )
        gateway.run_sdxl_assembly_task(
            MagicMock(), {}, 0, 1, 20, 0, None, "normal", loras=[]
        )

        mock_release.assert_not_called()

    @patch("backend.sdxl_assembly.lifecycle_coordinator.release_domains")
    def test_process_transition_same_family_calls_release_for_changes(self, mock_release_domains):
        current_key = process_transition.build_process_key(
            family=process_transition.PROCESS_FAMILY_SDXL,
            process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity=("model-a.safetensors", "clip-a.safetensors"),
        )
        requested_key = process_transition.build_process_key(
            family=process_transition.PROCESS_FAMILY_SDXL,
            process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity=("model-b.safetensors", "clip-a.safetensors"),
        )

        with patch("backend.resources.prepare_for_checkpoint_switch", lambda **kwargs: kwargs["release_callback"]()):
            process_transition.release_process_boundary(current_key, requested_key)

        self.assertGreater(mock_release_domains.call_count, 0)
        # All calls should specify MODEL_PROMPT
        for call_args_list in mock_release_domains.call_args_list:
            called_domains = call_args_list[0][0]
            self.assertIn(LifecycleDomain.MODEL_PROMPT, called_domains)
            self.assertNotIn(LifecycleDomain.STRUCTURAL_CN, called_domains)
            self.assertNotIn(LifecycleDomain.CONTEXTUAL_CN, called_domains)
            self.assertNotIn(LifecycleDomain.SPATIAL_VAE, called_domains)

    @patch("backend.sdxl_assembly.lifecycle_coordinator.release_domains")
    def test_process_transition_family_switch_calls_full_teardown(self, mock_release_domains):
        current_key = process_transition.build_process_key(
            family=process_transition.PROCESS_FAMILY_SDXL,
            process_class=process_transition.PROCESS_CLASS_STANDARD_SDXL,
            authoritative_identity=("model-a.safetensors", "clip-a.safetensors"),
        )
        requested_key = process_transition.build_process_key(
            family=process_transition.PROCESS_FAMILY_FLUX_FILL,
            process_class=process_transition.PROCESS_CLASS_FLUX_FILL,
            authoritative_identity=(("ae", "ae.safetensors"),),
        )

        with patch("backend.resources.prepare_for_checkpoint_switch", lambda **kwargs: kwargs["release_callback"]()):
            process_transition.release_process_boundary(current_key, requested_key)

        self.assertGreater(mock_release_domains.call_count, 0)
        # At least one call should specify FULL_TEARDOWN
        has_full_teardown = False
        for call_args_list in mock_release_domains.call_args_list:
            called_domains = call_args_list[0][0]
            if LifecycleDomain.FULL_TEARDOWN in called_domains:
                has_full_teardown = True
        self.assertTrue(has_full_teardown)
