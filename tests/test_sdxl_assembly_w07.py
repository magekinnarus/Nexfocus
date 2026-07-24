import unittest
import tempfile
import types
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    SDXLStructuralControlDescriptor,
    StructuralHintArtifact,
    make_spatial_image_descriptor,
    SDXLAssemblyValidationError,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
    determine_controlnet_type,
)
from backend.sdxl_assembly.stream_st_preprocess_worker import (
    StreamingStructuralPreprocessWorker,
    get_preprocess_cache_key,
)
from backend.sdxl_assembly.stream_st_cn_worker import StreamingStructuralControlWorker
from backend.sdxl_assembly.assembly import SDXLAssembly
from backend.sdxl_assembly.runtime_state import clear_all_caches
import modules.flags as flags
import modules.model_taxonomy as model_taxonomy

class TestSDXLAssemblyW07(unittest.TestCase):
    def setUp(self):
        clear_all_caches()

    def test_structural_descriptors_immutability_and_fingerprints(self):
        # Verify immutability and stable fingerprinting
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)
        
        control_desc = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={"low_threshold": 64, "high_threshold": 128},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("canny_model.safetensors"),
            checkpoint_sha256="fake_sha",
            checkpoint_type="controlnet",
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )
        
        # Verify frozen attribute access and mutation safety
        with self.assertRaises(Exception):
            control_desc.weight = 0.5

        key1 = get_preprocess_cache_key(control_desc)
        
        # Changing resolution invalidates cache key
        control_desc_diff_res = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={"low_threshold": 64, "high_threshold": 128},
            target_width=512,
            target_height=512,
            checkpoint_path=Path("canny_model.safetensors"),
            checkpoint_sha256="fake_sha",
            checkpoint_type="controlnet",
        )
        key2 = get_preprocess_cache_key(control_desc_diff_res)
        self.assertNotEqual(key1, key2)

        # Changing checkpoint does NOT invalidate preprocess cache key (model independent)
        control_desc_diff_ckpt = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={"low_threshold": 64, "high_threshold": 128},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("different_canny_model.safetensors"),
            checkpoint_sha256="different_fake_sha",
            checkpoint_type="controlnet",
        )
        key3 = get_preprocess_cache_key(control_desc_diff_ckpt)
        self.assertEqual(key1, key3)

        depth_desc = SDXLStructuralControlDescriptor(
            slot_index=2,
            control_type=flags.cn_depth,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_depth,
            preprocessor_path=Path("depth_a.pt"),
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("depth_model.safetensors"),
            checkpoint_sha256="fake_depth_sha",
            checkpoint_type="controlnet",
        )
        depth_desc_diff_path = SDXLStructuralControlDescriptor(
            slot_index=2,
            control_type=flags.cn_depth,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_depth,
            preprocessor_path=Path("depth_b.pt"),
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("depth_model.safetensors"),
            checkpoint_sha256="fake_depth_sha",
            checkpoint_type="controlnet",
        )
        self.assertNotEqual(get_preprocess_cache_key(depth_desc), get_preprocess_cache_key(depth_desc_diff_path))

    @patch("extras.preprocessors.canny_pyramid")
    def test_preprocess_cache_invalidation_and_reuse(self, mock_canny):
        mock_canny.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)
        
        control_desc = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={"low_threshold": 64, "high_threshold": 128},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("canny_model.safetensors"),
            checkpoint_sha256="fake_sha",
            checkpoint_type="controlnet",
        )
        
        req = MagicMock(spec=SDXLAssemblyRequest)
        req.structural_controls = (control_desc,)
        
        worker = StreamingStructuralPreprocessWorker(req)
        
        # First call: cache miss
        hints1 = worker.preprocess()
        self.assertIn(1, hints1)
        self.assertFalse(hints1[1].cache_hit)
        self.assertEqual(mock_canny.call_count, 1)
        
        # Second call: cache hit
        hints2 = worker.preprocess()
        self.assertTrue(hints2[1].cache_hit)
        self.assertEqual(mock_canny.call_count, 1)  # Canny preprocessor not called again

    @patch("backend.controlnet.load_controlnet")
    def test_control_worker_lifecycle_and_teardown(self, mock_load):
        # Setup mock control model
        mock_model = MagicMock()
        mock_copied_model = MagicMock()
        mock_model.copy.return_value = mock_copied_model
        mock_load.return_value = mock_model
        
        desc = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fake_img_fp",
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("canny_model.safetensors"),
            checkpoint_sha256="fake_sha",
            checkpoint_type="controlnet",
        )
        
        req = MagicMock(spec=SDXLAssemblyRequest)
        req.structural_controls = (desc,)
        
        worker = StreamingStructuralControlWorker(req)
        
        # Verify checkpoint loading and CPU offloading
        hint_artifact = StructuralHintArtifact(
            slot_index=1,
            control_type=flags.cn_canny,
            hint_tensor=torch.zeros((1, 3, 1024, 1024)),
            hint_fingerprint="hint_fp",
        )
        prepared_hints = {1: hint_artifact}
        
        cond = {"positive": [[torch.zeros(1), {"cross_attn": torch.zeros(1)}]], "negative": []}
        
        attached_cond = worker.attach_conditioning(cond, prepared_hints)
        
        # Check copy-on-request
        self.assertEqual(mock_model.copy.call_count, 1)
        # Check setting hint, weight, and ranges
        mock_copied_model.set_cond_hint.assert_called_once()
        
        # Check chaining logic
        self.assertIn("control", attached_cond["positive"][0][1])
        self.assertEqual(attached_cond["positive"][0][1]["control"], mock_copied_model)
        
        # Check end cleans up run-bound states
        worker.end()
        mock_copied_model.cleanup.assert_called_once()
        self.assertEqual(len(worker.active_copied_controls), 0)
        
        # Check release_owned_resources clears everything
        worker.release_owned_resources()
        mock_model.cleanup.assert_called_once()
        self.assertEqual(len(worker._SUPPORT_MODEL_CACHE), 0)

    @patch("backend.controlnet.load_controlnet")
    def test_control_worker_creates_distinct_copies_per_previous_chain(self, mock_load):
        base_model = MagicMock()
        copy_a = MagicMock()
        copy_b = MagicMock()
        base_model.copy.side_effect = [copy_a, copy_b]
        mock_load.return_value = base_model

        desc = SDXLStructuralControlDescriptor(
            slot_index=1,
            control_type=flags.cn_canny,
            image_pixels=torch.zeros((1, 64, 64, 3)),
            image_fingerprint="fake_img_fp",
            preprocessor_id=flags.cn_canny,
            preprocessor_path=None,
            preprocessor_params={},
            target_width=1024,
            target_height=1024,
            checkpoint_path=Path("canny_model.safetensors"),
            checkpoint_sha256="fake_sha",
            checkpoint_type="controlnet",
        )

        req = MagicMock(spec=SDXLAssemblyRequest)
        req.structural_controls = (desc,)

        worker = StreamingStructuralControlWorker(req)
        prepared_hints = {
            1: StructuralHintArtifact(
                slot_index=1,
                control_type=flags.cn_canny,
                hint_tensor=torch.zeros((1, 3, 1024, 1024)),
                hint_fingerprint="hint_fp",
            )
        }
        prev_a = object()
        prev_b = object()
        cond = {
            "positive": [
                [torch.zeros(1), {"cross_attn": torch.zeros(1), "control": prev_a}],
                [torch.zeros(1), {"cross_attn": torch.zeros(1), "control": prev_b}],
            ],
            "negative": [],
        }

        attached = worker.attach_conditioning(cond, prepared_hints)

        self.assertIs(attached["positive"][0][1]["control"], copy_a)
        self.assertIs(attached["positive"][1][1]["control"], copy_b)
        self.assertIsNot(attached["positive"][0][1]["control"], attached["positive"][1][1]["control"])
        copy_a.set_previous_controlnet.assert_called_once_with(prev_a)
        copy_b.set_previous_controlnet.assert_called_once_with(prev_b)

    @patch("backend.utils.load_torch_file")
    def test_control_lora_separation(self, mock_load):
        # 1. Test ControlNet Classification
        mock_load.return_value = {"zero_convs.0.0.weight": torch.zeros(1)}
        self.assertEqual(determine_controlnet_type("cn.safetensors"), "controlnet")
        
        # 2. Test ControlLora Classification
        mock_load.return_value = {"lora_controlnet": torch.zeros(1)}
        # Bust cache
        global _CONTROLNET_TYPE_CACHE
        import backend.sdxl_assembly.request_builder as rb
        rb._CONTROLNET_TYPE_CACHE.clear()
        self.assertEqual(determine_controlnet_type("lora.safetensors"), "control_lora")

    def test_eligibility_preserves_production_gates(self):
        task_state = types.SimpleNamespace(
            base_model_name="sdxl_base.safetensors",
            goals=["txt2img"],
            tiled=False,
            prepared_structural_cn_tasks={},
            prepared_contextual_cn_tasks={},
            initial_latent=None,
            requested_route_id="",
            requested_route_family="",
            current_tab="generate",
            input_image_checkbox=False,
            cn_tasks={},
        )
        
        # Should be eligible for normal plain text2img
        with patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as mock_tax, \
             patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list") as mock_get, \
             patch("os.path.exists", return_value=True):
            mock_tax.return_value = MagicMock(architecture=model_taxonomy.ARCHITECTURE_SDXL)
            mock_get.return_value = "sdxl_base.safetensors"
            
            eligible, reason = determine_eligibility(task_state)
            self.assertTrue(eligible)
            
            # Resolved asset maps alone are not active slots.
            eligible_cn, reason_cn = determine_eligibility(task_state, controlnet_paths={flags.cn_canny: "canny_path"})
            self.assertTrue(eligible_cn)
            self.assertIsNone(reason_cn)

            task_state.cn_tasks = {
                flags.cn_canny: [[np.zeros((64, 64, 3)), 1.0, 1.0, 0.0, 0]],
            }
            eligible_active_cn, reason_active_cn = determine_eligibility(
                task_state,
                controlnet_paths={flags.cn_canny: "canny_path"},
            )
            self.assertTrue(eligible_active_cn, reason_active_cn)

            # Force eligible allows them
            eligible_forced, reason_forced = determine_eligibility(task_state, controlnet_paths={flags.cn_canny: "canny_path"}, force_eligible=True)
            self.assertTrue(eligible_forced)

    @patch("backend.sdxl_assembly.request_builder.determine_controlnet_type", return_value="controlnet")
    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    def test_build_request_assigns_unique_structural_slots_across_types(self, mock_get_file, _mock_control_type):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_model = tmp / "base.safetensors"
            canny_model = tmp / "canny.safetensors"
            depth_model = tmp / "depth.safetensors"
            depth_preprocessor = tmp / "depth_preprocessor.pth"
            base_model.touch()
            canny_model.touch()
            depth_model.touch()
            depth_preprocessor.touch()
            mock_get_file.return_value = str(base_model)

            pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
            task_state = types.SimpleNamespace(
                base_model_name="base.safetensors",
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
                sharpness=2.0,
                adaptive_cfg=7.0,
                adm_scaler_positive=1.5,
                adm_scaler_negative=0.8,
                adm_scaler_end=0.3,
                canny_low_threshold=64,
                canny_high_threshold=128,
                skipping_cn_preprocessor=False,
                cn_tasks={
                    flags.cn_canny: [[pixels, 0.9, 0.5, 0.0, 1]],
                    flags.cn_depth: [[pixels, 0.8, 0.6, 0.0, 2]],
                    flags.cn_cpds: [],
                },
            )
            task_state.get_cn_tasks_for_channel = lambda channel: {
                flags.cn_canny: task_state.cn_tasks[flags.cn_canny],
                flags.cn_depth: task_state.cn_tasks[flags.cn_depth],
                flags.cn_cpds: task_state.cn_tasks[flags.cn_cpds],
            }

            request = build_assembly_request(
                task_state=task_state,
                task_dict={"task_prompt": "prompt", "task_negative_prompt": "neg", "task_seed": 1234},
                current_task_id=1,
                total_count=1,
                all_steps=20,
                preparation_steps=0,
                denoising_strength=None,
                final_scheduler_name="karras",
                loras=[],
                controlnet_paths={
                    flags.cn_canny: str(canny_model),
                    flags.cn_depth: str(depth_model),
                },
                image_input_result={"structural_preprocessor_paths": {flags.cn_depth: str(depth_preprocessor)}},
                force_eligible=True,
            )

        self.assertEqual([desc.slot_index for desc in request.structural_controls], [1, 2])
        self.assertEqual(
            [desc.control_type for desc in request.structural_controls],
            [flags.cn_canny, flags.cn_depth],
        )

    def test_clear_all_caches_clears_structural_worker_caches(self):
        StreamingStructuralPreprocessWorker._PREPROCESS_CACHE["hint"] = StructuralHintArtifact(
            slot_index=1,
            control_type=flags.cn_canny,
            hint_tensor=torch.zeros((1, 3, 64, 64)),
            hint_fingerprint="hint_fp",
        )

        model = MagicMock()
        patcher = MagicMock()
        model.control_model_wrapped = patcher
        StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE["sha"] = model

        clear_all_caches(reason="test_structural_cache_clear")

        self.assertEqual(len(StreamingStructuralPreprocessWorker._PREPROCESS_CACHE), 0)
        self.assertEqual(len(StreamingStructuralControlWorker._SUPPORT_MODEL_CACHE), 0)
        model.cleanup.assert_called_once()
        patcher.detach.assert_called_once()

    def test_assembly_ends_control_worker_when_attach_fails(self):
        request = MagicMock(spec=SDXLAssemblyRequest)
        request.validate = MagicMock()
        request.device = "cpu"
        request.metadata = {}
        request.spatial_context = None
        request.tiled_refinement = None
        request.color_extraction = None
        request.lora_specs = ()
        request.contextual_controls = ()
        request.structural_controls = (
            SDXLStructuralControlDescriptor(
                slot_index=1,
                control_type=flags.cn_canny,
                image_pixels=torch.zeros((1, 64, 64, 3)),
                image_fingerprint="fake_img_fp",
                preprocessor_id=flags.cn_canny,
                preprocessor_path=None,
                preprocessor_params={},
                target_width=1024,
                target_height=1024,
                checkpoint_path=Path(__file__),
                checkpoint_sha256="fake_sha",
                checkpoint_type="controlnet",
            ),
        )

        unet_spine = MagicMock()
        text_encode_worker = MagicMock()
        text_encode_worker.get_conditioning.return_value = {
            "positive": [[torch.zeros(1), {"cross_attn": torch.zeros(1)}]],
            "negative": [],
        }
        vae_decode_worker = MagicMock()
        vae_decode_worker.prepare_latents.return_value = types.SimpleNamespace(samples=torch.zeros((1, 4, 8, 8)))
        lora_worker = MagicMock()
        lora_worker.materialize_patches.return_value = []
        st_preprocess_worker = MagicMock()
        st_preprocess_worker.preprocess.return_value = {}
        st_control_worker = MagicMock()
        st_control_worker.attach_conditioning.side_effect = RuntimeError("attach failed")

        assembly = SDXLAssembly(
            unet_spine=unet_spine,
            text_encode_worker=text_encode_worker,
            vae_decode_worker=vae_decode_worker,
            lora_worker=lora_worker,
            st_preprocess_worker=st_preprocess_worker,
            st_control_worker=st_control_worker,
        )

        with self.assertRaisesRegex(RuntimeError, "Worker execution failed: attach failed"):
            assembly.execute(request)

        st_control_worker.end.assert_called_once()
        unet_spine.start.assert_not_called()

    def test_import_boundary_hygiene(self):
        # Assert that our workers don't import legacy runtime classes
        import sys
        legacy_modules = [
            "backend.sdxl_unified_runtime",
            "backend.sdxl_streaming_runtime",
            "backend.sdxl_resident_runtime",
            "backend.sdxl_runtime_policy",
            "backend.staging_manager",
            "backend.process_transition",
            "backend.memory_governor",
        ]
        
        # Check newly implemented files
        files_to_check = [
            Path("backend/sdxl_assembly/stream_st_preprocess_worker.py"),
            Path("backend/sdxl_assembly/stream_st_cn_worker.py"),
            Path("backend/sdxl_assembly/assembly.py"),
            Path("backend/sdxl_assembly/contracts.py"),
            Path("backend/sdxl_assembly/request_builder.py"),
        ]
        
        for file_path in files_to_check:
            content = file_path.read_text(encoding="utf-8")
            for legacy in legacy_modules:
                legacy_import = legacy.replace("backend.", "")
                self.assertNotIn(f"import {legacy}", content, f"Hygiene violation: {file_path} imports legacy {legacy}")
                self.assertNotIn(f"from backend import {legacy_import}", content, f"Hygiene violation: {file_path} imports legacy {legacy}")
