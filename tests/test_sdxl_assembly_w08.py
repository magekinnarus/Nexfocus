import unittest
import tempfile
import types
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    SDXLAssemblyEligibilityError,
    ResolvedFileIdentity,
    SDXLContextualControlDescriptor,
    ContextualPayloadArtifact,
    make_spatial_image_descriptor,
    SDXLAssemblyValidationError,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
)
from backend.sdxl_assembly.stream_ctx_cn_worker import (
    StreamingContextualControlWorker,
)
from backend.sdxl_assembly.assembly import SDXLAssembly
from backend.sdxl_assembly.runtime_state import clear_all_caches
import modules.flags as flags
import modules.model_taxonomy as model_taxonomy

class TestSDXLAssemblyW08(unittest.TestCase):
    def setUp(self):
        clear_all_caches()

    def _make_dummy_request(self, **overrides) -> SDXLAssemblyRequest:
        kwargs = {
            "request_id": "req_test",
            "route_id": "txt2img_assembly",
            "image_index": 0,
            "image_count": 1,
            "checkpoint": ResolvedFileIdentity(path=Path(__file__).resolve(), sha256="123", size_bytes=100, modified_ns=123),
            "vae": ResolvedFileIdentity(path=Path(__file__).resolve(), sha256="vae_sha", size_bytes=50, modified_ns=456),
            "model_variant_key": "sdxl",
            "prompt": "test prompt",
            "negative_prompt": "ugly",
            "positive_texts": ("test prompt",),
            "negative_texts": ("ugly",),
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "seed": 42,
            "device": "cpu",
        }
        kwargs.update(overrides)
        return SDXLAssemblyRequest(**kwargs)

    def test_contextual_descriptors_immutability(self):
        # Create a descriptor
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)

        control_desc = SDXLContextualControlDescriptor(
            ui_slot_index=1,
            control_type="ImagePrompt",
            image_pixels=desc.pixels,
            image_fingerprint=desc.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="fake_model_sha",
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )

        # Verify frozen immutability
        with self.assertRaises(Exception):
            control_desc.weight = 0.5

    def test_explicit_rejection_of_faceid_v2(self):
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)

        # 1. Verification of direct validation rejection
        with self.assertRaises(SDXLAssemblyValidationError):
            req = self._make_dummy_request(
                contextual_controls=(
                    SDXLContextualControlDescriptor(
                        ui_slot_index=0,
                        control_type="FaceID V2",  # Retired!
                        image_pixels=desc.pixels,
                        image_fingerprint=desc.fingerprint,
                        source_image_role="face_image",
                        model_path=Path("faceid.bin"),
                        model_sha256="fake_sha",
                    ),
                )
            )
            req.validate()

        # 2. Verification of determine_eligibility rejection
        task_state = MagicMock()
        task_state.cn_tasks = {"FaceID V2": [None, 1.0, 1.0]}
        eligible, reason = determine_eligibility(task_state, force_eligible=True)
        self.assertFalse(eligible)
        self.assertIn("retired", reason)

        # Check FaceSwap alias rejection
        task_state.cn_tasks = {"FaceSwap": [None, 1.0, 1.0]}
        eligible, reason = determine_eligibility(task_state, force_eligible=True)
        self.assertFalse(eligible)
        self.assertIn("retired", reason)

        task_state.cn_tasks = {"MLSD": [None, 1.0, 1.0]}
        eligible, reason = determine_eligibility(task_state, force_eligible=True)
        self.assertFalse(eligible)
        self.assertIn("retired", reason)

    def test_mlsd_retirement(self):
        # MLSD should be absent from structural types
        self.assertNotIn("MLSD", flags.cn_structural_types)
        
        # Test validation fails on MLSD in structural control
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)
        
        # Checking that MLSD cannot be loaded since it isn't a valid type
        from backend.sdxl_assembly.contracts import SDXLStructuralControlDescriptor
        with self.assertRaises(SDXLAssemblyValidationError):
            req = self._make_dummy_request(
                structural_controls=(
                    SDXLStructuralControlDescriptor(
                        slot_index=1,
                        control_type="MLSD",  # Retired!
                        image_pixels=desc.pixels,
                        image_fingerprint=desc.fingerprint,
                        preprocessor_id="MLSD",
                        preprocessor_path=None,
                        preprocessor_params={},
                        target_width=1024,
                        target_height=1024,
                        checkpoint_path=Path("dummy_mlsd.safetensors"),
                        checkpoint_sha256="fake_sha",
                        checkpoint_type="controlnet",
                    ),
                )
            )
            req.validate()

    @patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list")
    def test_build_request_assigns_legacy_contextual_slot_index(self, mock_get_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_model = tmp / "base.safetensors"
            clip_vision = tmp / "clip_vision.safetensors"
            ip_negative = tmp / "ip_negative.safetensors"
            ip_model = tmp / "ip_adapter.bin"
            for path in (base_model, clip_vision, ip_negative, ip_model):
                path.touch()

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
                cn_tasks={
                    flags.cn_ip: [[pixels, 0.8, 0.7, 0.0]],
                    flags.cn_pulid: [],
                },
            )
            task_state.get_cn_tasks_for_channel = lambda _channel: {
                flags.cn_ip: [[pixels, 0.8, 0.7, 0.0]],
                flags.cn_pulid: [],
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
                contextual_assets={
                    "contextual_model_paths": {flags.cn_ip: str(ip_model)},
                    "clip_vision_path": str(clip_vision),
                    "ip_negative_path": str(ip_negative),
                },
                force_eligible=True,
            )

            # The named legacy adapter preserves ordinal slot identity when
            # an older four-field task has no explicit UI slot field.
            self.assertEqual(len(request.contextual_controls), 1)
            self.assertEqual(request.contextual_controls[0].ui_slot_index, 0)

    def test_contextual_cache_key_invalidation_rules(self):
        pixels1 = np.ones((64, 64, 3), dtype=np.uint8) * 255
        pixels2 = np.ones((64, 64, 3), dtype=np.uint8) * 128
        desc1 = make_spatial_image_descriptor(pixels1)
        desc2 = make_spatial_image_descriptor(pixels2)

        # Baseline descriptor
        desc_base = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc1.pixels,
            image_fingerprint=desc1.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_1",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 224},
            weight=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )

        worker = StreamingContextualControlWorker(MagicMock())
        key_base = worker._get_contextual_cache_key(desc_base)

        # 1. Changing image fingerprint invalidates key
        desc_diff_img = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc2.pixels,
            image_fingerprint=desc2.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_1",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 224},
        )
        self.assertNotEqual(key_base, worker._get_contextual_cache_key(desc_diff_img))

        # 2. Changing image role invalidates key
        desc_diff_role = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc1.pixels,
            image_fingerprint=desc1.fingerprint,
            source_image_role="different_role",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_1",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 224},
        )
        self.assertNotEqual(key_base, worker._get_contextual_cache_key(desc_diff_role))

        # 3. Changing model sha invalidates key
        desc_diff_model = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc1.pixels,
            image_fingerprint=desc1.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_different",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 224},
        )
        self.assertNotEqual(key_base, worker._get_contextual_cache_key(desc_diff_model))

        # 4. Changing preprocess params invalidates key
        desc_diff_params = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc1.pixels,
            image_fingerprint=desc1.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_1",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 512},
        )
        self.assertNotEqual(key_base, worker._get_contextual_cache_key(desc_diff_params))

        # 5. Changing weight, start_percent, or end_percent does NOT invalidate key
        desc_diff_weight = SDXLContextualControlDescriptor(
            ui_slot_index=0,
            control_type="ImagePrompt",
            image_pixels=desc1.pixels,
            image_fingerprint=desc1.fingerprint,
            source_image_role="image_prompt",
            model_path=Path("ip_adapter.bin"),
            model_sha256="sha_1",
            clip_vision_path=Path("clip_vision.safetensors"),
            clip_vision_sha256="clip_sha",
            preprocess_params={"resize_to": 224},
            weight=0.5,
            start_percent=0.1,
            end_percent=0.9,
        )
        self.assertEqual(key_base, worker._get_contextual_cache_key(desc_diff_weight))

    @patch("backend.resources.load_model_gpu")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_contextual_model_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_clip_vision_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.ensure_ip_negative_local")
    @patch("backend.ip_adapter._sorted_kv_modules")
    def test_image_prompt_preprocessing_cache_miss_hit(self, mock_kv_modules, mock_negative, mock_clip_vision, mock_ctx_model, mock_load_gpu):
        # Setup mocks
        mock_ctx = MagicMock()
        mock_ctx["model"] = MagicMock()
        mock_ctx["model"].plus = False
        mock_ctx["model"].load_device = torch.device("cpu")
        mock_ctx["model"].dtype = torch.float32
        mock_ctx["image_proj_model"] = MagicMock()
        mock_ctx["ip_layers"] = MagicMock()
        mock_ctx_model.return_value = mock_ctx

        mock_clip = MagicMock()
        mock_clip.to.return_value = mock_clip
        mock_clip.encode_image.return_value.image_embeds = torch.ones((1, 10))
        mock_clip_vision.return_value = mock_clip

        mock_negative.return_value = torch.zeros((1, 10))
        
        # Return mock module layers
        dummy_layer1 = MagicMock(return_value=torch.ones((1, 128)))
        dummy_layer2 = MagicMock(return_value=torch.ones((1, 128)))
        mock_kv_modules.return_value = [dummy_layer1, dummy_layer2]

        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)

        req = self._make_dummy_request(
            contextual_controls=(
                SDXLContextualControlDescriptor(
                    ui_slot_index=0,
                    control_type="ImagePrompt",
                    image_pixels=desc.pixels,
                    image_fingerprint=desc.fingerprint,
                    source_image_role="image_prompt",
                    model_path=Path("ip_adapter.bin"),
                    model_sha256="fake_sha",
                    clip_vision_path=Path("clip_vision.safetensors"),
                    clip_vision_sha256="clip_sha",
                    ip_negative_path=Path("ip_neg.safetensors"),
                    ip_negative_sha256="neg_sha",
                ),
            )
        )

        worker = StreamingContextualControlWorker(req)
        # Clear static cache first
        worker.clear_payload_cache()

        # Miss
        res = worker.preprocess()
        self.assertEqual(len(res), 1)
        self.assertFalse(res[0].cache_hit)
        self.assertEqual(res[0].payload_fingerprint, desc.fingerprint)

        # Hit
        res2 = worker.preprocess()
        self.assertTrue(res2[0].cache_hit)

    @patch("backend.resources.load_model_gpu")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_contextual_model_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_eva_clip_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_face_parser_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker.load_insightface_local")
    @patch("backend.sdxl_assembly.stream_ctx_cn_worker.StreamingContextualControlWorker._detect_faces_local")
    @patch("insightface.utils.face_align.norm_crop")
    @patch("backend.ip_adapter._sorted_kv_modules")
    def test_pulid_preprocessing_workflow(self, mock_kv_modules, mock_norm_crop, mock_detect_faces, mock_insightface, mock_face_parser, mock_eva_clip, mock_ctx_model, mock_load_gpu):
        # Setup mocks
        mock_ctx = MagicMock()
        mock_ctx["model"] = MagicMock()
        mock_ctx["model"].load_device = torch.device("cpu")
        mock_ctx["model"].dtype = torch.float32
        mock_ctx["image_proj_model"] = MagicMock()
        mock_ctx["image_proj_model"].model.return_value = torch.ones((1, 5, 2048))
        mock_ctx["ip_layers"] = MagicMock()
        mock_ctx_model.return_value = mock_ctx

        mock_eva = MagicMock()
        mock_eva.to.return_value = mock_eva
        mock_eva.image_size = 336
        mock_eva.image_mean = [0.48, 0.45, 0.40]
        mock_eva.image_std = [0.22, 0.22, 0.22]
        mock_eva.return_value = (torch.ones((1, 1024)), [torch.ones((1, 1024))])
        mock_eva_clip.return_value = mock_eva

        mock_parser = MagicMock()
        mock_parser.return_value = [torch.zeros((1, 19, 512, 512))]
        mock_face_parser.return_value = mock_parser

        face_mock = MagicMock()
        face_mock.embedding = np.ones((512,), dtype=np.float32)
        face_mock.kps = np.ones((5, 2), dtype=np.float32)
        mock_detect_faces.return_value = [face_mock]
        mock_norm_crop.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        dummy_layer1 = MagicMock(return_value=torch.ones((1, 128)))
        mock_kv_modules.return_value = [dummy_layer1]

        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)

        req = self._make_dummy_request(
            contextual_controls=(
                SDXLContextualControlDescriptor(
                    ui_slot_index=1,
                    control_type="PuLID",
                    image_pixels=desc.pixels,
                    image_fingerprint=desc.fingerprint,
                    source_image_role="face_image",
                    model_path=Path("pulid.safetensors"),
                    model_sha256="fake_sha",
                    eva_clip_path=Path("eva_clip.pt"),
                    eva_clip_sha256="eva_sha",
                    insightface_model_names=("antelopev2",)
                ),
            )
        )

        worker = StreamingContextualControlWorker(req)
        worker.clear_payload_cache()

        res = worker.preprocess()
        self.assertEqual(len(res), 1)
        self.assertFalse(res[1].cache_hit)

    def test_unified_attention_patching_and_cleanup(self):
        # Create mockup of UNet spine and patches replace options
        unet_spine = MagicMock()
        unet_spine.unet = MagicMock()
        unet_spine.unet.model_options = {
            "transformer_options": {
                "patches_replace": {
                    "attn2": {}
                }
            }
        }

        # Mock request contextual descriptors
        pixels = np.ones((64, 64, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)

        req = self._make_dummy_request(
            contextual_controls=(
                SDXLContextualControlDescriptor(
                    ui_slot_index=0,
                    control_type="ImagePrompt",
                    image_pixels=desc.pixels,
                    image_fingerprint=desc.fingerprint,
                    source_image_role="image_prompt",
                    model_path=Path("ip_adapter.bin"),
                    model_sha256="fake_sha",
                ),
            )
        )

        worker = StreamingContextualControlWorker(req)
        worker.clear_payload_cache()

        # Manually populate payload cache
        cache_key = worker._get_contextual_cache_key(req.contextual_controls[0])
        mock_payload = ([torch.ones((1, 10))], [torch.zeros((1, 10))])
        worker._PAYLOAD_CACHE[cache_key] = ContextualPayloadArtifact(
            ui_slot_index=0,
            control_type="ImagePrompt",
            payload=mock_payload,
            payload_fingerprint=desc.fingerprint,
        )

        # Attach
        worker.attach_unet_patches(unet_spine)
        
        # Verify patches_replace options contain the patched layers
        patches = unet_spine.unet.model_options["transformer_options"]["patches_replace"]["attn2"]
        self.assertTrue(len(patches) > 0)
        self.assertIn(("input", 4, 0), patches)

        # End request & verify request-local cleanup
        worker.end()
        self.assertEqual(len(patches), 0)

if __name__ == "__main__":
    unittest.main()
