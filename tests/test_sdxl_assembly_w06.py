import unittest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

import modules.model_taxonomy as model_taxonomy
from backend.sdxl_assembly.contracts import (
    SDXLAssemblyRequest,
    ResolvedFileIdentity,
    SpatialImageDescriptor,
    SpatialMaskDescriptor,
    SpatialContextDescriptor,
    PreparedSpatialContext,
    make_spatial_image_descriptor,
    make_spatial_mask_descriptor
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
    build_spatial_context_descriptor
)
from backend.sdxl_assembly.spatial_context_worker import SpatialContextWorker
from backend.sdxl_assembly.vae_encode_worker import VaeEncodeWorker
from backend.sdxl_assembly.runtime_state import clear_all_caches
from modules.pipeline.workflow_compiler import compile_workflow_plan
from modules.pipeline.workflow_contracts import FrozenWorkflowSelection
from types import SimpleNamespace

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
        input_image_checkbox=False,
        current_tab="generate",
        requested_source_surface="",
        requested_route_id="",
        workflow_plan=None,
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


class TestSDXLAssemblyW06(unittest.TestCase):
    def test_frozen_descriptors_mutation_safety(self):
        # 1. Test image descriptor clone & normalize
        pixels = np.ones((512, 512, 3), dtype=np.uint8) * 255
        desc = make_spatial_image_descriptor(pixels)
        
        # Modify original pixels array
        pixels[0, 0, 0] = 0
        
        # Verify descriptor's pixels are cloned and unaffected, and properly scaled to [0, 1]
        self.assertAlmostEqual(desc.pixels[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(desc.pixels.shape, (1, 512, 512, 3))
        
        # 2. Test mask descriptor matching size & clone
        mask = np.ones((512, 512), dtype=np.uint8) * 255
        mask_desc = make_spatial_mask_descriptor(mask, desc)
        
        mask[0, 0] = 0
        
        # Verify descriptor's mask is cloned and binarized to 1.0
        self.assertAlmostEqual(mask_desc.mask[0, 0, 0].item(), 1.0)

    def test_cache_keys_and_invalidation(self):
        pixels = np.ones((64, 64, 3), dtype=np.float32)
        image_desc = make_spatial_image_descriptor(pixels)
        
        prepared1 = PreparedSpatialContext(
            mode="image",
            original_pixels=image_desc.pixels,
            original_mask=None,
            bb_pixels=image_desc.pixels,
            bb_mask=None,
            bbox=(0, 64, 0, 64),
            image_fingerprint=image_desc.fingerprint,
            bb_pixels_fingerprint=image_desc.fingerprint,
        )
        
        key1 = prepared1.get_cache_key("vae_a")
        key2 = prepared1.get_cache_key("vae_b")
        self.assertNotEqual(key1, key2)
        
        key3 = prepared1.get_cache_key("vae_a")
        self.assertEqual(key1, key3)

        prepared2 = PreparedSpatialContext(
            mode="image",
            original_pixels=image_desc.pixels,
            original_mask=None,
            bb_pixels=image_desc.pixels,
            bb_mask=None,
            bbox=(8, 56, 8, 56),
            bbox_area_ratio=0.5625,
            image_fingerprint="different_image_fingerprint",
            bb_pixels_fingerprint=image_desc.fingerprint,
        )
        key4 = prepared2.get_cache_key("vae_a")
        self.assertNotEqual(key1, key4)

    def test_eligibility_gate_preserves_txt2img(self):
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
        task_state.workflow_plan = compile_workflow_plan(
            FrozenWorkflowSelection(source_surface="normal_generate")
        )
        task_state.sdxl_assembly_posture = "streaming"
        task_state.sdxl_execution_policy = None
        task_state.objr_engine = None
        task_state.uov_method = ""
        task_state.inpaint_route = "sdxl"
        task_state.remove_bg_enabled = False
        task_state.remove_obj_enabled = False
        task_state.mixing_image_prompt_and_inpaint = False
        task_state.mixing_image_prompt_and_outpaint = False
        
        # Should be eligible for plain txt2img
        with patch("backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy") as mock_tax, \
             patch("backend.sdxl_assembly.request_builder.get_file_from_folder_list") as mock_get, \
             patch("os.path.exists", return_value=True):
            mock_tax.return_value = MagicMock(architecture=model_taxonomy.ARCHITECTURE_SDXL)
            mock_get.return_value = "sdxl_base.safetensors"
            
            eligible, reason = determine_eligibility(task_state)
            self.assertTrue(eligible)
            
            # Unplanned spatial payloads cannot override immutable txt2img truth.
            eligible, reason = determine_eligibility(task_state, image_input_result={"inpaint_image": np.zeros((1, 1))})
            self.assertTrue(eligible)
            self.assertIsNone(reason)
            
            # Reject when inpaint/outpaint goals are active
            task_state_inpaint = MagicMock()
            task_state_inpaint.goals = ["inpaint"]
            task_state_inpaint.input_image_checkbox = False
            task_state_inpaint.current_tab = "generate"
            task_state_inpaint.requested_source_surface = ""
            task_state_inpaint.requested_route_id = ""
            task_state_inpaint.workflow_plan = compile_workflow_plan(
                FrozenWorkflowSelection(source_surface="inpaint")
            )
            task_state_inpaint.sdxl_assembly_posture = "streaming"
            task_state_inpaint.sdxl_execution_policy = None
            task_state_inpaint.objr_engine = None
            task_state_inpaint.uov_method = ""
            task_state_inpaint.inpaint_route = "sdxl"
            task_state_inpaint.remove_bg_enabled = False
            task_state_inpaint.remove_obj_enabled = False
            task_state_inpaint.mixing_image_prompt_and_inpaint = False
            task_state_inpaint.mixing_image_prompt_and_outpaint = False
            eligible, reason = determine_eligibility(task_state_inpaint)
            self.assertFalse(eligible)

    def test_force_eligible_probe_bypass(self):
        task_state = MagicMock()
        task_state.goals = ["inpaint"]
        task_state.input_image_checkbox = False
        task_state.current_tab = "generate"
        task_state.requested_source_surface = ""
        task_state.requested_route_id = ""
        task_state.workflow_plan = compile_workflow_plan(
            FrozenWorkflowSelection(source_surface="inpaint")
        )
        task_state.sdxl_assembly_posture = "streaming"
        task_state.sdxl_execution_policy = None
        task_state.objr_engine = None
        task_state.uov_method = ""
        task_state.inpaint_route = "sdxl"
        task_state.remove_bg_enabled = False
        task_state.remove_obj_enabled = False
        task_state.mixing_image_prompt_and_inpaint = False
        task_state.mixing_image_prompt_and_outpaint = False
        eligible, reason = determine_eligibility(task_state)
        self.assertFalse(eligible)
        
        eligible, reason = determine_eligibility(task_state, force_eligible=True)
        self.assertTrue(eligible)

    @patch("backend.sdxl_assembly.vae_encode_worker._encode_attached_vae")
    @patch("backend.sdxl_assembly.vae_encode_worker.acquire_vae_component")
    def test_vae_encode_worker_transient_lifecycle(self, mock_acquire, mock_encode):
        mock_vae = MagicMock()
        mock_encode.return_value = {"samples": torch.zeros((1, 4, 8, 8))}
        mock_acquire.return_value = mock_vae
        
        checkpoint = ResolvedFileIdentity(Path("chk.safetensors"), "chk_sha", 1, 1)
        req = SDXLAssemblyRequest(
            request_id="req_test",
            route_id="probe",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="prompt",
            negative_prompt="",
            positive_texts=("prompt",),
            negative_texts=("",),
            width=64,
            height=64,
            steps=1,
            cfg=1.0,
            sampler="euler",
            scheduler="normal",
            seed=1,
            device="cpu",
        )
        
        image_desc = make_spatial_image_descriptor(np.ones((64, 64, 3)))
        prepared = PreparedSpatialContext(
            mode="image",
            original_pixels=image_desc.pixels,
            original_mask=None,
            bb_pixels=image_desc.pixels,
            bb_mask=None,
            bbox=(0, 64, 0, 64),
            image_fingerprint=image_desc.fingerprint,
            bb_pixels_fingerprint=image_desc.fingerprint,
        )
        
        worker = VaeEncodeWorker(req)
        worker._ENCODE_CACHE.clear()
        
        artifacts = worker.encode(prepared)
        
        # Verify CPU parked
        self.assertEqual(artifacts.route_latent.device, torch.device("cpu"))
        
        # Verify VAE was ejected
        self.assertIsNone(worker.vae)

    @patch("backend.sdxl_assembly.vae_encode_worker._encode_attached_vae")
    @patch("backend.sdxl_assembly.vae_encode_worker.acquire_vae_component")
    def test_vae_encode_cache_hit_preserves_blend_mask(self, mock_acquire, mock_encode):
        mock_vae = MagicMock()
        mock_encode.return_value = {"samples": torch.zeros((1, 4, 8, 8))}
        mock_acquire.return_value = mock_vae

        checkpoint = ResolvedFileIdentity(Path("chk.safetensors"), "chk_sha", 1, 1)
        req = SDXLAssemblyRequest(
            request_id="req_test",
            route_id="probe",
            image_index=0,
            image_count=1,
            checkpoint=checkpoint,
            vae=None,
            model_variant_key="sdxl",
            prompt="prompt",
            negative_prompt="",
            positive_texts=("prompt",),
            negative_texts=("",),
            width=64,
            height=64,
            steps=1,
            cfg=1.0,
            sampler="euler",
            scheduler="normal",
            seed=1,
            device="cpu",
        )

        image_desc = make_spatial_image_descriptor(np.ones((64, 64, 3)))
        mask_desc = make_spatial_mask_descriptor(np.ones((64, 64), dtype=np.uint8) * 255, image_desc)
        blend_mask = torch.ones((1, 64, 64), dtype=torch.float32)
        prepared = PreparedSpatialContext(
            mode="image",
            original_pixels=image_desc.pixels,
            original_mask=mask_desc.mask,
            bb_pixels=image_desc.pixels,
            bb_mask=mask_desc.mask,
            blend_mask=blend_mask,
            bbox=(0, 64, 0, 64),
            image_fingerprint=image_desc.fingerprint,
            mask_fingerprint=mask_desc.fingerprint,
            bb_pixels_fingerprint=image_desc.fingerprint,
            bb_mask_fingerprint=mask_desc.fingerprint,
        )

        worker = VaeEncodeWorker(req)
        worker._ENCODE_CACHE.clear()

        first = worker.encode(prepared)
        second = worker.encode(prepared)

        self.assertIsNotNone(first.blend_mask)
        self.assertTrue(second.cache_hit)
        self.assertIsNotNone(second.blend_mask)
        self.assertTrue(torch.equal(first.blend_mask, second.blend_mask))

    def test_clear_all_caches_clears_vae_encode_cache(self):
        VaeEncodeWorker._ENCODE_CACHE.clear()
        VaeEncodeWorker._ENCODE_CACHE["demo"] = {"route_latent": torch.zeros((1, 4, 8, 8))}
        clear_all_caches(reason="w06_test")
        self.assertEqual(len(VaeEncodeWorker._ENCODE_CACHE), 0)
        self.assertIsNotNone(second.blend_mask)
        self.assertTrue(torch.equal(first.blend_mask, second.blend_mask))

    def test_clear_all_caches_clears_vae_encode_cache(self):
        VaeEncodeWorker._ENCODE_CACHE.clear()
        VaeEncodeWorker._ENCODE_CACHE["demo"] = {"route_latent": torch.zeros((1, 4, 8, 8))}
        clear_all_caches(reason="w06_test")
        self.assertEqual(len(VaeEncodeWorker._ENCODE_CACHE), 0)


def test_production_eligibility_gate_admits_supported_image_inputs(tmp_path, monkeypatch):
    from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly
    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.get_file_from_folder_list',
        lambda _name, _folders: str(checkpoint_path),
    )
    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy',
        lambda _path: SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL),
    )

    state = _task_state()
    state.input_image_checkbox = True
    state.current_tab = "inpaint"
    state.inpaint_input_image = np.zeros((64, 64, 3))
    state.inpaint_mask_image = np.zeros((64, 64))
    state.goals = ["inpaint"]
    from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
    bind_legacy_workflow_plan(state)
    eligible, reason = is_eligible_for_sdxl_assembly(
        task_state=state,
        loras=[],
        image_input_result={
            "inpaint_image": np.zeros((64, 64, 3)),
            "inpaint_mask": np.zeros((64, 64)),
        },
    )
    assert eligible, f"Expected inpaint to be eligible under W09, got: {reason}"


def test_request_builder_forces_eligibility_and_captures_spatial_context(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / 'sdxl_base.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.get_file_from_folder_list',
        lambda _name, _folders: str(checkpoint_path),
    )
    monkeypatch.setattr(
        'backend.sdxl_assembly.request_builder.config.resolve_model_taxonomy',
        lambda _path: SimpleNamespace(architecture=model_taxonomy.ARCHITECTURE_SDXL),
    )

    inpaint_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    inpaint_mask = np.ones((64, 64), dtype=np.uint8) * 255

    state = _task_state(goals=['inpaint'])
    state.input_image_checkbox = True
    state.current_tab = "inpaint"
    state.inpaint_input_image = inpaint_img
    state.inpaint_mask_image = inpaint_mask
    from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
    bind_legacy_workflow_plan(state)

    request = build_assembly_request(
        task_state=state,
        task_dict=_task_dict(),
        current_task_id=0,
        total_count=1,
        all_steps=3,
        preparation_steps=0,
        denoising_strength=1.0,
        final_scheduler_name='karras',
        loras=[],
        image_input_result={
            'inpaint_image': inpaint_img,
            'inpaint_mask': inpaint_mask,
        },
        force_eligible=True,
    )

    assert request.spatial_context is not None
    assert request.spatial_context.mode == 'inpaint'
    assert request.spatial_context.source_image.pixels.shape == (1, 64, 64, 3)
    assert request.spatial_context.source_mask.mask.shape == (1, 64, 64)


def _identity(path: Path, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=path,
        sha256=sha,
        size_bytes=path.stat().st_size,
        modified_ns=path.stat().st_mtime_ns,
    )


def test_assembly_coordinates_spatial_context_worker_and_vae_encode_worker(tmp_path):
    checkpoint_path = tmp_path / 'checkpoint.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    from backend.sdxl_assembly.contracts import (
        make_spatial_image_descriptor,
        make_spatial_mask_descriptor,
        SpatialContextDescriptor,
    )
    img_desc = make_spatial_image_descriptor(np.ones((64, 64, 3)))
    mask_desc = make_spatial_mask_descriptor(np.ones((64, 64)), img_desc)
    spatial_desc = SpatialContextDescriptor(
        mode='inpaint',
        source_image=img_desc,
        source_mask=mask_desc,
        target_width=64,
        target_height=64,
    )

    request = SDXLAssemblyRequest(
        request_id='req_spatial_test',
        route_id='probe',
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
        spatial_context=spatial_desc,
    )

    from backend.sdxl_assembly.contracts import PreparedSpatialContext, SpatialAssemblyArtifacts
    from backend.sdxl_assembly.assembly import SDXLAssembly
    from types import SimpleNamespace
    
    prepared_context = PreparedSpatialContext(
        mode='inpaint',
        original_pixels=img_desc.pixels,
        original_mask=mask_desc.mask,
        bb_pixels=img_desc.pixels,
        bb_mask=mask_desc.mask,
        bbox=(0, 64, 0, 64),
        image_fingerprint=img_desc.fingerprint,
        mask_fingerprint=mask_desc.fingerprint,
        bb_pixels_fingerprint=img_desc.fingerprint,
        bb_mask_fingerprint=mask_desc.fingerprint,
    )

    denoise_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    spatial_artifacts = SpatialAssemblyArtifacts(
        route_latent=torch.zeros((1, 4, 8, 8)),
        bb_latent=torch.ones((1, 4, 8, 8)),
        denoise_mask=denoise_mask,
        source_fingerprint='src_fingerprint',
        image_fingerprint='img_fingerprint',
        route_latent_fingerprint='latent_fingerprint',
        bb_latent_fingerprint='bb_latent_fingerprint',
        denoise_mask_fingerprint='denoise_mask_fingerprint',
    )

    spatial_worker = SimpleNamespace(
        prepare=lambda: prepared_context,
        teardown_assembly_order=lambda: None
    )
    vae_encode_worker = SimpleNamespace(
        encode=lambda _ctx: spatial_artifacts,
        teardown_assembly_order=lambda: None
    )
    lora_worker = SimpleNamespace(materialize_patches=lambda: None, teardown_assembly_order=lambda: None)
    text_encode_worker = SimpleNamespace(get_conditioning=lambda **_kwargs: {}, teardown_assembly_order=lambda: None)
    vae_decode_worker = SimpleNamespace(
        decode=lambda _latent, _device: (np.zeros((64, 64, 3), dtype=np.uint8), 0.0, 0.0),
        teardown_assembly_order=lambda: None
    )
    denoise_inputs = {}

    def _denoise(latent, *args, **kwargs):
        denoise_inputs["latent"] = latent.detach().cpu().clone()
        denoise_inputs["denoise_mask"] = kwargs.get("denoise_mask")
        return latent

    unet_spine = SimpleNamespace(
        start=lambda **_kwargs: None,
        denoise=_denoise,
        end=lambda: None,
        teardown_assembly_order=lambda: None,
    )

    assembly = SDXLAssembly(
        unet_spine=unet_spine,
        text_encode_worker=text_encode_worker,
        vae_decode_worker=vae_decode_worker,
        lora_worker=lora_worker,
        spatial_context_worker=spatial_worker,
        vae_encode_worker=vae_encode_worker,
    )

    res = assembly.execute(request)
    assert res.output_image.shape == (64, 64, 3)
    assert torch.equal(denoise_inputs["latent"], spatial_artifacts.bb_latent)
    assert torch.equal(denoise_inputs["denoise_mask"], denoise_mask)
    assert res.metadata["spatial_contract"]["mode"] == "inpaint"
    assert res.metadata["spatial_contract"]["bb_latent_fingerprint"] == "bb_latent_fingerprint"
    assert res.metadata["spatial_contract"]["denoise_mask_fingerprint"] == "denoise_mask_fingerprint"


def test_assembly_composes_inpaint_patch_back_to_full_canvas(tmp_path):
    checkpoint_path = tmp_path / 'checkpoint.safetensors'
    checkpoint_path.write_bytes(b'checkpoint')

    from backend.sdxl_assembly.contracts import (
        PreparedSpatialContext,
        SpatialAssemblyArtifacts,
        SpatialContextDescriptor,
        make_spatial_image_descriptor,
        make_spatial_mask_descriptor,
    )
    from backend.sdxl_assembly.assembly import SDXLAssembly
    from types import SimpleNamespace

    source_pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    source_mask = np.zeros((8, 8), dtype=np.uint8)
    source_mask[2:6, 1:5] = 255
    bbox = (2, 6, 1, 5)

    img_desc = make_spatial_image_descriptor(source_pixels)
    mask_desc = make_spatial_mask_descriptor(source_mask, img_desc)
    spatial_desc = SpatialContextDescriptor(
        mode='inpaint',
        source_image=img_desc,
        source_mask=mask_desc,
        target_width=4,
        target_height=4,
        bbox=bbox,
    )

    request = SDXLAssemblyRequest(
        request_id='req_compose_inpaint',
        route_id='probe',
        image_index=0,
        image_count=1,
        checkpoint=_identity(checkpoint_path, 'checkpoint_sha'),
        vae=None,
        model_variant_key='sdxl',
        prompt='prompt',
        negative_prompt='negative',
        positive_texts=('prompt',),
        negative_texts=('negative',),
        width=4,
        height=4,
        steps=3,
        cfg=5.0,
        sampler='euler',
        scheduler='karras',
        seed=123,
        device='cpu',
        spatial_context=spatial_desc,
    )

    blend_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    blend_mask[:, 2:6, 1:5] = 1.0
    prepared_context = PreparedSpatialContext(
        mode='inpaint',
        original_pixels=img_desc.pixels,
        original_mask=mask_desc.mask,
        bb_pixels=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
        bb_mask=torch.ones((1, 4, 4), dtype=torch.float32),
        blend_mask=blend_mask,
        bbox=bbox,
        image_fingerprint=img_desc.fingerprint,
        mask_fingerprint=mask_desc.fingerprint,
        bb_pixels_fingerprint='bb_pixels',
        bb_mask_fingerprint='bb_mask',
    )

    spatial_artifacts = SpatialAssemblyArtifacts(
        route_latent=torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        bb_latent=torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        denoise_mask=torch.ones((1, 1, 4, 4), dtype=torch.float32),
        blend_mask=blend_mask,
        source_fingerprint=img_desc.fingerprint,
        image_fingerprint=img_desc.fingerprint,
        route_latent_fingerprint='route_latent',
        bb_latent_fingerprint='bb_latent',
        denoise_mask_fingerprint='denoise_mask',
        blend_mask_fingerprint='blend_mask',
        bbox=bbox,
    )

    spatial_worker = SimpleNamespace(
        prepare=lambda: prepared_context,
        teardown_assembly_order=lambda: None,
    )
    vae_encode_worker = SimpleNamespace(
        encode=lambda _ctx: spatial_artifacts,
        teardown_assembly_order=lambda: None,
    )
    lora_worker = SimpleNamespace(materialize_patches=lambda: None, teardown_assembly_order=lambda: None)
    text_encode_worker = SimpleNamespace(get_conditioning=lambda **_kwargs: {}, teardown_assembly_order=lambda: None)
    vae_decode_worker = SimpleNamespace(
        decode=lambda _latent, _device: (np.ones((4, 4, 3), dtype=np.uint8) * 255, 0.0, 0.0),
        teardown_assembly_order=lambda: None,
    )
    unet_spine = SimpleNamespace(
        start=lambda **_kwargs: None,
        denoise=lambda latent, *args, **kwargs: latent,
        end=lambda: None,
        teardown_assembly_order=lambda: None,
    )

    assembly = SDXLAssembly(
        unet_spine=unet_spine,
        text_encode_worker=text_encode_worker,
        vae_decode_worker=vae_decode_worker,
        lora_worker=lora_worker,
        spatial_context_worker=spatial_worker,
        vae_encode_worker=vae_encode_worker,
    )

    res = assembly.execute(request)

    assert res.output_image.shape == (8, 8, 3)
    assert res.width == 8
    assert res.height == 8
    assert np.all(res.output_image[2:6, 1:5] == 255)
    assert np.all(res.output_image[:2] == 0)
    assert np.all(res.output_image[6:] == 0)
    assert np.all(res.output_image[:, :1] == 0)
    assert np.all(res.output_image[:, 5:] == 0)
