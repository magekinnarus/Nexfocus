from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.model_taxonomy as model_taxonomy
from backend import resources
from backend.sdxl_assembly.assembly import SDXLAssembly
from backend.sdxl_assembly.contracts import ResolvedFileIdentity, SDXLAssemblyRequest
from backend.sdxl_assembly.gateway import is_eligible_for_sdxl_assembly, run_sdxl_assembly_task
from backend.sdxl_assembly.request_builder import build_assembly_request, determine_eligibility
from modules.pipeline import inference


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


def _identity(path: Path, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=path,
        sha256=sha,
        size_bytes=path.stat().st_size,
        modified_ns=path.stat().st_mtime_ns,
    )


def test_production_eligibility_gate_admits_supported_image_inputs(tmp_path, monkeypatch):
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

    request = build_assembly_request(
        task_state=_task_state(goals=['inpaint']),
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
    text_encode_worker = SimpleNamespace(get_conditioning=lambda: {}, teardown_assembly_order=lambda: None)
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
        start=lambda: None,
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
    text_encode_worker = SimpleNamespace(get_conditioning=lambda: {}, teardown_assembly_order=lambda: None)
    vae_decode_worker = SimpleNamespace(
        decode=lambda _latent, _device: (np.ones((4, 4, 3), dtype=np.uint8) * 255, 0.0, 0.0),
        teardown_assembly_order=lambda: None,
    )
    unet_spine = SimpleNamespace(
        start=lambda: None,
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
