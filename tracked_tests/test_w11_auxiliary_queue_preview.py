"""Focused W11e auxiliary queue/preview and custom GAN tile size tests."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace

from modules.route_intent import resolve_route_intent
from modules.runtime_surface_state import _resolve_task_display_fields
from backend.process_transition import (
    PROCESS_CLASS_STANDARD_SDXL,
    PROCESS_FAMILY_FLUX_FILL,
    PROCESS_FAMILY_SDXL,
    build_process_key,
    resolve_requested_process_key,
)
from backend import process_transition
from modules.pipeline.routes import build_generation_route
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
from modules.task_state import TaskState
from modules.upscale_engine import NexUpscaleEngine
from modules.upscale_tile_policy import normalize_gan_tile_size
import backend.resources as resources
from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL, OBJR_ENGINE_MAT
import modules.async_worker as async_worker


def _build_planned_route(state):
    bind_legacy_workflow_plan(state)
    return build_generation_route(state)


def test_route_intent_flux_removal():
    # 1. Flux remove: wants_removal, remove_obj_enabled=True, and OBJR_ENGINE_FLUX_FILL
    state = TaskState(
        input_image_checkbox=True,
        current_tab="remove",
        remove_base_image=np.zeros((64, 64, 3), dtype=np.uint8),
        remove_obj_enabled=True,
        objr_engine=OBJR_ENGINE_FLUX_FILL,
    )
    intent = resolve_route_intent(state)
    assert intent.wants_removal is True
    assert intent.route_id == "flux_removal"
    assert intent.route_family == "flux_fill"

    # 2. Auxiliary remove: wants_removal, but MAT engine or BG enabled
    state_aux = TaskState(
        input_image_checkbox=True,
        current_tab="remove",
        remove_base_image=np.zeros((64, 64, 3), dtype=np.uint8),
        remove_bg_enabled=True,
        objr_engine=OBJR_ENGINE_MAT,
    )
    intent_aux = resolve_route_intent(state_aux)
    assert intent_aux.wants_removal is True
    assert intent_aux.route_id == "removal"
    assert intent_aux.route_family == "removal"


def test_build_generation_route_flux_removal():
    state = TaskState(
        input_image_checkbox=True,
        current_tab="remove",
        remove_base_image=np.zeros((64, 64, 3), dtype=np.uint8),
        remove_obj_enabled=True,
        objr_engine=OBJR_ENGINE_FLUX_FILL,
    )
    route = _build_planned_route(state)
    assert route.route_id == "flux_removal"
    assert route.family == "flux_fill"
    assert route.display_name == "Flux Remove"


def test_runtime_surface_display_fields_flux_removal():
    # Active flux removal
    state = TaskState(
        input_image_checkbox=True,
        current_tab="remove",
        remove_base_image=np.zeros((64, 64, 3), dtype=np.uint8),
        remove_obj_enabled=True,
        objr_engine=OBJR_ENGINE_FLUX_FILL,
        runtime_route_id="flux_removal",
        remove_prompt="repair statue",
    )
    fields = _resolve_task_display_fields(state)
    assert fields["workflow_name"] == "Flux Fill Object Removal"
    assert fields["prompt_label"] == "Remove Prompt"
    assert fields["prompt_text"] == "repair statue"

    # Stale flags check: not a removal request, but stale remove flags exist
    state_stale = TaskState(
        prompt="beautiful sunset",
        uov_method="Disabled",
        remove_bg_enabled=True, # stale flag
        remove_obj_enabled=True, # stale flag
    )
    fields_stale = _resolve_task_display_fields(state_stale)
    assert fields_stale["workflow_name"] == "Txt2Img"
    assert fields_stale["prompt_label"] == "Prompt"
    assert fields_stale["prompt_text"] == "beautiful sunset"


def test_process_transition_flux_removal():
    state = TaskState(
        input_image_checkbox=True,
        current_tab="remove",
        remove_base_image=np.zeros((64, 64, 3), dtype=np.uint8),
        remove_obj_enabled=True,
        objr_engine=OBJR_ENGINE_FLUX_FILL,
    )
    route = _build_planned_route(state)
    
    # Resolving process key for flux_removal route should expect a flux process
    key = resolve_requested_process_key(state, route)
    assert key is not None
    assert key.family == PROCESS_FAMILY_FLUX_FILL


def test_color_enhancement_ignores_stale_flux_removal_engine_for_process_key(monkeypatch):
    sdxl_key = build_process_key(
        family=PROCESS_FAMILY_SDXL,
        process_class=PROCESS_CLASS_STANDARD_SDXL,
        authoritative_identity=("model-a.safetensors", "vae-a.safetensors", "clip-a.safetensors"),
    )
    monkeypatch.setattr(process_transition, "resolve_sdxl_process_key", lambda *_args, **_kwargs: sdxl_key)
    monkeypatch.setattr(
        process_transition,
        "resolve_flux_fill_process_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Color Enhancement must not resolve Flux")),
    )

    state = TaskState(
        input_image_checkbox=True,
        current_tab="uov",
        uov_method="Color Enhancement",
        uov_input_image=np.zeros((64, 64, 3), dtype=np.uint8),
        upscale_gan_output_image=np.zeros((128, 128, 3), dtype=np.uint8),
        objr_engine=OBJR_ENGINE_FLUX_FILL,
        remove_obj_enabled=True,
        sdxl_execution_policy=SimpleNamespace(enabled=True),
    )
    route = _build_planned_route(state)

    key = resolve_requested_process_key(state, route)

    assert route.route_id == "color_enhanced_upscale"
    assert key == sdxl_key


def test_gan_tile_size_normalizes_to_supported_rungs():
    assert normalize_gan_tile_size(192) == 256
    assert normalize_gan_tile_size(256) == 256
    assert normalize_gan_tile_size(319) == 256
    assert normalize_gan_tile_size(320) == 320
    assert normalize_gan_tile_size(1200) == 1024


def test_gan_tile_size_bypass_heuristics(monkeypatch):
    engine = NexUpscaleEngine()
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Track the tiles sizes used
    passed_tile_sizes = []
    def fake_process_tiled(img, upscale_fn, scale, tile_size, overlap, device, dtype, ow, oh):
        passed_tile_sizes.append(tile_size)
        return torch.zeros((1, 3, oh, ow))
        
    monkeypatch.setattr(engine, "_process_tiled", fake_process_tiled)
    monkeypatch.setattr(resources, "begin_memory_phase", lambda *args, **kwargs: None)
    
    def dummy_upscale_fn(t):
        return t
        
    engine.process(
        dummy_img,
        dummy_upscale_fn,
        scale=2,
        device=torch.device("cpu"),
        is_bgr=True,
        dtype=torch.float32,
        tile_size=320
    )
    
    assert len(passed_tile_sizes) > 0
    assert passed_tile_sizes[0] == 320


def test_worker_clears_interrupt_state(monkeypatch):
    # Set interrupt flag to True
    resources.interrupt_current_processing(True)
    assert resources.processing_interrupted() is True
    
    # Mock AsyncTask
    task = async_worker.AsyncTask(args=[])
    task.state = TaskState(aspect_ratios_selection="invalid")
    
    # Run handler, which will clear the interrupt and then fail on aspect ratio
    with pytest.raises(ValueError, match="Invalid aspect ratio selection"):
        async_worker.handler(task)
        
    # The interrupt state must have been cleared!
    assert resources.processing_interrupted() is False
