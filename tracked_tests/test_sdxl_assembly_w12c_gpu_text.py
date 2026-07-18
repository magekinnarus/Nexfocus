from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import torch

from backend.sdxl_assembly.contracts import (
    ResolvedFileIdentity,
    SDXLLoraSpec,
    SDXLAssemblyRequest,
    UNetPostureKind,
    TextEncoderPostureKind,
    VAEPostureKind,
    LoraPatchPostureKind,
)
from backend.sdxl_assembly.request_builder import (
    determine_eligibility,
    build_assembly_request,
)
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.assembler import SDXLAssemblyAssembler
from backend.sdxl_assembly.gpu_text_encode_worker import GpuTextEncodeWorker
from backend.sdxl_assembly.gpu_lora_worker import GpuLoraWorker
from backend.sdxl_assembly.runtime_state import (
    SDXLGpuTextKey,
    _GPU_TEXT_RUNTIME_STATE,
    acquire_active_gpu_text,
    release_active_gpu_text,
)
from modules.task_state import TaskState
from modules.pipeline.workflow_legacy_adapter import bind_legacy_workflow_plan
from modules.parameter_registry import _normalize_sdxl_assembly_posture_value
from modules.ui_components.advanced_panel import resolve_default_sdxl_assembly_posture
from backend import process_transition

def _identity(name: str, sha: str) -> ResolvedFileIdentity:
    return ResolvedFileIdentity(
        path=Path(name),
        sha256=sha,
        size_bytes=1,
        modified_ns=1,
    )


def _gpu_request(*, lora_specs=(), checkpoint_sha="checkpoint_sha") -> SDXLAssemblyRequest:
    return SDXLAssemblyRequest(
        request_id="test_req",
        route_id="txt2img_assembly",
        image_index=0,
        image_count=1,
        checkpoint=_identity("checkpoint.safetensors", checkpoint_sha),
        vae=None,
        model_variant_key="sdxl",
        prompt="positive prompt",
        negative_prompt="negative prompt",
        positive_texts=("positive prompt",),
        negative_texts=("negative prompt",),
        width=512,
        height=512,
        steps=30,
        cfg=7.0,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="karras",
        seed=42,
        device="cuda",
        lora_specs=tuple(lora_specs),
        unet_posture=UNetPostureKind.RESIDENT,
        clip_posture=TextEncoderPostureKind.GPU_PINNED,
        vae_posture=VAEPostureKind.TRANSIENT,
        lora_posture=LoraPatchPostureKind.RESIDENT,
    )


def _lora(name: str, *, unet_weight=1.0, clip_weight=1.0) -> SDXLLoraSpec:
    return SDXLLoraSpec(
        file_identity=_identity(name, f"sha_{name}"),
        unet_weight=unet_weight,
        clip_weight=clip_weight,
    )


def _fake_clip(events=None):
    events = events if events is not None else []
    model = SimpleNamespace(current_weight_patches_uuid=None, device=torch.device("cuda"))
    model.cpu = MagicMock(side_effect=lambda: events.append("cpu"))
    patcher = SimpleNamespace(
        model=model,
        patches={},
        weight_wrapper_patches={},
        backup={},
        object_patches_backup={},
        runtime_reload=lambda _model, _device: events.append("reload"),
    )
    return SimpleNamespace(patcher=patcher, cond_stage_model=model), model


def test_gpu_text_key_ignores_explicit_unet_only_lora_addition():
    dual = _lora("dual.safetensors", unet_weight=0.8, clip_weight=0.8)
    unet_only = _lora("inpaint.patch", unet_weight=1.0, clip_weight=0.0)

    dual_key = _GPU_TEXT_RUNTIME_STATE._build_key(_gpu_request(lora_specs=(dual,)))
    mixed_key = _GPU_TEXT_RUNTIME_STATE._build_key(_gpu_request(lora_specs=(dual, unet_only)))

    assert mixed_key == dual_key

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    def fake_get_identity(path):
        return _identity(str(path), f"sha_{Path(path).name}")
    monkeypatch.setattr("backend.sdxl_assembly.request_builder.get_file_identity", fake_get_identity)

    import modules.config as config
    monkeypatch.setattr(config, "paths_checkpoints", ["/mock/checkpoints"])

    def fake_folder_list(name, folders):
        return f"/mock/checkpoints/{name}"
    monkeypatch.setattr("backend.sdxl_assembly.request_builder.get_file_from_folder_list", fake_folder_list)

    from modules.model_taxonomy import ARCHITECTURE_SDXL
    taxonomy = SimpleNamespace(architecture=ARCHITECTURE_SDXL)
    monkeypatch.setattr(config, "resolve_model_taxonomy", lambda path: taxonomy)

    old_exists = os.path.exists
    def fake_exists(path):
        if "mock" in str(path) or "checkpoint" in str(path):
            return True
        try:
            return old_exists(path)
        except Exception:
            return False
    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: "mock" in str(self) or "checkpoint" in str(self))

def test_w12c_normalization():
    assert _normalize_sdxl_assembly_posture_value("gpu_text") == "gpu_text"
    assert _normalize_sdxl_assembly_posture_value("GPU-text") == "gpu_text"


def test_w12c_colab_free_ui_default_is_gpu_text_only():
    assert resolve_default_sdxl_assembly_posture(SimpleNamespace(name="colab_free")) == "gpu_text"
    assert resolve_default_sdxl_assembly_posture(SimpleNamespace(name="colab_pro")) == "auto"
    assert resolve_default_sdxl_assembly_posture(SimpleNamespace(name="local_normal")) == "auto"

def test_w12c_eligibility_and_vram_floor(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "gpu_text"

    # Under 10GB VRAM floor must fail closed
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 8.0 * 1024)
    eligible, reason = determine_eligibility(task)
    assert not eligible
    assert "requires at least 10 GB VRAM" in reason

    # At or above 10GB VRAM floor must succeed
    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 12.0 * 1024)
    eligible, reason = determine_eligibility(task)
    assert eligible, f"Failed: {reason}"

def test_w12c_posture_tuple_mapping(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "gpu_text"
    task_dict = {"task_seed": 12345}

    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 12.0 * 1024)
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )
    assert req.unet_posture == UNetPostureKind.RESIDENT
    assert req.clip_posture == TextEncoderPostureKind.GPU_PINNED
    assert req.vae_posture == VAEPostureKind.TRANSIENT
    assert req.lora_posture == LoraPatchPostureKind.RESIDENT

def test_w12c_director_worker_selection(monkeypatch):
    task = TaskState()
    task.base_model_name = "test_model.safetensors"
    task.sdxl_assembly_posture = "gpu_text"
    task_dict = {"task_seed": 12345}

    monkeypatch.setattr("backend.environment_profile.detect_total_vram_mb", lambda: 12.0 * 1024)
    req = build_assembly_request(
        task, task_dict, 0, 1, 30, 0, None, "karras", loras=[]
    )

    # Mock assembler acquire methods to return mocks
    mock_unet = MagicMock()
    mock_vae = MagicMock()
    mock_gpu_lora = MagicMock()

    monkeypatch.setattr(SDXLAssemblyAssembler, "acquire_unet_spine", lambda r, lora_worker: mock_unet)
    monkeypatch.setattr(SDXLAssemblyAssembler, "acquire_vae_decode_worker", lambda r: mock_vae)
    monkeypatch.setattr(SDXLAssemblyAssembler, "acquire_gpu_lora_worker", lambda r: mock_gpu_lora)

    # Select assembly and verify that the lora_worker is used for both UNet and text
    with patch.object(SDXLAssemblyAssembler, "acquire_text_encode_worker") as mock_acquire_text:
        SDXLAssemblyDirector.select_assembly(req)
        mock_acquire_text.assert_called_once()
        args, kwargs = mock_acquire_text.call_args
        # lora_worker must be mock_gpu_lora
        assert kwargs.get("lora_worker") is mock_gpu_lora

def test_gpu_text_encode_worker_conditioning(monkeypatch):
    req = SDXLAssemblyRequest(
        request_id="test_req",
        route_id="txt2img_assembly",
        image_index=0,
        image_count=1,
        checkpoint=_identity("checkpoint.safetensors", "checkpoint_sha"),
        vae=None,
        model_variant_key="sdxl",
        prompt="positive prompt",
        negative_prompt="negative prompt",
        positive_texts=("positive prompt",),
        negative_texts=("negative prompt",),
        width=512,
        height=512,
        steps=30,
        cfg=7.0,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="karras",
        seed=42,
        device="cuda",
        lora_specs=(),
        unet_posture=UNetPostureKind.RESIDENT,
        clip_posture=TextEncoderPostureKind.GPU_PINNED,
        vae_posture=VAEPostureKind.TRANSIENT,
        lora_posture=LoraPatchPostureKind.RESIDENT,
    )

    mock_clip = MagicMock()
    mock_clip.patcher = MagicMock()
    
    # Mock runtime state acquire
    monkeypatch.setattr("backend.sdxl_assembly.gpu_text_encode_worker.acquire_active_gpu_text", lambda r, lora_worker: (mock_clip, False))
    
    # Mock conditioning methods
    mock_cond_pair = {
        "positive": {"cond": torch.zeros((1, 77, 2048)), "pooled": torch.zeros((1, 1280))},
        "negative": {"cond": torch.zeros((1, 77, 2048)), "pooled": torch.zeros((1, 1280))},
    }
    mock_adm_pair = {
        "positive": torch.zeros((1, 6)),
        "negative": torch.zeros((1, 6)),
    }
    
    monkeypatch.setattr("backend.conditioning.encode_prompt_pair_sdxl", lambda *args, **kwargs: mock_cond_pair)
    monkeypatch.setattr("backend.conditioning.build_sdxl_adm_pair", lambda *args, **kwargs: mock_adm_pair)

    worker = GpuTextEncodeWorker(req)
    cond = worker.get_conditioning()

    assert "positive" in cond
    assert "negative" in cond
    assert cond["positive"][0][0] is mock_cond_pair["positive"]["cond"]
    assert cond["positive"][0][1]["pooled_output"] is mock_cond_pair["positive"]["pooled"]

def test_gpu_text_runtime_state_lifecycle(monkeypatch):
    _GPU_TEXT_RUNTIME_STATE.release()
    assert _GPU_TEXT_RUNTIME_STATE.get_active_key() is None

    req = _gpu_request()
    mock_clip = MagicMock()
    mock_clip.patcher = MagicMock()
    mock_clip.patcher.model = MagicMock()
    loader_calls = []
    monkeypatch.setattr(
        "backend.loader.load_sdxl_clip",
        lambda *args, **kwargs: loader_calls.append((args, kwargs)) or mock_clip,
    )

    clip, reused = acquire_active_gpu_text(req)
    assert clip is mock_clip
    assert not reused

    key = _GPU_TEXT_RUNTIME_STATE.get_active_key()
    assert key is not None
    assert key.checkpoint_sha256 == "checkpoint_sha"
    assert key.clip_posture == "gpu_pinned"
    assert loader_calls[0][1]["load_device"] == torch.device("cuda")
    assert loader_calls[0][1]["offload_device"] == torch.device("cuda")
    assert loader_calls[0][1]["dtype"] == torch.float32

    clip2, reused2 = acquire_active_gpu_text(req)
    assert clip2 is mock_clip
    assert reused2

    assert release_active_gpu_text(reason="test")
    assert _GPU_TEXT_RUNTIME_STATE.get_active_key() is None


def test_gpu_text_zero_actual_patch_change_bypasses_reload_and_compile():
    _GPU_TEXT_RUNTIME_STATE.release()
    events = []
    clip, _model = _fake_clip(events)
    old_request = _gpu_request()
    new_request = _gpu_request(lora_specs=(_lora("zero.safetensors"),))
    _GPU_TEXT_RUNTIME_STATE._clip = clip
    _GPU_TEXT_RUNTIME_STATE._key = _GPU_TEXT_RUNTIME_STATE._build_key(old_request)
    _GPU_TEXT_RUNTIME_STATE._actual_patch_count = 0

    class ZeroPatchWorker:
        def resolve_clip_patches(self, _clip):
            events.append("resolve")
            return ()

        def apply_clip_patches(self, _clip, *, resolved_patches):
            events.append("apply")
            assert resolved_patches == ()
            return 0

        def compile_clip_patches(self, _clip):
            events.append("compile")

    acquired, reused = acquire_active_gpu_text(new_request, lora_worker=ZeroPatchWorker())
    assert acquired is clip
    assert reused is False
    assert events == ["resolve", "apply"]
    assert _GPU_TEXT_RUNTIME_STATE.get_actual_patch_count() == 0
    _GPU_TEXT_RUNTIME_STATE.release()


def test_gpu_text_resolves_before_checkpoint_reload_on_patch_removal():
    _GPU_TEXT_RUNTIME_STATE.release()
    events = []
    clip, _model = _fake_clip(events)
    old_request = _gpu_request(lora_specs=(_lora("active.safetensors"),))
    new_request = _gpu_request()
    _GPU_TEXT_RUNTIME_STATE._clip = clip
    _GPU_TEXT_RUNTIME_STATE._key = _GPU_TEXT_RUNTIME_STATE._build_key(old_request)
    _GPU_TEXT_RUNTIME_STATE._actual_patch_count = 3

    class RemovalWorker:
        def resolve_clip_patches(self, _clip):
            events.append("resolve")
            return ()

        def apply_clip_patches(self, _clip, *, resolved_patches):
            events.append("apply")
            return 0

    acquired, reused = acquire_active_gpu_text(new_request, lora_worker=RemovalWorker())
    assert acquired is clip
    assert reused is False
    assert events == ["resolve", "reload", "apply"]
    assert _GPU_TEXT_RUNTIME_STATE.get_actual_patch_count() == 0
    _GPU_TEXT_RUNTIME_STATE.release()


def test_gpu_text_compile_failure_drops_owner_without_cpu_shadow():
    _GPU_TEXT_RUNTIME_STATE.release()
    events = []
    clip, model = _fake_clip(events)
    old_request = _gpu_request(lora_specs=(_lora("old.safetensors"),))
    new_request = _gpu_request(lora_specs=(_lora("new.safetensors"),))
    _GPU_TEXT_RUNTIME_STATE._clip = clip
    _GPU_TEXT_RUNTIME_STATE._key = _GPU_TEXT_RUNTIME_STATE._build_key(old_request)
    _GPU_TEXT_RUNTIME_STATE._actual_patch_count = 2

    class FailingWorker:
        def resolve_clip_patches(self, _clip):
            events.append("resolve")
            return (({"clip.key": object()}, 1.0),)

        def apply_clip_patches(self, _clip, *, resolved_patches):
            events.append("apply")
            return 1

        def compile_clip_patches(self, _clip):
            events.append("compile")
            raise torch.cuda.OutOfMemoryError("simulated")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
        acquire_active_gpu_text(new_request, lora_worker=FailingWorker())

    assert events == ["resolve", "reload", "apply", "compile"]
    assert _GPU_TEXT_RUNTIME_STATE.get_active_key() is None
    assert _GPU_TEXT_RUNTIME_STATE.get_actual_patch_count() == 0
    model.cpu.assert_not_called()


def test_gpu_text_and_unet_lora_signatures_invalidate_independently():
    from backend.sdxl_assembly.runtime_state import SDXLResidentRuntimeState

    clip_a = _gpu_request(lora_specs=(_lora("one.safetensors", unet_weight=0.0, clip_weight=0.5),))
    clip_b = _gpu_request(lora_specs=(_lora("one.safetensors", unet_weight=0.0, clip_weight=0.8),))
    unet_only = _gpu_request(lora_specs=(_lora("one.safetensors", unet_weight=1.0, clip_weight=0.5),))

    assert _GPU_TEXT_RUNTIME_STATE._build_key(clip_a) != _GPU_TEXT_RUNTIME_STATE._build_key(clip_b)
    assert _GPU_TEXT_RUNTIME_STATE._build_key(clip_a) == _GPU_TEXT_RUNTIME_STATE._build_key(unet_only)
    assert SDXLResidentRuntimeState._build_key(clip_a) == SDXLResidentRuntimeState._build_key(clip_b)
    assert SDXLResidentRuntimeState._build_key(clip_a) != SDXLResidentRuntimeState._build_key(unet_only)


def test_gpu_text_process_identity_comes_from_assembly_selector(monkeypatch):
    base_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        execution_family="standard_sdxl",
        residency_class="full_resident",
        route_family="sdxl",
    )
    monkeypatch.setattr(
        "modules.pipeline.inference.resolve_unified_sdxl_process_key",
        lambda *args, **kwargs: base_key,
    )
    task = SimpleNamespace(
        sdxl_assembly_posture="gpu_text",
        loras=[],
        base_model_additional_loras=[],
    )

    gpu_key = process_transition.resolve_sdxl_process_key(task)
    assert gpu_key.residency_class == "resident_unet_gpu_text"
    changes = process_transition.classify_sdxl_process_key_changes(base_key, gpu_key)
    assert [change.value for change in changes] == ["spine_posture_change"]


def test_gpu_text_posture_change_precedes_simultaneous_lora_reuse():
    registry = process_transition.SharedProcessRegistry()
    gpu_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip", "lora-a"),
        residency_class="resident_unet_gpu_text",
    )
    legacy_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip", "lora-b"),
        residency_class="full_resident",
    )

    registry.set_active_key(gpu_key)
    decision = registry.evaluate_transition(legacy_key)

    assert decision.reset_required is True
    assert decision.reason == "residency_class_change"


def test_inactive_outpaint_controlnet_slots_are_discarded():
    from modules import flags
    from modules.async_worker import AsyncTask

    queued = AsyncTask({
        "input_image_checkbox": True,
        "current_tab": "outpaint",
        "requested_route_id": "outpaint",
        "requested_route_family": "image_input",
        "outpaint_input_image": object(),
        "mixing_image_prompt_and_outpaint": False,
        "cn_0_image": object(),
        "cn_0_type": flags.cn_cpds,
    })

    assert all(not tasks for tasks in queued.state.cn_tasks.values())


def test_gpu_text_legacy_bypass_releases_before_legacy_load(monkeypatch):
    from backend.sdxl_assembly import gateway
    from backend.sdxl_assembly import runtime_state
    from modules.pipeline import inference

    gpu_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        execution_family="standard_sdxl",
        residency_class="resident_unet_gpu_text",
        route_family="sdxl",
    )
    legacy_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        execution_family="standard_sdxl",
        residency_class="full_resident",
        route_family="sdxl",
    )
    decision = process_transition.ProcessTransitionDecision(
        action="reset",
        reason="process_posture_change",
        reset_required=True,
        current_key=gpu_key,
        requested_key=legacy_key,
    )
    events = []

    monkeypatch.setattr(gateway, "is_eligible_for_sdxl_assembly", lambda **kwargs: (False, "forced bypass"))
    monkeypatch.setattr(process_transition, "get_active_process_key", lambda: gpu_key)
    monkeypatch.setattr(inference, "resolve_unified_sdxl_process_key", lambda *args, **kwargs: legacy_key)
    monkeypatch.setattr(
        process_transition,
        "apply_process_transition_gate",
        lambda key: events.append(("release", key)) or decision,
    )
    monkeypatch.setattr(runtime_state, "get_active_sdxl_resident_spine_key", lambda: None)
    monkeypatch.setattr(runtime_state, "get_active_gpu_text_key", lambda: None)
    monkeypatch.setattr(inference, "_ensure_supported_unified_runtime_request", lambda task: None)
    monkeypatch.setattr(
        inference,
        "_run_unified_sdxl_task",
        lambda *args, **kwargs: events.append(("legacy_load", None)) or ["image"],
    )
    monkeypatch.setattr(inference, "save_and_log", lambda *args, **kwargs: ["saved-path"])

    task = SimpleNamespace(
        last_stop=False,
        steps=1,
        height=64,
        width=64,
        use_expansion=False,
    )
    bind_legacy_workflow_plan(task)
    inference.process_task(
        task_state=task,
        task_dict={"task_seed": 1},
        current_task_id=0,
        total_count=1,
        all_steps=1,
        preparation_steps=0,
        denoising_strength=None,
        final_scheduler_name="karras",
        loras=[],
    )

    assert [event[0] for event in events] == ["release", "legacy_load"]
    released_key = events[0][1]
    assert released_key.family == legacy_key.family
    assert released_key.process_class == legacy_key.process_class
    assert released_key.authoritative_identity == legacy_key.authoritative_identity
    assert released_key.residency_class == legacy_key.residency_class
    assert released_key.composition_identity == task.workflow_plan.identity()


def test_gpu_text_legacy_bypass_fails_closed_if_owner_remains(monkeypatch):
    from backend.sdxl_assembly import runtime_state
    from modules.pipeline import inference

    gpu_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        residency_class="resident_unet_gpu_text",
    )
    legacy_key = process_transition.ProcessKey(
        family="sdxl",
        process_class="standard_sdxl",
        authoritative_identity=("checkpoint", "clip"),
        residency_class="full_resident",
    )
    decision = process_transition.ProcessTransitionDecision(
        action="reset",
        reason="process_posture_change",
        reset_required=True,
        current_key=gpu_key,
        requested_key=legacy_key,
    )

    monkeypatch.setattr(process_transition, "get_active_process_key", lambda: gpu_key)
    monkeypatch.setattr(inference, "resolve_unified_sdxl_process_key", lambda *args, **kwargs: legacy_key)
    monkeypatch.setattr(process_transition, "apply_process_transition_gate", lambda key: decision)
    monkeypatch.setattr(runtime_state, "get_active_sdxl_resident_spine_key", lambda: object())
    monkeypatch.setattr(runtime_state, "get_active_gpu_text_key", lambda: None)

    with pytest.raises(RuntimeError, match="resident_unet"):
        task = SimpleNamespace()
        bind_legacy_workflow_plan(task)
        inference._prepare_gpu_text_legacy_bypass_transition(
            task,
            loras=[],
        )


def test_gpu_clip_patch_count_reports_only_keys_accepted_by_patcher():
    request = _gpu_request(lora_specs=(_lora("mixed.safetensors"),))
    worker = GpuLoraWorker(request)
    patcher = MagicMock()
    patcher.add_patches.return_value = ["accepted.key"]
    clip = SimpleNamespace(patcher=patcher)

    count = worker.apply_clip_patches(
        clip,
        resolved_patches=(({"accepted.key": object(), "missing.key": object()}, 0.75),),
    )

    assert count == 1
    assert worker.clip_patch_count == 1


def test_gpu_text_lifecycle_retains_only_for_lora_reconfiguration(monkeypatch):
    from backend.sdxl_assembly.lifecycle_coordinator import LifecycleChange, release_for_changes
    from backend.sdxl_assembly import runtime_state

    gpu_releases = []
    monkeypatch.setattr(runtime_state, "release_active_sdxl_streaming_spine", lambda **kwargs: False)
    monkeypatch.setattr(runtime_state, "release_active_sdxl_resident_spine", lambda **kwargs: False)
    monkeypatch.setattr(runtime_state, "release_text_encoder_component_cache", lambda **kwargs: None)
    monkeypatch.setattr(runtime_state, "release_prompt_conditioning_cache", lambda **kwargs: None)
    monkeypatch.setattr(
        runtime_state,
        "release_active_gpu_text",
        lambda **kwargs: gpu_releases.append(kwargs.get("reason")) or True,
    )

    release_for_changes([LifecycleChange.LORA_STACK_CHANGE], reason="lora")
    assert gpu_releases == []

    release_for_changes([LifecycleChange.CHECKPOINT_CHANGE], reason="checkpoint")
    assert gpu_releases == ["checkpoint"]
