from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend import resources
from backend.sdxl_assembly.contracts import SDXLAssemblyEligibilityError, SDXLAssemblyResult, SDXLAssemblyRequest
from backend.sdxl_assembly.request_builder import determine_eligibility, build_assembly_request
from backend.sdxl_assembly.director import SDXLAssemblyDirector
from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback
from backend.sdxl_assembly.lifecycle_coordinator import release_for_changes, LifecycleChange
from backend.sdxl_assembly.runtime_state import _clip_lora_signature, _unet_lora_signature
from modules.pipeline.workflow_contracts import require_workflow_plan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GatewayRequestState:
    checkpoint_sha256: str
    vae_sha256: str | None
    unet_posture: str
    text_encoder_posture: str
    vae_posture: str
    lora_posture: str
    lora_stack_hash: str
    unet_lora_signature: tuple[tuple[str, float], ...]
    clip_lora_signature: tuple[tuple[str, float], ...]
    prompt_payload_hash: str
    spatial_signature: Any
    structural_signature: tuple[Any, ...]
    contextual_signature: tuple[Any, ...]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _resolved_file_label(file_identity: Any) -> str:
    path = getattr(file_identity, "path", None)
    if path is not None:
        try:
            return path.name
        except Exception:
            return str(path)
    return str(getattr(file_identity, "sha256", "unknown"))


def _summarize_additional_unet_only_loras(lora_specs: Any) -> list[str]:
    summary: list[str] = []
    for spec in tuple(lora_specs or ()):
        if not bool(getattr(spec, "enabled", True)):
            continue
        # Effective clip_weight=0 is not sufficient to identify an
        # additional/inpaint LoRA: recognized user-selected UNet-only assets
        # intentionally have the same effective channel shape. Preserve the
        # frozen request provenance instead of reconstructing it from weights.
        if getattr(spec, "provenance", None) != "additional":
            continue
        unet_weight = float(getattr(spec, "unet_weight", 0.0) or 0.0)
        clip_weight = float(getattr(spec, "clip_weight", 0.0) or 0.0)
        if unet_weight == 0.0 or clip_weight != 0.0:
            continue
        summary.append(
            f"{_resolved_file_label(getattr(spec, 'file_identity', None))}@{unet_weight:g}"
        )
    return summary


def _spatial_context_signature(spatial_context: Any) -> Any:
    if spatial_context is None:
        return None

    source_image = getattr(spatial_context, "source_image", None)
    source_mask = getattr(spatial_context, "source_mask", None)
    pre_bb_image = getattr(spatial_context, "pre_bb_image", None)
    pre_bb_mask = getattr(spatial_context, "pre_bb_mask", None)
    pre_blend_mask = getattr(spatial_context, "pre_blend_mask", None)

    return (
        str(getattr(spatial_context, "mode", "") or ""),
        getattr(source_image, "fingerprint", None),
        getattr(source_mask, "fingerprint", None),
        int(getattr(spatial_context, "target_width", 0) or 0),
        int(getattr(spatial_context, "target_height", 0) or 0),
        getattr(spatial_context, "denoise_strength", None),
        _freeze_value(getattr(spatial_context, "bbox", None)),
        float(getattr(spatial_context, "bbox_area_ratio", 1.0) or 1.0),
        getattr(pre_bb_image, "fingerprint", None),
        getattr(pre_bb_mask, "fingerprint", None),
        getattr(pre_blend_mask, "fingerprint", None),
        str(getattr(spatial_context, "outpaint_direction", "") or ""),
        int(getattr(spatial_context, "outpaint_expansion_size", 0) or 0),
        bool(getattr(spatial_context, "outpaint_pixelate", False)),
    )


def _structural_control_signature(control: Any) -> Any:
    slot_idx = getattr(control, "slot_index", None)
    slot_idx = -1 if slot_idx is None else int(slot_idx)
    return (
        slot_idx,
        str(getattr(control, "control_type", "") or ""),
        getattr(control, "image_fingerprint", None),
        str(getattr(control, "preprocessor_id", "") or ""),
        str(getattr(control, "preprocessor_path", "") or ""),
        _freeze_value(getattr(control, "preprocessor_params", {})),
        int(getattr(control, "target_width", 0) or 0),
        int(getattr(control, "target_height", 0) or 0),
        str(getattr(control, "checkpoint_path", "") or ""),
        str(getattr(control, "checkpoint_sha256", "") or ""),
        str(getattr(control, "checkpoint_type", "") or ""),
        _freeze_value(getattr(control, "unsupported_mode_errors", ())),
        _freeze_value(getattr(control, "extra_params", {})),
    )


def _contextual_control_signature(control: Any) -> Any:
    ui_slot_idx = getattr(control, "ui_slot_index", None)
    ui_slot_idx = -1 if ui_slot_idx is None else int(ui_slot_idx)
    return (
        ui_slot_idx,
        str(getattr(control, "control_type", "") or ""),
        getattr(control, "image_fingerprint", None),
        str(getattr(control, "source_image_role", "") or ""),
        str(getattr(control, "model_path", "") or ""),
        str(getattr(control, "model_sha256", "") or ""),
        str(getattr(control, "clip_vision_path", "") or ""),
        str(getattr(control, "clip_vision_sha256", "") or ""),
        str(getattr(control, "ip_negative_path", "") or ""),
        str(getattr(control, "ip_negative_sha256", "") or ""),
        str(getattr(control, "eva_clip_path", "") or ""),
        str(getattr(control, "eva_clip_sha256", "") or ""),
        _freeze_value(getattr(control, "insightface_model_names", ())),
        _freeze_value(getattr(control, "preprocess_params", {})),
        _freeze_value(getattr(control, "unsupported_mode_errors", ())),
    )


def _build_gateway_request_state(request: SDXLAssemblyRequest) -> _GatewayRequestState:
    structural_signature = tuple(
        sorted(
            (_structural_control_signature(control) for control in request.structural_controls),
            key=lambda item: item[0],
        )
    )
    contextual_signature = tuple(
        sorted(
            (_contextual_control_signature(control) for control in request.contextual_controls),
            key=lambda item: item[0],
        )
    )
    return _GatewayRequestState(
        checkpoint_sha256=str(request.checkpoint.sha256 or ""),
        vae_sha256=request.vae.sha256 if request.vae is not None else None,
        unet_posture=str(request.unet_posture.value),
        text_encoder_posture=str(request.clip_posture.value),
        vae_posture=str(request.vae_posture.value),
        lora_posture=str(request.lora_posture.value),
        lora_stack_hash=str(request.lora_stack_hash or ""),
        unet_lora_signature=_unet_lora_signature(request),
        clip_lora_signature=_clip_lora_signature(request),
        prompt_payload_hash=str(request.prompt_payload_hash or ""),
        spatial_signature=_spatial_context_signature(request.spatial_context),
        structural_signature=structural_signature,
        contextual_signature=contextual_signature,
    )


def _calculate_gateway_changes(
    previous_state: _GatewayRequestState,
    request_state: _GatewayRequestState,
) -> list[LifecycleChange]:
    changes: list[LifecycleChange] = []

    def add(change: LifecycleChange) -> None:
        if change not in changes:
            changes.append(change)

    if previous_state.checkpoint_sha256 != request_state.checkpoint_sha256:
        add(LifecycleChange.CHECKPOINT_CHANGE)
    if previous_state.vae_sha256 != request_state.vae_sha256:
        add(LifecycleChange.SPATIAL_VAE_CHANGE)
    if (
        previous_state.unet_posture != request_state.unet_posture
        or previous_state.lora_posture != request_state.lora_posture
    ):
        add(LifecycleChange.SPINE_POSTURE_CHANGE)
    if previous_state.text_encoder_posture != request_state.text_encoder_posture:
        add(LifecycleChange.TEXT_ENCODER_POSTURE_CHANGE)
    if previous_state.vae_posture != request_state.vae_posture:
        add(LifecycleChange.SPATIAL_VAE_CHANGE)
    unet_lora_changed = (
        previous_state.unet_lora_signature != request_state.unet_lora_signature
    )
    clip_lora_changed = (
        previous_state.clip_lora_signature != request_state.clip_lora_signature
    )
    # The resident-spine owner handles effective UNet LoRA changes during
    # acquire.  Only an effective CLIP change invalidates prompt/text-owned
    # state here; otherwise adding a recognized UNet-only LoRA needlessly
    # releases and recompiles the unchanged CPU text encoder. Keep the hash
    # fallback for synthetic/legacy requests with no effective LoRA entries.
    if clip_lora_changed or (
        previous_state.lora_stack_hash != request_state.lora_stack_hash
        and not unet_lora_changed
        and not previous_state.unet_lora_signature
        and not request_state.unet_lora_signature
        and not previous_state.clip_lora_signature
        and not request_state.clip_lora_signature
    ):
        add(LifecycleChange.LORA_STACK_CHANGE)
    if previous_state.prompt_payload_hash != request_state.prompt_payload_hash:
        add(LifecycleChange.PROMPT_CHANGE)
    if previous_state.spatial_signature != request_state.spatial_signature:
        add(LifecycleChange.SPATIAL_VAE_CHANGE)
    if previous_state.structural_signature != request_state.structural_signature:
        add(LifecycleChange.STRUCTURAL_CN_CHANGE)
    if previous_state.contextual_signature != request_state.contextual_signature:
        add(LifecycleChange.CONTEXTUAL_CN_CHANGE)

    return changes

def is_eligible_for_sdxl_assembly(
    task_state: Any,
    loras: List[Tuple[str, float]],
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Gateway check to determine if a request should go to the new assembly lane or old path."""
    try:
        workflow_plan = require_workflow_plan(task_state)
    except (RuntimeError, ValueError) as exc:
        return False, f"Invalid frozen workflow plan: {exc}"
    return determine_eligibility(
        task_state=task_state,
        loras=loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
        workflow_plan=workflow_plan,
        allow_legacy_adapter=False,
    )

_LAST_REQUEST_STATE: Optional[_GatewayRequestState] = None

def clear_gateway_state() -> None:
    """Clear last request tracker to prevent stale request comparisons."""
    global _LAST_REQUEST_STATE
    _LAST_REQUEST_STATE = None

def run_sdxl_assembly_task(
    task_state: Any,
    task_dict: Dict[str, Any],
    current_task_id: int,
    total_count: int,
    all_steps: int,
    preparation_steps: int,
    denoising_strength: Optional[float],
    final_scheduler_name: str,
    *,
    loras: List[Tuple[str, float]],
    controlnet_paths: Optional[Dict[str, str]] = None,
    contextual_assets: Optional[Dict[str, Any]] = None,
    base_model_additional_loras: Optional[List[Tuple[str, float]]] = None,
    image_input_result: Optional[Dict[str, Any]] = None,
    progressbar_callback: Optional[Any] = None,
    status_callback: Optional[Any] = None,
    preview_runtime_holder: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Gateway entry point that executes an eligible task via the SDXL Assembly lane."""
    import time
    import psutil
    import torch

    workflow_plan = require_workflow_plan(task_state)

    # 1. Build frozen request
    try:
        request = build_assembly_request(
            task_state=task_state,
            task_dict=task_dict,
            current_task_id=current_task_id,
            total_count=total_count,
            all_steps=all_steps,
            preparation_steps=preparation_steps,
            denoising_strength=denoising_strength,
            final_scheduler_name=final_scheduler_name,
            loras=loras,
            controlnet_paths=controlnet_paths,
            contextual_assets=contextual_assets,
            base_model_additional_loras=base_model_additional_loras,
            image_input_result=image_input_result,
            workflow_plan=workflow_plan,
            allow_legacy_adapter=False,
        )
    except resources.InterruptProcessingException:
        raise
    except Exception as e:
        logger.error(f"[SDXL Assembly] Request building/freeze failed: {e}")
        raise RuntimeError(f"Request building/freeze failed: {e}") from e

    # Telemetry envelope begin
    process = psutil.Process()
    rss_start = process.memory_info().rss / (1024 * 1024)
    cuda_alloc_start = 0.0
    cuda_reserved_start = 0.0
    if torch.cuda.is_available():
        try:
            cuda_alloc_start = torch.cuda.memory_allocated() / (1024 * 1024)
            cuda_reserved_start = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    composition = getattr(task_state, 'sdxl_assembly_posture', 'auto')
    req_id = getattr(request, 'request_id', f"req_{current_task_id}_{int(time.time())}")
    unet_p = getattr(getattr(request, 'unet_posture', None), 'value', 'unknown')
    clip_p = getattr(getattr(request, 'clip_posture', None), 'value', 'unknown')
    vae_p = getattr(getattr(request, 'vae_posture', None), 'value', 'unknown')
    lora_p = getattr(getattr(request, 'lora_posture', None), 'value', 'unknown')
    route_id = getattr(request, 'route_id', 'unknown')
    seed = getattr(request, 'seed', 'unknown')
    width = getattr(request, 'width', 'unknown')
    height = getattr(request, 'height', 'unknown')
    steps = getattr(request, 'steps', 'unknown')

    ckpt_name = 'unknown'
    ckpt = getattr(request, 'checkpoint', None)
    if ckpt is not None:
        ckpt_path = getattr(ckpt, 'path', None)
        if ckpt_path is not None:
            ckpt_name = ckpt_path.name
        else:
            ckpt_name = getattr(ckpt, 'sha256', 'unknown')

    lora_specs = getattr(request, 'lora_specs', [])
    unet_lora_count = sum(1 for spec in lora_specs if getattr(spec, 'unet_weight', 0.0) != 0.0)
    clip_lora_count = sum(1 for spec in lora_specs if getattr(spec, 'clip_weight', 0.0) != 0.0)
    additional_unet_only_loras = _summarize_additional_unet_only_loras(lora_specs)

    cn_count = len(getattr(request, 'structural_controls', ())) + len(getattr(request, 'contextual_controls', ()))

    logger.debug(
        f"[SDXL LORA ADMISSION] Route/Workflow: {route_id} | "
        f"Additional UNet-only LoRAs ({len(additional_unet_only_loras)}): {additional_unet_only_loras}"
    )

    print(
        f"[Nex] Inference: {route_id} | Checkpoint: {ckpt_name} | Seed: {seed} | "
        f"Dims: {width}x{height} | Steps: {steps}"
    )
    logger.debug(
        f"[SDXL RUN BEGIN] Correlation ID: {req_id} | "
        f"Composition: {composition} | "
        f"Postures: unet={unet_p}, clip={clip_p}, vae={vae_p}, lora={lora_p} | "
        f"Route/Workflow: {route_id} | Seed: {seed} | "
        f"Dims: {width}x{height} | Steps: {steps} | "
        f"Checkpoint: {ckpt_name} | "
        f"LoRAs: UNet={unet_lora_count}, CLIP={clip_lora_count} | "
        f"ControlNets: {cn_count} | "
        f"Memory: RSS={rss_start:.1f}MB, CUDA_Alloc={cuda_alloc_start:.1f}MB, CUDA_Res={cuda_reserved_start:.1f}MB"
    )

    start_time = time.perf_counter()
    status = "SUCCESS"
    err_msg = ""
    result = None
    assembly = None

    try:
        # 1.5. Detect changes from last request and trigger domain releases
        global _LAST_REQUEST_STATE
        request_state = _build_gateway_request_state(request)
        if _LAST_REQUEST_STATE is not None:
            changes = _calculate_gateway_changes(_LAST_REQUEST_STATE, request_state)
            if changes:
                release_for_changes(changes, reason="gateway_transition")

        _LAST_REQUEST_STATE = request_state

        # 2. Select assembly
        assembly = SDXLAssemblyDirector.select_assembly(
            request,
            status_callback=status_callback,
            progress_state=task_state,
        )

        if preview_runtime_holder is not None:
            preview_runtime_holder["assembly"] = assembly

        # 3. Create progress callback
        progress_cb = SDXLAssemblyProgressCallback(
            request,
            progressbar_callback,
            progress_state=task_state,
        )

        # 4. Execute
        result = assembly.execute(request, callback=progress_cb)
        return result.output_image

    except resources.InterruptProcessingException as exc:
        status = "INTERRUPT"
        raise exc
    except Exception as exc:
        status = "FAILURE"
        err_msg = str(exc)
        raise exc
    finally:
        # Telemetry envelope end
        if assembly is not None:
            try:
                assembly.close()
            except Exception:
                pass
        if preview_runtime_holder is not None:
            preview_runtime_holder["assembly"] = None

        duration = time.perf_counter() - start_time
        rss_end = process.memory_info().rss / (1024 * 1024)
        cuda_alloc_end = 0.0
        cuda_reserved_end = 0.0
        cuda_peak_end = 0.0
        if torch.cuda.is_available():
            try:
                cuda_alloc_end = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_reserved_end = torch.cuda.memory_reserved() / (1024 * 1024)
                cuda_peak_end = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                pass

        from backend.sdxl_assembly.runtime_state import debug_component_cache_report
        cache_report = debug_component_cache_report()
        resident_retained = "retained" if cache_report.get("active_resident_spine") else "released/none"
        gpu_text_retained = "retained" if cache_report.get("active_gpu_text") else "released/none"
        unet_bytes = cache_report.get("resident_unet_model_bytes", 0)
        gpu_text_bytes = cache_report.get("gpu_text_model_bytes", 0)
        gpu_text_patches = cache_report.get("gpu_text_actual_patch_count", 0)
        cpu_text_entries = cache_report.get("cpu_text_component_cache_entries", 0)
        cpu_patched_text = cache_report.get("cpu_patched_text_slot_active", False)
        clean_shadow = cache_report.get("clean_shadow_bytes", 0)

        output_info = ""
        if status == "SUCCESS" and result is not None:
            output_info = f" | Image: {result.output_image.shape[1]}x{result.output_image.shape[0]}"

        err_info = f" | Error: {err_msg}" if err_msg else ""

        logger.debug(
            f"[SDXL RUN END] Correlation ID: {getattr(request, 'request_id', 'unknown')} | "
            f"Status: {status}{err_info} | Duration: {duration:.3f}s{output_info} | "
            f"Resident Spine Retained: {resident_retained} | "
            f"GPU Text Retained: {gpu_text_retained} | "
            f"Resident UNet Bytes: {unet_bytes} | GPU Text Bytes: {gpu_text_bytes} | "
            f"GPU Text Patches: {gpu_text_patches} | CPU Text Cache Entries: {cpu_text_entries} | "
            f"CPU Patched Text Slot: {cpu_patched_text} | Clean Shadow Bytes: {clean_shadow} | "
            f"Memory: RSS={rss_end:.1f}MB, CUDA_Alloc={cuda_alloc_end:.1f}MB, CUDA_Res={cuda_reserved_end:.1f}MB, CUDA_Peak={cuda_peak_end:.1f}MB"
        )
        if status == "SUCCESS":
            print(f"[Nex] Inference complete in {duration:.2f}s{output_info}")
        elif status == "INTERRUPT":
            print(f"[Nex] Inference interrupted after {duration:.2f}s.")
        else:
            compact_error = " ".join(str(err_msg or "Unknown error").split())
            print(
                f"[Nex] Inference failed after {duration:.2f}s | Route: {route_id} | "
                f"Checkpoint: {ckpt_name} | Error: {compact_error}"
            )
